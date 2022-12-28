#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob
import json
import logging
import pickle
import time
import zlib
from typing import List, Tuple, Dict, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from .dpr.utils.data_utils import RepTokenSelector
from .dpr.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
from .dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from .dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from .dpr.models import init_biencoder_components
from .dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from .dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from .dpr.utils.data_utils import Tensorizer
from .dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token) for q in batch_questions
                    ]
                else:
                    batch_tensors = [tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions]
            elif isinstance(batch_questions[0], T):
                batch_tensors = [q for q in batch_questions]
            else:
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            # TODO: this only works for Wav2vec pipeline but will crash the regular text pipeline
            # max_vector_len = max(q_t.size(1) for q_t in batch_tensors)
            # min_vector_len = min(q_t.size(1) for q_t in batch_tensors)
            max_vector_len = max(q_t.shape[-1] for q_t in batch_tensors)
            min_vector_len = min(q_t.shape[-1] for q_t in batch_tensors)

            if max_vector_len != min_vector_len:
                # TODO: _pad_to_len move to utils
                from dpr.models.reader import _pad_to_len
                batch_tensors = [_pad_to_len(q.squeeze(0), 0, max_vector_len) for q in batch_tensors]

            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    # TODO Parishad commented
    # logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(self, questions: List[str], query_token: str = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        print('LocalFaissRetriever index', type(self.index))
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results


# works only with our distributed_faiss library
class DenseRPCRetriever(DenseRetriever):
    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index_cfg_path: str,
        dim: int,
        use_l2_conversion: bool = False,
        nprobe: int = 256,
    ):
        from distributed_faiss.client import IndexClient

        super().__init__(question_encoder, batch_size, tensorizer)
        self.dim = dim
        self.index_id = "dr"
        self.nprobe = nprobe
        logger.info("Connecting to index server ...")
        self.index_client = IndexClient(index_cfg_path)
        self.use_l2_conversion = use_l2_conversion
        logger.info("Connected")

    def load_index(self, index_id):
        from distributed_faiss.index_cfg import IndexCfg

        self.index_id = index_id
        logger.info("Loading remote index %s", index_id)
        idx_cfg = IndexCfg()
        idx_cfg.nprobe = self.nprobe
        if self.use_l2_conversion:
            idx_cfg.metric = "l2"

        self.index_client.load_index(self.index_id, cfg=idx_cfg, force_reload=False)
        logger.info("Index loaded")
        self._wait_index_ready(index_id)

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int = 1000,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        from distributed_faiss.index_cfg import IndexCfg

        buffer = []
        idx_cfg = IndexCfg()

        idx_cfg.dim = self.dim
        logger.info("Index train num=%d", idx_cfg.train_num)
        idx_cfg.faiss_factory = "flat"
        index_id = self.index_id
        self.index_client.create_index(index_id, idx_cfg)

        def send_buf_data(buf, index_client):
            buffer_vectors = [np.reshape(encoded_item[1], (1, -1)) for encoded_item in buf]
            buffer_vectors = np.concatenate(buffer_vectors, axis=0)
            meta = [encoded_item[0] for encoded_item in buf]
            index_client.add_index_data(index_id, buffer_vectors, meta)

        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                send_buf_data(buffer, self.index_client)
                buffer = []
        if buffer:
            send_buf_data(buffer, self.index_client)
        logger.info("Embeddings sent.")
        self._wait_index_ready(index_id)

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100, search_batch: int = 512
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :param search_batch:
        :return:
        """
        if self.use_l2_conversion:
            aux_dim = np.zeros(len(query_vectors), dtype="float32")
            query_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
            logger.info("query_hnsw_vectors %s", query_vectors.shape)
            self.index_client.cfg.metric = "l2"

        results = []
        for i in range(0, query_vectors.shape[0], search_batch):
            time0 = time.time()
            query_batch = query_vectors[i : i + search_batch]
            logger.info("query_batch: %s", query_batch.shape)
            # scores, meta = self.index_client.search(query_batch, top_docs, self.index_id)

            scores, meta = self.index_client.search_with_filter(
                query_batch, top_docs, self.index_id, filter_pos=3, filter_value=True
            )

            logger.info("index search time: %f sec.", time.time() - time0)
            results.extend([(meta[q], scores[q]) for q in range(len(scores))])
        return results

    def _wait_index_ready(self, index_id: str):
        from distributed_faiss.index_state import IndexState
        # TODO: move this method into IndexClient class
        while self.index_client.get_state(index_id) != IndexState.TRAINED:
            logger.info("Remote Index is not ready ...")
            time.sleep(60)
        logger.info(
            "Remote Index is ready. Index data size %d",
            self.index_client.get_ntotal(index_id),
        )


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def validate_from_meta(
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    meta_compressed: bool,
) -> List[List[bool]]:

    match_stats = calculate_matches_from_meta(
        answers, result_ctx_ids, workers_num, match_type, use_title=True, meta_compressed=meta_compressed
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        # if questions_extra_attr and questions_extra:
        #    extra = questions_extra[i]
        #    results_item[questions_extra_attr] = extra

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


# TODO: unify with save_results
def save_results_from_meta(
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    rpc_meta_compressed: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [doc for doc in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": docs[c][0],
                    "title": zlib.decompress(docs[c][2]).decode() if rpc_meta_compressed else docs[c][2],
                    "text": zlib.decompress(docs[c][1]).decode() if rpc_meta_compressed else docs[c][1],
                    "is_wiki": docs[c][3],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }
        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc


def validate_tables(
    passages: Dict[object, TableChunk],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_chunked_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_chunk_hits = match_stats.top_k_chunk_hits
    top_k_table_hits = match_stats.top_k_table_hits

    logger.info("Validation results: top k documents hits %s", top_k_chunk_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_chunk_hits]
    logger.info("Validation results: top k table chunk hits accuracy %s", top_k_hits)

    logger.info("Validation results: top k tables hits %s", top_k_table_hits)
    top_k_table_hits = [v / len(result_ctx_ids) for v in top_k_table_hits]
    logger.info("Validation results: top k tables accuracy %s", top_k_table_hits)

    return match_stats.top_k_chunk_hits


def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages


@hydra.main(config_path="../conf/dpr", config_name="dense_retriever", version_base="1.1")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []
    questions_text = []
    question_answers = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    total_queries = len(qa_src)
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)

    logger.info("questions len %d", len(questions))
    logger.info("questions_text len %d", len(questions_text))

    if cfg.rpc_retriever_cfg_file:
        index_buffer_sz = 1000
        retriever = DenseRPCRetriever(
            encoder,
            cfg.batch_size,
            tensorizer,
            cfg.rpc_retriever_cfg_file,
            vector_size,
            use_l2_conversion=cfg.use_l2_conversion,
        )
    else:
        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Local Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        index.init_index(vector_size)
        retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = retriever.generate_question_vectors(questions, query_token=qa_src.special_query_token)
    print('q tensor:', questions_tensor.shape, 'spec token', qa_src.special_query_token)
    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    index_path = cfg.index_path
    if cfg.rpc_retriever_cfg_file and cfg.rpc_index_id:
        retriever.load_index(cfg.rpc_index_id)
    elif index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        # send data for indexing
        id_prefixes = []
        ctx_sources = []
        for ctx_src in cfg.ctx_datatsets:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            logger.info("ctx_sources: %s", type(ctx_src))

        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # index all passages
        ctx_files_patterns = cfg.encoded_ctx_files

        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            assert (
                index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)
        if index_path:
            retriever.index.serialize(index_path)
    # get top k results
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)
    exit()

    if cfg.use_rpc_meta:
        questions_doc_hits = validate_from_meta(
            question_answers,
            top_results_and_scores,
            cfg.validation_workers,
            cfg.match,
            cfg.rpc_meta_compressed,
        )
        if cfg.out_file:
            save_results_from_meta(
                questions,
                question_answers,
                top_results_and_scores,
                questions_doc_hits,
                cfg.out_file,
                cfg.rpc_meta_compressed,
            )
    else:
        all_passages = get_all_passages(ctx_sources)
        if cfg.validate_as_tables:

            questions_doc_hits = validate_tables(
                all_passages,
                question_answers,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        else:
            questions_doc_hits = validate(
                all_passages,
                question_answers,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        if cfg.out_file:
            save_results(
                all_passages,
                questions_text if questions_text else questions,
                question_answers,
                top_results_and_scores,
                questions_doc_hits,
                cfg.out_file,
            )

    if cfg.kilt_out_file:
        kilt_ctx = next(iter([ctx for ctx in ctx_sources if isinstance(ctx, KiltCsvCtxSrc)]), None)
        if not kilt_ctx:
            raise RuntimeError("No Kilt compatible context file provided")
        assert hasattr(cfg, "kilt_out_file")
        kilt_ctx.convert_to_kilt(qa_src.kilt_gold_file, cfg.out_file, cfg.kilt_out_file)


if __name__ == "__main__":
    import os
    os.environ['HYDRA_FULL_ERROR'] = '1'
    main()

# data: data.retriever_results.nq.single.dev, data.retriever.qas.nq-dev, data.retriever.nq-dev
# checkpoint: checkpoint.retriever.single.nq.bert-base-encoder

# python -m DPR.dense_retriever model_file=/network/scratch/p/parishad.behnamghader/FiD/DPR/dpr/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp out_file=my_out_file qa_dataset=nq_dev ctx_datatsets=[dpr_wiki] encoded_ctx_files=[\"/network/scratch/p/parishad.behnamghader/FiD/DPR/outputs/2022-11-02/18-01-20/my_out_file_0\"]
