from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from .dense_retriever import LocalFaissRetriever
from .dpr.data.biencoder_data import BiEncoderPassage
from .dpr.models import init_biencoder_components
from .dpr.options import set_cfg_params_from_state, setup_cfg_gpu
from .dpr.utils.data_utils import Tensorizer
from .dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)


def gen_ctx_vectors(
        cfg: DictConfig,
        ctx_rows: List[Tuple[object, BiEncoderPassage]],
        model: nn.Module,
        tensorizer: Tensorizer,
        insert_title: bool = False,
) -> List[Tuple[object, np.array]]:
    dpr = cfg.dpr
    n = len(ctx_rows)
    bsz = dpr.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):
        batch = ctx_rows[batch_start: batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None) for ctx in batch
        ]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), dpr.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), dpr.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), dpr.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])
    return results


def retrieve_topk_docs(sample, doc_encoder=None, tensorizer=None, retriever=None, args=None):
    k = args.reason.k
    ctxs = []
    all_docs = []
    if args.reason.fact_type == 'facts':
        all_docs = [{'text': f} for f in sample['facts']].copy()
    elif args.reason.fact_type == 'gold_facts':
        all_docs = [{'text': f} for f in sample['gold_facts']].copy()
    elif args.reason.fact_type == 'single_fact':
        if 'hypothesis' not in sample:
            ValueError('no single fact is mentioned in sample:\n', sample)
        all_docs = [{'text': sample['hypothesis']}].copy()
    else:
        ValueError('{} is not a valid fact-type argument.'.format(args.reason.fact_type))
    if len(all_docs) == 0:
        return None
    sample['all_ctxs'] = all_docs
    for doc_id, doc in enumerate(all_docs):
        ctxs.append(('{}-{}'.format(sample['id'], doc_id), BiEncoderPassage(doc['text'], None)))

    res = gen_ctx_vectors(args, ctxs, doc_encoder, tensorizer, False)
    res = torch.from_numpy(np.concatenate([r[1].reshape(1, r[1].shape[0]) for r in res], axis=0))

    # time to get the best ones
    if args.reason.task == 'lm':
        q = sample['query'].replace('[MASK]', '<extra_id_0>')
        sample['question'] = q
        sample['target'] = ['<extra_id_0> ' + t for t in sample['target']]
    else:
        q = sample['question']
        sample['target'] = sample['answer']

    questions_tensor = retriever.generate_question_vectors([q], query_token=None)

    scores = torch.matmul(questions_tensor, res.T).reshape(-1)
    k = min(k, res.shape[0])
    topk_doc_idx = torch.topk(scores, k=k).indices
    sample['ctxs'] = []
    for i in topk_doc_idx:
        sample['ctxs'].append({'text': all_docs[i]['text'], 'score': scores[i]})
    return sample


def load_encoder_tensorizer(cfg):
    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    ctx_encoder = encoder.ctx_model
    q_encoder = encoder.question_model

    ctx_encoder, _ = setup_for_distributed_mode(
        ctx_encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    ctx_encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(ctx_encoder)
    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False)

    # question encoder
    q_encoder, _ = setup_for_distributed_mode(q_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    q_encoder.eval()

    model_to_load_q = get_model_obj(q_encoder)
    vector_size = model_to_load_q.get_out_size()

    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    retriever = LocalFaissRetriever(q_encoder, cfg.batch_size, tensorizer, index)
    return ctx_encoder, tensorizer, retriever
