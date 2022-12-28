# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import hydra
import numpy as np
import src.data
import src.evaluation
import src.model
import src.slurm
import src.util
import torch
import torch.distributed as dist
import transformers

from DPR.dpr_utils import load_encoder_tensorizer
from lm_utils import RetLM, load_dataloader, MyCollator
from misc_utils import save_lm_report, save_qa_report, compute_f1_score


def test_lm(ret_lm, dataloader, cfg, output_f=None, logger=None):
    lm = cfg.lm

    total = 0
    alternative_prediction_num, first_token_prediction_num = 0, 0
    save_every = 1000
    best_alternatives, predicted_tokens_list, datas, retrieved_statements = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, labels_, _, context_ids, context_mask) = batch
            # context_ids : 1xCx200. C: number of given contexts
            # labels_ : Tx20. true and alternative targets.

            o = ret_lm.do_lm(context_ids, context_mask, labels_, idx, args=cfg)
            predicted_alt = o['predicted_alt']
            example = o['example']

            best_alternatives.append(predicted_alt)
            alternative_prediction_num += int(predicted_alt == 0)
            first_token_prediction_num += o['hits1']

            retrieved_statements.append(o['retrieved'])
            datas.append(example)
            predicted_tokens_list.append(ret_lm.tokenizer.decode(o['first_token_pred']))
            total += 1
        if (i + 1) % save_every == 0:
            print('Writing up to sample #{} report...'.format(i))
            save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=output_f)
            datas, best_alternatives, predicted_alt, retrieved_statements, predicted_tokens_list = [], [], [], [], []

    if len(datas) > 0:
        print('Writing up to the last sample report...')
        save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=output_f)

    if lm.is_distributed:
        torch.distributed.barrier()
    logger.info(f'Hits@1: {(1.0 * first_token_prediction_num) / total:.4f}, Total number of example {total}')
    logger.info(
        f'% Correct Alternative Prediction: {(1.0 * alternative_prediction_num) / total:.4f}, Total number of example {total}')
    output_f.write(f'\nHits@1: {(1.0 * first_token_prediction_num) / total:.4f}')
    output_f.write(f'\n% Correct Alternative Prediction: {(1.0 * alternative_prediction_num) / total:.4f}')
    output_f.close()


def test_qa(ret_lm, dataloader, cfg, output_f=None, logger=None):
    lm = cfg.lm
    reason = cfg.reason
    total = 0
    p_scores, r_scores, f_scores = [], [], []
    save_every = 1000
    retrieved_statements, predicted_tokens_list, datas = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, labels_, _, context_ids, context_mask) = batch
            # context_ids : 1xCx200. C: number of given contexts
            # labels_ : Tx20. true and alternative targets.

            o = ret_lm.do_qa(context_ids, context_mask, idx, args=cfg)
            example = o['example']
            p, r, f1 = compute_f1_score(src.evaluation.normalize_answer(o['gen_ans']),
                                        src.evaluation.normalize_answer(example['target'][0]))
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(f1)

            datas.append(example)
            predicted_tokens_list.append(o['gen_ans'])
            retrieved_statements.append(o['retrieved'])
            total += 1
        if (i + 1) % save_every == 0:
            print('Writing up to sample #{} report...'.format(i))
            save_qa_report(datas, retrieved_statements, predicted_tokens_list, output_f=output_f)
            datas, enriched_queries, predicted_tokens_list = [], [], []

    if len(datas) > 0:
        print('Writing up to the last sample report...')
        save_qa_report(datas, retrieved_statements, predicted_tokens_list, output_f=output_f)

    if lm.is_distributed:
        torch.distributed.barrier()

    logger.info(
        f'F1 {np.mean(f_scores):.4f}, Precision {np.mean(p_scores):.4f}, Recall {np.mean(r_scores):.4f}, Total number of example {total}')
    output_f.write(
        f'\nF1 {np.mean(f_scores):.4f}, Precision {np.mean(p_scores):.4f}, Recall {np.mean(r_scores):.4f}, Total number of example {total}\n')
    output_f.close()


def test(cfg):
    lm = cfg.lm
    reason = cfg.reason
    dpr = cfg.dpr

    doc_encoder, tensorizer, retriever = load_encoder_tensorizer(dpr)
    device = src.slurm.init_distributed_mode(lm)
    src.slurm.init_signal_handler()
    lm.train_batch_size = lm.per_gpu_batch_size * max(1, lm.world_size)

    dir_path = Path(lm.checkpoint_dir) / lm.name
    if lm.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if lm.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(lm.is_main, lm.is_distributed, Path(lm.checkpoint_dir) / lm.name / 'run.log')

    output_f = open(reason.output_file, 'a+')
    print('output_file_name', reason.output_file)
    lm.eval_data = reason.data_file

    tokenizer, model = None, None
    q_prefix = 'question:'
    ctx_prefix = 'context:'
    if reason.lm == 'fid':
        tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        model_class = src.model.FiDT5
        model = model_class.from_pretrained(lm.model_path)
        model = model.to(lm.device)
    elif reason.lm == 'flan':
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model = model.to(lm.device)

        q_prefix = ''
        ctx_prefix = ''
    else:
        ValueError('Invalid lm requested.')

    # print('param#', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for name, parameter in model.named_parameters():
    #     print(name, parameter.numel())

    flan_prompt = ''
    if reason.task == 'qa':
        flan_prompt = 'Answer the following question.\n'
    elif reason.task == 'lm':
        flan_prompt = 'Find <extra_id_0>.\n'
    collator_function = MyCollator(lm.text_maxlength, tokenizer, lm=reason.lm, flan_prompt=flan_prompt)

    eval_dataloader, eval_dataset = load_dataloader(cfg, lm.eval_data, doc_encoder=doc_encoder, tensorizer=tensorizer,
                                                    retriever=retriever, q_prefix=q_prefix, ctx_prefix=ctx_prefix,
                                                    collate_fn=collator_function)
    ret_lm = RetLM(model, tokenizer, eval_dataset, args=cfg)

    logger.info("Start eval")
    if reason.task == 'lm':
        test_lm(ret_lm, eval_dataloader, cfg, output_f=output_f, logger=logger)

    elif reason.task == 'qa':
        test_qa(ret_lm, eval_dataloader, cfg, output_f=output_f, logger=logger)


@hydra.main(config_path="conf", config_name="general_lm_dpr", version_base='1.1')
def main(cfg):
    print('cfg', cfg)
    test(cfg)


if __name__ == "__main__":
    import os

    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main()
