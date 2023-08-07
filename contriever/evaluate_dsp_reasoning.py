# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from itertools import repeat
import json

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task
from src.index import DistributedIndex

from misc_utils import add_my_args, normalize_answer, compute_f1_score, save_qa_report

import dsp
from dsp_search import Contriever
from dsp_generate import multihop_QA_v2

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator, repeat(opt.reason_fact_type)))
    # data_iterator = filter(None, map(partial(task.process, opt.my_fact_type), data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size))
    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator)).cuda()
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])
    dist.barrier()
    return data_iterator


@torch.no_grad()
def evaluate_qa(model, opt, step=None, write=True):

    output_f = open(opt.reason_output_file, 'a+')
    print('output_file_name', opt.reason_output_file)


    opt.qa_prompt_format = '{question}'
    opt.qa_answer_format = '{target}'

    p_scores, r_scores, f_scores = [], [], []
    save_every, total = 100, 0
    datas, retrieved_statements, predicted_tokens_list, histories = [], [], [], []

    # DSP
    model.eval()
    index = DistributedIndex()
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)

    openai_key = opt.reason_openai_key
    if 'davinci' in opt.reason_lm:
        lm = dsp.GPT3(model=opt.reason_lm, api_key=openai_key)
    else:
        lm = dsp.HFModel(model=opt.reason_lm)
    rm = Contriever(unwrapped_model, index, opt=opt, task=task)
    dsp.settings.configure(lm=lm, rm=rm)
    
    data_iterator = _get_eval_data_iterator(opt, opt.reason_data_file, task)
    if not opt.reason_fewshot == 'boolean' and not opt.reason_fewshot == 'short':
        ValueError('--- Not using the fewshot template')
        opt.reason_fewshot = None
    reason_task = 'compare_qa' if opt.reason_dataset == 'strategyqa' else 'qa'


    for i, batch in enumerate(data_iterator):
        if len(batch["passages"][0]) < 1:
            continue
        query = batch.get("query", [""])
        x = multihop_QA_v2(query[0], batch["passages"][0], infer_k=opt.reason_k, answer_format=opt.reason_fewshot)
        pred = x.answer
        if isinstance(batch["answer"][0], list):
            gold = batch["answer"][0][0]
        else:
            gold = batch["answer"][0]
        p, r, f = compute_f1_score(normalize_answer(pred), normalize_answer(gold))
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)

        retrieved_statements.append(x.retrieveds)
        predicted_tokens_list.append(pred)
        datas.append({'question': query[0], 'answer': [gold]})
        histories.append({
            'h0_rationale': x['h0']['completions'].data[0]['rationale'],
            'h0_query': x['h0']['completions'].data[0]['query'],
            'h1_rationale': x['h1']['completions'].data[0]['rationale'],
            'h1_query': x['h1']['completions'].data[0]['query'],
            'final_rationale': x['completions'].data[0]['rationale']
            })

        total += 1

        if (i + 1) % save_every == 0:
            if write:
                print('Writing up to sample #{} report...'.format(i))
                save_qa_report(datas, retrieved_statements, predicted_tokens_list, output_f=output_f, histories=histories)
            datas, retrieved_statements, predicted_tokens_list, histories = [], [], [], []

    if len(histories) > 0:
        if write:
            print('Writing up to the last sample report...')
            save_qa_report(datas, retrieved_statements, predicted_tokens_list, output_f=output_f, histories=histories)

    if write:
        print(f'F1 {np.mean(f_scores):.4f}, Precision {np.mean(p_scores):.4f}, Recall {np.mean(r_scores):.4f}, Total number of example {total}')
        output_f.write(json.dumps({"scores": 
                                   {"F1": "{:.4f}".format(np.mean(f_scores)), 
                                    "Precision": "{:.4f}".format(np.mean(p_scores)), 
                                    "Recall": "{:.4f}".format(np.mean(r_scores))},
                                    "# examples": total}) + '\n')
        output_f.close()


if __name__ == "__main__":
    options = get_options()
    add_my_args(options)
    opt = options.parse()

    if opt.reason_task == 'qa':
        opt.task = 'qa'
    opt.n_context = opt.reason_k

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)
    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")
    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)
    logger.info("Start Evaluation")
    dist_utils.barrier()

    logger.info(f"Start Evaluation on {opt.reason_data_file}")

    assert not opt.retrieve_only

    if opt.task == 'qa' and opt.reason_task == 'qa':
        evaluate_qa(model, opt, step)
    else:
        ValueError('Invalid task params.')
