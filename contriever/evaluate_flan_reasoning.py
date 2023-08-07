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

from misc_utils import add_my_args, normalize_answer, compute_f1_score, save_qa_report, save_lm_report
from utils import RetFlan

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
def evaluate_lm(model, opt, step=None):
    output_f = open(opt.reason_output_file, 'a+')
    print('output_file_name', opt.reason_output_file)

    opt.lm_question_mask_token = '<extra_id_0>'
    opt.lm_answer_prefix = '<extra_id_0> '

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(opt.reason_lm, cache_dir='/network/scratch/p/parishad.behnamghader/.cache/')
    flan = AutoModelForSeq2SeqLM.from_pretrained(opt.reason_lm, cache_dir='/network/scratch/p/parishad.behnamghader/.cache/')
    flan = flan.to(opt.reason_device)

    alternative_prediction_num, first_token_prediction_num = 0, 0
    best_alternatives, retrieved_statements, predicted_tokens_list, datas = [], [], [], []
    save_every, total = 1000, 0

    model.eval()
    index = DistributedIndex()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)

    ret_lm = RetFlan(unwrapped_model, model, reader_tokenizer, flan, tokenizer, task=task, index=index, reason_task='lm')
    data_iterator = _get_eval_data_iterator(opt, opt.reason_data_file, task)

    for i, batch in enumerate(data_iterator):
        valid, o = ret_lm.get_answer(batch, opt=opt)
        if not valid:
            continue
        predicted_tokens_list += o['first_token_pred']
        best_alternatives += o['predicted_alt']
        retrieved_statements += o['retrieved']
        datas += o['example']
        first_token_prediction_num += o['hits1']
        for p_alt in o['predicted_alt']:
            alternative_prediction_num += int(p_alt == 0)
        total += len(o['example'])

        if (i + 1) % save_every == 0:
            print('Writing up to sample #{} report...'.format(i))
            save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=output_f)
            datas, retrieved_statements, best_alternatives, predicted_tokens_list = [], [], [], []

    if len(best_alternatives) > 0:
        print('Writing up to the last sample report...')
        save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=output_f)

    output_f.write(json.dumps({"scores": 
                                   {"Target ranking accuracy": "{:.4f}".format(float(alternative_prediction_num) / total),
                                    "Hits@1": "{:.4f}".format(float(first_token_prediction_num) / total)}, 
                                    "# examples": total}) + '\n')
    print(f'\nHits@1: {(1.0 * first_token_prediction_num) / total:.4f}')
    print(f'% Correct Alternative Prediction: {(1.0 * alternative_prediction_num) / total:.4f}')
    output_f.close()


@torch.no_grad()
def evaluate_qa(model, opt, step=None):
    output_f = open(opt.reason_output_file, 'a+')
    print('output_file_name', opt.reason_output_file)

    opt.qa_prompt_format = '{question}'
    opt.qa_answer_format = '{target}'

    p_scores, r_scores, f_scores = [], [], []
    save_every, total = 1000, 0
    datas, retrieved_statements, predicted_tokens_list = [], [], []

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(opt.reason_lm, cache_dir='/network/scratch/p/parishad.behnamghader/.cache/')
    flan = AutoModelForSeq2SeqLM.from_pretrained(opt.reason_lm, cache_dir='/network/scratch/p/parishad.behnamghader/.cache/')
    flan = flan.to(opt.reason_device)

    model.eval()
    index = DistributedIndex()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, opt.reason_data_file, task)
    if not opt.reason_fewshot == 'boolean' and not opt.reason_fewshot == 'short':
        print('--- Not using the fewshot template')
        opt.reason_fewshot = None
    else:
        print('--- Using the fewshot template', opt.reason_fewshot)
    reason_task = 'compare_qa' if opt.reason_dataset == 'strategyqa' and not opt.reason_fewshot else 'qa'
    print('--- task:', reason_task)
    ret_lm = RetFlan(unwrapped_model, model, reader_tokenizer, flan, tokenizer, task=task, index=index, reason_task=reason_task, few_shot=opt.reason_fewshot)
    # print(ret_lm.qa_template)

    for i, batch in enumerate(data_iterator):
        is_valid, o = ret_lm.get_answer(batch, opt=opt)
        if not is_valid:
            continue
        gold = o['example']['answer']
        # print(gold[0], o['gen_ans'])
        p, r, f = compute_f1_score(normalize_answer(o['gen_ans']), normalize_answer(gold[0]))
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)

        retrieved_statements.append(o['retrieved'])
        predicted_tokens_list.append(o['gen_ans'])
        datas.append(o['example'])
        total += 1

        if (i + 1) % save_every == 0:
            print('Writing up to sample #{} report...'.format(i))
            save_qa_report(datas, retrieved_statements, predicted_tokens_list, output_f=output_f)
            datas, retrieved_statements, predicted_tokens_list = [], [], []

    if len(datas) > 0:
        print('Writing up to the last sample report...')
        save_qa_report(datas, retrieved_statements, predicted_tokens_list, output_f=output_f)

    print(f'F1 {np.mean(f_scores):.4f}, Precision {np.mean(p_scores):.4f}, Recall {np.mean(r_scores):.4f}, '
          f'Total number of example {total}')
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
    elif opt.reason_task == 'lm':
        opt.task = 'my_lm'
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
    # print('param#', sum(p.numel() for p in model.reader.parameters() if p.requires_grad))
    # for name, parameter in model.reader.named_parameters():
    #     print(name, parameter.numel())
    # exit()
    logger.info("Start Evaluation")
    dist_utils.barrier()

    assert not opt.retrieve_only

    if opt.task == 'qa' and opt.reason_task == 'qa':
        evaluate_qa(model, opt, step)
    elif opt.task == 'my_lm' and opt.reason_task == 'lm':
        evaluate_lm(model, opt, step)
    else:
        ValueError('Invalid task and my_qa params.')
