import json
import logging
import os
import random

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data.dictionary import Dictionary

from misc_utils import add_reason_args, get_hits_at_k, compute_f1_score, save_lm_report, save_qa_report, clean_str
from utils import RetLM, Reason_KNN_Dstore

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.my_lm')


def initialize_seed(i, args):
    args.seed = i
    utils.set_torch_seed(i)
    torch.manual_seed(i)
    random.seed(i)
    np.random.seed(i)
    torch.cuda.manual_seed_all(i)


def test_lm(parsed_args):
    run_id = 0
    initialize_seed(run_id, parsed_args)

    utils.import_user_module(parsed_args)
    task = tasks.setup_task(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu
    if parsed_args.reason_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_f = open(parsed_args.reason_output_file, 'a+')
    print('output_file_name', parsed_args.reason_output_file)
    with open(parsed_args.reason_data_file) as f:
        dataset = json.load(f)

    logger.info('loading model from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    assert len(models) > 0
    logger.info('loading dictionary from {}'.format(parsed_args.reason_dict_dir))
    d = Dictionary()
    d.add_from_file(parsed_args.reason_dict_dir)
    knn_dstore = None
    if args.knnlm:
        knn_dstore = Reason_KNN_Dstore(args)

    ret_lm = RetLM(models, d, cuda=use_cuda, task=args.reason_task)

    logger.info('Start inference')
    hitsk = [1, 5]
    all_valid_samples, all_score_samples, alternative_prediction = 0, 0, 0
    top = {k: 0 for k in hitsk}
    save_every = 1000
    datas, retrieved_statements, predicted_tokens_list, best_alternatives = [], [], [], []
    sent_score = 0

    for sample_id, data in enumerate(dataset):
        query = clean_str(data['query'])
        target_tokens = clean_str(data['target'])
        facts = []
        if args.reason_fact_type == 'facts':
            facts = clean_str(data['facts'])
        elif args.reason_fact_type == 'gold_facts':
            facts = clean_str(data['gold_facts'])
        else:
            ValueError('{} is not a valid fact-type argument.'.format(args.reason_fact_type))
        if len(facts) < 1:
            continue

        facts_info = ret_lm.get_fact_keys_vals(facts, args=args)
        o = ret_lm.get_answer(query, sample_id, target_tokens, args=args, knn_dstore=knn_dstore, facts_info=facts_info)

        logits = o['logits']
        targets = o['target']
        predicted_tokens = torch.topk(logits, dim=-1, k=max(hitsk)).indices
        predicted_tokens_1 = predicted_tokens[:, 0]
        top_ = get_hits_at_k(predicted_tokens, targets, ks=hitsk)
        for k in hitsk:
            top[k] += top_[k]
        all_valid_samples += 1

        datas.append(data)
        retrieved_statements.append(o['retrieved'])
        predicted_tokens_list.append(predicted_tokens_1)

        if o['true_score'] != torch.inf:
            sent_score += o['true_score']
            all_score_samples += 1
        best_alternative = o['predicted_alt']
        if best_alternative == 0:
            alternative_prediction += 1
        best_alternatives.append(best_alternative.item())

        if (sample_id + 1) % save_every == 0:
            print('Writing up to sample #{} report...'.format(sample_id))
            save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=output_f,
                           dic=d)
            datas, retrieved_statements, predicted_tokens_list, best_alternatives = [], [], [], []
    if len(datas) > 0:
        print('Writing up to last sample report...')
        save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=output_f,
                       dic=d)
    top1, top5 = top[1], top[5]
    output_f.write('\nHits@1: {:.4f}, Hits@5: {:.4f}'.format(float(top1) / all_valid_samples,
                                                             float(top5) / all_valid_samples))
    print('\nHits@1: {:.4f}, Hits@5: {:.4f}'.format(float(top1) / all_valid_samples,
                                                    float(top5) / all_valid_samples))
    output_f.write('\n% Correct Alternative Prediction: {}\n'.format(float(alternative_prediction) / all_valid_samples))
    print('% Correct Alternative Prediction: {}'.format(float(alternative_prediction) / all_valid_samples))

    output_f.close()


def test_qa(parsed_args):
    run_id = 0
    initialize_seed(run_id, parsed_args)

    utils.import_user_module(parsed_args)
    task = tasks.setup_task(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu
    if parsed_args.reason_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_f = open(parsed_args.reason_output_file, 'a+')
    print('output_file_name', parsed_args.reason_output_file)
    with open(parsed_args.reason_data_file) as f:
        dataset = json.load(f)

    logger.info('loading model from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    assert len(models) > 0

    logger.info('loading dictionary from {}'.format(parsed_args.reason_dict_dir))
    d = Dictionary()
    d.add_from_file(parsed_args.reason_dict_dir)
    knn_dstore = None
    if args.knnlm:
        knn_dstore = Reason_KNN_Dstore(args)
    if args.reason_dataset == 'strategyqa':
        args.reason_task = 'compare_qa'
    ret_lm = RetLM(models, d, cuda=use_cuda, task=args.reason_task)
    # print('param#', sum(p.numel() for p in models[0].parameters()))

    logger.info('Start inference')
    q_suffix = ' answer'
    all_valid_samples = 0
    save_every = 1000
    datas, queries, retrieved_statements, answers, f_scores, p_scores, r_scores = [], [], [], [], [], [], []

    for sample_id, data in enumerate(dataset):
        query = clean_str(data['question']) + q_suffix
        answer = clean_str(data['answer'])
        facts = []
        if args.reason_fact_type == 'facts':
            facts = clean_str(data['facts'])
        elif args.reason_fact_type == 'gold_facts':
            facts = clean_str(data['gold_facts'])
        elif args.reason_fact_type == 'single_fact':
            if 'hypothesis' not in data:
                ValueError('no single fact is mentioned in sample:\n', data)
            facts = clean_str([data['hypothesis']])
        else:
            ValueError('{} is not a valid fact-type argument.'.format(args.reason_fact_type))

        if args.reason_dataset == 'strategyqa':
            options = answer.copy()
            options.sort()
            options.reverse()
            facts = [f + ' ' + ' / '.join(options) + ' .' for f in facts]
            query += ' [MASK]'
        
        if len(facts) < 1:
            continue

        facts_info = ret_lm.get_fact_keys_vals(facts, args=args)
        o = ret_lm.get_answer(query, answer, sample_id, args=args, knn_dstore=knn_dstore, facts_info=facts_info)

        p, r, f = compute_f1_score(o['answer'], answer[0], d)
        f_scores.append(f)
        p_scores.append(p)
        r_scores.append(r)
        all_valid_samples += 1
        answers.append(o['answer'])
        datas.append(data)
        retrieved_statements.append(o['retrieved'])
        queries.append(o['query'][:-len(q_suffix)])

        if (sample_id + 1) % save_every == 0:
            print('Writing up to sample #{} report...'.format(sample_id))
            # save_qa_report(datas, enriched_queries, answers, output_f)
            save_qa_report(datas, queries, retrieved_statements, answers, output_f=output_f)
            datas, queries, retrieved_statements, answers = [], [], [], []
    if len(datas) > 0:
        print('Writing up to last sample report...')
        save_qa_report(datas, queries, retrieved_statements, answers, output_f=output_f)

    print('F1 {:.4f}, Precision {:.4f}, Recall {:.4f}, Total number of example {}'.format(np.mean(f_scores),
                                                                                          np.mean(p_scores),
                                                                                          np.mean(r_scores),
                                                                                          all_valid_samples))
    output_f.write(
        '\nF1 {:.4f}, Precision {:.4f}, Recall {:.4f}, Total number of example {}\n'.format(np.mean(f_scores),
                                                                                            np.mean(p_scores),
                                                                                            np.mean(r_scores),
                                                                                            all_valid_samples))

    output_f.close()


if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    add_reason_args(parser)
    args = options.parse_args_and_arch(parser)
    print(args)
    if args.reason_task == 'lm':
        test_lm(args)
    elif args.reason_task == 'qa':
        test_qa(args)
    else:
        ValueError('Invalid task. `reason_task` arg should be lm or qa.')
