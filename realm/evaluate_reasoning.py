import json
import random

import numpy as np
import torch
import transformers

from misc_utils import arg_parser, compute_f1_score, get_hits_at_k, \
    save_lm_report_alt, save_lm_report_pred, save_qa_report
from utils import RetLM, load_models


def initialize_seed(i):
    torch.manual_seed(i)
    random.seed(i)
    np.random.seed(i)
    torch.cuda.manual_seed_all(i)


def test_qa(args):
    run_id = 0
    initialize_seed(run_id)

    if args.reason_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    transformers.logging.set_verbosity_error()

    models = load_models(args, device=device)
    tokenizer = models['tokenizer']
    qa_model = models['qa_model']
    ret_lm = RetLM(args.reason_task, tokenizer=tokenizer, qa_model=qa_model, device=args.reason_device)

    output_f = open(args.reason_output_file, 'a+')
    print('output_file_name', args.reason_output_file)
    with open(args.reason_data_file) as f:
        dataset = json.load(f)

    all_valid_samples = 0
    datas, queries, retrieved_statements, answers, f_scores, p_scores, r_scores = [], [], [], [], [], [], []

    for sample_id, d in enumerate(dataset):
        query = d['question']
        target = d['answer']
        facts = []
        if args.reason_fact_type == 'facts':
            facts = d['facts']
        elif args.reason_fact_type == 'gold_facts':
            facts = d['gold_facts']
        elif args.reason_fact_type == 'single_fact':
            if 'hypothesis' not in d:
                ValueError('no single fact is mentioned in sample:\n', d)
            facts = [d['hypothesis']]
        else:
            ValueError('{} is not a valid fact-type argument.'.format(args.reason_fact_type))

        if len(facts) < 1:
            continue
        candidates_info = {'text': facts}
        retrieve_k = min(len(facts), args.reason_k)
        o = ret_lm.get_answer(query, candidates_info, retrieve_k)
        p, r, f = compute_f1_score(o['answer'], target[0])
        f_scores.append(f)
        p_scores.append(p)
        r_scores.append(r)

        answers.append(o['answer'])
        datas.append(d)
        queries.append(o['query'])
        retrieved_statements.append(o['retrieved'])
        all_valid_samples += 1

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


def test_lm(args):
    assert args.reason_lm_task in {'alt', 'pred'}

    hitsk = [1, 5]
    run_id = 0
    initialize_seed(run_id)

    if args.reason_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    transformers.logging.set_verbosity_error()

    models = load_models(args, device=device)
    tokenizer = models['tokenizer']
    qa_model = models['qa_model']
    query_embedder = models['query_embedder']
    encoder = models['encoder']
    # print('param #', sum(p.numel() for p in encoder.parameters()))
    ret_lm = RetLM(args.reason_task, lm_task=args.reason_lm_task, tokenizer=tokenizer, qa_model=qa_model,
                   query_embedder=query_embedder, encoder=encoder, device=device)

    output_f = open(args.reason_output_file, 'a+')
    print('output_file_name', args.reason_output_file)
    with open(args.reason_data_file) as f:
        dataset = json.load(f)

    all_valid_samples = 0
    top = {k: 0 for k in hitsk}
    alternative_prediction = 0
    save_every = 1000
    datas, retrieved_statements, predicted_tokens_list, best_alternatives, alternatives = [], [], [], [], []
    for sample_id, d in enumerate(dataset):
        query = d['query']
        targets = d['target']
        facts = []
        if args.reason_fact_type == 'facts':
            facts = d['facts']
        elif args.reason_fact_type == 'gold_facts':
            facts = d['gold_facts']
        elif args.reason_fact_type == 'single_fact':
            if 'hypothesis' not in d:
                ValueError('no single fact is mentioned in sample:\n', d)
            facts = d['hypothesis']
        else:
            ValueError('{} is not a valid fact-type argument.'.format(args.reason_fact_type))

        if len(facts) < 1:
            continue
        candidates_info = {'text': facts}
        retrieve_k = min(len(facts), args.reason_k)

        is_valid, o = ret_lm.get_answer(query, targets, candidates_info, retrieve_k)
        if not is_valid:
            continue
        if args.reason_lm_task == 'alt':
            predicted_alt = o['predicted_alt']
            if predicted_alt == 0:
                alternative_prediction += 1
            all_valid_samples += 1

            datas.append(d)
            retrieved_statements.append(o['retrieved'])
            best_alternatives.append(predicted_alt)

            if (sample_id + 1) % save_every == 0:
                save_lm_report_alt(datas, retrieved_statements, best_alternatives, output_f=output_f)
                datas, retrieved_statements, best_alternatives = [], [], []

        elif args.reason_lm_task == 'pred':
            logits = o['logits']
            target = o['target']
            token_probs = torch.softmax(logits, dim=-1)
            predicted_tokens = torch.topk(token_probs, dim=-1, k=max(hitsk)).indices
            predicted_tokens_1 = predicted_tokens[:, 0]
            top_ = get_hits_at_k(predicted_tokens, target, ks=hitsk)
            for k in hitsk:
                top[k] += top_[k]
            all_valid_samples += 1

            datas.append(d)
            retrieved_statements.append(o['retrieved'])
            predicted_tokens_list.append(predicted_tokens_1)

            if (sample_id + 1) % save_every == 0:
                save_lm_report_pred(datas, retrieved_statements, predicted_tokens_list, output_f=output_f,
                                    tokenizer=tokenizer)
                datas, enriched_queries, predicted_tokens_list = [], [], []
    if args.reason_lm_task == 'pred':
        if len(datas) > 0:
            save_lm_report_pred(datas, retrieved_statements, predicted_tokens_list, output_f=output_f,
                                tokenizer=tokenizer)
        top1, top5 = top[1], top[5]
        output_f.write(
            '\nHits@1: {:.4f}, Hits@5: {:.4f}'.format(float(top1) / all_valid_samples, float(top5) / all_valid_samples))
        print(
            '\nHits@1: {:.4f}, Hits@5: {:.4f}'.format(float(top1) / all_valid_samples, float(top5) / all_valid_samples))
    if args.reason_lm_task == 'alt':
        if len(datas) > 0:
            save_lm_report_alt(datas, retrieved_statements, best_alternatives, output_f=output_f)
        output_f.write(
            '\n% Correct Alternative Prediction: {}\n'.format(float(alternative_prediction) / all_valid_samples))
        print('\n% Correct Alternative Prediction: {}'.format(float(alternative_prediction) / all_valid_samples))
    output_f.close()


if __name__ == '__main__':
    args = arg_parser().parse_args()
    print(args)
    if args.reason_task == 'lm':
        test_lm(args)
    elif args.reason_task == 'qa':
        test_qa(args)
