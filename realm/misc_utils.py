import string
from argparse import ArgumentParser

import regex
import torch


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--reason_device", type=str, default="cuda")
    parser.add_argument("--reason_k", type=int, default=5)
    parser.add_argument("--reason_data_file", type=str, required=True)
    parser.add_argument("--reason_dataset", type=str, required=True)
    parser.add_argument("--reason_output_file", type=str, required=True)
    parser.add_argument("--reason_info", type=str, default="")

    parser.add_argument("--reason_lm_task", type=str, default='target_ranking')  # target_ranking, prediction
    parser.add_argument("--reason_fact_type", type=str, default='facts')  # facts, gold_facts, single_fact
    parser.add_argument("--reason_task", type=str, default='lm')  # lm, qa

    return parser


def save_lm_report_prediction(datas, retrieved_statements, predicted_tokens_list, output_f=None, tokenizer=None):
    for i, d in enumerate(datas):
        output_f.write('Query: {}\n'.format(d['query']))
        output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
        output_f.write('Expected: {}\n'.format(d['target'][0]))
        output_f.write('Generated: {}\n'.format(tokenizer.decode([predicted_tokens_list[i][0]])))

        output_f.write('\n')


def save_lm_report_target_ranking(datas, retrieved_statements, predicted_alt, output_f=None):
    for i, d in enumerate(datas):
        output_f.write('Query: {}\n'.format(d['query']))
        output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
        output_f.write('Alternatives: {}\n'.format(d['target']))
        output_f.write(
            '{} Preferred: {}\n'.format('+' if predicted_alt[i] == 0 else '-', d['target'][predicted_alt[i]]))

        output_f.write('\n')


def save_qa_report(datas, queries, retrieved_statements, predicted_ans_list, output_f=None):
    for i, data in enumerate(datas):
        output_f.write('Query: {}\n'.format(queries[i]))
        output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
        output_f.write('Expected: {}\n'.format(data['answer'][0]))
        output_f.write('Generated: {}\n'.format(predicted_ans_list[i]))

        output_f.write('\n')


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1_score(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return prec, rec, 2 * (prec * rec) / (prec + rec)


def get_hits_at_k(predicted_tokens, target, ks=[1, 5]):
    target = target[0]
    predicteds_ = {}
    most_predicted_counts = {}
    top = {k: 0 for k in ks}
    tokens_, counts_ = {}, {}
    most_predicted = {}

    predicted_tokens_1 = predicted_tokens[:, 0]
    for k in ks:
        predicteds_[k] = []
    for k in ks:
        if k == 1:
            predicteds_[k].append(predicted_tokens_1[0].item())
        else:
            predicteds_[k] += predicted_tokens[0].tolist()
    for k in ks:
        tokens_[k], counts_[k] = torch.unique(torch.tensor(predicteds_[k]), return_counts=True)
        predicteds_[k] = sorted(list(zip(counts_[k].tolist(), tokens_[k].tolist())))
        most_predicted_counts[k] = set([x[0] for x in predicteds_[k][-k:]])
        most_predicted[k] = []
        for idx in range(len(predicteds_[k])):
            if predicteds_[k][idx][0] in most_predicted_counts[k]:
                most_predicted[k].append(predicteds_[k][idx][1])
        for trg in target:
            if trg in most_predicted[k]:
                top[k] += 1
                break
    return top
