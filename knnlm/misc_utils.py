import torch
import json


def add_reason_args(parser):
    group = parser.add_argument_group("Reason Args")
    group.add_argument("--reason_device", type=str, default="cuda")
    group.add_argument("--reason_output_file", type=str, required=True)
    group.add_argument("--reason_data_file", type=str, required=True)
    group.add_argument("--reason_dict_dir", type=str, default="data-bin/wikitext-103/dict.txt")
    group.add_argument("--reason_info", type=str, default="")
    group.add_argument("--reason_task", type=str, default="lm")  # lm, qa
    group.add_argument("--reason_fact_type", type=str, default='facts')  # facts, gold_facts, single_fact
    group.add_argument("--reason_dataset", type=str, required=True)  # entailmentbank, strategyqa
    return group


def get_hits_at_k(predicted_tokens, targets, ks=[1, 5]):
    predicteds_ = {}
    most_predicted_counts = {}
    top = {k: 0 for k in ks}
    tokens_, counts_ = {}, {}
    most_predicted = {}

    targets = targets[0]

    predicted_tokens_1 = predicted_tokens[:, 0]
    if True:
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

            for target in targets:
                if target in most_predicted[k]:
                    top[k] += 1
                    break
    return top


def compute_f1_score(prediction, truth, d):
    from utils import binarize
    prediction = prediction + ' .'
    truth = truth + ' .'
    pred_tokens = binarize(prediction, 0, d)[0]['net_input']['src_tokens'][0].tolist()
    truth_tokens = binarize(truth, 0, d)[0]['net_input']['src_tokens'][0].tolist()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return prec, rec, 2 * (prec * rec) / (prec + rec)


def save_lm_report(datas, retrieved_statements, predicted_alt, predicted_tokens_list, output_f=None, dic=None):
    # for i, data in enumerate(datas):
    #     output_f.write('Query: {}\n'.format(data['query']))
    #     output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
    #     gen = dic[predicted_tokens_list[i][0]] if dic else predicted_tokens_list[i][0]
    #     output_f.write('Generated: {}\n'.format(gen))
    #     output_f.write('Alternatives: {}\n'.format(data['target']))
    #     output_f.write(
    #         '{} Preferred: {}\n'.format('+' if predicted_alt[i] == 0 else '-', data['target'][predicted_alt[i]]))

    #     output_f.write('\n')
    for i, d in enumerate(datas):
        o = {}
        o['query'] = d['query']
        o['retrieved_statements'] = retrieved_statements[i]
        o['alternatives'] = d['target']
        o['ranked_target'] = d['target'][predicted_alt[i]]
        o['ranked_correctly'] = True if predicted_alt[i] == 0 else False
        o['predicted'] = dic[predicted_tokens_list[i][0]] if dic else predicted_tokens_list[i][0]

        output_f.write(json.dumps(o) + '\n')


def save_qa_report(datas, queries, retrieved_statements, predicted_ans_list, output_f=None):
    # for i, data in enumerate(datas):
    #     output_f.write('Query: {}\n'.format(queries[i]))
    #     output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
    #     output_f.write('Expected: {}\n'.format(data['answer'][0]))
    #     output_f.write('Generated: {}\n'.format(predicted_ans_list[i]))

    #     output_f.write('\n')
    for i, data in enumerate(datas):
        o = {}
        o['query'] = queries[i]
        o['retrieved_statements'] = retrieved_statements[i]
        o['answer'] = data['answer'][0]
        o['response'] = predicted_ans_list[i]

        output_f.write(json.dumps(o) + '\n')


def id_to_txt_from_dictionary(ids, d):
    def id_to_txt(doc_token_ids):
        tokens = [d[i] for i in doc_token_ids]
        return " ".join(tokens) + " "

    if isinstance(ids, list):
        res = []
        for doc_token_ids in ids:
            res.append(id_to_txt(doc_token_ids))
    else:
        res = id_to_txt(ids)
    return res


def clean_str(ss):
    if isinstance(ss, list):
        if len(ss) == 0:
            return []
        assert isinstance(ss[0], str)
        return [clean_str(s) for s in ss]
    return ss.replace('.', ' . ').replace(',', ' , ').replace('!', ' ! ').replace(';', ' ; ').replace('?',
                                                                                                      ' ? ').replace(
        '-', ' - ').replace('(', ' ( ').replace(')', ' ) ').replace('"', ' " ').replace("'", " ' ")
