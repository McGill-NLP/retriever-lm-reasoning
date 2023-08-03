import json

def save_lm_report(datas, retrieved_statements, predicted_alt, predicted_tokens_list, output_f=None):   
    for i, d in enumerate(datas):
        o = {}
        o['query'] = d['question']
        o['retrieved_statements'] = retrieved_statements[i]
        o['alternatives'] = d['target']
        o['ranked_target'] = d['target'][predicted_alt[i]]
        o['ranked_correctly'] = True if predicted_alt[i] == 0 else False
        o['predicted'] = predicted_tokens_list[i]

        output_f.write(json.dumps(o) + '\n')


def save_qa_report(datas, retrieved_statements, predicted_ans_list, output_f=None):
    for i, data in enumerate(datas):
        o = {}
        o['query'] = data['question']
        o['retrieved_statements'] = retrieved_statements[i]
        o['answer'] = data['answer'][0]
        o['response'] = predicted_ans_list[i]

        output_f.write(json.dumps(o) + '\n')


def compute_f1_score(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return prec, rec, 2 * (prec * rec) / (prec + rec)
