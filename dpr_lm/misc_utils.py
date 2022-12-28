

def save_lm_report(datas, retrieved_statements, predicted_alt, predicted_tokens_list, output_f=None):
    for i, d in enumerate(datas):
        output_f.write('Query: {}\n'.format(d['question']))
        output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
        output_f.write('Generated: {}\n'.format(predicted_tokens_list[i]))
        output_f.write('Alternatives: {}\n'.format(d['target']))
        output_f.write(
            '{} Preferred: {}\n'.format('+' if predicted_alt[i] == 0 else '-', d['target'][predicted_alt[i]]))

        output_f.write('\n')


def save_qa_report(datas, retrieved_statements, predicted_ans_list, output_f=None):
    for i, data in enumerate(datas):
        output_f.write('Query: {}\n'.format(data['question']))
        output_f.write('Retrieved: {}\n'.format(' | '.join(retrieved_statements[i])))
        output_f.write('Expected: {}\n'.format(data['answer'][0]))
        output_f.write('Generated: {}\n'.format(predicted_ans_list[i]))

        output_f.write('\n')


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
