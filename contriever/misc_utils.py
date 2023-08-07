import json


def add_my_args(options):
    group = options.parser
    group.add_argument("--reason_device", type=str, default="cuda")
    group.add_argument("--reason_output_file", type=str)
    group.add_argument("--reason_data_file", type=str)
    group.add_argument("--reason_info", type=str, default="")
    group.add_argument("--reason_fact_type", type=str, default='facts')  # facts, gold_facts, single_fact
    group.add_argument("--reason_k", type=int, default=5)
    group.add_argument("--reason_task", type=str, default='lm')
    group.add_argument("--reason_dataset", type=str, required=True)  # strategyqa, entailmentbank
    group.add_argument("--reason_lm", type=str)  # text-davinci-002, google/flan-t5-base, etc.
    group.add_argument("--reason_openai_key", type=str)  # for openai models

    return group


def normalize_answer(s):
    import regex
    import string

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
    pred_tokens = prediction.split()
    truth_tokens = truth.split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return prec, rec, 2 * (prec * rec) / (prec + rec)


def save_lm_report(datas, retrieved_statements, best_alternatives, predicted_tokens_list, output_f=None):
    for i, d in enumerate(datas):
        o = {}
        o['query'] = d['query']
        o['retrieved_statements'] = retrieved_statements[i]
        o['alternatives'] = d['target']
        o['ranked_target'] = d['target'][best_alternatives[i]]
        o['ranked_correctly'] = True if best_alternatives[i] == 0 else False
        o['predicted'] = predicted_tokens_list[i]

        output_f.write(json.dumps(o) + '\n')


def save_qa_report(datas, retrieved_statements, predicted_ans_list, output_f=None, histories=None):
    for i, data in enumerate(datas):
        o = {}
        o['query'] = data['question']
        o['retrieved_statements'] = retrieved_statements[i]
        o['answer'] = data['answer'][0]
        o['response'] = predicted_ans_list[i]
        if histories:
            o['histories'] = histories

        output_f.write(json.dumps(o) + '\n')
