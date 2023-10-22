import json


def load_data_variables(var_type):
    if var_type == "single_fact" or var_type == "gold_facts" or var_type == "facts":
        models = ["knnlm", "realm", "fid", "atlas", "flan-t5-base"]
        model_name_map = {
            "knnlm": "kNN-LM",
            "realm": "REALM",
            "fid": "DPR + FiD",
            "atlas": "Contriever + ATLAS",
            "flan-t5-base": "Contriever + Flan-T5"
            }
        if var_type == "single_fact":
            ks = {model: 1 for model in models}
            ks["knnlm"] = 100
        else:
            ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 100]

        return {
            "ks": ks,
            "models": models,
            "model_name_map": model_name_map
        }
    if var_type == "retriever" or var_type == "retriever-acc":
        ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 100]
        rets = ["knnlm", "realm", "fid", "atlas"]
        ret_name_map = {
            "knnlm": "kNN-LM",
            "realm": "REALM",
            "fid": "DPR",
            "atlas": "Contriever"
            }
        if var_type == "retriever-acc":
            ks = {ret: 25 for ret in rets}
            ks["knnlm"] = 100

        return {
            "ks": ks,
            "rets": rets,
            "ret_name_map": ret_name_map
        }
    if var_type == "flans":
        models = ["flan-t5-small", "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"]
        model_name_map = {
            "flan-t5-small": "Flan-T5-small",
            "flan-t5-base": "Flan-T5-base",
            "flan-t5-large": "Flan-T5-large",
            "flan-t5-xl": "Flan-T5-xl",
            "flan-t5-xxl": "Flan-T5-xxl"
            }
        ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 100]

        return {
            "ks": ks,
            "models": models,
            "model_name_map": model_name_map
        }
    
    if var_type == "dsp":
        models = ["flan-t5-base", "flan-t5-xxl", "text-davinci-002",]
        model_name_map = {
            "text-davinci-002": "GPT-3",
            "flan-t5-base": "Flan-T5-base",
            "flan-t5-xxl": "Flan-T5-xxl"
            }
        ks = [5]

        return {
            "ks": ks,
            "models": models,
            "model_name_map": model_name_map
        }
        

def create_facts_scores_dict(log_dir):
    vars = load_data_variables("facts")
    ks = vars["ks"]
    models = vars["models"]
    model_name_map = vars["model_name_map"]

    facts = {}
    facts['qa'] = {1: {}, 2: {}, 3: {}, 4: {}}
    facts['lm'] = {1: {}, 2: {}, 3: {}, 4: {}}

    for task in facts.keys():
        for ds in facts[task].keys():
            print(f"-- Exporting scores for task {task} on dataset {ds}")
            for model in models:
                res = dict()
                for k in ks:
                    if ds <= 3:
                        addr = f"{task}/entailmentbank_{ds}_{model}_k_{k}_facts.jsonl"
                    else:
                        if task == 'lm':
                            addr = f"{task}/strategyqa_{model}_k_{k}_facts.jsonl"
                        else:
                            addr = f"{task}/strategyqa_yesno_{model}_k_{k}_facts.jsonl"
                    try:
                        f = open(log_dir + f"{addr}", "r")
                        score_line = f.readlines()[-1]
                        scores = json.loads(score_line)["scores"]
                        for score_k, score_v in scores.items():
                            if score_k in res:
                                res[score_k].append(score_v)
                            else:
                                res[score_k] = [score_v]
                    except Exception as error:
                        print(error)
                        if len(res.keys()) == 0:
                            exit(-1)
                        else:
                            for score_k in res.keys():
                                res[score_k].append(res[score_k][-1])
                facts[task][ds][model_name_map[model]] = res
        output = f"{task}_facts = " + json.dumps(facts[task], indent=4)
        print(f"-- Exported scores for task {task} facts:")
        print(output)
    return facts


def create_gold_facts_scores_dict(log_dir):
    vars = load_data_variables("gold_facts")
    ks = vars["ks"]
    models = vars["models"]
    model_name_map = vars["model_name_map"]

    facts = {}
    facts['qa'] = {2: {}}
    facts['lm'] = {2: {}}

    for task in facts.keys():
        for ds in facts[task].keys():
            print(f"-- Exporting scores for task {task} on dataset {ds}")
            for model in models:
                res = dict()
                for k in ks:
                    addr = f"{task}/entailmentbank_{ds}_{model}_k_{k}_gold_facts.jsonl"
                    try:
                        f = open(log_dir + f"{addr}", "r")
                        score_line = f.readlines()[-1]
                        scores = json.loads(score_line)["scores"]
                        for score_k, score_v in scores.items():
                            if score_k in res:
                                res[score_k].append(score_v)
                            else:
                                res[score_k] = [score_v]
                    except Exception as error:
                        print(error)
                        if len(res.keys()) == 0:
                            exit(-1)
                        else:
                            for score_k in res.keys():
                                res[score_k].append(res[score_k][-1])
                facts[task][ds][model_name_map[model]] = res
        output = f"{task}_facts = " + json.dumps(facts[task], indent=4)
        print(f"-- Exported scores for task {task} gold facts:")
        print(output)
    return facts


def create_single_fact_scores_dict(log_dir):
    vars = load_data_variables("single_fact")
    models = vars["models"]
    model_name_map = vars["model_name_map"]
    ks = vars["ks"]

    facts = {}
    facts['qa'] = {2: {}}

    for task in facts.keys():
        for ds in facts[task].keys():
            print(f"-- Exporting scores for task {task} on dataset {ds}")
            for model in models:
                res = dict()
                addr = f"{task}/entailmentbank_{ds}_{model}_k_{ks[model]}_single_fact.jsonl"
                try:
                    f = open(log_dir + f"{addr}", "r")
                    score_line = f.readlines()[-1]
                    scores = json.loads(score_line)["scores"]
                    for score_k, score_v in scores.items():
                        if score_k in res:
                            res[score_k].append(score_v)
                        else:
                            res[score_k] = [score_v]
                except Exception as error:
                    print(error)
                    if len(res.keys()) == 0:
                        exit(-1)
                    else:
                        for score_k in res.keys():
                            res[score_k].append(res[score_k][-1])
                facts[task][ds][model_name_map[model]] = res
        output = f"{task}_facts = " + json.dumps(facts[task], indent=4)
        print(f"-- Exported scores for task {task} single fact:")
        print(output)
    return facts


def create_retriever_recall_dict(log_dir, data_dir):
    vars = load_data_variables("retriever")
    ks = vars["ks"]
    rets = vars["rets"]
    ret_name_map = vars["ret_name_map"]

    ret_recalls = {"qa": {}, "lm": {}}

    all_valid = 0
    for task in ret_recalls.keys():
        for ret in rets:
            ret_recall = list()
            for k in ks:
                sample_recall, sample_true_ret = [], []
                log_addr = f"{task}/entailmentbank_2_{ret}_k_{k}_facts.jsonl"
                data_addr = f"{task}/entailment_2_test.json"
                try:
                
                    datas = json.load(open(data_dir + f"{data_addr}", "r"))
                    logs = open(log_dir + f"{log_addr}", "r").readlines()
                    assert (len(datas) + 1) == len(logs), f"datas and logs len does not match in {log_addr}. datas: {len(datas)}, logs: {len(logs)}"
                    for data, log in zip(datas, logs):
                        log = json.loads(log)
                        ret_facts = [s.lower().strip() for s in log["retrieved_statements"]]
                        ret_facts_set = set(ret_facts)
                        gold_facts = set([s.lower().strip() for s in data['gold_facts']])
                        all_facts = set([s.lower().strip() for s in data['facts']])

                        if len(gold_facts) > 0:
                            if ret == "knnlm":
                                true_ret_facts = {}
                                for r_fact in ret_facts_set:
                                    for g_fact in gold_facts:
                                        if g_fact.startswith(r_fact):
                                            if g_fact not in true_ret_facts:
                                                true_ret_facts[g_fact] = 1
                                                break
                                sample_true_ret.append(len(true_ret_facts))
                            else:
                                sample_true_ret.append(len(gold_facts.intersection(ret_facts_set)))
                            r = sample_true_ret[-1] / len(gold_facts)
                            sample_recall.append(r)
                        all_valid += 1
                    ret_recall.append(sum(sample_recall) / len(sample_recall))
                except Exception as error:
                
                    print(error)
                    if len(ret_recall) == 0:
                        print("len(ret_recall) = 0")
                        exit(-1)
                    else:
                        ret_recall.append(ret_recall[-1])

            ret_recalls[task][ret_name_map[ret]] = ret_recall

        output = f"retriever_recalls = " + json.dumps(ret_recalls[task], indent=4)
        print(f"-- Exported recall for retrievers in task {task}:")
        print(output)
    return ret_recalls


def create_retriever_accuracy_dict(log_dir, data_dir):
    vars = load_data_variables("retriever-acc")
    ks = vars["ks"]
    rets = vars["rets"]
    ret_name_map = vars["ret_name_map"]

    ret_accs = {"qa": {}, "lm": {}}

    all_valid = 0
    for task in ret_accs.keys():
        for ret in rets:
            ret_acc = list()
            sample_top_acc = []
            log_addr = f"{task}/entailmentbank_2_{ret}_k_{ks[ret]}_facts.jsonl"
            data_addr = f"{task}/entailment_2_test.json"
            try:
                datas = json.load(open(data_dir + f"{data_addr}", "r"))
                logs = open(log_dir + f"{log_addr}", "r").readlines()
                assert (len(datas)+1) == len(logs)
                for data, log in zip(datas, logs):
                    log = json.loads(log)
                    ret_facts = [s.lower().strip() for s in log["retrieved_statements"]]
                    top_ret_facts = set(ret_facts[:len(data['gold_facts'])])
                    gold_facts = set([s.lower().strip() for s in data['gold_facts']])

                    if len(gold_facts) > 0:
                        if ret == "knnlm":
                            true_ret_facts = {}
                            for r_fact in top_ret_facts:
                                for g_fact in gold_facts:
                                    if g_fact.startswith(r_fact):
                                        if g_fact not in true_ret_facts:
                                            true_ret_facts[g_fact] = 1
                                            break
                            top_a = len(true_ret_facts)
                        else:
                            top_a = len(gold_facts.intersection(top_ret_facts))
                        sample_top_acc.append(top_a / len(gold_facts))
                    all_valid += 1
                ret_acc.append(sum(sample_top_acc) / len(sample_top_acc))
            except Exception as error:
                print(error)
                if len(ret_acc) == 0:
                    exit(-1)
                else:
                    ret_acc.append(ret_acc[-1])

            ret_accs[task][ret_name_map[ret]] = ret_acc

        output = f"retriever_accuracies = " + json.dumps(ret_accs[task], indent=4)
        print(f"-- Exported accuracy for retrievers in task {task}:")
        print(output)
    return ret_accs


def create_flan_scores_dict(log_dir):
    vars = load_data_variables("flans")
    ks = vars["ks"]
    models = vars["models"]
    model_name_map = vars["model_name_map"]

    facts = {}
    facts['qa'] = {1: {}, 2: {}, 3: {}, 4: {}}

    for task in facts.keys():
        for ds in facts[task].keys():
            print(f"-- Exporting scores for task {task} on dataset {ds}")
            for model in models:
                res = dict()
                for k in ks:
                    if ds <= 3:
                        addr = f"{task}/entailmentbank_{ds}_{model}_k_{k}_facts.jsonl"
                    else:
                        if task == 'lm':
                            addr = f"{task}/strategyqa_{model}_k_{k}_facts.jsonl"
                        else:
                            addr = f"{task}/strategyqa_yesno_{model}_k_{k}_facts.jsonl"
                    try:
                        f = open(log_dir + f"{addr}", "r")
                        score_line = f.readlines()[-1]
                        scores = json.loads(score_line)["scores"]
                        for score_k, score_v in scores.items():
                            if score_k in res:
                                res[score_k].append(score_v)
                            else:
                                res[score_k] = [score_v]
                    except Exception as error:
                        print(error)
                        if len(res.keys()) == 0:
                            exit(-1)
                        else:
                            for score_k in res.keys():
                                res[score_k].append(res[score_k][-1])
                facts[task][ds][model_name_map[model]] = res
        output = f"{task}_facts = " + json.dumps(facts[task], indent=4)
        print(f"-- Exported scores for task {task} facts:")
        print(output)
    return facts


def create_dsp_scores_dict(log_dir):
    vars = load_data_variables("dsp")
    ks = vars["ks"]
    models = vars["models"]
    model_name_map = vars["model_name_map"]

    facts = {}
    facts['no-dsp'] = {1: {}, 2: {}, 4: {}}
    facts['dsp'] = {1: {}, 2: {}, 4: {}}

    for task in facts.keys():
        file_suffix = "fewshot" if task == "no-dsp" else "dsp"
        for ds in facts[task].keys():
            print(f"-- Exporting scores for {task} on dataset {ds}")
            for model in models:
                res = dict()
                for k in ks:
                    if ds <= 3:
                        addr = f"qa/entailmentbank_{ds}_{model}_k_{k}_facts_{file_suffix}.jsonl"
                    else:
                        addr = f"qa/strategyqa_yesno_{model}_k_{k}_facts_{file_suffix}.jsonl"
                    try:
                        f = open(log_dir + f"{addr}", "r")
                        score_line = f.readlines()[-1]
                        scores = json.loads(score_line)["scores"]
                        for score_k, score_v in scores.items():
                            if score_k in res:
                                res[score_k].append(score_v)
                            else:
                                res[score_k] = [score_v]
                    except Exception as error:
                        print(error)
                        if len(res.keys()) == 0:
                            exit(-1)
                        else:
                            for score_k in res.keys():
                                res[score_k].append(res[score_k][-1])
                facts[task][ds][model_name_map[model]] = res
        output = f"{task}_facts = " + json.dumps(facts[task], indent=4)
        print(f"-- Exported scores for {task}:")
        print(output)
    return facts


log_file = "visualization.log"
log_dir = "<parent dir of the log file>"
data_dir = "<parent dir of the data file>"
log_file = open(log_dir + log_file, 'a+')

o = create_facts_scores_dict(log_dir)
log_file.write("Model scores using facts=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')

o = create_gold_facts_scores_dict(log_dir)
log_file.write("Model scores using gold facts=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')

o = create_single_fact_scores_dict(log_dir)
log_file.write("Model scores using single fact=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')

o = create_retriever_recall_dict(log_dir, data_dir)
log_file.write("Retriever recall scores=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')

o = create_retriever_accuracy_dict(log_dir, data_dir)
log_file.write("Retriever accuracy scores=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')

o = create_flan_scores_dict(log_dir)
log_file.write("Flan scores using facts=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')

o = create_dsp_scores_dict(log_dir)
log_file.write("DSP scores using facts=\n")
log_file.write(json.dumps(o, indent=4) + '\n\n')
