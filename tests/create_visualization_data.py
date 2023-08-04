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
    if var_type == "retriever":
        ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 100]
        rets = ["knnlm", "realm", "fid", "atlas"]
        ret_name_map = {
            "knnlm": "kNN-LM",
            "realm": "REALM",
            "fid": "DPR",
            "atlas": "Contriever"
            }

        return {
            "ks": ks,
            "rets": rets,
            "ret_name_map": ret_name_map
        }
        

def create_facts_scores_dict(log_dir):
    vars = load_data_variables("facts")
    ks = vars["ks"]
    models = vars["models"]
    model_name_map = vars["model_name_map"]

    facts = {}
    facts['qa'] = {1: {}, 2: {}}  #, 3: {}, 4: {}}
    # facts['lm'] = {1: {}, 2: {}}

    for task in facts.keys():
        for ds in facts[task].keys():
            print(f"-- Exporting scores for task {task} on dataset {ds}")
            for model in models:
                res = dict()
                for k in ks:
                    if ds <= 3:
                        addr = f"{task}/entailmentbank_{ds}_{model}_k_{k}_facts.jsonl"
                    else:
                        addr = f"{task}/strategyqa_{model}_k_{k}_facts.jsonl"
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
        print("-- Exported scores for task {task} facts:")
        print(output)


def create_gold_facts_scores_dict(log_dir):
    vars = load_data_variables("gold_facts")
    ks = vars["ks"]
    models = vars["models"]
    model_name_map = vars["model_name_map"]

    facts = {}
    facts['qa'] = {2: {}}

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
        print("-- Exported scores for task {task} gold facts:")
        print(output)


def create_single_fact_scores_dict(log_dir):
    vars = load_data_variables("single_fact")
    models = vars["models"]
    model_name_map = vars["model_name_map"]

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
        print("-- Exported scores for task {task} single fact:")
        print(output)


def create_retriever_recall_dict(log_dir):
    vars = load_data_variables("retriever")
    ks = vars["ks"]
    rets = vars["rets"]
    ret_name_map = vars["ret_name_map"]

    ret_recalls = {}

    for ret in rets:
        ret_recall = list()
        addr = f"qa/entailmentbank_2_{model}_k_{ks[model]}_facts.jsonl"
        try:
            f = open(log_dir + f"{addr}", "r")
            pass


        except Exception as error:
            print(error)
            if len(ret_recall) == 0:
                exit(-1)
            else:
                pass
        ret_recalls[ret_name_map[ret]] = ret_recall
    output = f"recalls = " + json.dumps(ret_recalls, indent=4)
    print("-- Exported retriever scores:")
    print(output)


    
log_dir = "/network/scratch/p/parishad.behnamghader/mcgill-nlp/retriever-lm-reasoning/my_logs/"
# create_facts_scores_dict(log_dir)
# create_gold_facts_scores_dict(log_dir)
# create_single_fact_scores_dict(log_dir)
# create_retriever_recall_dict(log_dir)