import pickle
import re
import random
import jsonlines
import numpy as np

from utils.data_utils import load_entail_dataset


def write_all_steps(part='train', data_dir="data/entailmentbank/"):
    dataset = load_entail_dataset(addr=data_dir + "task_1", part=part)
    steps_list = []
    for data_id, data in enumerate(dataset):
        if data_id % 1000 == 0:
            print('id', data_id)
        sample_facts_map = {}
        sample_facts = re.split(r'\W*sent[0-9]+:\W*', data['context'])[1:]
        for i in range(len(sample_facts)):
            sample_facts_map['sent{}'.format(i + 1)] = sample_facts[i] + '.'
        steps = data['proof'].split('; ')[:-1]
        for step in steps:
            valid = True
            step_components = step.split(' -> ')
            if step_components[1] == 'hypothesis':
                conc = data['hypothesis'] + '.'
            elif step_components[1].startswith('int'):
                tmp = step_components[1].index(':')
                conc = step_components[1][tmp + 2:] + '.'
                sample_facts_map[step_components[1][:tmp]] = conc
            else:
                print('not valid conclusion component.', step)
                valid = False
            if step_components[0].count('&') == 1:
                fact_names = step_components[0].split(' & ')
                try:
                    fact_a = sample_facts_map[fact_names[0]]
                    fact_b = sample_facts_map[fact_names[1]]
                except:
                    print('not valid fact components.', step)
                    valid = False
            else:
                valid = False

            if valid:
                steps_list.append([fact_a, fact_b, conc])
    random.shuffle(steps_list)
    print(len(steps_list))
    with open(data_dir + 'task_1_{}_steps.json'.format(part), 'wb') as f:
        pickle.dump(steps_list, f)


def write_goal_steps(part='train', data_dir="data/entailmentbank/"):
    dataset = load_entail_dataset(addr=data_dir + "task_1", part=part)
    steps_list = []
    for data_id, data in enumerate(dataset):
        if data_id % 1000 == 0:
            print('id', data_id)
        sample_facts_map = {}
        sample_facts = re.split(r'\W*sent[0-9]+:\W*', data['context'])[1:]
        for i in range(len(sample_facts)):
            sample_facts_map['sent{}'.format(i + 1)] = sample_facts[i] + '.'
        steps = data['proof'].split('; ')[:-1]
        for step in steps:
            valid = False
            step_components = step.split(' -> ')
            if step_components[1] == 'hypothesis':
                conc = data['hypothesis'] + '.'
                if step_components[0].count('&') == 1:
                    fact_names = step_components[0].split(' & ')
                    try:
                        fact_a = sample_facts_map[fact_names[0]]
                        fact_b = sample_facts_map[fact_names[1]]
                        valid = True
                    except:
                        print('not valid fact components.', step)
            elif step_components[1].startswith('int'):
                tmp = step_components[1].index(':')
                conc = step_components[1][tmp + 2:] + '.'
                sample_facts_map[step_components[1][:tmp]] = conc
            else:
                print('not valid conclusion component.', step)
            if valid:
                steps_list.append([fact_a, fact_b, conc])
    random.shuffle(steps_list)
    print(len(steps_list))
    with open(data_dir + 'task_1_{}_goal_steps.json'.format(part), 'wb') as f:
        pickle.dump(steps_list, f)


def check_fact_stats(ds, part='train', task=1, data_dir="data/mlm/"):
    if ds == 'ftrace-abs':
        datastore_addr = data_dir + 'ftrace_1700000.jsonl'
    elif ds == 'entailmentbank':
        datastore_addr = data_dir + 'entailment_{}_{}.jsonl'.format(task, part)
    if ds == 'strategyqa':
        datastore_addr = data_dir + 'strategyqa.jsonl'
    dataset = jsonlines.open(datastore_addr)

    fact_len = []
    for sample_id, d in enumerate(dataset):
        if sample_id > 5000:
            break
        facts = d['facts']
        fact_len.append(len(facts))

    fact_len = np.array(fact_len)
    print(ds, part, task)
    print('mean', fact_len.mean(), 'std', fact_len.std(), 'min', fact_len.min(), 'max', fact_len.max())

# write_all_steps(part='dev')
# write_goal_steps(part='train')
# check the output file
# with open('data/entailmentbank/task_1_train_goal_steps.json', 'rb') as f:
#     l = pickle.load(f)
#     print(len(l))
#     print(l[:2])

# check_fact_stats('entailmentbank', part='test', task=1)
# check_fact_stats('entailmentbank', part='test', task=2)
# check_fact_stats('entailmentbank', part='test', task=3)
# check_fact_stats('ftrace-abs', part='test', task=1)
# check_fact_stats('strategyqa', part='test', task=1)

# python -m data.entailmentbank.retrieve_steps.py
