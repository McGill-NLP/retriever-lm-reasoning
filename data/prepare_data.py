import json
import random
import re
from argparse import ArgumentParser


custom_masking_tokens = ['moon', 'earth', 'sun', 'galaxy', 'leo', 'sky', 'star', 'stars', 'venus', 'planet',
                         'planets', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                         'november', 'december', 'alaska', 'hemisphere', 'season', 'winter', 'summer', 'spring',
                         'fall', 'florida', 'neptune', 'solar', 'east', 'north', 'south', 'west', 'human',
                         'humans', 'satellite', 'crater', 'craters', 'air', 'phobos', 'gravity', 'pluto',
                         'light', 'heat', 'candle', 'candles', 'electron', 'electrons', 'proton', 'protons',
                         'neutron', 'neutrons', 'atom', 'atoms', 'hydrogen', 'oxygen', 'helium', 'energy',
                         'mars', 'uranus', 'space', 'volcano', 'land', 'glacier', 'glaciers', 'river', 'sulfur',
                         'rivers', 'delta', 'water', 'canyon', 'canyons', 'fossil', 'fossils', 'dinosaurs',
                         'dinosaur', 'mountain', 'ohio', 'ocean', 'oceans', 'antarctica', 'sediment', 'plant',
                         'plants', 'animal', 'animals', 'rock', 'rocks', 'bird', 'birds', 'horse', 'horses',
                         'elephant', 'elephants', 'savanna', 'mineral', 'crystal', 'diamond', 'quartz',
                         'halite', 'minerals', 'diamonds', 'crystals', 'tide', 'tides', 'ice', 'comet', 'cliff',
                         'comets', 'mountains', 'telescope', 'microscope', 'gas', 'gases', 'volcanoes', 'sea',
                         'granite', 'sediments', 'sand', 'coal', 'oil', 'gasoline', 'rain', 'earthquake',
                         'limestone', 'lithosphere', 'atmosphere', 'soil', 'forest', 'forests', 'fire', 'fires',
                         'forests', 'tree', 'trees', 'equator', 'wind', 'stream', 'surface', 'surfaces',
                         'magnet', 'magnets', 'machine', 'machines', 'injury', 'injuries', 'feather', 'wing',
                         'feathers', 'wings', 'ancestor', 'adult', 'adults', 'kid', 'kids', 'egg', 'eggs',
                         'frog', 'frogs', 'caterpillar', 'caterpillars', 'butterfly', 'butterflies', 'skunk',
                         'skunks', 'sugar', 'seed', 'seeds', 'bear', 'bears', 'alligator', 'cow', 'alligators',
                         'cows', 'sodium', 'chlorine', 'nucleus', 'ion', 'ions', 'molecule', 'molecules',
                         'clay', 'magnetic', 'iron', 'electricity', 'metal', 'airport', 'traffic', 'congestion',
                         'lake', 'lakes', 'shape', 'volume', 'mass', 'length', 'weight', 'solid', 'liquid',
                         'object', 'objects', 'tool', 'tools', 'appliance', 'appliances', 'cell', 'insect', 'wave',
                         'waves']


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='entailmentbank')  # entailmentbank, strategyqa
    parser.add_argument("--input_file", type=str, default='')  # relative address from "data" folder
    parser.add_argument("--output_file", type=str, default='')  # file output name, without format suffix
    parser.add_argument("--split", type=int, default=0)  # 1 if you want to split to train/dev/test
    parser.add_argument("--qa", type=int, default=0)
    parser.add_argument("--lm", type=int, default=0)
    return parser


def get_gold_facts(data):
    sample_facts_map = {}
    sample_facts = re.split(r'\W*sent[0-9]+:\W*', data['context'])[1:]
    for i in range(len(sample_facts)):
        sample_facts_map['sent{}'.format(i + 1)] = sample_facts[i].capitalize() + '.'
    steps = data['proof'].split('; ')[:-1]
    gold_facts = []
    if 'distractors' in data:
        # distractors already exist in data
        for k, fact in sample_facts_map.items():
            if k not in data['distractors']:
                gold_facts.append(fact)
    else:
        for step in steps:
            step_components = step.split(' -> ')
            if step_components[0].count('&') >= 1:
                fact_names = step_components[0].split(' & ')
                try:
                    for f_n in fact_names:
                        if f_n.startswith('sent'):
                            gold_facts.append(sample_facts_map[f_n])
                except:
                    print('not valid fact components.', step)
    return gold_facts


def get_alternative_targets(facts, target, tagger=None):
    facts_text = ' '.join([fact[:-1] + ' .' for fact in facts]).replace(',', ' ,')
    sentence = tagger(facts_text)
    ents = []
    if len(sentence.ents) > 0:
        for ent in sentence.ents:
            if ent.text.lower() != target.lower():
                ents.append(ent.text.lower() if ent.text.lower() in facts_text else ent.text)
    if len(ents) > 0:
        return [target, random.choice(ents)]

    # no entity mentions found by Spacy, let's try search for some pre-defined tokens
    if len(target.split()) == 1:
        f_toks = facts_text.lower().split()
        for tok in custom_masking_tokens:
            if tok in f_toks and tok != target.lower():
                ents.append(tok)
    if len(ents) > 0:
        return [target, random.choice(ents)]

    return [target, '']


def process_query(d, tagger=None):
    sent = d['hypothesis']
    sentence = tagger(sent)

    def are_tgts_valid(tgts):
        return len(tgts) > 1 and tgts[1] != ''

    if len(sentence.ents) > 0:
        masked_ent = sentence.ents[-1]
        if masked_ent.start_char > 0:
            final_masked_text = '{}{}{}'.format(sent[:masked_ent.start_char], '[MASK]', sent[masked_ent.end_char:])
            alternative_tgts = get_alternative_targets(d['facts'], masked_ent.text, tagger=tagger)
            is_valid = are_tgts_valid(alternative_tgts)
            return final_masked_text, alternative_tgts, is_valid

    sent_toks = sent[:-1].split()
    l = len(sent_toks)
    mask_idx = None
    for i in range(l):
        if sent_toks[l - i - 1] in custom_masking_tokens:
            mask_idx = l - i - 1
            break
    if mask_idx is None:
        return '', [], False
    final_masked_text = '{} {} {}{}'.format(" ".join(sent_toks[:mask_idx]), '[MASK]',
                                            " ".join(sent_toks[mask_idx + 1:]), sent[-1])
    alternative_tgts = get_alternative_targets(d['facts'], sent_toks[mask_idx], tagger=tagger)
    is_valid = are_tgts_valid(alternative_tgts)
    return final_masked_text, alternative_tgts, is_valid


def get_hypothesis_and_facts(ds, data):
    if ds == 'entailmentbank':
        sample_hyp = data['hypothesis'].capitalize() + '.'
        sample_facts = re.split(r'\W*sent[0-9]+:\W*', data['context'])[1:]
        for i in range(len(sample_facts)):
            sample_facts[i] = sample_facts[i].capitalize() + '.'
    elif ds == 'strategyqa':
        sample_facts = data['facts']
        sample_hyp = data['hypothesis']
    return sample_hyp, sample_facts


def create_and_write_lm(args=None):
    output_dataset = []
    dataset = []
    assert args.dataset in ['strategyqa', 'entailmentbank'], 'Add the new dataset\'s preparation process'
    if args.dataset == 'strategyqa':
        dataset = json.load(open(args.input_file, 'r'))
    elif args.dataset == 'entailmentbank':
        with open(args.input_file, "r") as f:
            for line in f:
                dataset.append(json.loads(line))

    import spacy
    tagger = spacy.load('en_core_web_sm')

    valid_samples, data_id = 0, 0
    for data_id, data in enumerate(dataset):
        sample_hyp, sample_facts = get_hypothesis_and_facts(args.dataset, data)
        d = {'hypothesis': sample_hyp, 'facts': sample_facts}
        masked_query, target_tokens, d_is_valid = process_query(d, tagger=tagger)

        if d_is_valid:  # there were at least two entities in the facts and query, as the gold and alternative targets.
            valid_samples += 1
            data_sample = {'dataset_id': data_id, 'query': masked_query, 'target': target_tokens, 'facts': sample_facts}
            if args.dataset == 'strategyqa':
                data_sample['question'] = data['question']
                data_sample['answer'] = data['answer']
            elif args.dataset == 'entailmentbank':
                data_sample['gold_facts'] = get_gold_facts(data)

            output_dataset.append(data_sample)
    if args.split == 1:
        random.shuffle(output_dataset)
        data_num = len(output_dataset)
        dev_num = int(0.25 * data_num)
        test_num = int(0.35 * data_num)
        with open('lm/{}_test.json'.format(args.output_file), 'w') as outfile:
            json.dump(output_dataset[:test_num], outfile)
        with open('lm/{}_dev.json'.format(args.output_file), 'w') as outfile:
            json.dump(output_dataset[test_num:test_num+dev_num], outfile)
        with open('lm/{}_train.json'.format(args.output_file), 'w') as outfile:
            json.dump(output_dataset[test_num+dev_num:], outfile)
    else:
        with open('lm/{}.json'.format(args.output_file), 'w') as outfile:
            json.dump(output_dataset, outfile)
    print('LM Done! Total samples {}, Valid samples {}'.format(data_id + 1, valid_samples))


def create_and_write_qa(args=None):
    dataset, output_dataset = [], []
    assert args.dataset in ['entailmentbank', 'strategyqa'], 'Add the new dataset\'s preparation process'
    if args.dataset == 'entailmentbank':
        with open(args.input_file, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
    if args.dataset == 'strategyqa':
        dataset = json.load(open(args.input_file, 'r'))

    valid_samples, data_id = 0, 0
    for data_id, data in enumerate(dataset):
        if not 'question' in data or not 'answer' in data:
            continue
        sample_q = data['question']
        if args.dataset == 'strategyqa':
            sample_q = data['question'] + " yes or no?"
        sample_a = data['answer']
        sample_facts = [f.capitalize() + '.' for f in re.split(r'\W*sent[0-9]+:\W*', data['context'])[1:]]

        valid_samples += 1
        d = {'dataset_id': data_id,
             'question': sample_q,
             'answer': [sample_a],
             'facts': sample_facts,
             'gold_facts': get_gold_facts(data),
             'hypothesis': data.get('hypothesis', '').capitalize() + '.'}
        if args.dataset == 'strategyqa':
            d['answer'] = ["yes", "no"] if d['answer'] == "true" else ["no", "yes"]
        output_dataset.append(d)
    with open('qa/{}.json'.format(args.output_file), 'w') as outfile:
        json.dump(output_dataset, outfile)
    print('QA Done! Total samples {}, Valid samples {}'.format(data_id + 1, valid_samples))


if __name__ == '__main__':
    args = arg_parser().parse_args()
    print(args)

    # lm
    if args.lm > 0:
        create_and_write_lm(args=args)
    # qa
    if args.qa > 0:
        create_and_write_qa(args=args)


# cd data
# python prepare_data.py --input_file raw/entailmentbank/task_1/dev.jsonl --output_file entailmentbank_1_dev --dataset entailmentbank --qa 1 --lm 1
