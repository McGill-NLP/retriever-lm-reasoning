import json
import random

import torch
from src.data import encode_passages
from torch.utils.data import DataLoader, SequentialSampler

from DPR.dpr_utils import retrieve_topk_docs


class RetLM():
    def __init__(self, model, tokenizer, dataset, args=None):
        self.model = model
        self.model.eval()
        if hasattr(self.model, "module"):
            self.model = self.model.module
        if args.lm.write_crossattention_scores:
            self.model.overwrite_forward_crossattention()
            self.model.reset_score_storage()
        self.tokenizer = tokenizer
        self.dataset = dataset
        if args.reason.task == 'qa':
            self.get_answer = self.do_qa
        elif args.reason.task == 'compare_qa':
            self.get_answer = self.do_comparison_qa
        elif args.reason.task == 'lm':
            self.get_answer = self.do_lm

    @torch.no_grad()
    def do_lm(self, context_ids, context_mask, labels_, sample_id, args=None):
        if args.lm.write_crossattention_scores:
            self.model.reset_score_storage()
        outputs = self.model.generate(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            max_length=50,
        )
        # outputs: 1xt. generated answer.

        alt_num = labels_.shape[0]
        losses = torch.zeros(alt_num)
        for alt_i in range(alt_num):
            l = labels_[alt_i].unsqueeze(0)
            labels_output = self.model(input_ids=context_ids.cuda(),
                                       attention_mask=context_mask.cuda(),
                                       labels=l.cuda())
            losses[alt_i] = labels_output[0]

        best_alternative = torch.argmin(losses)
        example = self.dataset.data[sample_id[0]]

        for k, o in enumerate(outputs):
            ans = self.tokenizer.decode(o, skip_special_tokens=True)
            example = self.dataset.data[sample_id[k]]
            example_targets = [' '.join(tg.split()[1:]) for tg in example['target']]

            ans_ids = self.tokenizer(ans).input_ids
            first_token_pred = ans_ids[0]
            tgt_ids = self.tokenizer(example_targets[0]).input_ids
        return {'hits1': int(first_token_pred in tgt_ids),
                'first_token_pred': first_token_pred,
                'retrieved': [ctx['text'] for ctx in example['ctxs']],
                'predicted_alt': best_alternative,
                'example': example
                }

    def do_qa(self, context_ids, _, context_mask, sample_id, args=None):
        if args.lm.write_crossattention_scores:
            self.model.reset_score_storage()

        outputs = self.model.generate(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            max_length=50,
        )
        # outputs: 1xt. generated answer.

        for k, o in enumerate(outputs):
            ans = self.tokenizer.decode(o, skip_special_tokens=True)
            example = self.dataset.data[sample_id[k]]
        return {'query': example['question'],
                'retrieved': [ctx['text'] for ctx in example['ctxs']],
                'gen_ans': ans,
                'example': example}
    
    def do_comparison_qa(self, context_ids, labels, context_mask, sample_id, args=None):
        if args.lm.write_crossattention_scores:
            self.model.reset_score_storage()

        alt_num = labels.shape[0]
        losses = torch.zeros(alt_num)
        for alt_i in range(alt_num):
            l = labels[alt_i].unsqueeze(0)
            labels_output = self.model(input_ids=context_ids.cuda(),
                                       attention_mask=context_mask.cuda(),
                                       labels=l.cuda())
            losses[alt_i] = labels_output[0]

        predicted_alt = torch.argmin(losses)
        example = self.dataset.data[sample_id[0]]
        return {'query': example['question'],
                'retrieved': [ctx['text'] for ctx in example['ctxs']],
                'gen_ans': example['target'][predicted_alt],
                'example': example}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            if isinstance(target, str):
                return target + ' </s>'
            if isinstance(target, list):
                return [t + '</s>' for t in target]
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            'index': index,
            'question': question,
            'target': target,
            'passages': passages,
            'scores': scores,
            'gold_facts': example.get('gold_facts'),
            'original_answer': example.get('original_answer')
        }

    def sort_data(self):
        if self.n_context is None or self.n_context == 0 or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]


class MyCollator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, lm='fid', flan_prompt='Find <extra_id_0>.\n'):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.lm = lm
        self.flan_prompt = flan_prompt

    def __call__(self, batch):
        assert (batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        multi_targets = isinstance(batch[0]['target'], list)
        if not multi_targets:
            target = [ex['target'] for ex in batch]
        else:
            target = []
            for ex in batch:
                target += [t for t in ex['target']]
        if self.lm == 'flan':
            target = self.tokenizer.batch_encode_plus(
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                padding=True,
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False,
            )
        elif self.lm == 'fid':
            target = self.tokenizer.batch_encode_plus(
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False,
            )
        else:
            ValueError('Invalid lm requested.')

        # print(target)
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if self.lm == 'fid':
                if example['passages'] is None or len(example['passages']) == 0:
                    return [example['question']]
                return [example['question'] + " " + t for t in example['passages']]
            elif self.lm == 'flan':
                return ['{}{} {}'.format(self.flan_prompt, " ".join(example['passages']), example['question'])]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength,
                                                     lm=self.lm)
        if self.lm == 'flan':
            passage_ids, passage_masks = passage_ids.squeeze(1), passage_masks.squeeze(1)

        return (index, target_ids, target_mask, passage_ids, passage_masks)


def my_load_data_with_retrieved_docs(data_path=None, global_rank=-1, world_size=-1, doc_encoder=None, tensorizer=None,
                                     retriever=None, cfg=None):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        # if global_rank > -1 and not k % world_size == global_rank:
        #     continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k

        # Retrieve the topk docs
        example = retrieve_topk_docs(example, doc_encoder=doc_encoder, tensorizer=tensorizer, retriever=retriever,
                                     args=cfg)
        if example is None:
            continue
        for c in example['ctxs']:
            if 'score' not in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()
    return examples


def load_dataloader(cfg, data, doc_encoder=None, tensorizer=None, retriever=None, q_prefix='', ctx_prefix='',
                    collate_fn=None):
    lm = cfg.lm
    reason = cfg.reason
    eval_examples = my_load_data_with_retrieved_docs(
        data,
        global_rank=lm.global_rank,
        world_size=lm.world_size,
        doc_encoder=doc_encoder,
        tensorizer=tensorizer,
        retriever=retriever,
        cfg=cfg
    )
    lm.n_context = reason.k
    eval_dataset = MyDataset(
        eval_examples,
        lm.n_context,
        question_prefix=q_prefix,
        passage_prefix=ctx_prefix
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=lm.per_gpu_batch_size,
        num_workers=4,
        collate_fn=collate_fn
    )
    return eval_dataloader, eval_dataset
