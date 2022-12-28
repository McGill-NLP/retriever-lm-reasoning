from collections import Counter

import numpy as np
import torch
from fairseq import utils
from fairseq.data import data_utils
from fairseq.tokenizer import tokenize_line


class Reason_KNN_Dstore(object):
    def __init__(self, args):
        self.k = args.k
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func

    def get_keys_vals(self, facts_info, args=None):
        keys, target_vals, src_vals = facts_info['dstore_keys'], facts_info['dstore_target_vals'], facts_info[
            'dstore_src_vals']
        keys = keys.cpu().numpy().astype(np.float16 if args.dstore_fp16 else np.float32)
        target_vals = target_vals.cpu().numpy().astype(np.int16 if args.dstore_fp16 else np.int)
        src_vals = [v.cpu().numpy().astype(np.int16 if args.dstore_fp16 else np.int) for v in src_vals]
        return keys, target_vals, src_vals

    def get_knns(self, queries):
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx, facts_info, target_position, d=None, args=None):
        keys, target_vals, src_vals = self.get_keys_vals(facts_info, args=args)

        def dist_func(q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                if self.metric_type == 'l2':
                    knns_vecs = torch.from_numpy(keys).cuda()
                    query_vecs = q.view(q.shape[0], 1, q.shape[-1]).repeat(1, knns_vecs.shape[0], 1)  ## TODO
                    l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=-1)
                    return -1 * l2

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(keys).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            raise ValueError("Invalid knn similarity function!")

        sent_query = queries.reshape(-1, queries.shape[-1])
        dists = dist_func(sent_query, function=self.sim_func)
        k = min(self.k, dists.shape[1])
        topk_retrieved = torch.topk(dists, k=k)

        topk_indices_all = topk_retrieved.indices.cpu()
        probs_all = utils.softmax(topk_retrieved.values, dim=-1)

        knn_probs = torch.full((topk_indices_all.shape[0], len(d.symbols)), -10000.0,
                               dtype=torch.float).cuda()  # -10000 as in the original kNN=LM's code
        topk_retrieved_tokens_all = []
        # merge values for same token values in vals
        for q_id in range(topk_indices_all.shape[0]):
            topk_indices = topk_indices_all[q_id]
            probs = probs_all[q_id]
            seen_vals = {}
            for i in range(k):
                if target_vals[topk_indices[i]][0] in seen_vals:
                    probs[seen_vals[target_vals[topk_indices[i]][0]]] += probs[i]
                else:
                    seen_vals[target_vals[topk_indices[i]][0]] = i
            seen_vals_indices = torch.tensor(list(seen_vals.values()))

            topk_probs = torch.log(probs[seen_vals_indices])
            topk_indices = topk_indices[seen_vals_indices]
            topk_retrieved_tokens = [np.append(src_vals[ret_i], target_vals[ret_i]) for ret_i in topk_indices]
            topk_retrieved_tokens_all.append(topk_retrieved_tokens)

            knn_probs[q_id, torch.tensor(target_vals[topk_indices], dtype=torch.int64).view(-1).cuda()] = topk_probs
        knn_probs = utils.log_softmax(knn_probs, dim=-1)
        return knn_probs, topk_retrieved_tokens_all[target_position[0][0].item()]


class RetLM():
    def __init__(self, models, d, cuda=False):
        assert len(models) == 1, 'len(models)==1 for kNN-LM experiments.'
        self.models = models
        self.dic = d
        self.cuda = cuda

    def get_fact_keys_vals(self, facts, args=None):
        facts_info = {'sample': [], 'fact_unks': [], 'fact_replaced': [], 'hypos': []}
        dstore_keys, dstore_target_vals, dstore_src_vals = None, None, None

        for fact_id, fact in enumerate(facts):
            fact_sample, fact_unks, fact_replaced = binarize(fact, fact_id, self.dic)
            facts_info['sample'].append(fact_sample)
            facts_info['fact_unks'].append(fact_unks)
            facts_info['fact_replaced'].append(fact_replaced)
            fact_sample = utils.move_to_cuda(fact_sample) if self.cuda else fact_sample
            hypo = self.model_encode(fact_sample, args=args, is_fact=True)[0]
            facts_info['hypos'].append(hypo)

            src_tokens = [fact_sample['net_input']['src_tokens'][:, :i + 1].to(torch.int) for i in
                          range(fact_sample['ntokens'])]

            if dstore_keys is None:
                dstore_keys = hypo['dstore_keys'].view(-1, args.decoder_embed_dim).to(torch.float32)
                dstore_target_vals = hypo['tokens'].view(-1, 1).to(torch.int)
                dstore_src_vals = src_tokens
            else:
                dstore_keys = torch.cat(
                    (dstore_keys, hypo['dstore_keys'].view(-1, args.decoder_embed_dim).to(torch.float32)), dim=0)
                dstore_target_vals = torch.cat((dstore_target_vals, hypo['tokens'].view(-1, 1).to(torch.int)), dim=0)
                dstore_src_vals += src_tokens
        facts_info['dstore_keys'] = dstore_keys
        facts_info['dstore_target_vals'] = dstore_target_vals
        facts_info['dstore_src_vals'] = dstore_src_vals
        return facts_info

    @torch.no_grad()
    def model_encode(self, sample, args=None, is_fact=False):
        pad = self.dic.pad()

        net_input = sample['net_input']

        # for model in self.models:
        model = self.models[0]
        model.eval()
        decoder_out = model(**net_input)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)

        if is_fact:
            hypos = []
            bsz = len(sample['start_indices'])
            start_idxs = sample['start_indices']
            for i in range(bsz):
                ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], pad) \
                    if sample['target'] is not None else None
                hypos.append({
                    'tokens': ref,
                    'dstore_keys': decoder_out[1][args.knn_keytype][start_idxs[i]:, i, :]
                })
            return hypos
        return decoder_out, attn

    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, coeff):
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - coeff)
        coeffs[1] = np.log(coeff)
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
        return curr_prob

    @torch.no_grad()
    def do_lm(self, query, sample_id, target_tokens, args=None, compute_alignment=False, knn_dstore=None,
              facts_info=None):

        sample, query_unks, query_replaced, target_ids = binarize_with_target(query, sample_id, target_tokens, self.dic)
        sample = utils.move_to_cuda(sample) if self.cuda else sample

        softmax_batch = args.softmax_batch
        eos = self.dic.eos()
        pad = self.dic.pad()
        knn = args.knnlm
        orig_target = sample['target']

        avg_probs = None
        avg_attn = None
        retrieved_facts = []

        def batch_for_softmax(dec_out, target):
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(dim=2, index=target.unsqueeze(-1))
            return probs

        model = self.models[0]
        decoder_out, attn = self.model_encode(sample, args=args)
        batched = batch_for_softmax(decoder_out, orig_target)
        probs, idx = None, 0
        for i, (bd, tgt, is_single) in enumerate(batched):
            sample['target'] = tgt
            curr_prob = model.get_normalized_probs(bd, log_probs=len(self.models) == 1, sample=sample).data

            if is_single:
                probs = gather_target_probs(curr_prob, orig_target)
            else:
                if probs is None:
                    probs = curr_prob.new(orig_target.numel())
                step = curr_prob.size(0) * curr_prob.size(1)
                end = step + idx
                tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                probs[idx:end] = tgt_probs.view(-1)
                idx = end
            sample['target'] = orig_target

        # 2 x len
        probs = probs.view(sample['target'].shape)
        # 1 x vocab
        curr_prob = torch.where(curr_prob.lt(0), curr_prob, torch.tensor(-torch.inf).cuda())
        curr_prob = curr_prob[:1, sample['target_indices'][0][0], :]  # the first token of true target

        true_target_pos = sample['target_indices'][0][0]

        if knn:
            queries = bd[1][args.knn_keytype]
            if len(self.models) != 1:
                raise ValueError('Only knn *log* probs are supported.')

            queries = queries.permute(1, 0, 2)
            # (2 x len) x vocab
            knn_prob, retrieved_facts = knn_dstore.get_knn_log_prob(queries, orig_target, pad, facts_info,
                                                                    sample['target_indices'], d=self.dic, args=args)

            # 1 x vocab
            curr_prob = self.combine_knn_and_vocab_probs(knn_prob[true_target_pos].view(-1), curr_prob.view(-1),
                                                         args.lmbda).reshape(1, -1)

            knn_prob = knn_prob.reshape(queries.shape[0], queries.shape[1], -1)
            probs = self.combine_knn_and_vocab_probs(gather_target_probs(knn_prob, orig_target).squeeze(-1), probs,
                                                     args.lmbda)

        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs.add_(probs)
        if attn is not None and torch.is_tensor(attn):
            attn = attn.data
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_probs.div_(len(self.models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(self.models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            next_token_score = curr_prob[0, sample['target'][0, true_target_pos]]
            score_i = avg_probs_i.sum() / tgt_len
            # remove padding from ref
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        pad,
                        eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append({
                'tokens': ref,
                'score': score_i,
                'true_next_token_score': next_token_score,
                'attention': avg_attn_i,
                'alignment': alignment,
                'target_dist': curr_prob,
                'positional_scores': avg_probs_i,
                'retrieved_docs': retrieved_facts,
                # 'dstore_keys': decoder_out[1][args.knn_keytype][start_idxs[i]:, i, :] if is_fact else None,
            })

        alt_scores = torch.tensor([hypo['score'] for hypo in hypos])
        best_alternative = torch.argmax(alt_scores, dim=-1)
        from misc_utils import id_to_txt_from_dictionary
        retrieveds = id_to_txt_from_dictionary(hypos[0]['retrieved_docs'], self.dic)
        return {'logits': hypos[0]['target_dist'],
                'target': target_ids,
                'retrieved': retrieveds,
                'predicted_alt': best_alternative,
                'true_score': hypos[0]['score']
                }

    def do_qa(self, query, sample_id, args=None, compute_alignment=False, knn_dstore=None, facts_info=None):
        sample, query_unks, query_replaced = binarize(query, sample_id, self.dic)
        sample = utils.move_to_cuda(sample) if self.cuda else sample

        eos = self.dic.eos()
        pad = self.dic.pad()
        dot = self.dic.index('.')
        knn = args.knnlm

        net_input = sample['net_input']
        EOS = False
        answer_tokens, answer_ids = [], []
        retrieveds = []
        model = self.models[0]
        while not EOS and len(answer_ids) < 100:
            model.eval()
            decoder_out = model(**net_input)

            curr_prob = model.get_normalized_probs(decoder_out, log_probs=len(self.models) == 1, sample=sample).data
            curr_prob = torch.where(curr_prob.lt(0), curr_prob, torch.tensor(-torch.inf).cuda())
            curr_prob = curr_prob[:1, -1, :]

            if knn:
                queries = decoder_out[1][args.knn_keytype]
                if len(self.models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                queries = queries.permute(1, 0, 2)
                target_index = torch.tensor([[net_input['src_lengths'][0] - 1]])
                knn_prob, retrieved_facts = knn_dstore.get_knn_log_prob(queries, None, pad, facts_info,
                                                                        target_index, d=self.dic, args=args)
                if len(retrieveds) == 0:
                    retrieveds = retrieved_facts
                # 1 x vocab
                curr_prob = self.combine_knn_and_vocab_probs(knn_prob[-1].view(-1), curr_prob.view(-1),
                                                             args.lmbda).reshape(1, -1)

            next_token = torch.argmax(curr_prob[0])
            if next_token.item() == pad or next_token.item() == eos or next_token.item() == dot:
                EOS = True
            answer_ids.append(next_token.item())
            answer_tokens.append(self.dic[next_token])
            net_input['src_tokens'] = torch.cat(
                (net_input['src_tokens'], torch.tensor([[next_token]]).to(net_input['src_tokens'].device)), dim=-1)
            net_input['src_lengths'] = [net_input['src_lengths'][0] + 1]

        pred_ans = ' '.join(answer_tokens)
        from misc_utils import id_to_txt_from_dictionary
        retrieveds = id_to_txt_from_dictionary(retrieveds, self.dic)
        return {'query': query, 'retrieved': retrieveds, 'answer': pred_ans}


def collate(sample, pad_idx, eos_idx):
    src_tokens = data_utils.collate_tokens([sample['source']], pad_idx, eos_idx, left_pad=False)
    assert sample['target'] is not None
    target = [sample['target']]

    return {
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.LongTensor(sample['source'].numel()),
        },
        'target': target,
    }


def binarize(txt, txt_id, diction, tokenize=tokenize_line, append_eos=False, reverse_order=False):
    replaced = Counter()

    def replaced_consumer(word, idx):
        if idx == diction.unk_index and word != diction.unk_word:
            replaced.update([word])

    ids = diction.encode_line(
        line=txt,
        line_tokenizer=tokenize,
        add_if_not_exist=False,
        consumer=replaced_consumer,
        append_eos=append_eos,
        reverse_order=reverse_order,
    )
    ids = ids.reshape(1, -1)

    return {
               "id": [txt_id],
               "nsentences": 1,
               "ntokens": len(ids[0]) - 1,
               "target": ids[:, 1:].to(torch.int64),
               "start_indices": [0],
               "net_input": {
                   "src_tokens": ids[:, :-1],
                   "src_lengths": [len(ids[0]) - 1],
               }
           }, sum(replaced.values()), replaced


def binarize_with_target(query, txt_id, target_options, diction, tokenize=tokenize_line, append_eos=False,
                         reverse_order=False):
    replaced = Counter()
    query = query.replace('[MASK]', ' [MASK] ')
    query_sp = query.split(' [MASK] ')
    before = query_sp[0]

    def replaced_consumer(word, idx):
        if idx == diction.unk_index and word != diction.unk_word:
            replaced.update([word])

    before_ids = diction.encode_line(
        line=before,
        line_tokenizer=tokenize,
        add_if_not_exist=False,
        consumer=replaced_consumer,
        append_eos=append_eos,
        reverse_order=reverse_order,
    )
    mask_idx = len(before_ids)

    alternatives = [diction.encode_line(
        line=target_seq,
        line_tokenizer=tokenize,
        add_if_not_exist=False,
        consumer=replaced_consumer,
        append_eos=append_eos,
        reverse_order=reverse_order,
    ).to(torch.int64) for target_seq in target_options]
    alt_num = len(target_options)
    target_indices = [torch.tensor(list(range(mask_idx - 1, mask_idx + len(alternatives[i]) - 1))) for i in
                      range(alt_num)]

    ids = [diction.encode_line(
        line=query.replace('[MASK]', target_options[i]),
        line_tokenizer=tokenize,
        add_if_not_exist=False,
        consumer=replaced_consumer,
        append_eos=append_eos,
        reverse_order=reverse_order,
    ) for i in range(alt_num)]

    max_ids_len = 0
    for sent in ids:
        if len(sent) > max_ids_len:
            max_ids_len = len(sent)
    ids_tensor = torch.full((alt_num, max_ids_len), 1, dtype=torch.int32)
    for i in range(alt_num):
        ids_tensor[i, :len(ids[i])] = ids[i]

    return {
               "id": [txt_id] * alt_num,
               "nsentences": alt_num,
               "ntokens": torch.sum(ids_tensor != 1),
               "target": ids_tensor[:, 1:].to(torch.int64),
               "target_indices": target_indices,
               "start_indices": [0] * alt_num,
               "net_input": {
                   "src_tokens": ids_tensor,
                   "src_lengths": torch.tensor([len(ids[i]) for i in range(alt_num)]),
               }
           }, sum(replaced.values()), replaced, alternatives
