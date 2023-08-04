import random

import numpy as np
import torch
from transformers.models.realm.modeling_realm import RealmKnowledgeAugEncoder, RealmForOpenQA, RealmEmbedder
from transformers.models.realm.retrieval_realm import RealmRetriever
from transformers.models.realm.tokenization_realm import RealmTokenizer


class CandidateScorer():
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self,
                 input_ids=None,
                 candidate_embs=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 inputs_embeds=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 return_dict=None):
        query_outputs = self.embedder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_score = query_outputs.projected_score
        relevance_score = torch.einsum("BD,ND->BN", query_score.reshape(input_ids.shape[0], -1), candidate_embs)
        return relevance_score

    def embed(self, inputs):
        outputs = self.embedder(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            token_type_ids=inputs.token_type_ids,
        )
        return outputs.projected_score


class RetLM():
    def __init__(self, task, tokenizer=None, qa_model=None, query_embedder=None, encoder=None, device=None,
                 lm_task=None, dataset=None):
        self.task = task
        self.tokenizer = tokenizer
        self.qa_model = qa_model
        self.query_embedder = query_embedder
        self.encoder = encoder
        self.device = device
        if task == 'qa':
            self.get_answer = self.do_qa
        elif task == 'compare_qa':
            self.get_answer = self.do_comparison_qa
        elif task == 'lm':
            self.lm_task = lm_task
            self.get_answer = self.do_lm
        else:
            ValueError('Invalid task for RetLM.')

    def tokenize_lm_query(self, query, target_texts):
        if self.lm_task == 'target_ranking':
            target_texts = target_texts
            targets = self.tokenizer(target_texts, return_tensors="pt", max_length=512, truncation=True,
                                     padding=True).input_ids[:, 1:-1]

            query = query.split(' [MASK]')
            if len(query) < 2:
                query = query[0]
                assert '[MASK]' in query
                query = query.replace('[MASK]', ' [MASK]')
                query = query.split(' [MASK]')
            target_ = torch.where(targets != 102, targets, 0)
            targets = [targets[i][:(target_[i] > 0).sum()] for i in range(targets.shape[0])]
            query_texts = [query[0] + ' [MASK]' * (target_[i] > 0).sum() + query[-1] for i in range(len(targets))]

            inputs = self.tokenizer(query_texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
            masked_idx = (inputs.input_ids == 103).nonzero(as_tuple=True)
            query_is_valid = len(masked_idx[1]) > 0
            return {'inputs': inputs,
                    'masked_idx': masked_idx,
                    'targets': targets,
                    'query_is_valid': query_is_valid
                    }
        elif self.lm_task == 'prediction':
            target_texts = target_texts[:1]
            targets = self.tokenizer(target_texts, return_tensors="pt", max_length=512, truncation=True,
                                     padding=True).input_ids[:, 1:-1]

            one_token_inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
            one_token_masked_idx = (one_token_inputs.input_ids == 103).nonzero(as_tuple=True)[1]

            query_is_valid = len(one_token_masked_idx) > 0
            return {'inputs': one_token_inputs,
                    'masked_idx': one_token_masked_idx,
                    'targets': targets,
                    'query_is_valid': query_is_valid
                    }

    def get_topk_candidates(self, inputs, candidates_embs, my_scorer, k):
        candidates_embs = candidates_embs.to(self.device)
        relevance_score = my_scorer(**inputs, candidate_embs=candidates_embs)
        relevant_candidate_indices = torch.topk(relevance_score, k=k, dim=-1).indices
        return relevant_candidate_indices, relevance_score

    def do_lm(self, query, targets_text, candidates_info, k):
        query_info = self.tokenize_lm_query(query, targets_text)
        query_is_valid = query_info['query_is_valid']
        if not query_is_valid:
            return False, None

        def do_next_token_prediction(query_info, candidates_info):
            masked_query_inputs = query_info['inputs'].to(self.device)
            masked_idx = query_info['masked_idx']
            targets = query_info['targets'].to(self.device)

            input_ids = list(masked_query_inputs.input_ids[0])
            try:
                sep_idx = input_ids.index(102)  # [SEP]
            except:
                sep_idx = len(input_ids)
            masked_query_text = self.tokenizer.decode(input_ids[1:sep_idx])

            candidates_texts = candidates_info['text']
            my_scorer = CandidateScorer(self.query_embedder)
            candidates_inputs = self.tokenizer.batch_encode_candidates(candidates_info['text'], max_length=512,
                                                                       truncation=True,
                                                                       padding=True, return_tensors="pt").to(
                self.device)
            candidates_info['emb'] = my_scorer.embed(candidates_inputs)
            relevant_candidates, rel_scores = self.get_topk_candidates(masked_query_inputs, candidates_info['emb'],
                                                                       my_scorer,
                                                                       k=k)
            relevant_candidates = relevant_candidates[0]
            # rel_scores = rel_scores[0]
            texts = []
            combined_cand = ''
            retrieved_statements = []
            for j in range(len(relevant_candidates)):
                tmp = candidates_texts[relevant_candidates[j]]
                if isinstance(tmp, str):
                    candidate_text_ = tmp
                else:
                    candidate_text_ = tmp.decode('UTF-8')
                retrieved_statements.append(candidate_text_)
                combined_cand += candidate_text_ + ' '
            # rel_scores = rel_scores.sum(dim=-1, keepdim=True)
            texts.append('{} [SEP] {}'.format(masked_query_text, combined_cand))
            inputs = self.tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(
                self.device)

            outputs = self.encoder(**inputs)

            return {
                'logits': outputs.logits[torch.arange(outputs.logits.shape[0]), masked_idx, :],
                'target': targets,
                'retrieved': retrieved_statements
            }

        def do_model_preference(query_info, candidates_info):
            masked_query_inputs = query_info['inputs']
            targets = query_info['targets']

            alt_num = len(targets)

            masked_query_text = []
            for i in range(alt_num):
                sep_idx = (masked_query_inputs.input_ids[i] == 102).nonzero(as_tuple=True)[0]
                if len(sep_idx) == 0:
                    sep_idx = len(masked_query_inputs.input_ids[i])
                else:
                    sep_idx = sep_idx[0]
                masked_query_text.append(self.tokenizer.decode(masked_query_inputs.input_ids[i, 1:sep_idx]))

            candidates_texts = candidates_info['text']
            my_scorer = CandidateScorer(self.query_embedder)
            candidates_inputs = self.tokenizer.batch_encode_candidates(candidates_info['text'], max_length=512,
                                                                       truncation=True,
                                                                       padding=True, return_tensors="pt").to(
                self.device)
            candidates_info['emb'] = my_scorer.embed(candidates_inputs)

            relevant_candidates, rel_scores = self.get_topk_candidates(masked_query_inputs.to(self.device),
                                                                       candidates_info['emb'], my_scorer, k)
            relevant_cand_scores = torch.log_softmax(rel_scores, dim=-1)[torch.arange(rel_scores.shape[0]).unsqueeze(1), relevant_candidates]
            texts = []
            retrieved_statements = [[] for _ in range(alt_num)]
            true_tokens = masked_query_inputs.input_ids.clone()
            sent_scores = None
            mask_scores = torch.zeros(alt_num, 1)
            enriched_q = []
            for i in range(alt_num):
                texts = []
                true_tokens[i][true_tokens[i] == 103] = targets[i].to(self.device)
                enriched_q.append(masked_query_text[i] + " [SEP] ")
                for j in range(len(relevant_candidates[i])):
                    tmp = candidates_texts[relevant_candidates[i, j]]
                    if isinstance(tmp, str):
                        candidate_text_ = tmp
                    else:
                        candidate_text_ = tmp.decode('UTF-8')
                    retrieved_statements[i].append(candidate_text_)
                    enriched_q[-1] += candidate_text_ + " | "
                    texts.append('{} [SEP] {}'.format(masked_query_text[i], candidate_text_))
                # print(texts)
                inputs = self.tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(
                    self.device)
                outputs = self.encoder(**inputs)

                cand_sent_scores = []
                cand_mask_scores = torch.zeros(len(relevant_candidates[i]))
                tokens_num_wout_pad = (true_tokens[i] > 0).sum()
                for cand in range(len(relevant_candidates[i])):
                    cand_sent_scores.append(torch.log_softmax(
                        outputs.logits[cand, torch.arange(tokens_num_wout_pad), :], dim=-1)[torch.arange(tokens_num_wout_pad), true_tokens[i][:tokens_num_wout_pad]])
                    cand_mask_scores[cand] = torch.mean(
                        cand_sent_scores[-1][masked_query_inputs.input_ids[i][:tokens_num_wout_pad] == 103], dim=-1)
                cand_mask_scores = cand_mask_scores.to(relevant_cand_scores.device)
                
                # log (1/C \sum{ p(z|x)p(y|z) })
                mask_scores[i] = torch.logsumexp((relevant_cand_scores[i] + cand_mask_scores), 0) - torch.log(torch.tensor(len(relevant_candidates[i])*1.0))

            return {
                'alternative_mask_scores': mask_scores,
                'alternative_sentence_scores': sent_scores,
                'alternative_targets': targets,
                'retrieved': retrieved_statements[0],
            }

        if self.lm_task == 'target_ranking':
            o = do_model_preference(query_info, candidates_info)
            alt_tgt_scores = o['alternative_mask_scores'].view(-1)
            retrieveds = o['retrieved']
            best_alternative = torch.argmax(alt_tgt_scores, dim=-1)
            return True, {'predicted_alt': best_alternative, 'query': query, 'retrieved': retrieveds}
        elif self.lm_task == 'prediction':
            o = do_next_token_prediction(query_info, candidates_info)
            return True, o
        else:
            ValueError('Invalid lm method. Should be either target_ranking or prediction')

    def do_qa(self, query, _, candidates_info, k):
        self.qa_model.config.reader_beam_size = k
        self.qa_model.reader.reader_beam_size = k
        self.qa_model.config.searcher_beam_size = k

        if k == 0:
            candidates_info['text'] = ['']
            k = 1
        query_ids = self.tokenizer(query, return_tensors="pt").to(self.device)

        candidates_texts = candidates_info['text']
        scorer = CandidateScorer(self.qa_model.embedder)
        candidates_inputs = self.tokenizer.batch_encode_candidates(candidates_texts, max_length=512, truncation=True,
                                                                   padding=True, return_tensors="pt").to(self.device)
        candidates_info['emb'] = scorer.embed(candidates_inputs)

        # original  qa_model.block_emb 13353718 x 128
        # new       qa_model.block_emb F x 128
        self.qa_model.block_emb = candidates_info['emb']
        self.qa_model.retriever.block_records = np.char.encode(np.array(candidates_info['text']))
        reader_output, predicted_answer_ids, pred_block, retrieved_blocks_ids = self.qa_model(**query_ids,
                                                                                              return_dict=False,
                                                                                              k=k)
        predicted_answer = self.tokenizer.decode(predicted_answer_ids)
        if len(retrieved_blocks_ids.shape) == 0:
            retrieveds = [candidates_texts[retrieved_blocks_ids]]
        else:
            retrieveds = [candidates_texts[i] for i in retrieved_blocks_ids]

        return {'query': query, 'retrieved': retrieveds, 'answer': predicted_answer}
    
    def do_comparison_qa(self, query, options, candidates_info, k):
        self.qa_model.config.reader_beam_size = k
        self.qa_model.reader.reader_beam_size = k
        self.qa_model.config.searcher_beam_size = k

        if k == 0:
            candidates_info['text'] = ['']
            k = 1
        query_ids = self.tokenizer(query, return_tensors="pt").to(self.device)

        option_ids = []
        for option in options:
            option_ids.append(self.tokenizer([option], add_special_tokens=False, return_token_type_ids=False, return_attention_mask = False).input_ids)

        candidates_texts = candidates_info['text']
        scorer = CandidateScorer(self.qa_model.embedder)
        candidates_inputs = self.tokenizer.batch_encode_candidates(candidates_texts, max_length=512, truncation=True,
                                                                   padding=True, return_tensors="pt").to(self.device)
        candidates_info['emb'] = scorer.embed(candidates_inputs)

        # original  qa_model.block_emb 13353718 x 128
        # new       qa_model.block_emb F x 128
        self.qa_model.block_emb = candidates_info['emb']
        self.qa_model.retriever.block_records = np.char.encode(np.array(candidates_info['text']))
        options_loss = torch.zeros(len(options))
        for option_i, option_id in enumerate(option_ids):
            reader_output, predicted_answer_ids, pred_block, retrieved_blocks_ids = self.qa_model(**query_ids, 
                                                                                                  answer_ids=option_id,
                                                                                                  return_dict=False,
                                                                                                  k=k)
            options_loss[option_i] = reader_output.loss
        if len(retrieved_blocks_ids.shape) == 0:
            retrieveds = [candidates_texts[retrieved_blocks_ids]]
        else:
            retrieveds = [candidates_texts[i] for i in retrieved_blocks_ids]

        return {'query': query, 'retrieved': retrieveds, 'answer': options[torch.argmin(options_loss)]}


def load_models(args, device=None):
    print('Loading models...')
    tokenizer, encoder, retriever, qa_model, query_embedder = None, None, None, None, None
    if args.reason_task == 'lm':
        tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-scorer")
        query_embedder = RealmEmbedder.from_pretrained("google/realm-orqa-nq-openqa").to(device)
        encoder = RealmKnowledgeAugEncoder.from_pretrained("google/realm-cc-news-pretrained-encoder",
                                                           num_candidates=args.reason_k).to(device)
    elif args.reason_task == 'qa':
        tokenizer = RealmTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
        query_embedder = RealmEmbedder.from_pretrained("google/realm-orqa-nq-openqa").to(device)
        retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
        qa_model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=retriever).to(device)
    print('Loaded models.')
    return {'tokenizer': tokenizer, 'encoder': encoder, 'retriever': retriever, 'qa_model': qa_model,
            'query_embedder': query_embedder}


def truncate_and_pad_candidates(facts, MAX_CANDIDATE_NUM=None, MIN_CANDIDATE_NUM=None):
    candidate_num = max(min(len(facts), MAX_CANDIDATE_NUM), MIN_CANDIDATE_NUM)
    if len(facts) < candidate_num:
        facts += [''] * (candidate_num - len(facts))
    else:
        facts = random.sample(facts, candidate_num)
    return facts
