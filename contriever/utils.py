import torch
from misc_utils import normalize_answer


class RetLM():
    def __init__(self, unwrapped_model, model, reader_tokenizer, task=None, index=None, reason_task=''):
        self.unwrapped_model = unwrapped_model
        self.model = model
        self.reader_tokenizer = reader_tokenizer
        self.task = task
        self.index = index

        self.reason_task = reason_task
        if reason_task == 'qa':
            self.get_answer = self.do_qa
        elif reason_task == 'compare_qa':
            self.get_answer = self.do_comparison_qa
        elif reason_task == 'lm':
            self.get_answer = self.do_lm
        else:
            ValueError('Invalid task for RetLM.')

    def do_lm(self, batch, opt=None):
        pass

    def do_qa(self, batch, opt=None):
        pass

    def do_comparison_qa(self, batch, opt=None):
        pass

    def retrieve_topk(self, query, query_enc, batch, batch_metadata=None, opt=None):
        assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
        all_passages = batch["passages"][0]
        self.index.init_embeddings(all_passages)
        self.model.build_index(self.index, all_passages, opt.per_gpu_embedder_batch_size, logger=None)
        query_ids_retriever = query_enc["input_ids"].cuda()
        query_mask_retriever = query_enc["attention_mask"].cuda()
        retrieved_passages, _ = self.unwrapped_model.retrieve(
            self.index,
            min(opt.n_context, len(all_passages)),
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata=batch_metadata,
            filtering_fun=self.task.filter,
        )
        return retrieved_passages


class RetAtlas(RetLM):
    def __init__(self, unwrapped_model, model, reader_tokenizer, task=None, index=None, reason_task=''):
        super().__init__(unwrapped_model, model, reader_tokenizer, task=task, index=index, reason_task=reason_task)
  

    def do_lm(self, batch, opt=None):
        best_alternatives, predicted_tokens_list, examples, retrieved = [], [], [], []
        alternative_prediction_num, first_token_prediction_num, total = 0, 0, 0
        batch_query = batch.get("query", [""])
        batch_answers = batch.get("answer", [""])
        batch_metadata = batch.get("metadata")
        batch_target_tokens = None

        for k in range(len(batch_answers)):
            if len(batch["passages"][k]) < 1:
                continue
            retrieved_passages = None
            alt_num = len(batch_answers[k])
            query = [batch_query[k]]

            examples.append({'query': query[k], 'target': batch_answers[k]})

            target_losses = torch.zeros(alt_num)
            for alt_i in range(alt_num):
                # retrieve
                query_enc, labels, decoder_input_ids = self.unwrapped_model.tokenize(query, [batch_answers[k][alt_i]],
                                                                                     target_tokens=batch_target_tokens)
                if not retrieved_passages:
                    retrieved_passages = self.retrieve_topk(query, query_enc, batch, batch_metadata=batch_metadata,
                                                            opt=opt)
                reader_tokens, _ = self.unwrapped_model.tokenize_passages(query, retrieved_passages)

                loss, _ = self.unwrapped_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
                target_losses[alt_i] = loss
            predicted_alt = torch.argmin(target_losses)
            best_alternatives.append(predicted_alt)
            # alternatives.append(batch_answers[k])
            alternative_prediction_num += int(predicted_alt == 0)

            generation = self.unwrapped_model.generate(
                reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
            )
            g = generation[0]
            if opt.decoder_prompt_format is not None:
                query_ids = self.reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                g = g[len(query_ids) + 1:]
            pred = self.reader_tokenizer.decode(g, skip_special_tokens=True)

            ans_ids = self.reader_tokenizer(pred).input_ids
            first_token_pred = ans_ids[0]
            tgt_ids = self.reader_tokenizer(batch_answers[k][0]).input_ids
            first_token_prediction_num += int(first_token_pred in tgt_ids)

            retrieved.append([p['text'] for p in retrieved_passages[0]])
            predicted_tokens_list.append(self.reader_tokenizer.decode(first_token_pred))
            total += 1
            return {'hits1': first_token_prediction_num,
                    'first_token_pred': predicted_tokens_list,
                    'retrieved': retrieved,
                    'predicted_alt': best_alternatives,
                    'example': examples
                    }

    def do_qa(self, batch, opt=None):
        if len(batch["passages"][0]) < 1:
            return False, None
        query = batch.get("query", [""])
        answers = batch.get("answer", [""])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")

        query_enc, _, _ = self.unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
        # retrieve
        retrieved_passages = self.retrieve_topk(query, query_enc, batch, batch_metadata=batch_metadata, opt=opt)

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            return False, None

        reader_tokens, _ = self.unwrapped_model.tokenize_passages(query, retrieved_passages)

        generation = self.unwrapped_model.generate(
            reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
        )
        g = generation[0]

        if opt.decoder_prompt_format is not None:
            query_ids = self.reader_tokenizer.encode(
                opt.decoder_prompt_format.format_map({"query": query[0]}), add_special_tokens=False
            )
            g = g[len(query_ids) + 1:]
        pred = self.reader_tokenizer.decode(g, skip_special_tokens=True)
        gold = ' '.join(batch["answer"][0].split()[1:])

        return True, {'query': query[0],
                      'retrieved': [p['text'] for p in retrieved_passages[0]],
                      'gen_ans': pred,
                      'example': {'question': query[0], 'answer': [gold]}}

    def do_comparison_qa(self, batch, opt=None):
        if len(batch["passages"][0]) < 1:
            return False, None
        query = batch.get("query", [""])
        answers = batch.get("answer", [[""]])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")

        retrieved_passages = None
        alt_num = len(answers[0])
        query = [query[0]]
        target_losses = torch.zeros(alt_num)
        for alt_i in range(alt_num):
            # retrieve
            query_enc, labels, decoder_input_ids = self.unwrapped_model.tokenize(query, [answers[0][alt_i]],
                                                                                 target_tokens=target_tokens)
            if not retrieved_passages:
                retrieved_passages = self.retrieve_topk(query, query_enc, batch, batch_metadata=batch_metadata, opt=opt)
            reader_tokens, _ = self.unwrapped_model.tokenize_passages(query, retrieved_passages)

            loss, _ = self.unwrapped_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
            target_losses[alt_i] = loss
        predicted_alt = torch.argmin(target_losses)
        return True, {'query': query[0],
                      'retrieved': [p['text'] for p in retrieved_passages[0]],
                      'gen_ans': ' '.join(answers[0][predicted_alt].split()[1:]),
                      'example': {'question': query[0], 'answer': [' '.join(answers[0][0].split()[1:])]}}
