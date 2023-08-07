from __future__ import annotations
import dsp
from dsp.utils import deduplicate, dotdict


class Contriever:
    def __init__(self, rm, index, opt=None, task=None):
        self.rm = rm
        self.index = index
        self.opt = opt
        self.task = task

    def __call__(self, query, facts, answers, k=2):
        query = [query]
        batch_metadata = [{}]
        target_tokens = None

        # Contriever retriever
        query_enc, _, _ = self.rm.tokenize(query, answers, target_tokens=target_tokens)
        # retrieve
        # retrieved_passages = retrieve_topk(query, query_enc, batch, batch_metadata=batch_metadata, index=index,
        #                                    model=model, unwrapped_model=unwrapped_model, opt=opt, task=task)
        all_passages = [{"title": "", "text": t} for t in facts]
        self.index.init_embeddings(all_passages)
        self.rm.build_index(self.index, all_passages, self.opt.per_gpu_embedder_batch_size, logger=None)
        query_ids_retriever = query_enc["input_ids"].cuda()
        query_mask_retriever = query_enc["attention_mask"].cuda()
        retrieved_passages, scores = self.rm.retrieve(
            self.index,
            min(k, len(all_passages)),
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata=batch_metadata,
            filtering_fun=self.task.filter,
        )
        topk = [{'long_text': p['text'], 'score': s} for p, s in zip(retrieved_passages[0], scores[0])]
        return [dotdict(psg) for psg in topk]


def retrieve(query: str, facts: list[str], k: int) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, facts, [""], k=k)
    # print("passages in retrieve", passages)
    passages = [psg.long_text for psg in passages]

    return passages


def retrieveEnsemble(queries: list[str], facts: list[str], k: int, by_prob: bool = True) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")

    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, facts, [""], k=k * 3):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages
