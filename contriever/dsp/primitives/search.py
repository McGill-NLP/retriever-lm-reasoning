import dsp
from typing import Dict, List


def retrieve(query: str, k: int) -> List[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=k)
    passages = [psg.long_text for psg in passages]

    return passages


def retrieveEnsemble(queries: List[str], k: int, by_prob: bool = True) -> List[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")

    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, k=k * 3):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages
