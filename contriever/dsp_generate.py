from __future__ import annotations
import dsp
from dsp.utils import deduplicate, dotdict
from dsp_search import retrieveEnsemble, retrieve
from few_shot_data import short_train, yes_no_train

from typing import Dict


def multihop_QA_v2(question, facts, demo_k=2, infer_k=5, answer_format=None, output_f=None) -> str:
    if answer_format == 'short':
        train = [dsp.Example(question=d['question'], answer=d['answer'], facts=d['context']) for d in short_train]

        Question = dsp.Type(prefix="Question:", 
                            desc="${the question to be answered}")

        Answer = dsp.Type(prefix="Answer:", 
                        desc="${a short factoid answer, often between 1 and 5 words}", 
                        format=dsp.format_answers)
        qa_template = dsp.Template(instructions="Answer questions with short factoid answers.",
                                question=Question(),
                                answer=Answer())
    elif answer_format == 'boolean':
        train = [dsp.Example(question=d['question'], answer=d['answer'], facts=d['context']) for d in yes_no_train]

        Question = dsp.Type(prefix="Question:", 
                            desc="${the question to be answered}")

        Answer = dsp.Type(prefix="Answer:", 
                          desc="${a yes or no answer.}", 
                          format=dsp.format_answers)
        qa_template = dsp.Template(instructions="Answer questions with yes or no.",
                                   question=Question(),
                                   answer=Answer())
    else:
        ValueError('Invalid answer reason_fewshot arg', answer_format)

    Context = dsp.Type(
        prefix="Context:\n",
        desc="${sources that may contain relevant content}",
        format=dsp.passages2text
    )

    SearchRationale = dsp.Type(
        prefix="Rationale: Let's think step by step. To answer this question, we first need to find out",
        desc="${the missing information}"
    )

    SearchQuery = dsp.Type(
        prefix="Search Query:",
        desc="${a simple question for seeking the missing information}"
    )

    Rationale = dsp.Type(
        prefix="Rationale: Let's think step by step.",
        desc="${a step-by-step deduction that identifies the correct response, which will be provided below}"
    )


    @dsp.transformation
    def QA_predict(example: dsp.Example, sc=True):
        if sc:
            example, completions = dsp.generate(qa_template_with_CoT, n=1, temperature=0.7)(example, stage='qa')
            completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(qa_template_with_CoT)(example, stage='qa')

        return example.copy(answer=completions.answer)


    qa_template_with_CoT = dsp.Template(
        instructions=qa_template.instructions,
        context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
    )

    rewrite_template = dsp.Template(
        instructions="Write a search query that will help answer a complex question.",
        question=Question(), rationale=SearchRationale(), query=SearchQuery()
    )

    CondenseRationale = dsp.Type(
        prefix="Rationale: Let's think step by step. Based on the context, we have learned the following.",
        desc="${information from the context that provides useful clues}"
    )

    hop_template = dsp.Template(
        instructions=rewrite_template.instructions,
        context=Context(), question=Question(), rationale=CondenseRationale(), query=SearchQuery()
    )


    @dsp.transformation
    def multihop_search_v1(example: dsp.Example, max_hops=2, k=2) -> dsp.Example:
        example.context = []

        for hop in range(max_hops):
            # Generate a query based
            template = rewrite_template if hop == 0 else hop_template
            example, completions = dsp.generate(template)(example, stage=f'h{hop}')

            # Retrieve k results based on the query generated
            passages = retrieve(completions.query, example.facts, k=k)
            # Update the context by concatenating old and new passages
            example.context = deduplicate(example.context + passages)

        return example


    def multihop_QA_v1(question: str, facts: list[str]) -> str:
        demos = dsp.sample(train, k=7)
        x = dsp.Example(question=question, facts=facts, demos=demos)

        x = multihop_search_v1(x, k=k)
        x = QA_predict(x, sc=False)

        return x.answer


    @dsp.transformation
    def multihop_attempt(d: dsp.Example, k=2) -> dsp.Example:
        # Prepare unaugmented demonstrations for the example.
        x = dsp.Example(question=d.question, demos=dsp.all_but(train, d), facts=d.facts)

        # Search. And skip examples where search fails.
        # Annotate demonstrations for multihop_search_v2 with the simpler multihop_search_v1 pipeline.
        x = multihop_search_v1(x, k=k)
        if not dsp.passage_match(x.context, d.answer): return None

        # Predict. And skip examples where predict fails.
        x = QA_predict(x, sc=False)
        if not dsp.answer_match(x.answer, d.answer): return None

        return d.copy(**x)


    @dsp.transformation
    def multihop_demonstrate(x: dsp.Example, k=2) -> dsp.Example:
        demos = dsp.sample(train, k=2)
        x.demos = dsp.annotate(multihop_attempt)(demos, k=k, return_all=True)
        return x


    @dsp.transformation
    def multihop_search_v2(example: dsp.Example, max_hops=2, k=5) -> dsp.Example:
        example.context = []
        retrieveds = []
        # print('example', example)
        for hop in range(max_hops):
            # Generate queries
            template = rewrite_template if hop == 0 else hop_template
            example, completions = dsp.generate(template, n=2, temperature=0.7)(example, stage=f'h{hop}')

            # Collect the queries and search with result fusion
            queries = [c.query for c in completions] + [example.question]
            _rets = retrieveEnsemble(queries, example.facts, k=k, by_prob=False)
            example.context = _rets
            retrieveds += _rets

            # Arrange the passages for the next hop
            if hop > 0:
                example.context = [completions[0].rationale] + example.context

        example.retrieveds = retrieveds
        return example
    
    x = dsp.Example(question=question, facts=[f['text'] for f in facts], retrieveds=[])
    x = multihop_demonstrate(x, k=demo_k)
    x = multihop_search_v2(x, k=infer_k)
    x = QA_predict(x)
    return x

