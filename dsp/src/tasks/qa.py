# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from src.evaluation import exact_match_score, f1_score, normalize_answer
from src.options import Options
from src.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["exact_match", "f1", "eval_loss"]

    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()
        self.qa_prompt_format_str = opt.qa_prompt_format
        self.qa_answer_format_str = opt.qa_answer_format

    def get_qa_prompt(self, question: str) -> str:
        return self.qa_prompt_format_str.format(question=question)

    def process(self, example, fact_type=None, *args, **kwargs):
        if "answer" in example:
            target = example["answer"][0]
        elif "answers" in example:
            target = random.choice(example["answers"])
        else:
            target = None

        assert fact_type in [None, 'facts', 'gold_facts', 'single_fact']
        if not fact_type or fact_type == 'facts':
            example["passages"] = [{"title": "", "text": t} for t in example['facts']]
        elif fact_type == 'gold_facts':
            example["passages"] = [{"title": "", "text": t} for t in example['gold_facts']]
        elif fact_type == 'single_fact':
            assert 'hypothesis' in example
            example["passages"] = [{"title": "", "text": example['hypothesis']}]
        example["metadata"] = example.get("metadata", {})
        example["query"] = self.get_qa_prompt(example["question"])
        if target is not None:
            # example["answer"] = f"<extra_id_0> {target}"
            example["answer"] = self.qa_answer_format_str.format(target=target)

        return example

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
