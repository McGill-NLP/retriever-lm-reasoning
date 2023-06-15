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
    metrics = ["hits1", "eval_loss"]

    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()

    def get_q_prompt(self, question: str) -> str:
        return question.replace('[MASK]', '<extra_id_0>')

    def process(self, example, fact_type=None, *args, **kwargs):
        if "target" in example:
            target = example["target"]
        elif "answers" in example:
            target = random.choice(example["answers"])
        else:
            target = None

        assert fact_type in [None, 'facts', 'gold_facts']
        if not fact_type or fact_type == 'facts':
            example["passages"] = [{"title": "", "text": t} for t in example['facts']]
        elif fact_type == 'gold_facts':
            example["passages"] = [{"title": "", "text": t} for t in example['gold_facts']]

        example["metadata"] = example.get("metadata", {})
        example["query"] = self.get_q_prompt(example["query"])
        if target is not None:
            if isinstance(target, str):
                example["answer"] = f"<extra_id_0> {target}"
            elif isinstance(target, list):
                example["answer"] = [f"<extra_id_0> {t}" for t in target]

        return example

    def evaluation(self, prediction, ground_truths):
        return
