## Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model
> Parishad BehnamGhader, Santiago Miret, Siva Reddy

This repository contains the source code for the evaluations in [Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model](https://arxiv.org/abs/2212.09146).

---

### Preparing Datasets

This section includes the code for preparing datasets for both QA and LM experiments.

#### Dependencies

- [Spacy](https://spacy.io/usage) (tested with version 3.4.3)

#### Preparation

You can use the following script to generate QA and LM json datastores for EntailmentBank and StrategyQA datasets. Note that in the experiments on StrategyQA, we have changed the question and answers to a declarative format to perform LM.
```bash
cd data
python prepare_data.py --input_file raw/entailmentbank/task_1/dev.jsonl --output_file entailmentbank_1_dev --dataset entailmentbank --qa 1 --lm 1
python prepare_data.py --input_file raw/strategyqa/strategyqa_declarative_train.json --output_file strategyqa --dataset strategyqa --split 1 --lm 1
```
If you wish to use your own data samples, you must follow the following json data format:

**QA**
```json
[
  {
    "question": "Which event occurs on a daily cycle?", 
    "answer": ["The Sun rises and sets."], %(for cases like StrategyQA, the answer would be like ["yes", "no"] with "yes" being the correct answer)
    "facts": ["The sun rising / setting occurs once per day."],
    "gold_facts": ["The sun rising / setting occurs once per day."], %(used when evaluating the models with only ground-truth facts)
    "hypothesis": "The sun rising and setting is the event that occurs once per day." %(used when evaluating the models with one single hypothesis sentence.)
  },
]
```

**LM**
```json
[
  {
    "query": "As the distance of the star to earth decreases, the [MASK] will appear brighter.",
    "target": ["star", "space"], %(the first alternative target is the ground-truth masked entity)
    "facts": ["A star produces light and heat."], 
    "gold_facts":["A star produces light and heat."] %(used when evaluating the models with only ground-truth facts)
  },
]
```
---
### Experiments

We evaluate the reasoning abilities of the following retriever-augmented language models:
1. [REALM](https://huggingface.co/docs/transformers/model_doc/realm)
2. [kNN-LM](https://github.com/urvashik/knnlm)
3. [DPR + FiD](https://github.com/facebookresearch/FiD)
4. [Contriever + ATLAS](https://github.com/facebookresearch/atlas)
5. [DPR + Flan-T5](https://huggingface.co/google/flan-t5-base)


You may find the visualization of the results in `visualization.ipynb`. The dependencies for each model is mentioned briefly. You may want to look at each model's repository for more information.

<details><summary>1. REALM</summary>
<p>

##### Dependencies
- python 3 (tested with 3.7)
- pytorch (tested with 1.11.0)
- transformers (tested with 4.20.1)
- numpy

You may want to use `realm/environment.yml` as well.

##### Experiments
The following scripts run all kinds of experiments
```bash
cd realm

#QA
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_task qa \
  --reason_k 5 \
  --reason_dataset <entailmentbank / strategyqa>
  
#LM (target ranking)
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_task lm \
  --reason_k 5 \
  --reason_dataset <entailmentbank / strategyqa>
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_lm_task`: 'target_ranking' (model preference) | 'prediction' (masked token prediction)
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'entailmentbank' | 'strategyqa'
</p></details>

<details><summary>2. kNN-LM</summary>
<p>

##### Dependencies
- python 3 (tested with 3.7)
- pytorch (tested with 1.11.0)
- faiss-gpu (tested with 1.7.1)
- numpy

You may want to use `knnlm/environment.yml` as well.

##### Experiments
In order to run the kNN-LM experiments, you must first download the checkpoint provided in the [paper's code repository](https://github.com/urvashik/knnlm), prepare the datastores and dictionary. The following scripts run all kinds of experiments
```bash
cd knnlm

#QA
python evaluate_reasoning.py data-bin/wikitext-103 \
  --path checkpoints/checkpoint_best.pt --sample-break-mode complete \
  --max-tokens 3072 --context-window 2560 --softmax-batch 1024 \
  --model-overrides "{'knn_keytype': 'last_ffn_input'}" --knn-keytype 'last_ffn_input' \
  --knnlm --k 5 --lmbda 0.65 \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_task qa \
  --reason_dataset <entailmentbank / strategyqa>
  
#LM
python evaluate_reasoning.py data-bin/wikitext-103 \
  --path checkpoints/checkpoint_best.pt --sample-break-mode complete \
  --max-tokens 3072 --context-window 2560 --softmax-batch 1024 \
  --model-overrides "{'knn_keytype': 'last_ffn_input'}" --knn-keytype 'last_ffn_input' \
  --knnlm --k 5 --lmbda 0.65 \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_task lm \
  --reason_dataset <entailmentbank / strategyqa>
```

A list of the script arguments is explained below:
- `k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'entailmentbank' | 'strategyqa'
</p></details>

<details><summary>3. DPR + FiD</summary>
<p>

##### Dependencies
- python 3 (tested with 3.7)
- pytorch (tested with 1.6.0)
- transformers (tested with 3.0.2)
- faiss-cpu (tested with 1.6.1)
- numpy

You may want to use `dpr/fid_environment.yml` as well.

##### Experiments
In order to run the fid experiments, you must first prepare the [DPR retriever](https://github.com/facebookresearch/DPR). In these experiments, we load the `nq_retriever` checkpoints. 
Also, we use the `nq_reader_base` checkpoint for the FiD model available in [the model's github repository](https://github.com/facebookresearch/FiD).
The following scripts run all kinds of experiments. `dpr` arguments refer to the retriever's arguments, `lm` arguments refer to the model's arguments, and `reason` arguments refer to the specific arguments for our experiments. 
```bash
cd dpr

#QA
python evaluate_reasoning.py \
  dpr.model_file=<absolute address of the retriever .cp file> \
  lm.model_path=<absolute address of the pretrained reader directory> \
  lm.per_gpu_batch_size=1 lm.name=reason lm.checkpoint_dir=checkpoint \
  reason.data_file=<absolute address of the preprocessed json data file> \
  reason.output_file=<absolute address of a report.jsonl file> \
  reason.k=5 \
  reason.task=qa \
  reason.dataset=<entailmentbank / strategyqa>

  
#LM
python evaluate_reasoning.py \
  dpr.model_file=<absolute address of the retriever .cp file> \
  lm.model_path=<absolute address of the pretrained reader directory> \
  lm.per_gpu_batch_size=1 lm.name=reason lm.checkpoint_dir=checkpoint \
  reason.data_file=<absolute address of the preprocessed json data file> \
  reason.output_file=<absolute address of a report.jsonl file> \
  reason.k=5 \
  reason.task=lm \
  reason.dataset=<entailmentbank / strategyqa>
```

A list of the script arguments is explained below:
- `k`: number of retrieved statements
- `data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `output_file`: absolute address of a report.jsonl file
- `task`: 'qa' | 'lm'
- `fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `dataset`: 'strategyqa' | 'entailmentbank'
</p></details>

<details><summary>4. Contriever + ATLAS</summary>
<p>

##### Dependencies
- python 3 (tested with 3.8)
- pytorch (tested with 1.11.0)
- transformers (tested with 4.18.0)
- faiss-gpu (tested with 1.7.2)
- numpy

You may want to use `contriever/contriever_environment.yml` as well.

##### Experiments
In order to run the ATLAS experiments, you must first download the preferred model from [ATLAS github](https://github.com/facebookresearch/atlas). In our experiments we load the `models/atlas_nq/base` ATLAS model.
The following scripts run all kinds of experiments.
```bash
cd contriever
port=$(shuf -i 15000-16000 -n 1)

#QA
python evaluate_atlas_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512 \
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task qa \
  --reason_dataset <entailmentbank / strategyqa>
  
#LM
python evaluate_atlas_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512\
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task lm \
  --reason_dataset <entailmentbank / strategyqa>
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'strategyqa' | 'entailmentbank'
</p></details>

<details><summary>5. Contriever + Flan-T5</summary>
<p>

##### Dependencies
- python 3 (tested with 3.8)
- pytorch (tested with 1.11.0)
- transformers (tested with 4.18.0)
- faiss-gpu (tested with 1.7.2)
- numpy

You may want to use `contriever/contriever_environment.yml` as well.

##### Experiments
In order to run the flan-t5 experiments, you must first download the preferred model from [ATLAS github](https://github.com/facebookresearch/atlas). In our experiments we load the `models/atlas_nq/base` ATLAS model.
The following scripts run all kinds of experiments.
```bash
cd contriever
port=$(shuf -i 15000-16000 -n 1)

#QA
python evaluate_flan_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512 \
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task qa \
  --reason_dataset <entailmentbank / strategyqa>
  --reason_lm <google/flan-t5-small, google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl, google/flan-t5-xxl>
  
#LM
python evaluate_flan_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512\
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task lm \
  --reason_dataset <entailmentbank / strategyqa>
  --reason_lm <google/flan-t5-small, google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl, google/flan-t5-xxl>
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'strategyqa' | 'entailmentbank'
- `reason_lm`: optional flan model to evaluate (used basically for model size evaluations) 'google/flan-t5-small' | 'google/flan-t5-base' | 'google/flan-t5-large' | 'google/flan-t5-xl' | 'google/flan-t5-xxl'
- `reason_fewshot`: optional, if you want to use fewshot examples, use 'boolean' for StrategyQA or 'short' for Entailmentbank experiments. This arg was used to compare fewshot Flan and GPT performance vs. the DSP variant.
</p></details>

<details><summary>6. Demonstrate-Search-Predict</summary>
<p>

##### Dependencies
- python 3 (tested with 3.8)
- pytorch (tested with 1.11.0)
- transformers (tested with 4.18.0)
- faiss-gpu (tested with 1.7.2)
- dsp

You may want to use `contriever/dsp_environment.yml` as well.

##### Experiments
First download the preferred model from [ATLAS github](https://github.com/facebookresearch/atlas). In our experiments we load the `models/atlas_nq/base` ATLAS model.
```bash
cd contriever
port=$(shuf -i 15000-16000 -n 1)

#QA
python evaluate_dsp_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512 \
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task qa \
  --reason_dataset <entailmentbank / strategyqa> \
  --reason_lm <google/flan-t5-base, google/flan-t5-xxl, text-davinci-002> \
  --reason_openai_key <the openai key>
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'strategyqa' | 'entailmentbank'
- `reason_lm`: optional flan model to evaluate (used basically for model size evaluations) 'google/flan-t5-base' | 'google/flan-t5-xxl' | 'text-davinci-002'
- `reason_fewshot`: use 'boolean' for StrategyQA and 'short' for Entailmentbank experiments
</p></details>
In order to reproduce the visualizations in the paper, please run `tests/create_visualization_data.py` to export the results. Then, you might want to copy the results to the `visualization.ipynb`.
