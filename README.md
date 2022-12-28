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
    "answer": ["The Sun rises and sets."], 
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
3. [FiD + DPR](https://github.com/facebookresearch/FiD)
4. [ATLAS + Contriever](https://github.com/facebookresearch/atlas)
5. [Flan-T5 + DPR](https://huggingface.co/google/flan-t5-base)


The dependencies for each model is mentioned briefly. You may want to look at each model's repository for more information.

<details><summary>1. REALM</summary>
<p>

##### Dependencies
- Python 3 (tested with 3.7)
- Pytorch (tested with 1.11.0)
- Transformers (tested with 4.20.1)
- NumPy

You may want to use `realm/environment.yml` as well.

##### Experiments
The following scripts run all kinds of experiments
```bash
cd realm

#QA
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.txt file> \
  --reason_task qa
  --reason_k 5
  
#LM (model preference)
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.txt file> \
  --reason_task lm
  --reason_k 5
  --reason_lm_task alt

#LM (masked token prediction)
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.txt file> \
  --reason_task lm
  --reason_k 5
  --reason_lm_task pred
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.txt file
- `reason_task`: 'qa' | 'lm'
- `reason_lm_task`: 'alt' (model preference) | 'pred' (masked token prediction)
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
</p></details>

<details><summary>2. kNN-LM</summary>
<p>

##### Dependencies
- Python 3 (tested with 3.7)
- Pytorch (tested with 1.11.0)
- faiss-gpu (tested with 1.7.1)
- NumPy

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
  --reason_output_file <absolute address of a report.txt file> \
  --reason_task qa
  
#LM
python evaluate_reasoning.py data-bin/wikitext-103 \
  --path checkpoints/checkpoint_best.pt --sample-break-mode complete \
  --max-tokens 3072 --context-window 2560 --softmax-batch 1024 \
  --model-overrides "{'knn_keytype': 'last_ffn_input'}" --knn-keytype 'last_ffn_input' \
  --knnlm --k 5 --lmbda 0.65 \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.txt file> \
  --reason_task lm
```

A list of the script arguments is explained below:
- `k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.txt file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
</p></details>

<details><summary>3. FiD + DPR</summary>
<p>

##### Dependencies
- Python 3 (tested with 3.7)
- Pytorch (tested with 1.11.0)
- Transformers (tested with 4.20.1)
- faiss-cpu (tested with 1.6.1)
- NumPy

You may want to use `dpr_lm/flan_environment.yml` as well.

##### Experiments
In order to run the flan-t5 experiments, you must first prepare the [DPR retriever](https://github.com/facebookresearch/DPR). In these experiments, we load the `nq_retriever` checkpoints. 
Also, we use the `nq_reader_base` checkpoint for the FiD model available in [the model's github repository](https://github.com/facebookresearch/FiD).
The following scripts run all kinds of experiments. `dpr` arguments refer to the retriever's arguments, 'lm' arguments refer to the model's arguments, and `reason` arguments refer to the specific arguments for our experiments. 
```bash
cd dpr_lm

#QA
python evaluate_reasoning.py \
  dpr.model_file=<absolute address of the retriever .cp file> \
  lm.model_path=<absolute address of the pretrained reader directory> \
  lm.per_gpu_batch_size=1 lm.name=reason lm.checkpoint_dir=checkpoint \
  reason.data_file=<absolute address of the preprocessed json data file> \
  reason.output_file=<absolute address of a report.txt file> \
  reason.k=5 \
  reason.lm=fid \
  reason.task=qa
  
#LM
python evaluate_reasoning.py \
  dpr.model_file=<absolute address of the retriever .cp file> \
  lm.model_path=<absolute address of the pretrained reader directory> \
  lm.per_gpu_batch_size=1 lm.name=reason lm.checkpoint_dir=checkpoint \
  reason.data_file=<absolute address of the preprocessed json data file> \
  reason.output_file=<absolute address of a report.txt file> \
  reason.k=5 \
  reason.lm=fid \
  reason.task=lm
```

A list of the script arguments is explained below:
- `k`: number of retrieved statements
- `data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `output_file`: absolute address of a report.txt file
- `task`: 'qa' | 'lm'
- `fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `lm`: 'fid' ('flan' and 'fid' models use the same base code in our experiments.)
</p></details>

<details><summary>4. ATLAS + Contriever</summary>
<p>

##### Dependencies
- Python 3 (tested with 3.7)
- Pytorch (tested with 1.11.0)
- Transformers (tested with 4.18.0)
- faiss-gpu (tested with 1.7.2)
- NumPy

You may want to use `contriever_atlas/environment.yml` as well.

##### Experiments
In order to run the ATLAS experiments, you must first download the preferred model from [ATLAS github](https://github.com/facebookresearch/atlas). In our experiments we load the `models/atlas_nq/base` ATLAS model.
The following scripts run all kinds of experiments.
```bash
cd contriever_atlas
port=$(shuf -i 15000-16000 -n 1)

#QA
python evaluate_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512 \
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.txt file> \
  --reason_k 5 \
  --reason_task qa
  
#LM
python evaluate_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512\
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.txt file> \
  --reason_k 5 \
  --reason_task lm
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.txt file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
</p></details>

<details><summary>5. Flan-T5 + DPR</summary>
<p>

##### Dependencies
- Python 3 (tested with 3.7)
- Pytorch (tested with 1.11.0)
- Transformers (tested with 4.20.1)
- faiss (tested with 1.5.3)
- faiss-cpu (tested with 1.6.1)
- NumPy

You may want to use `dpr_lm/flan_environment.yml` as well.

##### Experiments
In order to run the flan-t5 experiments, you must first prepare the [DPR retriever](https://github.com/facebookresearch/DPR). In these experiments, we load the `nq_retriever` checkpoints. As the model, we load the `google/flan-t5-base` checkpoints of the Flan-T5 model available in [HuggingFace](https://huggingface.co/google/flan-t5-base). The following scripts run all kinds of experiments. `dpr` arguments refer to the retriever's arguments, and `reason` arguments, refer to the specific arguments for our experiments. 
```bash
cd dpr_lm

#QA
python evaluate_reasoning.py \
  dpr.model_file=<absolute address of the retriever .cp file> \
  reason.data_file=<absolute address of the preprocessed json data file> \
  reason.output_file=<absolute address of a report.txt file> \
  reason.k=5 \
  reason.lm=flan \
  reason.task=qa
  
#LM
python evaluate_reasoning.py \
  dpr.model_file=<absolute address of the retriever .cp file> \
  reason.data_file=<absolute address of the preprocessed json data file> \
  reason.output_file=<absolute address of a report.txt file> \
  reason.k=5 \
  reason.lm=flan \
  reason.task=lm
```

A list of the script arguments is explained below:
- `k`: number of retrieved statements
- `data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `output_file`: absolute address of a report.txt file
- `task`: 'qa' | 'lm'
- `fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `lm`: 'flan' ('flan' and 'fid' models use the same base code in our experiments.)
</p></details>
