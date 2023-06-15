from argparse import ArgumentParser
import json


def arg_parser():
    parser = ArgumentParser()

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="/network/scratch/p/parishad.behnamghader/FiD/open_domain_data/reason/")
    parser.add_argument("--output-dir", type=str, default="/network/scratch/p/parishad.behnamghader/FiD/my_output/")
    parser.add_argument("--cache-dir", type=str, default="/network/scratch/p/parishad.behnamghader/.cache/huggingface/")
    parser.add_argument("--lm-result-suffix", type=str, default="")
    parser.add_argument("--dataset", type=str,
                        default="entailmentbank")
    parser.add_argument("--part", type=str, default="test")  # train, test, dev used for entailmentbank
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--sample-num", type=int, default=5000)
    parser.add_argument("--write", type=int, default=1)  # 0, 1
    parser.add_argument("--lm", type=str, default="fid")  # train, test, dev used for entailmentbank
    parser.add_argument("--qa", type=int, default=0)  # 0, 1
    parser.add_argument("--fact-type", type=str, default="facts")  # facts, gold_facts

    return parser


def get_retriever_performance(my_args):
    if args.qa == 1:
        if args.dataset == 'entailmentbank':
            dataset_file_name = '{}_{}'.format(args.dataset, args.task)
            datastore_addr = my_args.data_dir + 'qa_entailment_{}_{}.json'.format(my_args.task, my_args.part)
        else:
            dataset_file_name = args.dataset

        file_name = '{}{}/qa/ds_{}_part_{}_num_{}_k_{}_{}_{}.txt'.format(my_args.output_dir, my_args.lm,
                                                                   dataset_file_name, my_args.part, my_args.sample_num,
                                                                   my_args.k, my_args.fact_type, my_args.lm_result_suffix)
    elif args.qa == 0:
        if args.dataset == 'entailmentbank':
            dataset_file_name = '{}_{}'.format(args.dataset, args.task)
            datastore_addr = my_args.data_dir + 'entailment_{}_{}_custom.json'.format(my_args.task, my_args.part)
        elif args.dataset == 'strategyqa':
            dataset_file_name = '{}_{}'.format(args.dataset, args.sqa_yes)
            datastore_addr = my_args.data_dir + 'strategyqa.json'
        else:
            dataset_file_name = args.dataset

        file_name = '{}{}/ds_{}_part_{}_num_{}_k_{}_{}.txt'.format(my_args.output_dir, my_args.lm,
                                                                   dataset_file_name, my_args.part, my_args.sample_num,
                                                                   my_args.k, my_args.lm_result_suffix)
    # file_name = args.output_dir + 'ds_{}_enc_{}_cand_{}_comb_{}_{}_part_{}_num_{}_cot_{}_target_{}_k_{}.txt'.format(
    #     dataset_file_name, args.encoder_model, args.candidate, args.cand_combination, args.mlm_result_suffix, args.part,
    #     args.sample_num, args.cot, args.target, args.selecting_cand_num)
    output_f = open(file_name, 'r')
    with open(datastore_addr, 'r') as fin:
        dataset_f = json.load(fin)
    data_id = 0
    sample_prec, sample_all_gold, sample_recall, sample_f1, sample_all_ret, sample_true_ret = [], [], [], [], [], []
    q_prefix = 'original query: '
    for line in output_f:
        if line.startswith(q_prefix):
            line = line[len(q_prefix):]
            data = dataset_f[data_id]
            # print(data_id)
            # print(type(data['question']), type(line))
            q = line[:len(data['question'])]
            ret_facts = set([fact.lower().strip() + '.' for fact in line[len(data['question']):].split('.')][:-1])
            # print(q, ret_facts)
            gold_facts = set([fact.lower().strip() for fact in data['gold_facts']])
            all_facts = set([fact['text'].lower().strip() for fact in data['ctxs']])
            # sents = line.split('.')
            # q = sents[0] + '.'
            # ret_facts = set([fact.lower().strip() + '.' for fact in sents[1:-1]])

            assert len(all_facts.intersection(ret_facts)) == len(ret_facts)
            if q.lower().strip() != data['question'].lower().strip():
                print(q.lower().strip(), '|', data['question'].lower().strip())

            if len(gold_facts) > 0:
                sample_all_ret.append(min(len(gold_facts), args.k))
                sample_all_gold.append(len(gold_facts))
                sample_true_ret.append(len(gold_facts.intersection(ret_facts)))
                p = len(gold_facts.intersection(ret_facts)) / min(len(gold_facts), args.k)
                r = len(gold_facts.intersection(ret_facts)) / len(gold_facts)
                sample_prec.append(p)
                sample_recall.append(r)
                if (p+r) != 0:
                    sample_f1.append(2 * p * r / (p+r))
                else:
                    sample_f1.append(0)

            data_id += 1

    output_f.close()
    if args.write == 1:
        output_f = open(file_name, 'a+')
        # output_f.write(
        #     '\nRetriever Precision: \tMicro {:.4f}, Macro {:.4f}\n'.format(sum(sample_prec) / len(sample_prec),
        #                                                                    sum(sample_true_ret) / sum(sample_all_ret)))
        # output_f.write(
        #     'Retriever Recall: \tMicro {:.4f}, Macro {:.4f}\n'.format(sum(sample_recall) / len(sample_recall),
        #                                                               sum(sample_true_ret) / sum(sample_all_gold)))
        output_f.write(
            'Retriever F1: \t\tMicro {:.4f}\n'.format(sum(sample_f1) / len(sample_f1)))

    print('Retriever Precision: \tMicro {:.4f}, Macro {:.4f}'.format(sum(sample_prec) / len(sample_prec), sum(sample_true_ret) / sum(sample_all_ret)))
    print('Retriever Recall: \tMicro {:.4f}, Macro {:.4f}'.format(sum(sample_recall) / len(sample_recall), sum(sample_true_ret) / sum(sample_all_gold)))
    print('Retriever F1: \t\tMicro {:.4f}'.format(sum(sample_f1) / len(sample_f1)))


if __name__ == '__main__':
    args = arg_parser().parse_args()
    # print_incorrect_predictions(args)
    # python check_realm_outputs.py --file ds_entailmentbank_1_cand_custom_comb_concat__num_200.txt
    get_retriever_performance(args)

