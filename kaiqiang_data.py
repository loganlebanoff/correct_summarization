import itertools
import os
from tqdm import tqdm
import numpy as np
from absl import flags
from absl import app
import pickle
import util
import sys
import glob
import data
import rouge_functions

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'train', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')
if 'mode' not in flags.FLAGS:
    flags.DEFINE_string('mode', 'write', 'Can be {write, evaluate}')
if 'train_dataset' not in flags.FLAGS:
    flags.DEFINE_string('train_dataset', 'cnn_dm', 'Can be {cnn_dm, gigaword}')

FLAGS(sys.argv)


import convert_data
# import lambdamart_scores_to_summaries
# import preprocess_for_lambdamart_no_flags

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('doc_indices', 'delimited_list')]
if FLAGS.dataset_name == 'duc_2004':
    names_to_types[2] = ('summary_text', 'string_list')
# names_to_types = [('raw_article_sents', 'string_list'), ('article', 'string'), ('abstract', 'string_list'), ('doc_indices', 'string')]
# names_to_types = [('raw_article_sents', 'string_list')]
min_matched_tokens = 1

def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    out_dir =  os.path.join(os.path.expanduser('~') + '/data/kaiqiang_data', FLAGS.dataset_name)
    if FLAGS.mode == 'write':
        util.create_dirs(out_dir)
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        elif FLAGS.dataset_split == 'all':
            dataset_splits = ['test', 'val', 'train']
        else:
            dataset_splits = [FLAGS.dataset_split]

        for dataset_split in dataset_splits:

            if dataset_split == 'test':
                ssi_data_path = os.path.join('logs/%s_bert_both_sentemb_artemb_plushidden' % FLAGS.dataset_name, 'ssi.pkl')
                print (util.bcolors.OKGREEN + "Loading SSI from BERT at %s" % ssi_data_path + util.bcolors.ENDC)
                with open(ssi_data_path) as f:
                    ssi_triple_list = pickle.load(f)

            source_dir = os.path.join(data_dir, FLAGS.dataset_name)
            source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

            total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
            example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False,
                                                       should_check_valid=False)

            out_document_path = os.path.join(out_dir, dataset_split + '.Ndocument')
            out_summary_path = os.path.join(out_dir, dataset_split + '.Nsummary')
            out_example_idx_path = os.path.join(out_dir, dataset_split + '.Nexampleidx')


            doc_writer = open(out_document_path, 'w')
            if dataset_split != 'test':
                sum_writer = open(out_summary_path, 'w')
            ex_idx_writer = open(out_example_idx_path, 'w')


            for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                    break
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, doc_indices = util.unpack_tf_example(
                    example, names_to_types)
                article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
                if FLAGS.dataset_name == 'duc_2004':
                    groundtruth_summ_sents = [[sent.strip() for sent in gt_summ_text.strip().split('\n')] for gt_summ_text in groundtruth_summary_text]
                else:
                    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
                if doc_indices is None:
                    doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
                doc_indices = [int(doc_idx) for doc_idx in doc_indices]
                # rel_sent_indices, _, _ = preprocess_for_lambdamart_no_flags.get_rel_sent_indices(doc_indices, article_sent_tokens)

                if dataset_split == 'test':
                    if example_idx >= len(ssi_triple_list):
                        raise Exception('Len of ssi list (%d) is less than number of examples (>=%d)' % (len(ssi_triple_list), example_idx))
                    ssi_length_extractive = ssi_triple_list[example_idx][2]
                    if ssi_length_extractive > 1:
                        a=0
                    ssi = ssi_triple_list[example_idx][1]
                    ssi = ssi[:ssi_length_extractive]
                    groundtruth_similar_source_indices_list = ssi
                else:
                    groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)

                for ssi_idx, ssi in enumerate(groundtruth_similar_source_indices_list):
                    if len(ssi) == 0:
                        continue
                    my_article = ' '.join(util.reorder(raw_article_sents, ssi))
                    doc_writer.write(my_article + '\n')
                    if dataset_split != 'test':
                        sum_writer.write(groundtruth_summ_sents[0][ssi_idx] + '\n')
                    ex_idx_writer.write(str(example_idx) + '\n')
    elif FLAGS.mode == 'evaluate':
        summary_dir = '/home/logan/data/kaiqiang_data/logan_ACL/trained_on_' + FLAGS.train_dataset + '/' + FLAGS.dataset_name
        out_summary_path = os.path.join(summary_dir, 'test' + 'Summary.txt')
        out_example_idx_path = os.path.join(out_dir, 'test' + '.Nexampleidx')
        decode_dir = 'logs/kaiqiang_%s_trainedon%s' % (FLAGS.dataset_name, FLAGS.train_dataset)
        rouge_ref_dir = os.path.join(decode_dir, 'reference')
        rouge_dec_dir = os.path.join(decode_dir, 'decoded')
        util.create_dirs(rouge_ref_dir)
        util.create_dirs(rouge_dec_dir)

        def num_lines_in_file(file_path):
            with open(file_path) as f:
                num_lines = sum(1 for line in f)
            return num_lines

        def process_example(sents, ex_idx, groundtruth_summ_sents):
            final_decoded_words = []
            for sent in sents:
                final_decoded_words.extend(sent.split(' '))
            rouge_functions.write_for_rouge(groundtruth_summ_sents, None, ex_idx, rouge_ref_dir, rouge_dec_dir, decoded_words=final_decoded_words, log=False)

        num_lines_summary = num_lines_in_file(out_summary_path)
        num_lines_example_indices = num_lines_in_file(out_example_idx_path)
        if num_lines_summary != num_lines_example_indices:
            raise Exception('Num lines summary != num lines example indices: (%d, %d)' % (num_lines_summary, num_lines_example_indices))

        source_dir = os.path.join(data_dir, FLAGS.dataset_name)
        example_generator = data.example_generator(source_dir + '/' + 'test' + '*', True, False,
                                                   should_check_valid=False)

        sum_writer = open(out_summary_path)
        ex_idx_writer = open(out_example_idx_path)
        prev_ex_idx = 0
        sents = []

        for line_idx in tqdm(range(num_lines_summary)):
            line = sum_writer.readline()
            ex_idx = int(ex_idx_writer.readline())

            if ex_idx == prev_ex_idx:
                sents.append(line)
            else:
                example = example_generator.next()
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, doc_indices = util.unpack_tf_example(
                    example, names_to_types)
                if FLAGS.dataset_name == 'duc_2004':
                    groundtruth_summ_sents = [[sent.strip() for sent in gt_summ_text.strip().split('\n')] for gt_summ_text in groundtruth_summary_text]
                else:
                    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
                process_example(sents, ex_idx, groundtruth_summ_sents)
                prev_ex_idx = ex_idx
                sents = [line]

        example = example_generator.next()
        raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, doc_indices = util.unpack_tf_example(
            example, names_to_types)
        if FLAGS.dataset_name == 'duc_2004':
            groundtruth_summ_sents = [[sent.strip() for sent in gt_summ_text.strip().split('\n')] for gt_summ_text in groundtruth_summary_text]
        else:
            groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
        process_example(sents, ex_idx, groundtruth_summ_sents)

        print("Now starting ROUGE eval...")
        if FLAGS.dataset_name == 'xsum':
            l_param = 100
        else:
            l_param = 100
        results_dict = rouge_functions.rouge_eval(rouge_ref_dir, rouge_dec_dir, l_param=l_param)
        rouge_functions.rouge_log(results_dict, decode_dir)





    else:
        raise Exception('mode flag was not evaluate or write.')


if __name__ == '__main__':
    app.run(main)



