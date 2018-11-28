import itertools
import os
from tqdm import tqdm
import numpy as np
from absl import flags
from absl import app
import cPickle
import util
import sys
import glob
import data

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'val', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)


import convert_data
import lambdamart_scores_to_summaries
import preprocess_for_lambdamart_no_flags

data_dir = 'tf_data/with_coref_and_ssi'
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]
min_matched_tokens = 1

def main(unused_argv):

    print 'Running statistics on %s' % FLAGS.dataset_name

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.dataset_name == 'duc_2004':
        FLAGS.dataset_split = 'test'

    ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name, FLAGS.dataset_split + '_ssi.pkl')

    with open(ssi_path) as f:
        ssi_list = cPickle.load(f)

    if FLAGS.dataset_name == 'duc_2004':
        for abstract_idx in [1,2,3]:
            ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name, FLAGS.dataset_split + '_ssi_' + str(abstract_idx) + '.pkl')
            with open(ssi_path) as f:
                temp_ssi_list = cPickle.load(f)
            ssi_list.extend(temp_ssi_list)


    num_extracted = [len(ssi) for ssi in util.flatten_list_of_lists(ssi_list)]
    hist_num_extracted = np.histogram(num_extracted, bins=6)
    print 'Histogram of number of sentences merged: ', util.hist_as_pdf_str(hist_num_extracted)


    source_dir = os.path.join(data_dir, FLAGS.dataset_name)
    source_files = sorted(glob.glob(source_dir + '/' + FLAGS.dataset_split + '*'))

    total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + FLAGS.dataset_split + '*', True, False,
                                               should_check_valid=False)

    all_possible_singles = 0
    all_possible_pairs = 0
    all_filtered_pairs = 0
    all_all_combinations = 0
    all_ssi_pairs = 0
    ssi_pairs_with_shared_coref = 0
    ssi_pairs_with_shared_word = 0
    ssi_pairs_with_either_coref_or_word = 0
    actual_total = 0
    rel_positions_primary = []
    rel_positions_secondary = []
    rel_positions_all = []
    for example_idx, example in enumerate(tqdm(example_generator, total=total)):
        raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(
            example, names_to_types)
        article_sent_tokens = [convert_data.process_sent(sent) for sent in raw_article_sents]
        groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
        if doc_indices is None:
            doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
        doc_indices = [int(doc_idx) for doc_idx in doc_indices]
        rel_sent_indices = preprocess_for_lambdamart_no_flags.get_rel_sent_indices(doc_indices, article_sent_tokens)
        groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)

        if FLAGS.dataset_name == 'duc_2004':
            first_k_indices = lambdamart_scores_to_summaries.get_indices_of_first_k_sents_of_each_article(rel_sent_indices, FLAGS.first_k)
        else:
            first_k_indices = [idx for idx in range(len(raw_article_sents))]

        possible_pairs = [x for x in list(itertools.combinations(first_k_indices, 2))]  # all pairs
        filtered_possible_pairs = preprocess_for_lambdamart_no_flags.filter_pairs_by_criteria(raw_article_sents, possible_pairs, corefs)
        removed_pairs = list(set(possible_pairs) - set(filtered_possible_pairs))
        possible_singles = [(i,) for i in first_k_indices]
        all_combinations = filtered_possible_pairs + possible_singles

        all_possible_singles += len(possible_singles)
        all_possible_pairs += len(possible_pairs)
        all_filtered_pairs += len(filtered_possible_pairs)
        all_all_combinations += len(all_combinations)

        # for ssi in groundtruth_similar_source_indices_list:
        #     if len(ssi) > 0:
        #         idx = rel_sent_indices[ssi[0]]
        #         rel_positions_primary.append(idx)
        #         rel_positions_all.append(idx)
        #     if len(ssi) > 1:
        #         idx = rel_sent_indices[ssi[1]]
        #         rel_positions_secondary.append(idx)
        #         rel_positions_all.append(idx)



        coref_pairs = preprocess_for_lambdamart_no_flags.get_coref_pairs(corefs)
        # DO OVER LAP PAIRS BETTER
        overlap_pairs = preprocess_for_lambdamart_no_flags.filter_by_overlap(article_sent_tokens, possible_pairs)

        for ssi in groundtruth_similar_source_indices_list:
            if len(ssi) == 2:
                all_ssi_pairs += 1
                do_share_coref = ssi in coref_pairs
                do_share_words = ssi in overlap_pairs
                if do_share_coref:
                    ssi_pairs_with_shared_coref += 1
                if do_share_words:
                    ssi_pairs_with_shared_word += 1
                if do_share_coref or do_share_words:
                    ssi_pairs_with_either_coref_or_word += 1


        actual_total += 1

    # print 'Possible_singles\tPossible_pairs\tFiltered_pairs\tAll_combinations: \n%.2f\t%.2f\t%.2f\t%.2f' % (all_possible_singles*1./actual_total, \
    #     all_possible_pairs*1./actual_total, all_filtered_pairs*1./actual_total, all_all_combinations*1./actual_total)

    # print 'Relative positions of groundtruth source sentences in document:\nPrimary\tSecondary\tBoth\n%.2f\t%.2f\t%.2f' % (np.mean(rel_positions_primary), np.mean(rel_positions_secondary), np.mean(rel_positions_all))

    print 'Pair statistics:\nShare_coref\tShare_word\tShare_either\n%.2f\t%.2f\t%.2f' \
          % (ssi_pairs_with_shared_coref*100./all_ssi_pairs, ssi_pairs_with_shared_word*100./all_ssi_pairs, ssi_pairs_with_either_coref_or_word*100./all_ssi_pairs)





if __name__ == '__main__':
    app.run(main)



