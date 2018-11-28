from tqdm import tqdm
from scoop import futures
import rouge_functions
from absl import flags
from absl import app
import convert_data
import time
import subprocess
import itertools
import glob
import numpy as np
import data
import os
import sys
from collections import defaultdict
import util
from scipy import sparse
from count_merged import html_highlight_sents_in_article, get_simple_source_indices_list
import cPickle
from profilestats import profile

FLAGS = flags.FLAGS
if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'singles', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'mode' not in flags.FLAGS:
    flags.DEFINE_string('mode', 'write_to_file', 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'start_over' not in flags.FLAGS:
    flags.DEFINE_boolean('start_over', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'first_k' not in flags.FLAGS:
    flags.DEFINE_integer('first_k', 20, 'Specifies k, where we consider only the first k sentences of each article. Only applied when [running on both singles and pairs, and not running on cnn_dm]')
if 'upper_bound' not in flags.FLAGS:
    flags.DEFINE_boolean('upper_bound', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'use_pair_criteria' not in flags.FLAGS:
    flags.DEFINE_boolean('use_pair_criteria', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')

FLAGS(sys.argv)

from preprocess_for_lambdamart_no_flags import get_features, get_single_sent_features, get_pair_sent_features, \
    Lambdamart_Instance, format_to_lambdamart, filter_pairs_by_criteria, get_rel_sent_indices

_exp_name = 'lambdamart'
model = FLAGS.dataset_name + '_ndcg5_%s_10000' % FLAGS.singles_and_pairs
tfidf_model = 'all'
dataset_split = 'test'
importance = True
filter_sentences = True
num_instances = -1
random_seed = 123
max_sent_len_feat = 20
min_matched_tokens = 2
# singles_and_pairs = 'singles'
include_tfidf_vec = True

data_dir = 'tf_data/with_coref_and_ssi'
temp_dir = 'data/temp/' + FLAGS.dataset_name + '/scores'
lambdamart_in_dir = 'data/temp/' + FLAGS.dataset_name + '/to_lambdamart'
lambdamart_out_dir = 'data/temp/' + FLAGS.dataset_name + '/lambdamart_results'
ssi_out_dir = 'data/temp/' + FLAGS.dataset_name + '/ssi'
log_dir = 'logs'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]

if FLAGS.singles_and_pairs == 'both':
    exp_name = FLAGS.dataset_name + '_' + _exp_name + '_both'
    dataset_articles = FLAGS.dataset_name
else:
    exp_name = FLAGS.dataset_name + '_' + _exp_name + '_singles'
    dataset_articles = FLAGS.dataset_name + '_singles'

if FLAGS.upper_bound:
    exp_name = exp_name + '_upperbound'

if FLAGS.singles_and_pairs == 'singles':
    sentence_limit = 1
else:
    sentence_limit = 2

if FLAGS.dataset_name == 'xsum':
    l_param = 40
else:
    l_param = 100

temp_in_dir = os.path.join(lambdamart_in_dir, 'lambdamart_' + FLAGS.singles_and_pairs)
temp_out_dir = os.path.join(lambdamart_out_dir, 'lambdamart_' + FLAGS.singles_and_pairs)
temp_in_path = temp_in_dir + '.txt'
temp_out_path = temp_out_dir + '.txt'
util.create_dirs(temp_in_dir)
util.create_dirs(temp_out_dir)
my_log_dir = os.path.join(log_dir, exp_name)
dec_dir = os.path.join(my_log_dir, 'decoded')
ref_dir = os.path.join(my_log_dir, 'reference')
html_dir = os.path.join(my_log_dir, 'hightlighted_html')
util.create_dirs(dec_dir)
util.create_dirs(ref_dir)
util.create_dirs(html_dir)
util.create_dirs(temp_dir)
util.create_dirs(ssi_out_dir)

tfidf_vec_path = 'data/tfidf/' + tfidf_model + '_tfidf_vec_5.pkl'
with open(tfidf_vec_path, 'rb') as f:
    tfidf_vectorizer = cPickle.load(f)

# @profile
def read_lambdamart_scores(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data = [[float(x) for x in line.split('\t')] for line in lines]
    data = np.array(data)
    return data

# @profile
def get_qid_source_indices(line):
    items = line.split(' ')
    for item in items:
        if 'qid' in item:
            qid = int(item.split(':')[1])
            break
    comment = line.strip().split('#')[1]
    source_indices_str = comment.split(',')[0]
    source_indices = source_indices_str.split(':')[1].split(' ')
    source_indices = [int(x) for x in source_indices]

    inst_id_str = comment.split(',')[1]
    inst_id = int(inst_id_str.split(':')[1])

    return qid, inst_id, source_indices

# @profile
def read_source_indices_from_lambdamart_input(file_path):
    out_dict = {}
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        qid, inst_id, source_indices = get_qid_source_indices(line)
        if qid not in out_dict:
            out_dict[qid] = {}
        # out_dict[(qid,inst_id)] = source_indices
        out_dict[qid][inst_id] = tuple(source_indices)
    return out_dict

# def get_source_indices(data):
#     similar_source_indices = []
#     possible_qids = [example_idx*10 + item for item in list(xrange(10))]
#     for qid in possible_qids:
#         if qid not in qid_to_source_indices:
#             break
#         similar_source_indices.append(qid_to_source_indices[qid])
#     return similar_source_indices

# @profile
def get_features_all_combinations(example_idx, raw_article_sents, article_sent_tokens, corefs, rel_sent_indices, first_k_indices, mmrs, single_feat_len, pair_feat_len, singles_and_pairs):
    # sent_term_matrix = util.get_tfidf_matrix(raw_article_sents)
    article_text = ' '.join(raw_article_sents)
    sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, raw_article_sents, article_text)
    doc_vector = np.mean(sent_term_matrix, axis=0)

    possible_pairs = [x for x in list(itertools.combinations(first_k_indices, 2))]   # all pairs
    if FLAGS.use_pair_criteria:
        possible_pairs = filter_pairs_by_criteria(raw_article_sents, possible_pairs, corefs)
    possible_singles = [(i,) for i in first_k_indices]
    if singles_and_pairs == 'pairs':
        all_combinations = possible_pairs
    elif singles_and_pairs == 'singles':
        all_combinations = possible_singles
    else:
        all_combinations = possible_pairs + possible_singles
    instances = []
    for source_indices in all_combinations:
        features = get_features(source_indices, sent_term_matrix, article_sent_tokens, rel_sent_indices,
                                single_feat_len, pair_feat_len, mmrs, singles_and_pairs)
        instances.append(Lambdamart_Instance(features, 0, example_idx, source_indices))
    return instances

# @profile
def get_sent_or_sents(article_sent_tokens, source_indices):
    chosen_sent_tokens = [article_sent_tokens[idx] for idx in source_indices]
    # sents = util.flatten_list_of_lists(chosen_sent_tokens)
    return chosen_sent_tokens

# @profile
def get_lambdamart_scores_for_singles_pairs(data, inst_id_to_source_indices):
    out_dict = {}
    for row in data:
        qid, inst_id, score = row
        source_indices = inst_id_to_source_indices[qid][inst_id]
        if qid not in out_dict:
            out_dict[qid] = {}
        out_dict[qid][source_indices] = score
    return out_dict

# @profile
def rank_source_sents(temp_in_path, temp_out_path):
    inst_id_to_source_indices = read_source_indices_from_lambdamart_input(temp_in_path)
    data = read_lambdamart_scores(temp_out_path)
    source_indices_to_scores = get_lambdamart_scores_for_singles_pairs(data, inst_id_to_source_indices)
    return source_indices_to_scores

# @profile
def get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices):
    if len(already_used_source_indices) == 0:
        source_indices = max(mmr_dict, key=mmr_dict.get)
    else:
        best_value = -9999999
        best_source_indices = ()
        for key, val in mmr_dict.iteritems():
            if val > best_value and not any(i in list(key) for i in already_used_source_indices):
                best_value = val
                best_source_indices = key
        source_indices = best_source_indices
    sents = get_sent_or_sents(article_sent_tokens, source_indices)
    return sents, source_indices

# @profile
def get_instances(example_idx, raw_article_sents, article_sent_tokens, corefs, rel_sent_indices, first_k_indices, temp_in_path, temp_out_path, single_feat_len, pair_feat_len, singles_and_pairs):
    tfidfs = util.get_tfidf_importances(tfidf_vectorizer, raw_article_sents)
    instances = get_features_all_combinations(example_idx, raw_article_sents, article_sent_tokens, corefs, rel_sent_indices, first_k_indices, tfidfs, single_feat_len, pair_feat_len, singles_and_pairs)
    return instances

# @profile
def generate_summary(article_sent_tokens, qid_ssi_to_importances, example_idx):
    qid = example_idx

    summary_sent_tokens = []
    summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
    already_used_source_indices = []
    similar_source_indices_list = []
    summary_sents_for_html = []
    ssi_length_extractive = None
    while len(summary_tokens) < 1000:
        if len(summary_tokens) >= l_param and ssi_length_extractive is None:
            ssi_length_extractive = len(similar_source_indices_list)
        if FLAGS.dataset_name == 'xsum' and len(summary_tokens) > 0:
            ssi_length_extractive = len(similar_source_indices_list)
            break
        mmr_dict = util.calc_MMR_source_indices(article_sent_tokens, summary_tokens, None, qid_ssi_to_importances, qid=qid)
        sents, source_indices = get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices)
        if len(source_indices) == 0:
            break
        summary_sent_tokens.extend(sents)
        summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
        similar_source_indices_list.append(source_indices)
        summary_sents_for_html.append(' <br> '.join([' '.join(sent) for sent in sents]))
        if filter_sentences:
            already_used_source_indices.extend(source_indices)
    if ssi_length_extractive is None:
        ssi_length_extractive = len(similar_source_indices_list)
    selected_article_sent_indices = util.flatten_list_of_lists(similar_source_indices_list[:ssi_length_extractive])
    summary_sents = [' '.join(sent) for sent in util.reorder(article_sent_tokens, selected_article_sent_indices)]
    # summary = '\n'.join([' '.join(tokens) for tokens in summary_sent_tokens])
    return summary_sents, similar_source_indices_list, summary_sents_for_html, ssi_length_extractive

def example_generator_extended(example_generator, total, single_feat_len, pair_feat_len, singles_and_pairs):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
    # for example in example_generator:
        example_idx += 1
        if num_instances != -1 and example_idx >= num_instances:
            break
        yield (example, example_idx, single_feat_len, pair_feat_len, singles_and_pairs)

# @profile
def write_highlighted_html(html, out_dir, example_idx):
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    with open(path, 'w') as f:
        f.write(html)

def get_indices_of_first_k_sents_of_each_article(rel_sent_indices, k):
    indices = [idx for idx, rel_sent_idx in enumerate(rel_sent_indices) if rel_sent_idx < k]
    return indices


# @profile
def write_to_lambdamart_examples_to_file(ex):
    example, example_idx, single_feat_len, pair_feat_len, singles_and_pairs = ex
    print example_idx
    # example_idx += 1
    temp_in_path = os.path.join(temp_in_dir, '%06d.txt' % example_idx)
    if not FLAGS.start_over and os.path.exists(temp_in_path):
        return
    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [convert_data.process_sent(sent) for sent in raw_article_sents]
    if doc_indices is None:
        doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
    doc_indices = [int(doc_idx) for doc_idx in doc_indices]
    rel_sent_indices = get_rel_sent_indices(doc_indices, article_sent_tokens)
    groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, sentence_limit)
    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
    groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]
    # summ_sent_tokens = [sent.strip().split() for sent in summary_text.strip().split('\n')]

    if FLAGS.dataset_name == 'duc_2004':
        first_k_indices = get_indices_of_first_k_sents_of_each_article(rel_sent_indices, FLAGS.first_k)
    else:
        first_k_indices = [idx for idx in range(len(raw_article_sents))]



    if importance:
        instances = get_instances(example_idx, raw_article_sents, article_sent_tokens, corefs, rel_sent_indices, first_k_indices, temp_in_path, temp_out_path,
                                    single_feat_len, pair_feat_len, singles_and_pairs)
    out_str = ''
    instances = sorted(instances, key=lambda x: (x.qid, x.source_indices))
    for inst_id, instance in enumerate(instances):
        instance.inst_id = inst_id
        lambdamart_str = format_to_lambdamart(instance, single_feat_len)
        out_str += lambdamart_str + '\n'
    with open(temp_in_path, 'w') as f:
        f.write(out_str)

# @profile
def evaluate_example(ex):
    example, example_idx, qid_ssi_to_importances, _, _ = ex
    print example_idx
    # example_idx += 1
    qid = example_idx
    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [convert_data.process_sent(sent) for sent in raw_article_sents]
    enforced_groundtruth_ssi_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, sentence_limit)
    groundtruth_summ_sent_tokens = []
    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
    groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]

    if FLAGS.upper_bound:
        selected_article_sent_indices = util.flatten_list_of_lists(enforced_groundtruth_ssi_list)
        summary_sents = [' '.join(sent) for sent in util.reorder(article_sent_tokens, selected_article_sent_indices)]
        similar_source_indices_list = groundtruth_similar_source_indices_list
        ssi_length_extractive = len(similar_source_indices_list)
    else:
        summary_sents, similar_source_indices_list, summary_sents_for_html, ssi_length_extractive = generate_summary(article_sent_tokens, qid_ssi_to_importances, example_idx)
        similar_source_indices_list_trunc = similar_source_indices_list[:ssi_length_extractive]
        summary_sents_for_html_trunc = summary_sents_for_html[:ssi_length_extractive]
        if example_idx <= 100:
            summary_sent_tokens = [sent.split(' ') for sent in summary_sents_for_html_trunc]
            extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list_trunc,
                                            article_sent_tokens, doc_indices=doc_indices)
            write_highlighted_html(extracted_sents_in_article_html, html_dir, example_idx)

            groundtruth_ssi_list, lcs_paths_list, article_lcs_paths_list = get_simple_source_indices_list(
                                            groundtruth_summ_sent_tokens,
                                           article_sent_tokens, None, sentence_limit, min_matched_tokens)
            groundtruth_highlighted_html = html_highlight_sents_in_article(groundtruth_summ_sent_tokens, groundtruth_ssi_list,
                                            article_sent_tokens, lcs_paths_list=lcs_paths_list, article_lcs_paths_list=article_lcs_paths_list, doc_indices=doc_indices)
            all_html = '<u>System Summary</u><br><br>' + extracted_sents_in_article_html + '<u>Groundtruth Summary</u><br><br>' + groundtruth_highlighted_html
            write_highlighted_html(all_html, html_dir, example_idx)
    rouge_functions.write_for_rouge(groundtruth_summ_sents, summary_sents, example_idx, ref_dir, dec_dir)
    return (groundtruth_similar_source_indices_list, similar_source_indices_list, ssi_length_extractive)

def sent_selection_eval(ssi_list, operation_on_gt):
    if FLAGS.dataset_name == 'cnn_dm':
        sys_max_sent_len = 4
    elif FLAGS.dataset_name == 'duc_2004':
        sys_max_sent_len = 5
    elif FLAGS.dataset_name == 'xsum':
        sys_max_sent_len = 1
    sys_pos = 0
    sys_neg = 0
    gt_pos = 0
    gt_neg = 0
    for gt, sys_, ext_len in ssi_list:
        gt = operation_on_gt(gt)
        sys_ = sys_[:sys_max_sent_len]
        sys_ = util.flatten_list_of_lists(sys_)
        # sys_ = sys_[:ext_len]
        for ssi in sys_:
            if ssi in gt:
                sys_pos += 1
            else:
                sys_neg += 1
        for ssi in gt:
            if ssi in sys_:
                gt_pos += 1
            else:
                gt_neg += 1
    prec = float(sys_pos) / (sys_pos + sys_neg)
    rec = float(gt_pos) / (gt_pos + gt_neg)
    if sys_pos + sys_neg == 0 or gt_pos + gt_neg == 0:
        f1 = 0
    else:
        f1 = 2.0 * (prec * rec) / (prec + rec)
    prec *= 100
    rec *= 100
    f1 *= 100
    suffix = '%.2f\t%.2f\t%.2f\t' % (prec, rec, f1)
    print 'Lambdamart P/R/F: '
    print suffix
    return suffix

def all_sent_selection_eval(ssi_list):
    def flatten(gt):
        return util.flatten_list_of_lists(gt)
    def primary(gt):
        return util.flatten_list_of_lists(util.enforce_sentence_limit(gt, 1))
    def secondary(gt):
        return [ssi[1] for ssi in gt if len(ssi) == 2]
    # def single(gt):
    #     return util.flatten_list_of_lists([ssi for ssi in gt if len(ssi) == 1])
    # def pair(gt):
    #     return util.flatten_list_of_lists([ssi for ssi in gt if len(ssi) == 2])
    operations_on_gt = [flatten, primary, secondary]
    suffixes = []
    for op in operations_on_gt:
        suffix = sent_selection_eval(ssi_list, op)
        suffixes.append(suffix)
    combined_suffix = '\n' + '\n'.join(suffixes)
    print combined_suffix
    return combined_suffix


def main(unused_argv):
# def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    print 'Running statistics on %s' % exp_name

    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
    ex_sents = ['single .', 'sentence .']
    article_text = ' '.join(ex_sents)
    sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, ex_sents, article_text)
    if FLAGS.singles_and_pairs == 'pairs':
        single_feat_len = 0
    else:
        single_feat_len = len(get_single_sent_features(0, sent_term_matrix,
                                                   [['single', '.'], ['sentence', '.']], [0, 0], 0))
    if FLAGS.singles_and_pairs == 'singles':
        pair_feat_len = 0
    else:
        pair_feat_len = len(
            get_pair_sent_features([0, 1], sent_term_matrix,
                               [['single', '.'], ['sentence', '.']], [0, 0], [0, 0]))


    total = len(source_files)*1000 if 'cnn' or 'newsroom' in dataset_articles else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)

    if FLAGS.mode == 'write_to_file':
        ex_gen = example_generator_extended(example_generator, total, single_feat_len, pair_feat_len, FLAGS.singles_and_pairs)
        print 'Creating list'
        ex_list = [ex for ex in ex_gen]
        print 'Converting...'
        # if len(sys.argv) > 1 and sys.argv[1] == '-m':
        instances_list = list(futures.map(write_to_lambdamart_examples_to_file, ex_list))
        # else:
        #     instances_list = []
        #     for ex in tqdm(ex_list):
        #         instances_list.append(write_to_lambdamart_examples_to_file(ex))

        file_names = sorted(glob.glob(os.path.join(temp_in_dir, '*')))
        instances_str = ''
        for file_name in tqdm(file_names):
            with open(file_name) as f:
                instances_str += f.read()
        with open(temp_in_path, 'wb') as f:
            f.write(instances_str)


    # RUN LAMBDAMART SCORING COMMAND HERE


    if FLAGS.mode == 'generate_summaries':
        qid_ssi_to_importances = rank_source_sents(temp_in_path, temp_out_path)
        ex_gen = example_generator_extended(example_generator, total, qid_ssi_to_importances, pair_feat_len, FLAGS.singles_and_pairs)
        print 'Creating list'
        ex_list = [ex for ex in ex_gen]
        ssi_list = list(futures.map(evaluate_example, ex_list))
        with open(os.path.join(ssi_out_dir, exp_name), 'w') as f:
            cPickle.dump(ssi_list, f)

        # save ssi_list
        with open(os.path.join(my_log_dir, 'ssi.pkl'), 'w') as f:
            cPickle.dump(ssi_list, f)
        with open(os.path.join(my_log_dir, 'ssi.pkl')) as f:
            ssi_list = cPickle.load(f)
        print 'Evaluating Lambdamart model F1 score...'
        suffix = all_sent_selection_eval(ssi_list)
        #
        # # for ex in tqdm(ex_list, total=total):
        # #     load_and_evaluate_example(ex)
        #
        print 'Evaluating ROUGE...'
        results_dict = rouge_functions.rouge_eval(ref_dir, dec_dir, l_param=l_param)
        # print("Results_dict: ", results_dict)
        rouge_functions.rouge_log(results_dict, my_log_dir, suffix=suffix)

    util.print_execution_time(start_time)


if __name__ == '__main__':
    # main()
    app.run(main)




































