from tqdm import tqdm
from scoop import futures
import rouge_eval_references
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
from collections import defaultdict
import util
from preprocess_lr import get_features, get_single_sent_features, get_pair_sent_features, \
    Lambdamart_Instance, format_to_lambdamart
from scipy import sparse
from count_merged import html_highlight_sents_in_article, get_simple_source_indices_list
import pickle
import dill
import tempfile
tempfile.tempdir = os.path.expanduser('~') + "/tmp"

exp_name = 'lambdamart_singles_lr'
dataset = 'cnn_dm_singles_lr'
dataset_articles = 'cnn_dm'
dataset_model = 'singles'
dataset_split = 'test'
model = 'ndcg5'
importance = True
filter_sentences = True
num_instances = -1
random_seed = 123
max_sent_len_feat = 20
sentence_limit = 2
min_matched_tokens = 2
singles_and_pairs = 'singles'
include_tfidf_vec = True

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
model_dir = 'data/lambdamart_models'
temp_dir = 'data/temp/scores'
lambdamart_in_dir = 'data/temp/to_lambdamart'
lambdamart_out_dir = 'data/temp/lambdamart_results'
log_dir = 'logs'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'), ('summary_text', 'string')]

temp_in_dir = os.path.join(lambdamart_in_dir, dataset_model, exp_name)
temp_out_dir = os.path.join(lambdamart_out_dir, dataset_model, exp_name)
util.create_dirs(temp_in_dir)
util.create_dirs(temp_out_dir)
my_log_dir = os.path.join(log_dir, dataset_model + '_' + exp_name)
dec_dir = os.path.join(my_log_dir, 'decoded')
ref_dir = os.path.join(my_log_dir, 'reference')
html_dir = os.path.join(my_log_dir, 'hightlighted_html')
util.create_dirs(dec_dir)
util.create_dirs(ref_dir)
util.create_dirs(html_dir)
util.create_dirs(temp_dir)

tfidf_vec_path = 'data/tfidf/' + dataset_articles + '_tfidf_vec.pkl'
with open(tfidf_vec_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open(os.path.join(model_dir, dataset + '.pkl'), 'rb') as f:
    lr_model = dill.load(f)

def read_lambdamart_scores(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data = [[float(x) for x in line.split('\t')] for line in lines]
    data = np.array(data)
    return data

# def get_best_lambdamart_scores(data):
#     qid_to_inst_id = defaultdict(lambda:(None, -1000000))
#     for row in data:
#         qid, inst_id, score = row
#         current_inst_id, current_best = qid_to_inst_id[qid]
#         if score > current_best:
#             qid_to_inst_id = (inst_id, score)
#     return qid_to_inst_id

def get_best_lambdamart_score_and_source_indices(data, inst_id_to_source_indices):
    current_inst_id = -1
    current_best = -1000000
    for row in data:
        qid, inst_id, score = row
        if score > current_best:
            current_inst_id = inst_id_to_source_indices[inst_id]
            current_best = score
    return current_best, current_inst_id

def get_best_source_indices(qid_to_inst_id, qid_inst_id_to_source_indices):
    out_dict = {}
    for qid, inst_id in qid_to_inst_id.items():
        out_dict[qid] = qid_inst_id_to_source_indices[qid, inst_id]
    return out_dict

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

def read_source_indices_from_lambdamart_input(file_path):
    out_dict = {}
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        qid, inst_id, source_indices = get_qid_source_indices(line)
        # out_dict[(qid,inst_id)] = source_indices
        out_dict[inst_id] = tuple(source_indices)
    return out_dict

def read_articles_abstracts(source_dir, dataset_split):
    source_dir = os.path.join(data_dir, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)

    print('Creating list')
    ex_list = [ex for ex in example_generator]
    a=0

def get_source_indices(example_idx, qid_to_source_indices):
    similar_source_indices = []
    possible_qids = [example_idx*10 + item for item in list(range(10))]
    for qid in possible_qids:
        if qid not in qid_to_source_indices:
            break
        similar_source_indices.append(qid_to_source_indices[qid])
    return similar_source_indices

# def get_source_indices(data):
#     similar_source_indices = []
#     possible_qids = [example_idx*10 + item for item in list(xrange(10))]
#     for qid in possible_qids:
#         if qid not in qid_to_source_indices:
#             break
#         similar_source_indices.append(qid_to_source_indices[qid])
#     return similar_source_indices

def get_features_all_combinations(raw_article_sents, article_sent_tokens, mmrs, single_feat_len, pair_feat_len):
    # sent_term_matrix = util.get_tfidf_matrix(raw_article_sents)
    article_text = ' '.join(raw_article_sents)
    sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, raw_article_sents, article_text)
    doc_vector = np.mean(sent_term_matrix, axis=0)

    possible_pairs = [list(x) for x in list(itertools.combinations(list(range(len(raw_article_sents))), 2))]   # all pairs
    possible_singles = [[i] for i in range(len(raw_article_sents))]
    if singles_and_pairs == 'pairs':
        all_combinations = possible_pairs
    elif singles_and_pairs == 'singles':
        all_combinations = possible_singles
    else:
        all_combinations = possible_pairs + possible_singles
    instances = []
    for source_indices in all_combinations:
        features = get_features(source_indices, sent_term_matrix, article_sent_tokens,
                                single_feat_len, pair_feat_len, mmrs)
        instances.append(Lambdamart_Instance(features, 0, 0, source_indices))
    return instances

def write_to_file(instances, out_path, single_feat_len):
    out_str = ''
    instances = sorted(instances, key=lambda x: (x.qid, x.source_indices))
    for inst_id, instance in enumerate(instances):
        instance.inst_id = inst_id
        lambdamart_str = format_to_lambdamart(instance, single_feat_len)
        out_str += lambdamart_str + '\n'
    with open(out_path, 'w') as f:
        f.write(out_str)

def rank_instances(in_path, out_path):
    command = 'java -jar ' + os.path.expanduser('~') + '/ranklib/bin/RankLib-2.10.jar '\
    + '-rank ' + in_path + ' '\
    + '-score ' + out_path + ' -ranker 6 -metric2t MAP -metric2T MAP '\
    + '-load ' + os.path.expanduser('~') + '/data/discourse/lambdamart/' + model + '.txt -sparse'
    subprocess.check_output(command.split(' '))

def get_sent_or_sents(article_sent_tokens, source_indices):
    chosen_sent_tokens = [article_sent_tokens[idx] for idx in source_indices]
    # sents = util.flatten_list_of_lists(chosen_sent_tokens)
    return chosen_sent_tokens

def rank_and_get_source_sents(instances, article_sent_tokens, temp_in_path, temp_out_path, single_feat_len):
    x = [inst.features for inst in instances]
    x = np.array(x)
    scores = lr_model.decision_function(x)
    source_indices = [np.argmax(scores)]

    # rank_instances(temp_in_path, temp_out_path)
    # data = read_lambdamart_scores(temp_out_path)
    # score, source_indices = get_best_lambdamart_score_and_source_indices(data, inst_id_to_source_indices)
    sents = get_sent_or_sents(article_sent_tokens, source_indices)
    return sents, source_indices

def generate_summary(raw_article_sents, article_sent_tokens, temp_in_path, temp_out_path, single_feat_len, pair_feat_len):
    summary_tokens = []

    while len(summary_tokens) < 120:
        mmrs = util.calc_MMR(raw_article_sents, article_sent_tokens, summary_tokens, None)
        instances = get_features_all_combinations(raw_article_sents, article_sent_tokens, mmrs, single_feat_len, pair_feat_len)
        sents, source_indices = rank_and_get_source_sents(instances, article_sent_tokens, temp_in_path, temp_out_path, single_feat_len)
        summary_tokens.extend(sents)
    summary_sent_tokens = [sent + ['.'] for sent in util.split_list_by_item(summary_tokens, '.')]
    summary_sents = [' '.join(sent) for sent in summary_sent_tokens]
    # summary = '\n'.join([' '.join(tokens) for tokens in summary_sent_tokens])
    return summary_sents



def get_lambdamart_scores_for_singles_pairs(data, inst_id_to_source_indices):
    out_dict = {}
    for row in data:
        qid, inst_id, score = row
        source_indices = inst_id_to_source_indices[inst_id]
        out_dict[source_indices] = score
    return out_dict

def rank_source_sents(instances, temp_in_path, temp_out_path, single_feat_len):
    # write_to_file(instances, temp_in_path, single_feat_len)
    # inst_id_to_source_indices = read_source_indices_from_lambdamart_input(temp_in_path)
    x = [inst.features for inst in instances]
    x = np.array(x)
    scores = lr_model.decision_function(x)
    # source_indices = [np.argmax(scores)]
    source_indices_to_scores = {}
    for score_idx, score in enumerate(scores):
        source_indices_to_scores[(score_idx,)] = score

    # source_indices_to_scores = get_lambdamart_scores_for_singles_pairs(data, inst_id_to_source_indices)
    return source_indices_to_scores

def get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices):
    if len(already_used_source_indices) == 0:
        source_indices = max(mmr_dict, key=mmr_dict.get)
    else:
        best_value = -9999999
        best_source_indices = ()
        for key, val in mmr_dict.items():
            if val > best_value and not any(i in list(key) for i in already_used_source_indices):
                best_value = val
                best_source_indices = key
        source_indices = best_source_indices
    sents = get_sent_or_sents(article_sent_tokens, source_indices)
    return sents, source_indices

def generate_summary_importance(raw_article_sents, article_sent_tokens, temp_in_path, temp_out_path, single_feat_len, pair_feat_len):
    tfidfs = util.get_tfidf_importances(tfidf_vectorizer, raw_article_sents)
    instances = get_features_all_combinations(raw_article_sents, article_sent_tokens, tfidfs, single_feat_len, pair_feat_len)
    source_indices_to_importances = rank_source_sents(instances, temp_in_path, temp_out_path, single_feat_len)
    summary_sent_tokens = []
    summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
    already_used_source_indices = []
    similar_source_indices_list = []
    summary_sents_for_html = []
    while len(summary_tokens) < 120:
        mmr_dict = util.calc_MMR_source_indices(article_sent_tokens, summary_tokens, None, source_indices_to_importances)
        sents, source_indices = get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices)
        if len(source_indices) == 0:
            break
        summary_sent_tokens.extend(sents)
        summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
        similar_source_indices_list.append(source_indices)
        summary_sents_for_html.append(' <br> '.join([' '.join(sent) for sent in sents]))
        if filter_sentences:
            already_used_source_indices.extend(source_indices)
    summary_sents = [' '.join(sent) for sent in summary_sent_tokens]
    # summary = '\n'.join([' '.join(tokens) for tokens in summary_sent_tokens])
    return summary_sents, similar_source_indices_list, summary_sents_for_html


def example_generator_extended(example_generator, total, single_feat_len, pair_feat_len):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
    # for example in example_generator:
        example_idx += 1
        if num_instances != -1 and example_idx >= num_instances:
            break
        yield (example, example_idx, single_feat_len, pair_feat_len)

def write_highlighted_html(html, out_dir, example_idx):
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    with open(path, 'w') as f:
        f.write(html)

def load_and_evaluate_example(ex):
    example, example_idx, single_feat_len, pair_feat_len = ex
    print(example_idx)
    # example_idx += 1
    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [convert_data.process_sent(sent) for sent in raw_article_sents]
    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
    groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]
    # summ_sent_tokens = [sent.strip().split() for sent in summary_text.strip().split('\n')]
    temp_in_path = os.path.join(temp_in_dir, '%06d.txt' % example_idx)
    temp_out_path = os.path.join(temp_out_dir, '%06d.txt' % example_idx)

    if importance:
        summary_sents, similar_source_indices_list, summary_sents_for_html = generate_summary_importance(raw_article_sents, article_sent_tokens, temp_in_path, temp_out_path,
                                    single_feat_len, pair_feat_len)
    else:
        summary_sents = generate_summary(raw_article_sents, article_sent_tokens, temp_in_path, temp_out_path, single_feat_len, pair_feat_len)
    if example_idx <= 100:
        summary_sent_tokens = [sent.split(' ') for sent in summary_sents_for_html]
        extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list,
                                        article_sent_tokens)
        write_highlighted_html(extracted_sents_in_article_html, html_dir, example_idx)

        groundtruth_similar_source_indices_list, lcs_paths_list, article_lcs_paths_list = get_simple_source_indices_list(
                                        groundtruth_summ_sent_tokens,
                                       article_sent_tokens, None, sentence_limit, min_matched_tokens)
        groundtruth_highlighted_html = html_highlight_sents_in_article(groundtruth_summ_sent_tokens, groundtruth_similar_source_indices_list,
                                        article_sent_tokens, lcs_paths_list=lcs_paths_list, article_lcs_paths_list=article_lcs_paths_list)
        all_html = '<u>System Summary</u><br><br>' + extracted_sents_in_article_html + '<u>Groundtruth Summary</u><br><br>' + groundtruth_highlighted_html
        write_highlighted_html(all_html, html_dir, example_idx)
    rouge_eval_references.write_for_rouge(groundtruth_summ_sents, summary_sents, example_idx, ref_dir, dec_dir)


def main(unused_argv):
    print('Running statistics on %s' % exp_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
    ex_sents = ['single .', 'sentence .']
    article_text = ' '.join(ex_sents)
    sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, ex_sents, article_text)
    if singles_and_pairs == 'pairs':
        single_feat_len = 0
    else:
        single_feat_len = len(get_single_sent_features(0, sent_term_matrix,
                                                   [['single', '.'], ['sentence', '.']], [0, 0]))
    if singles_and_pairs == 'singles':
        pair_feat_len = 0
    else:
        pair_feat_len = len(
            get_pair_sent_features([0, 1], sent_term_matrix,
                               [['single', '.'], ['sentence', '.']], [0, 0]))


    total = len(source_files)*1000 if 'cnn' or 'newsroom' in dataset_articles else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)


    ex_gen = example_generator_extended(example_generator, total, single_feat_len, pair_feat_len)
    print('Creating list')
    ex_list = [ex for ex in ex_gen]
    print('Converting...')
    list(futures.map(load_and_evaluate_example, ex_list))
    # for ex in tqdm(ex_list, total=total):
    #     load_and_evaluate_example(ex)

    print('Evaluating ROUGE...')
    results_dict = rouge_eval_references.rouge_eval(ref_dir, dec_dir)
    # print("Results_dict: ", results_dict)
    rouge_eval_references.rouge_log(results_dict, my_log_dir)

    util.print_execution_time(start_time)


if __name__ == '__main__':

    app.run(main)




































