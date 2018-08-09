import time
import itertools
import write_data
import numpy as np
import data
from tqdm import tqdm
import util
from absl import flags
from absl import app
import sys
import os
import hashlib
import struct
import subprocess
import collections
import glob
from tensorflow.core.example import example_pb2
from scipy import sparse
from scoop import futures
from collections import defaultdict
# from multiprocessing.dummy import Pool as ThreadPool
# pool = ThreadPool(12)

# FLAGS = FLAGS

exp_name = 'reference'
in_dataset = 'cnn_dm_coref'
out_dataset = 'cnn_dm_coref_importance'
dataset_split = 'train'
num_instances = -1,
random_seed = 123
max_sent_len_feat = 20
balance = False
importance = True

data_dir = '/home/logan/data/multidoc_summarization/merge_indices_tf_examples'
log_dir = '/home/logan/data/multidoc_summarization/logs/'
out_dir = '/home/logan/data/discourse/to_lambdamart'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

data_path = '/home/logan/data'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'), ('summary_text', 'string')]


def get_tf_example(source_file):
    reader = open(source_file, 'rb')
    len_bytes = reader.read(8)
    if not len_bytes: return  # finished reading this file
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    e = example_pb2.Example.FromString(example_str)
    return e

def get_summary_text(summary_file):
    with open(summary_file) as f:
        summary_text = f.read()
    return summary_text

def get_summary_from_example(e):
    summary_texts = []
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract)  # the abstracts texts was saved under the key 'abstract' in the data files
    all_abstract_sentences = [[sent.strip() for sent in data.abstract2sents(
        abstract)] for abstract in summary_texts]
    summary_text = '\n'.join(all_abstract_sentences[0])
    return summary_text


def get_human_summary_texts(summary_file):
    summary_texts = []
    e = get_tf_example(summary_file)
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract)  # the abstracts texts was saved under the key 'abstract' in the data files
    all_abstract_sentences = [[sent.strip() for sent in data.abstract2sents(
        abstract)] for abstract in summary_texts]
    summary_text = '\n'.join(all_abstract_sentences[0])
    return summary_text

def split_into_tokens(text):
    tokens = text.split()
    tokens = [t for t in tokens if t != '<s>' and t != '</s>']
    return tokens

def split_into_sent_tokens(text):
    sent_tokens = [[t for t in tokens.strip().split() if t != '<s>' and t != '</s>'] for tokens in text.strip().split('\n')]
    return sent_tokens

def limit_to_n_tokens(sent_tokens, n):
    res = []
    count = 0
    for sent in sent_tokens:
        out_sent = []
        for token in sent:
            if count < n:
                out_sent.append(token)
                count += 1
        if len(out_sent) > 0:
            res.append(out_sent)
    return res

def split_by_periods(tokens):
    period_indices = [idx for idx in range(len(tokens)) if tokens[idx] == '.']
    cur_idx = 0
    sents = []
    for period_idx in period_indices:
        sent = tokens[cur_idx:period_idx]
        cur_idx = period_idx + 1
        sents.append(sent)
    # sent = tokens[cur_idx:len(tokens)]
    # sents.append(sent)
    sents = [sent for sent in sents if len(sent) > 0]
    return sents

def convert_to_one_hot(value, bins, range):
    hist, _ = np.histogram(value, bins=bins, range=range)
    return hist.tolist()


def get_single_sent_features(sent_idx, sent_term_matrix, doc_vector, article_sent_tokens, mmr):
    is_sent_single = [1]  # is_sent_single
    doc_similarity = util.cosine_similarity(sent_term_matrix[sent_idx], doc_vector)[0][0]
    sent_len = len(article_sent_tokens[sent_idx])
    sent_len = min(max_sent_len_feat, sent_len)
    my_mmr = mmr[sent_idx]

    sent_idx, _ = np.histogram(sent_idx, bins=10, range=(1,10))
    doc_similarity, _ = np.histogram(doc_similarity, bins=5, range=(0,1))
    sent_len, _ = np.histogram(sent_len, bins=10, range=(1,max_sent_len_feat))
    my_mmr = convert_to_one_hot(my_mmr, 5, (0,1))
    return is_sent_single + sent_idx.tolist() + doc_similarity.tolist() + sent_len.tolist() + my_mmr

def get_pair_sent_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens, mmr):
    features = []
    features.append(1)  # is_sent_pair
    # features.extend([0, 0, 0]) # single-sent features
    sent_idx1, sent_idx2 = similar_source_indices[0], similar_source_indices[1]
    sent1_features = get_single_sent_features(sent_idx1,
                         sent_term_matrix, doc_vector, article_sent_tokens, mmr)
    features.extend(sent1_features) # sent_idx, doc_similarity, sent_len
    sent2_features = get_single_sent_features(sent_idx2,
                         sent_term_matrix, doc_vector, article_sent_tokens, mmr)
    features.extend(sent2_features) # sent_idx, doc_similarity, sent_len
    average_mmr = (mmr[sent_idx1] + mmr[sent_idx2])/2
    features.extend(convert_to_one_hot(average_mmr, 5, (0,1)))
    sents_similarity = util.cosine_similarity(sent_term_matrix[sent_idx1], sent_term_matrix[sent_idx2])[0][0]
    features.extend(convert_to_one_hot(sents_similarity, 5, (0,1))) # sents_similarity
    features.extend(convert_to_one_hot(abs(sent_idx1 - sent_idx2), 10, (0,20))) # sents_dist
    return features


def get_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens, single_feat_len,
                 pair_feat_len, mmr):
    if len(similar_source_indices) == 1:
        sent_idx = similar_source_indices[0]
        features = get_single_sent_features(sent_idx, sent_term_matrix, doc_vector, article_sent_tokens, mmr)
        features.extend([0]*pair_feat_len)
    elif len(similar_source_indices) == 2:
        features = [0]*single_feat_len
        features.extend(get_pair_sent_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens, mmr))
    elif len(similar_source_indices) == 0:
        return None
    else:
        raise Exception("Shouldn't be here")
    return features

first_pair_feature = 4

def format_to_lambdamart(inst):
    features, relevance, query_id, source_indices, inst_id = inst.features, inst.relevance, inst.qid, inst.source_indices, inst.inst_id
    out_str = str(relevance) + ' qid:' + str(query_id)

    # for feat_idx, feat in enumerate(features):
    #     out_str += ' %d:%0.6f' % (feat_idx+1, feat)
    # if features[0] == 1:
    #     which_features = range(first_pair_feature)
    # else:
    #     which_features = range(first_pair_feature, len(features))
    # for feat_idx in which_features:
    #     feat = features[feat_idx]
    #     out_str += ' %d:%0.6f' % (feat_idx+1, feat)

    for feat_idx, feat in enumerate(features):
        if feat != 0 or feat_idx == len(features)-1:
            out_str += ' %d:%d' % (feat_idx+1, feat)
    out_str += ' #source_indices:'
    for idx, source_idx in enumerate(source_indices):
        out_str += str(source_idx)
        if idx != len(source_indices) - 1:
            out_str += ' '
    out_str += ',inst_id:' + str(inst_id)
    return out_str

class Lambdamart_Instance:
    def __init__(self, features, relevance, qid, source_indices):
        self.features = features
        self.relevance = relevance
        self.qid = qid
        self.source_indices = source_indices
        self.inst_id = -1

def assign_inst_ids(instances):
    qid_cur_inst_id = defaultdict(int)
    for instance in instances:
        instance.inst_id = qid_cur_inst_id[instance.qid]
        qid_cur_inst_id[instance.qid] += 1


# @ray.remote
def convert_article_to_lambdamart_features(ex):
    # example_idx += 1
    # if num_instances != -1 and example_idx >= num_instances:
    #     break
    example, example_idx, single_feat_len, pair_feat_len = ex
    print example_idx
    raw_article_sents, similar_source_indices_list, summary_text = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [write_data.process_sent(sent) for sent in raw_article_sents]
    summ_sent_tokens = [sent.strip().split() for sent in summary_text.strip().split('\n')]

    # return [len(ssi) for ssi in similar_source_indices_list]
    sent_term_matrix = util.get_tfidf_matrix(raw_article_sents)
    doc_vector = np.mean(sent_term_matrix, axis=0)

    out_str = ''
    # ssi_idx_cur_inst_id = defaultdict(int)
    instances = []

    if importance:
        importances = util.special_squash(util.get_tfidf_importances(raw_article_sents))
        possible_pairs = [list(x) for x in list(itertools.combinations(list(xrange(len(raw_article_sents))), 2))]   # all pairs
        possible_singles = [[i] for i in range(len(raw_article_sents))]
        possible_combinations = possible_pairs + possible_singles
        positives = [ssi for ssi in similar_source_indices_list]
        negatives = [ssi for ssi in possible_combinations if not (ssi in positives or ssi[::-1] in positives)]

        qid = example_idx * 10
        for similar_source_indices in positives:
            # True sentence single/pair
            relevance = 1
            features = get_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens, single_feat_len, pair_feat_len, importances)
            if features is None:
                continue
            instances.append(Lambdamart_Instance(features, relevance, qid, similar_source_indices))
            a=0
        for negative_indices in negatives:
            neg_relevance = 0
            neg_features = get_features(negative_indices, sent_term_matrix, doc_vector, article_sent_tokens, single_feat_len, pair_feat_len, importances)
            if neg_features is None:
                continue
            instances.append(Lambdamart_Instance(neg_features, neg_relevance, qid, negative_indices))
    else:
        mmr_all = util.calc_MMR_all(raw_article_sents, article_sent_tokens, summ_sent_tokens, None) # the size is (# of summary sents, # of article sents)

        possible_pairs = [list(x) for x in list(itertools.combinations(list(xrange(len(raw_article_sents))), 2))]   # all pairs
        possible_singles = [[i] for i in range(len(raw_article_sents))]
        # negative_pairs = [x for x in possible_pairs if not (x in similar_source_indices_list or x[::-1] in similar_source_indices_list)]
        # negative_singles = [x for x in possible_singles if not (x in similar_source_indices_list or x[::-1] in similar_source_indices_list)]
        #
        # random_negative_pairs = np.random.permutation(len(negative_pairs)).tolist()
        # random_negative_singles = np.random.permutation(len(negative_singles)).tolist()

        all_combinations = list(itertools.product(possible_pairs + possible_singles, list(xrange(len(summ_sent_tokens)))))
        positives = [(similar_source_indices, summ_sent_idx) for summ_sent_idx, similar_source_indices in enumerate(similar_source_indices_list)]
        negatives = [(ssi, ssi_idx) for ssi, ssi_idx in all_combinations if not ((ssi, ssi_idx) in positives or (ssi[::-1], ssi_idx) in positives)]

        for similar_source_indices, ssi_idx in positives:
            # True sentence single/pair
            relevance = 1
            qid = example_idx * 10 + ssi_idx
            features = get_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens, single_feat_len, pair_feat_len, mmr_all[ssi_idx])
            if features is None:
                continue
            # inst_id = ssi_idx_cur_inst_id[ssi_idx]
            instances.append(Lambdamart_Instance(features, relevance, qid, similar_source_indices))
            # ssi_idx_cur_inst_id[ssi_idx] += 1
            a=0

            if balance:
                # False sentence single/pair
                is_pair = len(similar_source_indices) == 2
                if is_pair:
                    if len(random_negative_pairs) == 0:
                        continue
                    negative_indices = possible_pairs[random_negative_pairs.pop()]
                else:
                    if len(random_negative_singles) == 0:
                        continue
                    negative_indices = possible_singles[random_negative_singles.pop()]
                neg_relevance = 0
                neg_features = get_features(negative_indices, sent_term_matrix, doc_vector, article_sent_tokens, single_feat_len, pair_feat_len)
                if neg_features is None:
                    continue
                neg_lambdamart_str = format_to_lambdamart(neg_features, neg_relevance, qid, negative_indices)
                out_str += neg_lambdamart_str + '\n'

        if not balance:
            for negative_indices, ssi_idx in negatives:
                neg_relevance = 0
                qid = example_idx * 10 + ssi_idx
                neg_features = get_features(negative_indices, sent_term_matrix, doc_vector, article_sent_tokens, single_feat_len, pair_feat_len, mmr_all[ssi_idx])
                if neg_features is None:
                    continue
                # inst_id = ssi_idx_cur_inst_id[ssi_idx]
                instances.append(Lambdamart_Instance(neg_features, neg_relevance, qid, negative_indices))
                # ssi_idx_cur_inst_id[ssi_idx] += 1

    sorted_instances = sorted(instances, key=lambda x: (x.qid, x.source_indices))
    assign_inst_ids(sorted_instances)
    for instance in sorted_instances:
        lambdamart_str = format_to_lambdamart(instance)
        out_str += lambdamart_str + '\n'
    return out_str

def example_generator_extended(example_generator, total, single_feat_len, pair_feat_len):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
    # for example in example_generator:
        example_idx += 1
        if num_instances != -1 and example_idx >= num_instances:
            break
        yield (example, example_idx, single_feat_len, pair_feat_len)

# ####Delete all flags before declare#####
#
# def del_all_flags(FLAGS):
#     flags_dict = _flags()
#     keys_list = [keys for keys in flags_dict]
#     for keys in keys_list:
#         __delattr__(keys)

# del_all_flags(FLAGS)
def main(unused_argv):
    print 'Running statistics on %s' % exp_name

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, in_dataset)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
    pros = {'annotators': 'dcoref', 'outputFormat': 'json', 'timeout': '5000000'}
    single_feat_len = len(get_single_sent_features(0, sparse.csr_matrix(np.array([[0,0],[0,0]])), np.array([[0,0]]), [['single','.'],['sentence','.']], [0,0]))
    pair_feat_len = len(get_pair_sent_features([0,1], sparse.csr_matrix(np.array([[0,0],[0,0]])), np.array([[0,0]]), [['single','.'],['sentence','.']], [0,0]))
    util.print_vars(single_feat_len, pair_feat_len)

    example_idx = -1
    util.create_dirs(os.path.join(out_dir, out_dataset))
    out_path = os.path.join(out_dir, out_dataset, dataset_split + '.txt')
    writer = open(out_path, 'wb')
    total = len(source_files)*1000 if 'cnn' or 'newsroom' in in_dataset else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)
    # for example in tqdm(example_generator, total=total):
    ex_gen = example_generator_extended(example_generator, total, single_feat_len, pair_feat_len)
    print 'Creating list'
    ex_list = [ex for ex in ex_gen]
    print 'Converting...'
    # all_features = pool.map(convert_article_to_lambdamart_features, ex_list)



    # all_features = ray.get([convert_article_to_lambdamart_features.remote(ex) for ex in ex_list])

    all_features = list(futures.map(convert_article_to_lambdamart_features, ex_list))

    # all_features = []
    # for example  in tqdm(ex_gen, total=total):
    #     all_features.append(convert_article_to_lambdamart_features(example))

    writer.write(''.join(all_features))
    # all_features = util.flatten_list_of_lists(all_features)
    # num1 = sum(x == 1 for x in all_features)
    # num2 = sum(x == 2 for x in all_features)
    # print 'Single sent: %d instances. Pair sent: %d instances.' % (num1, num2)

    # for example in tqdm(ex_gen, total=total):
    #     features = convert_article_to_lambdamart_features(example)
    #     writer.write(features)

    writer.close()
    util.print_execution_time(start_time)


if __name__ == '__main__':

    app.run(main)






























