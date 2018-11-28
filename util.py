# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications made 2018 by Logan Lebanoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""
import math

import tensorflow as tf
import time
import os
import numpy as np
from absl import flags
import itertools
import data
from absl import logging
from sumy.nlp.tokenizers import Tokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dill
import random
import inspect, re
import string
import struct
import rouge_functions
import json
FLAGS = flags.FLAGS

stop_words = set(stopwords.words('english'))
CHUNK_SIZE = 1000

def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
            if FLAGS.use_pretrained:
                ckpt_dir = os.path.join(FLAGS.pretrained_path, 'train')
            else:
                ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l.
    """
    chunk_list = []
    for i in xrange(0, len(chunkable), n):
        chunk_list.append( chunkable[i:i+n])
    return chunk_list

def is_list_type(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

def get_first_item(lst):
    if not is_list_type(lst):
        return lst
    for item in lst:
        result = get_first_item(item)
        if result is not None:
            return result
    return None

def remove_period_ids(lst, vocab):
    first_item = get_first_item(lst)
    if first_item is None:
        return lst
    if vocab is not None and type(first_item) == int:
        period = vocab.word2id(data.PERIOD)
    else:
        period = '.'

    if is_list_type(lst[0]):
        return [[item for item in inner_list if item != period] for inner_list in lst]
    else:
        return [item for item in lst if item != period]

def to_unicode(text):
    try:
        text = unicode(text, errors='replace')
    except TypeError:
        return text
    return text

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

def calc_ROUGE_L_score(candidate, reference, metric='f1'):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    beta = 1.2
    prec = []
    rec = []

    if len(reference) == 0:
        return 0.

    if type(reference[0]) is not list:
        reference = [reference]

    for ref in reference:
        # compute the longest common subsequence
        lcs = my_lcs(ref, candidate)
        try:
            prec.append(lcs / float(len(candidate)))
            rec.append(lcs / float(len(ref)))
        except:
            print 'Candidate', candidate
            print 'Reference', ref
            raise


    prec_max = max(prec)
    rec_max = max(rec)

    if metric == 'f1':
        if (prec_max != 0 and rec_max != 0):
            score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
        else:
            score = 0.0
    elif metric == 'precision':
        score = prec_max
    elif metric == 'recall':
        score = rec_max
    else:
        raise Exception('Invalid metric argument: %s. Must be one of {f1,precision,recall}.' % metric)
    return score

def create_token_to_indices(lst):
    token_to_indices = {}
    for token_idx, token in enumerate(lst):
        if token in token_to_indices:
            token_to_indices[token].append(token_idx)
        else:
            token_to_indices[token] = [token_idx]
    return token_to_indices

def matching_unigrams(summ_sent, article_sent, should_remove_stop_words=False):
    if should_remove_stop_words:
        summ_sent = remove_stopwords_punctuation(summ_sent)
        article_sent = remove_stopwords_punctuation(article_sent)
    matches = []
    summ_indices = []
    article_indices = []
    summ_token_to_indices = create_token_to_indices(summ_sent)
    article_token_to_indices = create_token_to_indices(article_sent)
    for token in summ_token_to_indices.keys():
        if token in article_token_to_indices:
            summ_indices.extend(summ_token_to_indices[token])
            article_indices.extend(article_token_to_indices[token])
            matches.extend([token] * len(summ_token_to_indices[token]))
    summ_indices = sorted(summ_indices)
    article_indices = sorted(article_indices)
    return matches, (summ_indices, article_indices)

def is_punctuation(word):
    is_punctuation = [ch in string.punctuation for ch in word]
    if all(is_punctuation):
        return True
    return False

def is_stopword_punctuation(word):
    if word in stop_words or word in ('<s>', '</s>'):
        return True
    is_punctuation = [ch in string.punctuation for ch in word]
    if all(is_punctuation):
        return True
    return False

def remove_stopwords_punctuation(sent):
    new_sent = [token for token in sent if not is_stopword_punctuation(token)]
    return new_sent

'''
Functions for computing sentence similarity between a set of source sentences and a set of summary sentences

'''
def get_similarity(enc_tokens, summ_tokens, vocab):
    metric = 'precision'
    summ_tokens_combined = flatten_list_of_lists(summ_tokens)
    importances_hat = rouge_l_similarity(enc_tokens, summ_tokens_combined, vocab, metric=metric)
    return importances_hat

def rouge_l_similarity(article_sents, abstract_sents, vocab, metric='f1'):
    sentence_similarity = np.zeros([len(article_sents)], dtype=float)
    abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        rouge_l = calc_ROUGE_L_score(article_sent, abstract_sents_removed_periods, metric=metric)
        sentence_similarity[article_sent_idx] = rouge_l
    return sentence_similarity

def rouge_l_similarity_matrix(article_sents, abstract_sents, vocab, metric='f1'):
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents_removed_periods):
            rouge_l = calc_ROUGE_L_score(article_sent, abstract_sent, metric=metric)
            abs_similarities.append(rouge_l)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge_l
    return sentence_similarity_matrix

def rouge_1_similarity_matrix(article_sents, abstract_sents, vocab, metric, should_remove_stop_words):
    if should_remove_stop_words:
        article_sents = [remove_stopwords_punctuation(sent) for sent in article_sents]
        abstract_sents = [remove_stopwords_punctuation(sent) for sent in abstract_sents]
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge = rouge_functions.rouge_1(article_sent, abstract_sent, 0.5, metric=metric)
            abs_similarities.append(rouge)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge
    return sentence_similarity_matrix

def rouge_2_similarity_matrix(article_sents, abstract_sents, vocab, metric, should_remove_stop_words):
    if should_remove_stop_words:
        article_sents = [remove_stopwords_punctuation(sent) for sent in article_sents]
        abstract_sents = [remove_stopwords_punctuation(sent) for sent in abstract_sents]
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge = rouge_functions.rouge_2(article_sent, abstract_sent, 0.5, metric=metric)
            abs_similarities.append(rouge)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge
    return sentence_similarity_matrix


tokenizer = Tokenizer('english')
stemmer = PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))
tfidf_vectorizer = StemmedTfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=0.7)
err_tfidf_vectorizer = StemmedTfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=1.0)

def get_tfidf_matrix(sentences):
    try:
        sent_term_matrix = tfidf_vectorizer.fit_transform(sentences)
    except:
        try:
            sent_term_matrix = err_tfidf_vectorizer.fit_transform(sentences)
        except:
            raise
    return sent_term_matrix

def write_to_temp_files(string_list, temp_dir):
    file_paths = []
    for s_idx, s in enumerate(string_list):
        file_path = os.path.join(temp_dir, '%06d.txt' % s_idx)
        file_paths.append(file_path)
        with open(file_path, 'wb') as f:
            f.write(s)
    return file_paths


def get_doc_substituted_tfidf_matrix(tfidf_vectorizer, sentences, article_text):
    # file_paths = write_to_temp_files([article_text], temp_dir)
    # doc_vec = tfidf_vectorizer.transform(file_paths)
    # file_paths = write_to_temp_files(sentences, temp_dir)
    # sent_term_matrix = tfidf_vectorizer.transform(file_paths)
    doc_vec = tfidf_vectorizer.transform([article_text])
    sent_term_matrix = tfidf_vectorizer.transform(sentences)
    nonzero_rows, nonzero_cols = sent_term_matrix.nonzero()
    nonzero_indices = zip(nonzero_rows, nonzero_cols)
    for idx in nonzero_indices:
        val = doc_vec[0, idx[1]]
        sent_term_matrix[idx] = val
    return sent_term_matrix

def chunk_file(set_name, out_full_dir, out_dir, chunk_size=1000):
  in_file = os.path.join(out_full_dir, '%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(out_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(chunk_size):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def decode_text(text):
    try:
        text = text.decode('utf-8')
    except:
        try:
            text = text.decode('latin-1')
        except:
            raise
    return text

def unpack_tf_example(example, names_to_types):
    def get_string(name):
        return decode_text(example.features.feature[name].bytes_list.value[0])
    def get_string_list(name):
        texts = get_list(name)
        texts = [decode_text(text) for text in texts]
        return texts
    def get_list(name):
        return example.features.feature[name].bytes_list.value
    def get_delimited_list(name):
        text = get_string(name)
        return text.strip().split(' ')
    def get_delimited_list_of_lists(name):
        text = get_string(name)
        return [[int(i) for i in (l.strip().split(' ') if l != '' else [])] for l in text.strip().split(';')]
    def get_delimited_list_of_tuples(name):
        list_of_lists = get_delimited_list_of_lists(name)
        return [tuple(l) for l in list_of_lists]
    def get_json(name):
        text = get_string(name)
        return json.loads(text)
    func = {'string': get_string,
            'list': get_list,
            'string_list': get_string_list,
            'delimited_list': get_delimited_list,
            'delimited_list_of_lists': get_delimited_list_of_lists,
            'delimited_list_of_tuples': get_delimited_list_of_tuples,
            'json': get_json}

    res = []
    for name, type in names_to_types:
        if name not in example.features.feature:
            if name == 'doc_indices':
                res.append(None)
                continue
            else:
                return [None] * len(names_to_types)
                # print example
                # raise Exception('%s is not a feature of TF Example' % name)
        res.append(func[type](name))
    return res

# def get_tfidf_importances(raw_article_sents, tfidf_model_path=None):
def get_tfidf_importances(tfidf_vectorizer, raw_article_sents):
    article_text = ' '.join(raw_article_sents)
    sent_reps = get_doc_substituted_tfidf_matrix(tfidf_vectorizer, raw_article_sents, article_text)
    cluster_rep = np.mean(sent_reps, axis=0)
    similarity_matrix = cosine_similarity(sent_reps, cluster_rep)
    return np.squeeze(similarity_matrix, 1)

def singles_to_singles_pairs(distribution):
    possible_pairs = [tuple(x) for x in
                      list(itertools.combinations(list(xrange(len(distribution))), 2))]  # all pairs
    possible_singles = [tuple([i]) for i in range(len(distribution))]
    all_combinations = possible_pairs + possible_singles
    out_dict = {}
    for single in possible_singles:
        out_dict[single] = distribution[single[0]]
    for pair in possible_pairs:
        average = (distribution[pair[0]] + distribution[pair[1]]) / 2.0
        out_dict[pair] = average
    return out_dict

def combine_sim_and_imp(logan_similarity, logan_importances, lambda_val=0.6):
    mmr = lambda_val*logan_importances - (1-lambda_val)*logan_similarity
    mmr = np.maximum(mmr, 0)
    return mmr

def combine_sim_and_imp_dict(similarities_dict, importances_dict, lambda_val=0.6):
    mmr = {}
    for key in importances_dict.keys():
        mmr[key] = combine_sim_and_imp(similarities_dict[key], importances_dict[key], lambda_val=lambda_val)
    return mmr

def calc_MMR(raw_article_sents, article_sent_tokens, summ_tokens, vocab, importances=None):
    if importances is None:
        importances = get_tfidf_importances(raw_article_sents)
    importances = special_squash(importances)
    similarities = rouge_l_similarity(article_sent_tokens, summ_tokens, vocab, metric='precision')
    mmr = special_squash(combine_sim_and_imp(similarities, importances))
    return mmr

def calc_MMR_source_indices(article_sent_tokens, summ_tokens, vocab, importances_dict, qid=None):
    if qid is not None:
        importances_dict = importances_dict[qid]
    importances_dict = special_squash_dict(importances_dict)
    similarities = rouge_l_similarity(article_sent_tokens, summ_tokens, vocab, metric='precision')
    similarities_dict = singles_to_singles_pairs(similarities)
    mmr_dict = special_squash_dict(combine_sim_and_imp_dict(similarities_dict, importances_dict))
    return mmr_dict

def calc_MMR_all(raw_article_sents, article_sent_tokens, summ_sent_tokens, vocab):
    all_mmr = []
    summ_tokens_so_far = []
    mmr = calc_MMR(raw_article_sents, article_sent_tokens, summ_tokens_so_far, vocab)
    all_mmr.append(mmr)
    for summ_sent in summ_sent_tokens:
        summ_tokens_so_far.extend(summ_sent)
        mmr = calc_MMR(raw_article_sents, article_sent_tokens, summ_tokens_so_far, vocab)
        all_mmr.append(mmr)
    all_mmr = np.stack(all_mmr)
    return all_mmr

def special_squash(distribution):
    res = distribution - np.min(distribution)
    if np.max(res) == 0:
        print('All elements in distribution are 0, so setting all to 0')
        res.fill(0)
    else:
        res = res / np.max(res)
    return res

def special_squash_dict(distribution_dict):
    distribution = distribution_dict.values()
    values = special_squash(distribution)
    keys = distribution_dict.keys()
    items = zip(keys, values)
    out_dict = {}
    for key, val in items:
        out_dict[key] = val
    return out_dict

def print_execution_time(start_time):
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Finished at: ", localtime)
    time_taken = time.time() - start_time
    if time_taken < 60:
        print('Execution time: ', time_taken, ' sec')
    elif time_taken < 3600:
        print('Execution time: ', time_taken/60., ' min')
    else:
        print('Execution time: ', time_taken/3600., ' hr')

def split_list_by_item(lst, item):
    return [list(y) for x, y in itertools.groupby(lst, lambda z: z == item) if not x]

def show_callers_locals():
    """Print the local variables in the caller's frame."""
    callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    return callers_local_vars

def varname(my_var):
    callers_locals = show_callers_locals()
    return [var_name for var_name, var_val in callers_locals if var_val is my_var]

def print_vars(*args):
    for v in args:
        print varname(v), v

def reorder(l, ordering):
    return [l[i] for i in ordering]

def shuffle(*args):
    if len(args) == 0:
        raise Exception('No lists to shuffle')
    permutation = np.random.permutation(len(args[0]))
    return [reorder(arg, permutation) for arg in args]

def create_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def reshape_like(to_reshape, thing_with_shape):
    res = []
    if len(to_reshape) != len(flatten_list_of_lists(thing_with_shape)):
        print 'Len of to_reshape (' + str(len(to_reshape)) + ') does not equal len of thing_with_shape (' + str(len(flatten_list_of_lists(thing_with_shape))) + ')'
    idx = 0
    for lst in thing_with_shape:
        list_to_add = []
        for _ in lst:
            list_to_add.append(to_reshape[idx])
            idx += 1
        res.append(list_to_add)
    return res

def enforce_sentence_limit(groundtruth_similar_source_indices_list, sentence_limit):
    enforced_groundtruth_ssi_list = [ssi[:sentence_limit] for ssi in groundtruth_similar_source_indices_list]
    return enforced_groundtruth_ssi_list

def hist_as_pdf_str(hist):
    vals, bins = hist
    length = np.sum(vals)
    pdf = vals * 100.0 / length
    return '%.2f\t'*len(pdf) % tuple(pdf.tolist())

def find_largest_ckpt_folder(my_dir):
    folder_names = os.listdir(my_dir)
    folder_ckpt_nums = []
    for folder_name in folder_names:
        if '-' not in folder_name:
            ckpt_num = -1
        else:
            ckpt_num = int(folder_name.split('-')[-1].split('_')[0])
        folder_ckpt_nums.append(ckpt_num)
    max_idx = np.argmax(folder_ckpt_nums)
    return folder_names[max_idx]

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)





















