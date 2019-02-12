

import sys
import nltk
import numpy as np
import struct
from tensorflow.core.example import example_pb2
import os
import glob
from . import convert_data
from absl import flags
from absl import app
import pickle
from . import util
from .data import Vocab
from difflib import SequenceMatcher
import itertools
from tqdm import tqdm
from . import importance_features
from . import data
import json
import copy
# from pycorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('http://localhost:9000')
#
# path = "/home/logan/data/discourse/newsroom/data/test-stats.jsonl"
# data = []
#
# with open(path) as f:
#     for ln in f:
#         obj = json.loads(ln)
#         data.append(obj)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

FLAGS = flags.FLAGS


data_dir = '/home/logan/data/tf_data/with_coref'
log_dir = 'logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120

threshold = 0.9
default_exp_name = 'duc_2004_reservoir_lambda_0.6_mute_7_tfidf'

colors = [bcolors.OKBLUE, bcolors.OKGREEN, bcolors.WARNING, bcolors.FAIL]
colors_html = ['blue', 'green', 'gold', 'red']
html_dir = 'data/highlight'
ssi_dir = 'data/ssi'
kaiqiang_dir = 'data/kaiqiang_single_sent_data'
lambdamart_dir = '/home/logan/data/tf_data/with_coref_and_ssi'
highlight_colors = ['aqua', 'lime', 'yellow', '#FF7676', '#B9968D', '#D7BDE2', '#D6DBDF', '#F852AF', '#00FF8B', '#FD933A', '#8C8DFF', '#965DFF']
hard_highlight_colors = ['#00BBFF', '#00BB00', '#F4D03F', '#BB5454', '#A16252', '#AF7AC5', '#AEB6BF', '#FF008F', '#0ECA74', '#FF7400', '#6668FF', '#7931FF']
# hard_highlight_colors = ['blue', 'green', 'orange', 'red']

def start_tag(color):
    return "<font color='" + color + "'>"

def start_tag_highlight(color):
    return "<mark style='background-color: " + color + ";'>"

def html_friendly_fusions(summ_sent, similar_source_indices, lcs_paths, article_lcs_paths, article_sent_tokens):
    end_tag = "</font>"
    out_str = ''
    for token_idx, token in enumerate(summ_sent):
        insert_string = token
        for source_indices_idx, source_indices in enumerate(similar_source_indices):
            if source_indices_idx >= len(colors_html):
                insert_string = token
                break
            if token_idx in lcs_paths[source_indices_idx]:
                insert_string = start_tag(colors_html[source_indices_idx]) + token + end_tag
                break
        out_str += insert_string + ' '
    out_str += '<br><br>'
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        if source_indices_idx >= len(colors_html):
            break
        source_sentence = article_sent_tokens[source_indices[0]]
        for token_idx, token in enumerate(source_sentence):
            insert_string = token
            if token_idx in article_lcs_paths[source_indices_idx]:
                insert_string = start_tag(colors_html[source_indices_idx]) + token + end_tag
            out_str += insert_string + ' '
        out_str += '<br><br>'
    out_str += '-----------------------------------<br>'
    return out_str

def get_idx_for_source_idx(similar_source_indices, source_idx):
    summ_sent_indices = []
    priorities = []
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        for idx_idx, idx in enumerate(source_indices):
            if source_idx == idx:
                summ_sent_indices.append(source_indices_idx)
                priorities.append(idx_idx)
    if len(summ_sent_indices) == 0:
        return None, None
    else:
        return summ_sent_indices, priorities

def html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list,
                                    article_sent_tokens, doc_indices=None, lcs_paths_list=None, article_lcs_paths_list=None):
    end_tag = "</mark>"
    out_str = ''

    for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
        try:
            similar_source_indices = similar_source_indices_list[summ_sent_idx]
        except:
            similar_source_indices = []
            a=0
        # lcs_paths = lcs_paths_list[summ_sent_idx]


        for token_idx, token in enumerate(summ_sent):
            insert_string = token + ' '
            for source_indices_idx, source_indices in enumerate(similar_source_indices):
                if source_indices_idx == 0:
                    # print summ_sent_idx
                    try:
                        color = hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                    except:
                        print(summ_sent_idx)
                        print(summary_sent_tokens)
                        print('\n')
                else:
                    color = highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                # if token_idx in lcs_paths[source_indices_idx]:
                # if lcs_paths_list is not None:
                #     lcs_paths_list[summ_sent_idx][source_indices_idx]
                if lcs_paths_list is None or token_idx in lcs_paths_list[summ_sent_idx][source_indices_idx]:
                    insert_string = start_tag_highlight(color) + token + ' ' + end_tag
                    break
                # else:
                #     insert_string = start_tag_highlight(highlight_colors[source_indices_idx]) + token + end_tag
                #     break
            out_str += insert_string
        out_str += '<br><br>'

    cur_token_idx = 0
    cur_doc_idx = 0
    for sent_idx, sent in enumerate(article_sent_tokens):
        if doc_indices is not None:
            if cur_token_idx >= len(doc_indices):
                print("Warning: cur_token_idx is greater than len of doc_indices")
            elif doc_indices[cur_token_idx] != cur_doc_idx:
                cur_doc_idx = doc_indices[cur_token_idx]
                out_str += '<br>'
        summ_sent_indices, priorities = get_idx_for_source_idx(similar_source_indices_list, sent_idx)
        if priorities is None:
            colors = ['black']
            hard_colors = ['black']
        else:
            colors = [highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
            hard_colors = [hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
            # article_lcs_paths = article_lcs_paths_list[summ_sent_idx]
        source_sentence = article_sent_tokens[sent_idx]
        for token_idx, token in enumerate(source_sentence):
            if priorities is None:
                insert_string = token + ' '
            # elif token_idx in article_lcs_paths[priority]:
            else:
                insert_string = token + ' '
                for priority_idx in reversed(list(range(len(priorities)))):
                    summ_sent_idx = summ_sent_indices[priority_idx]
                    priority = priorities[priority_idx]
                    if article_lcs_paths_list is None or token_idx in article_lcs_paths_list[summ_sent_idx][priority]:
                        if priority == 0:
                            insert_string = start_tag_highlight(hard_colors[priority_idx]) + token + ' ' + end_tag
                        else:
                            insert_string = start_tag_highlight(colors[priority_idx]) + token + ' ' + end_tag
            # else:
                # insert_string = start_tag_highlight(highlight_colors[priority]) + token + end_tag
            cur_token_idx += 1
            out_str += insert_string
        out_str += '<br>'
    out_str += '<br>------------------------------------------------------<br><br>'
    return out_str


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
    if FLAGS.dataset_name == 'duc_2004':
        abstract_idx = FLAGS.abstract_idx
    else:
        abstract_idx = 0
    summary_text = '\n'.join(all_abstract_sentences[abstract_idx])
    return summary_text

def split_into_tokens(text):
    tokens = text.split()
    tokens = [t for t in tokens if t != '<s>' and t != '</s>']
    return tokens

def split_into_sent_tokens(text):
    sent_tokens = [[t for t in tokens.strip().split() if t != '<s>' and t != '</s>'] for tokens in text.strip().split('\n')]
    return sent_tokens

def get_sent_similarities(summ_sent, article_sent_tokens, vocab, only_rouge_l=False, remove_stop_words=True):
    similarity_matrix = util.rouge_l_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall')
    similarities = np.squeeze(similarity_matrix, 1)

    if not only_rouge_l:
        rouge_l = similarities
        rouge_1 = np.squeeze(util.rouge_1_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', remove_stop_words), 1)
        rouge_2 = np.squeeze(util.rouge_2_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', False), 1)
        similarities = (rouge_l + rouge_1 + rouge_2) / 3.0

    return similarities

def get_top_similar_sent(summ_sent, article_sent_tokens, vocab, remove_stop_words=True, multiple_ssi=False):
    try:
        similarities = get_sent_similarities(summ_sent, article_sent_tokens, vocab, remove_stop_words=remove_stop_words)
        top_similarity = np.max(similarities)
    except:
        print(summ_sent)
        print(article_sent_tokens)
        raise
    # sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim == top_similarity]
    if multiple_ssi:
        sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim > top_similarity * 0.75]
    else:
        sent_indices = [np.argmax(similarities)]
    return sent_indices, top_similarity

def replace_with_blanks(summ_sent, selection):
    replaced_summ_sent = [summ_sent[token_idx] if token_idx in selection else '' for token_idx, token in enumerate(summ_sent)]
    return  replaced_summ_sent

def get_similar_source_sents_by_lcs(summ_sent, selection, article_sent_tokens, vocab, similarities, depth, sentence_limit, min_matched_tokens, remove_stop_words=True, multiple_ssi=False):
    remove_unigrams = True
    if sentence_limit == 1:
        if depth > 2:
            return [[]], [[]], [[]]
    elif len(selection) < 3 or depth >= sentence_limit:      # base case: when summary sentence is too short
        return [[]], [[]], [[]]

    all_sent_indices = []
    all_lcs_paths = []
    all_article_lcs_paths = []

    # partial_summ_sent = util.reorder(summ_sent, selection)
    top_sent_indices, top_similarity = get_top_similar_sent(summ_sent, article_sent_tokens, vocab, remove_stop_words, multiple_ssi=multiple_ssi)
    top_similarities = util.reorder(similarities, top_sent_indices)
    top_sent_indices = [x for _, x in sorted(zip(top_similarities, top_sent_indices), key=lambda pair: pair[0])][::-1]
    for top_sent_idx in top_sent_indices:
        # top_sent_idx = top_sent_indices[0]
        if remove_unigrams:
            nonstopword_matches, _ = util.matching_unigrams(summ_sent, article_sent_tokens[top_sent_idx], should_remove_stop_words=remove_stop_words)
            lcs_len, (summ_lcs_path, article_lcs_path) = util.matching_unigrams(summ_sent, article_sent_tokens[top_sent_idx])
        if len(nonstopword_matches) < min_matched_tokens:
            continue
        # new_selection = [selection[idx] for idx in summ_lcs_path]
        # leftover_selection = [val for idx, val in enumerate(selection) if idx not in summ_lcs_path]
        # partial_summ_sent = replace_with_blanks(summ_sent, leftover_selection)
        leftover_selection = [idx for idx in range(len(summ_sent)) if idx not in summ_lcs_path]
        partial_summ_sent = replace_with_blanks(summ_sent, leftover_selection)

        sent_indices, lcs_paths, article_lcs_paths = get_similar_source_sents_by_lcs(
            partial_summ_sent, leftover_selection, article_sent_tokens, vocab, similarities, depth+1,
            sentence_limit, min_matched_tokens, remove_stop_words)   # recursive call

        combined_sent_indices = [[top_sent_idx] + indices for indices in sent_indices]      # append my result to the recursive collection
        combined_lcs_paths = [[summ_lcs_path] + paths for paths in lcs_paths]
        combined_article_lcs_paths = [[article_lcs_path] + paths for paths in article_lcs_paths]

        all_sent_indices.extend(combined_sent_indices)
        all_lcs_paths.extend(combined_lcs_paths)
        all_article_lcs_paths.extend(combined_article_lcs_paths)
    if len(all_sent_indices) == 0:
        return [[]], [[]], [[]]
    return all_sent_indices, all_lcs_paths, all_article_lcs_paths

def cluster_similar_source_sents(article_sent_tokens, similar_source_indices, vocab, threshold):
    chosen_article_sents = [sent for i, sent in enumerate(article_sent_tokens) if i in similar_source_indices]
    temp_similarity_matrix = util.rouge_l_similarity_matrix(chosen_article_sents,
                                         chosen_article_sents, vocab, 'f1')
    similarity_matrix = np.zeros([len(article_sent_tokens), len(article_sent_tokens)], dtype=float)
    for row_idx in range(len(temp_similarity_matrix)):
        for col_idx in range(len(temp_similarity_matrix)):
            similarity_matrix[similar_source_indices[row_idx], similar_source_indices[col_idx]] = temp_similarity_matrix[row_idx, col_idx]

    groups = [[similar_source_indices[0]]]
    for sent_idx in similar_source_indices[1:]:
        found_group = False
        for group in groups:
            for group_member in group:
                similarity = similarity_matrix[sent_idx, group_member]
                if similarity >= threshold:
                    found_group = True
                    group.append(sent_idx)
                    break
            if found_group:
                break
        if not found_group:
            groups.append([sent_idx])
    return groups

def get_shortest_distance(indices1, indices2, relative_to_article, rel_sent_positions):
    if relative_to_article:
        indices1 = [rel_sent_positions[idx] for idx in indices1]
        indices2 = [rel_sent_positions[idx] for idx in indices2]
    pairs = list(itertools.product(indices1, indices2))
    min_dist = min([abs(x - y) for x,y in pairs])
    return min_dist

def get_merge_example(similar_source_indices, article_sent_tokens, summ_sent, corefs):
    # restricted_source_indices = []
    # for source_indices_idx, source_indices in enumerate(similar_source_indices):
    #     if source_indices_idx >= FLAGS.sentence_limit:
    #         break
    #     restricted_source_indices.append(source_indices[0])
    merged_example_sentences = [' '.join(sent) for sent in util.reorder(article_sent_tokens, similar_source_indices)]
    merged_example_article_text = ' '.join(merged_example_sentences)
    merged_example_abstracts = [[' '.join(summ_sent)]]
    merge_example = convert_data.make_example(merged_example_article_text, merged_example_abstracts, None, merged_example_sentences, corefs)
    return merge_example

def write_lambdamart_example(simple_similar_source_indices, raw_article_sents, summary_text, corefs_str, doc_indices, writer):
    tf_example = example_pb2.Example()
    source_indices_str = ';'.join([' '.join(str(i) for i in source_indices) for source_indices in simple_similar_source_indices])
    tf_example.features.feature['similar_source_indices'].bytes_list.value.extend([source_indices_str.encode("utf8")])
    for sent in raw_article_sents:
        s = sent.encode('utf-8').strip()
        tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([s])
    tf_example.features.feature['summary_text'].bytes_list.value.extend([summary_text.encode("utf8")])
    if doc_indices is not None:
        tf_example.features.feature['doc_indices'].bytes_list.value.extend([doc_indices])
    tf_example.features.feature['corefs'].bytes_list.value.extend([corefs_str])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def get_kaiqiang_article_abstract(similar_source_indices, raw_article_sents, summ_sent):
    source_idx = similar_source_indices[0][0]
    article_text = raw_article_sents[source_idx]
    abstract_text = ' '.join(summ_sent)
    return article_text, abstract_text

def get_single_sent_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens):
    sent_idx = similar_source_indices[0]
    doc_similarity = util.cosine_similarity(sent_term_matrix[sent_idx], doc_vector)
    sent_len = len(article_sent_tokens[sent_idx])
    return sent_idx, doc_similarity, sent_len

def get_simple_source_indices_list(summary_sent_tokens, article_sent_tokens, vocab, sentence_limit, min_matched_tokens, remove_stop_words=True, lemmatize=True, multiple_ssi=False):
    if lemmatize:
        article_sent_tokens_lemma = util.lemmatize_sent_tokens(article_sent_tokens)
        summary_sent_tokens_lemma = util.lemmatize_sent_tokens(summary_sent_tokens)
    else:
        article_sent_tokens_lemma = article_sent_tokens
        summary_sent_tokens_lemma = summary_sent_tokens

    similar_source_indices_list = []
    lcs_paths_list = []
    article_lcs_paths_list = []
    for summ_sent in summary_sent_tokens_lemma:
        remove_lcs = True
        similarities = get_sent_similarities(summ_sent, article_sent_tokens_lemma, vocab)
        if remove_lcs:
            similar_source_indices, lcs_paths, article_lcs_paths = get_similar_source_sents_by_lcs(
                summ_sent, list(range(len(summ_sent))), article_sent_tokens_lemma, vocab, similarities, 0,
                sentence_limit, min_matched_tokens, remove_stop_words=remove_stop_words, multiple_ssi=multiple_ssi)
            similar_source_indices_list.append(similar_source_indices)
            lcs_paths_list.append(lcs_paths)
            article_lcs_paths_list.append(article_lcs_paths)
    deduplicated_similar_source_indices_list = []
    for sim_source_ind in similar_source_indices_list:
        dedup_sim_source_ind = []
        for ssi in sim_source_ind:
            if not (ssi in dedup_sim_source_ind or ssi[::-1] in dedup_sim_source_ind):
                dedup_sim_source_ind.append(ssi)
        deduplicated_similar_source_indices_list.append(dedup_sim_source_ind)
    # for sim_source_ind_idx, sim_source_ind in enumerate(deduplicated_similar_source_indices_list):
    #     if len(sim_source_ind) > 1:
    #         print ' '.join(summary_sent_tokens[sim_source_ind_idx])
    #         print '-----------'
    #         for ssi in sim_source_ind:
    #             for idx in ssi:
    #                 print ' '.join(article_sent_tokens[idx])
    #             print '-------------'
    #         print '\n\n'
    #         a=0
    simple_similar_source_indices = [tuple(sim_source_ind[0]) for sim_source_ind in deduplicated_similar_source_indices_list]
    lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in lcs_paths_list]
    article_lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in article_lcs_paths_list]
    return simple_similar_source_indices, lcs_paths_list, article_lcs_paths_list

ngram_orders = [1, 2, 3, 4, 'sentence']


def main(unused_argv):

    print('Running statistics on %s' % FLAGS.exp_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.only_highlight:
        FLAGS.sent_dataset = False
        FLAGS.ssi_dataset = False
        FLAGS.print_output = False
        FLAGS.highlight = True

    original_dataset_name = 'xsum' if 'xsum' in FLAGS.dataset_name else 'cnn_dm' if ('cnn_dm' in FLAGS.dataset_name or 'duc_2004' in FLAGS.dataset_name) else ''
    vocab = Vocab(FLAGS.vocab_path + '_' + original_dataset_name, FLAGS.vocab_size) # create a vocabulary

    source_dir = os.path.join(data_dir, FLAGS.dataset_name)
    util.create_dirs(html_dir)

    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for dataset_split in dataset_splits:
        source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
        if FLAGS.exp_name == 'reference':
            # summary_dir = log_dir + default_exp_name + '/decode_test_' + str(max_enc_steps) + \
            #                 'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/reference'
            # summary_files = sorted(glob.glob(summary_dir + '/*_reference.A.txt'))
            summary_dir = source_dir
            summary_files = source_files
        else:
            if FLAGS.exp_name == 'cnn_dm':
                summary_dir = log_dir + FLAGS.exp_name + '/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
            else:
                ckpt_folder = util.find_largest_ckpt_folder(log_dir + FLAGS.exp_name)
                summary_dir = log_dir + FLAGS.exp_name + '/' + ckpt_folder + '/decoded'
                # summary_dir = log_dir + FLAGS.exp_name + '/decode_test_' + str(max_enc_steps) + \
                #             'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/decoded'
            summary_files = sorted(glob.glob(summary_dir + '/*'))
        if len(summary_files) == 0:
            raise Exception('No files found in %s' % summary_dir)
        example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False)
        pros = {'annotators': 'dcoref', 'outputFormat': 'json', 'timeout': '5000000'}
        all_merge_examples = []
        num_extracted_list = []
        distances = []
        relative_distances = []
        html_str = ''
        extracted_sents_in_article_html = ''
        name = FLAGS.dataset_name + '_' + FLAGS.exp_name
        if FLAGS.coreference_replacement:
            name += '_coref'
        highlight_file_name = os.path.join(html_dir, FLAGS.dataset_name + '_' + FLAGS.exp_name)
        if FLAGS.consider_stopwords:
            highlight_file_name += '_stopwords'
        if FLAGS.highlight:
            extracted_sents_in_article_html_file = open(highlight_file_name + '_extracted_sents.html', 'wb')
        if FLAGS.kaiqiang:
            kaiqiang_article_texts = []
            kaiqiang_abstract_texts = []
            util.create_dirs(kaiqiang_dir)
            kaiqiang_article_file = open(os.path.join(kaiqiang_dir, FLAGS.dataset_name + '_' + dataset_split + '_' + str(FLAGS.min_matched_tokens) + '_articles.txt'), 'wb')
            kaiqiang_abstract_file = open(os.path.join(kaiqiang_dir, FLAGS.dataset_name + '_' + dataset_split + '_' + str(FLAGS.min_matched_tokens)  + '_abstracts.txt'), 'wb')
        if FLAGS.ssi_dataset:
            lambdamart_out_dir = os.path.join(lambdamart_dir, FLAGS.dataset_name)
            if FLAGS.sentence_limit == 1:
                lambdamart_out_dir += '_singles'
            if FLAGS.consider_stopwords:
                lambdamart_out_dir += '_stopwords'
            lambdamart_out_full_dir = os.path.join(lambdamart_out_dir, 'all')
            util.create_dirs(lambdamart_out_full_dir)
            lambdamart_writer = open(os.path.join(lambdamart_out_full_dir, dataset_split + '.bin'), 'wb')

        simple_similar_source_indices_list_plus_empty = []
        example_idx = -1
        instance_idx = 0
        total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
        for example in tqdm(example_generator, total=total):
            example_idx += 1
            if FLAGS.num_instances != -1 and instance_idx >= FLAGS.num_instances:
                break
        # for file_idx in tqdm(range(len(source_files))):
        #     example = get_tf_example(source_files[file_idx])
            article_text = example.features.feature['article'].bytes_list.value[0].lower()
            if FLAGS.exp_name == 'reference':
                summary_text = get_summary_from_example(example)
            else:
                summary_text = get_summary_text(summary_files[example_idx])
            article_tokens = split_into_tokens(article_text)
            if 'raw_article_sents' in example.features.feature and len(example.features.feature['raw_article_sents'].bytes_list.value) > 0:
                raw_article_sents = example.features.feature['raw_article_sents'].bytes_list.value

                raw_article_sents = [sent for sent in raw_article_sents if sent.strip() != '']
                article_sent_tokens = [convert_data.process_sent(sent) for sent in raw_article_sents]
            else:
                article_text = util.to_unicode(article_text)
                # sent_pros = {'annotators': 'ssplit', 'outputFormat': 'json', 'timeout': '5000000'}
                # sents_result_dict = nlp.annotate(str(article_text), properties=sent_pros)
                # article_sent_tokens = [[token['word'] for token in sent['tokens']] for sent in sents_result_dict['sentences']]

                raw_article_sents = nltk.tokenize.sent_tokenize(article_text)
                article_sent_tokens = [convert_data.process_sent(sent) for sent in raw_article_sents]
            if FLAGS.top_n_sents != -1:
                article_sent_tokens = article_sent_tokens[:FLAGS.top_n_sents]
                raw_article_sents = raw_article_sents[:FLAGS.top_n_sents]
            article_sents = [' '.join(sent) for sent in article_sent_tokens]
            try:
                article_tokens_string = str(' '.join(article_sents))
            except:
                try:
                    article_tokens_string = str(' '.join([sent.decode('latin-1') for sent in article_sents]))
                except:
                    raise


            if len(article_sent_tokens) == 0:
                continue

            summary_sent_tokens = split_into_sent_tokens(summary_text)
            if 'doc_indices' in example.features.feature and len(example.features.feature['doc_indices'].bytes_list.value) > 0:
                doc_indices_str = example.features.feature['doc_indices'].bytes_list.value[0]
                if '1' in doc_indices_str:
                    doc_indices = [int(x) for x in doc_indices_str.strip().split()]
                    rel_sent_positions = importance_features.get_sent_indices(article_sent_tokens, doc_indices)
                else:
                    num_tokens_total = sum([len(sent) for sent in article_sent_tokens])
                    rel_sent_positions = list(range(len(raw_article_sents)))
                    doc_indices = [0] * num_tokens_total

            else:
                rel_sent_positions = None
                doc_indices = None
                doc_indices_str = None
            if 'corefs' in example.features.feature and len(
                    example.features.feature['corefs'].bytes_list.value) > 0:
                corefs_str = example.features.feature['corefs'].bytes_list.value[0]
                corefs = json.loads(corefs_str)
            # summary_sent_tokens = limit_to_n_tokens(summary_sent_tokens, 100)

            similar_source_indices_list_plus_empty = []

            simple_similar_source_indices, lcs_paths_list,article_lcs_paths_list =  get_simple_source_indices_list(
                summary_sent_tokens, article_sent_tokens, vocab, FLAGS.sentence_limit, FLAGS.min_matched_tokens, not FLAGS.consider_stopwords, lemmatize=FLAGS.lemmatize,
                multiple_ssi=FLAGS.multiple_ssi)

            restricted_source_indices = util.enforce_sentence_limit(simple_similar_source_indices, FLAGS.sentence_limit)
            for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
                if FLAGS.sent_dataset:
                    if len(restricted_source_indices[summ_sent_idx]) == 0:
                        continue
                    merge_example = get_merge_example(restricted_source_indices[summ_sent_idx], article_sent_tokens, summ_sent, corefs)
                    all_merge_examples.append(merge_example)

            simple_similar_source_indices_list_plus_empty.append(simple_similar_source_indices)
            if FLAGS.ssi_dataset:
                write_lambdamart_example(simple_similar_source_indices, raw_article_sents, summary_text, corefs_str, doc_indices_str, lambdamart_writer)


            if FLAGS.highlight:
                # simple_ssi_plus_empty = [ [s[0] for s in sim_source_ind] for sim_source_ind in simple_similar_source_indices]
                extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, simple_similar_source_indices,
                                                                                  article_sent_tokens, doc_indices,
                                                                                  lcs_paths_list, article_lcs_paths_list)
                extracted_sents_in_article_html_file.write(extracted_sents_in_article_html)
            a=0

            instance_idx += 1


        if FLAGS.ssi_dataset:
            lambdamart_writer.close()
            if FLAGS.dataset_name == 'cnn_dm' or FLAGS.dataset_name == 'newsroom' or FLAGS.dataset_name == 'xsum':
                chunk_size = 1000
            else:
                chunk_size = 1
            util.chunk_file(dataset_split, lambdamart_out_full_dir, lambdamart_out_dir, chunk_size=chunk_size)

        if FLAGS.sent_dataset:
            out_dir = os.path.join(data_dir, FLAGS.dataset_name + '_sent')
            if FLAGS.sentence_limit == 1:
                out_dir += '_singles'
            if FLAGS.consider_stopwords:
                out_dir += '_stopwords'
            util.create_dirs(out_dir)
            if FLAGS.coreference_replacement:
                out_dir += '_coref'
            if FLAGS.top_n_sents != -1:
                out_dir += '_n=' + str(FLAGS.top_n_sents)
            convert_data.write_with_generator(iter(all_merge_examples), len(all_merge_examples), out_dir, dataset_split)

        if FLAGS.print_output:
            # html_str = FLAGS.dataset + ' | ' + FLAGS.exp_name + '<br><br><br>' + html_str
            # save_fusions_to_file(html_str)
            ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name)
            if FLAGS.consider_stopwords:
                ssi_path += '_stopwords'
            util.create_dirs(ssi_path)
            if FLAGS.dataset_name == 'duc_2004' and FLAGS.abstract_idx != 0:
                abstract_idx_str = '_%d' % FLAGS.abstract_idx
            else:
                abstract_idx_str = ''
            with open(os.path.join(ssi_path, dataset_split + '_ssi' + abstract_idx_str + '.pkl'), 'wb') as f:
                pickle.dump(simple_similar_source_indices_list_plus_empty, f)

        if FLAGS.kaiqiang:
            # kaiqiang_article_file.write('\n'.join(kaiqiang_article_texts))
            # kaiqiang_abstract_file.write('\n'.join(kaiqiang_abstract_texts))
            kaiqiang_article_file.close()
            kaiqiang_abstract_file.close()
        if FLAGS.highlight:
            extracted_sents_in_article_html_file.close()
        a=0


if __name__ == '__main__':
    flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
                               ' If you want to run on human summaries, then enter "reference".')
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
    flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
    flags.DEFINE_integer('num_instances', -1, 'Number of instances to run for before stopping. Use -1 to run on all instances.')
    flags.DEFINE_string('vocab_path', 'logs/vocab', 'Path expression to text vocabulary file.')
    flags.DEFINE_boolean('only_rouge_l', False, 'Whether to use only R-L in calculating similarity or whether to average over R-1, R-2, and R-L.')
    flags.DEFINE_boolean('coreference_replacement', False, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('kaiqiang', False, 'Whether to save the single sentences as a dataset for Kaiqiang.')
    flags.DEFINE_integer('top_n_sents', -1, 'Number of sentences to take from the beginning of the article. Use -1 to run on entire article.')
    flags.DEFINE_integer('min_matched_tokens', 2, 'Number of tokens required that still counts a source sentence as matching a summary sentence.')
    flags.DEFINE_integer('abstract_idx', 0, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('consider_stopwords', False, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('print_output', True, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('highlight', True, 'Whether to save an html file that shows the selected sentences as highlighted in the article.')
    flags.DEFINE_boolean('sent_dataset', True, 'Whether to save the merged sentences as a dataset.')
    flags.DEFINE_boolean('ssi_dataset', True, 'Whether to save features as a dataset that will be used to predict which sentences should be merged, using the LambdaMART system.')
    flags.DEFINE_boolean('only_highlight', False, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('lemmatize', True, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('multiple_ssi', False, 'Allow multiple singles are pairs to be chosen for each summary sentence, rather than just the top similar sentence.')

    app.run(main)

















