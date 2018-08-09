import sys
import nltk
import numpy as np
import struct
from tensorflow.core.example import example_pb2
import os
import glob
import write_data
from absl import flags
from absl import app
import cPickle
import util
from data import Vocab
from difflib import SequenceMatcher
import itertools
from tqdm import tqdm
import importance_features
import data
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


data_dir = '/home/logan/data/multidoc_summarization/tf_examples'
log_dir = '/home/logan/data/multidoc_summarization/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120

threshold = 0.9
default_exp_name = 'duc_2004_reservoir_lambda_0.6_mute_7_tfidf'

colors = [bcolors.OKBLUE, bcolors.OKGREEN, bcolors.WARNING, bcolors.FAIL]
colors_html = ['blue', 'green', 'gold', 'red']
html_dir = '/home/logan/data/discourse'
kaiqiang_dir = '/home/logan/data/discourse/kaiqiang_single_sent_data'
lambdamart_dir = '/home/logan/data/multidoc_summarization/merge_indices_tf_examples'
# lambdamart_chunked_dir = '/home/logan/data/multidoc_summarization/merge_indices_tf_examples'
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
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        for idx_idx, idx in enumerate(source_indices):
            if source_idx == idx:
                return source_indices_idx, idx_idx
    return None, None

def html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list,
                                    article_sent_tokens, doc_indices=None, lcs_paths_list=None, article_lcs_paths_list=None):
    end_tag = "</mark>"
    out_str = ''

    for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
        similar_source_indices = similar_source_indices_list[summ_sent_idx]
        # lcs_paths = lcs_paths_list[summ_sent_idx]


        for token_idx, token in enumerate(summ_sent):
            insert_string = token + ' '
            for source_indices_idx, source_indices in enumerate(similar_source_indices):
                if source_indices_idx == 0:
                    # print summ_sent_idx
                    color = hard_highlight_colors[summ_sent_idx]
                else:
                    color = highlight_colors[summ_sent_idx]
                # if token_idx in lcs_paths[source_indices_idx]:
                if lcs_paths_list is None or lcs_paths_list[summ_sent_idx][source_indices_idx]:
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
        if doc_indices is not None and doc_indices[cur_token_idx] != cur_doc_idx:
            cur_doc_idx = doc_indices[cur_token_idx]
            out_str += '<br>'
        summ_sent_idx, priority = get_idx_for_source_idx(similar_source_indices_list, sent_idx)
        if priority is None:
            color = 'black'
            hard_color = 'black'
        else:
            color = highlight_colors[summ_sent_idx]
            hard_color = hard_highlight_colors[summ_sent_idx]
            # article_lcs_paths = article_lcs_paths_list[summ_sent_idx]
        source_sentence = article_sent_tokens[sent_idx]
        for token_idx, token in enumerate(source_sentence):
            if priority is None:
                insert_string = token + ' '
            # elif token_idx in article_lcs_paths[priority]:
            elif article_lcs_paths_list is None or article_lcs_paths_list[summ_sent_idx][priority]:
                if priority == 0:
                    insert_string = start_tag_highlight(hard_color) + token + ' ' + end_tag
                else:
                    insert_string = start_tag_highlight(color) + token + ' ' + end_tag
            else:
                # insert_string = start_tag_highlight(highlight_colors[priority]) + token + end_tag
                insert_string = token + ' '
            cur_token_idx += 1
            out_str += insert_string
        out_str += '<br>'
    out_str += '<br>------------------------------------------------------<br><br>'
    return out_str



def save_fusions_to_file(out_str):
    name = FLAGS.dataset + '_' + FLAGS.exp_name
    if FLAGS.coreference_replacement:
        name += '_coref'
    file_name = os.path.join(html_dir, name + '.html')
    with open(file_name, 'wb') as f:
        f.write(out_str)


def get_nGram(l, n = 2):
    l = list(l)
    return set(zip(*[l[i:] for i in range(n)]))

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

def does_sent_appear_in_source(sent_tokens, article_tokens):
    len_sent_grams = list(get_nGram(article_tokens, n=len(sent_tokens)))
    return tuple(sent_tokens) in len_sent_grams

def find_extracted_phrase(article_sent_tokens, summ_sent, top_sent_idx):
    article_sent = article_sent_tokens[top_sent_idx]
    match = SequenceMatcher(None, summ_sent, article_sent).find_longest_match(0, len(summ_sent), 0, len(article_sent))
    start_idx = match.a
    end_idx = start_idx + match.size
    return start_idx, end_idx

def get_sent_similarities(summ_sent, article_sent_tokens, vocab):
    remove_stop_words = True
    similarity_matrix = util.Similarity_Functions.rouge_l_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', False)
    similarities = np.squeeze(similarity_matrix, 1)

    if not FLAGS.only_rouge_l:
        rouge_l = similarities
        rouge_1 = np.squeeze(util.Similarity_Functions.rouge_1_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', remove_stop_words), 1)
        rouge_2 = np.squeeze(util.Similarity_Functions.rouge_2_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', False), 1)
        similarities = (rouge_l + rouge_1 + rouge_2) / 3.0

    return similarities

def get_top_similar_sent(summ_sent, article_sent_tokens, vocab):
    try:
        similarities = get_sent_similarities(summ_sent, article_sent_tokens, vocab)
        top_similarity = np.max(similarities)
    except:
        print summ_sent
        print article_sent_tokens
        raise
    sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim == top_similarity]
    return sent_indices, top_similarity

def get_similar_source_sents(summ_sent, article_sent_tokens, vocab, threshold):
    top_sent_indices, top_similarity = get_top_similar_sent(summ_sent, article_sent_tokens, vocab)
    if top_similarity >= threshold:
        return [top_sent_indices], None, None
    else:
        top_sent_idx = top_sent_indices[0]
        start_idx, end_idx = find_extracted_phrase(article_sent_tokens, summ_sent, top_sent_idx)
        leftover_sent = summ_sent[:start_idx] + summ_sent[end_idx:]
        if len(leftover_sent) >= 3:
            second_sent_indices, second_similarity = get_top_similar_sent(leftover_sent, article_sent_tokens, vocab)
            return [top_sent_indices, second_sent_indices], start_idx, end_idx
        else:
            return [top_sent_indices], None, None

def get_similar_source_sents_by_lcs(summ_sent, selection, article_sent_tokens, vocab, similarities, depth):
    remove_unigrams = True
    if len(selection) < 3 or depth >= FLAGS.sentence_limit:      # base case: when summary sentence is too short
        return [], [], []
    partial_summ_sent = util.reorder(summ_sent, selection)
    top_sent_indices, top_similarity = get_top_similar_sent(partial_summ_sent, article_sent_tokens, vocab)
    top_similarities = util.reorder(similarities, top_sent_indices)
    top_sent_indices = [x for _, x in sorted(zip(top_similarities, top_sent_indices), key=lambda pair: pair[0])][::-1]
    top_sent_idx = top_sent_indices[0]
    if remove_unigrams:
        nonstopword_matches, _ = util.matching_unigrams(partial_summ_sent, article_sent_tokens[top_sent_idx], should_remove_stop_words=True)
        lcs_len, (summ_lcs_path, article_lcs_path) = util.matching_unigrams(partial_summ_sent, article_sent_tokens[top_sent_idx])
    else:
        lcs_len, (summ_lcs_path, article_lcs_path) = util.lcs(partial_summ_sent, article_sent_tokens[top_sent_idx])
    if len(nonstopword_matches) < FLAGS.min_matched_tokens:
        return [], [], []
    new_selection = [selection[idx] for idx in summ_lcs_path]
    leftover_selection = [val for idx, val in enumerate(selection) if idx not in summ_lcs_path]

    sent_indices, lcs_paths, article_lcs_paths = get_similar_source_sents_by_lcs(
        summ_sent, leftover_selection, article_sent_tokens, vocab, similarities, depth+1)   # recursive call

    sent_indices = [top_sent_indices] + sent_indices      # append my result to the recursive collection
    lcs_paths = [new_selection] + lcs_paths
    article_lcs_paths = [article_lcs_path] + article_lcs_paths
    return sent_indices, lcs_paths, article_lcs_paths

def cluster_similar_source_sents(article_sent_tokens, similar_source_indices, vocab, threshold):
    chosen_article_sents = [sent for i, sent in enumerate(article_sent_tokens) if i in similar_source_indices]
    temp_similarity_matrix = util.Similarity_Functions.rouge_l_similarity_matrix(chosen_article_sents,
                                         chosen_article_sents, vocab, 'f1', False)
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

def get_merge_example(similar_source_indices, article_sent_tokens, summ_sent):
    restricted_source_indices = []
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        if source_indices_idx >= FLAGS.sentence_limit:
            break
        restricted_source_indices.append(source_indices[0])
    merged_example_sentences = [' '.join(sent) for sent in util.reorder(article_sent_tokens, restricted_source_indices)]
    merged_example_article_text = ' '.join(merged_example_sentences)
    merged_example_abstracts = [[' '.join(summ_sent)]]
    merge_example = write_data.get_example_from_article(merged_example_article_text, merged_example_abstracts, tokenize_abstract=False)
    return merge_example

def write_lambdamart_example(simple_similar_source_indices, raw_article_sents, summary_text, writer):
    tf_example = example_pb2.Example()
    source_indices_str = ';'.join([' '.join(str(i) for i in source_indices) for source_indices in simple_similar_source_indices])
    tf_example.features.feature['similar_source_indices'].bytes_list.value.extend([source_indices_str])
    for sent in raw_article_sents:
        s = sent.encode('utf-8').strip()
        tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([s])
    tf_example.features.feature['summary_text'].bytes_list.value.extend([summary_text])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def get_kaiqiang_article_abstract(similar_source_indices, raw_article_sents, summ_sent):
    source_idx = similar_source_indices[0][0]
    article_text = raw_article_sents[source_idx]
    abstract_text = ' '.join(summ_sent)
    return article_text, abstract_text

def find_largest_ckpt_folder(my_dir):
    folder_names = os.listdir(my_dir)
    folder_ckpt_nums = []
    for folder_name in folder_names:
        if '-' not in folder_name:
            ckpt_num = -1
        else:
            ckpt_num = int(folder_name.split('-')[-1])
        folder_ckpt_nums.append(ckpt_num)
    max_idx = np.argmax(folder_ckpt_nums)
    return folder_names[max_idx]

def get_single_sent_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens):
    sent_idx = similar_source_indices[0]
    doc_similarity = util.cosine_similarity(sent_term_matrix[sent_idx], doc_vector)
    sent_len = len(article_sent_tokens[sent_idx])
    return sent_idx, doc_similarity, sent_len

ngram_orders = [1, 2, 3, 4, 'sentence']


def main(unused_argv):

    print 'Running statistics on %s' % FLAGS.exp_name

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    source_dir = os.path.join(data_dir, FLAGS.dataset)
    source_files = sorted(glob.glob(source_dir + '/' + FLAGS.dataset_split + '*'))
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
            ckpt_folder = find_largest_ckpt_folder(log_dir + FLAGS.exp_name)
            summary_dir = log_dir + FLAGS.exp_name + '/' + ckpt_folder + '/decoded'
            # summary_dir = log_dir + FLAGS.exp_name + '/decode_test_' + str(max_enc_steps) + \
            #             'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/decoded'
        summary_files = sorted(glob.glob(summary_dir + '/*'))
    if len(summary_files) == 0:
        raise Exception('No files found in %s' % summary_dir)
    example_generator = data.example_generator(source_dir + '/' + FLAGS.dataset_split + '*', True, False)
    pros = {'annotators': 'dcoref', 'outputFormat': 'json', 'timeout': '5000000'}
    all_merge_examples = []
    num_extracted_list = []
    distances = []
    relative_distances = []
    html_str = ''
    extracted_sents_in_article_html = ''
    name = FLAGS.dataset + '_' + FLAGS.exp_name
    if FLAGS.coreference_replacement:
        name += '_coref'
    file_name = os.path.join(html_dir, name + '.html')
    extracted_sents_in_article_html_file = open(os.path.join(html_dir, FLAGS.dataset + '_' + FLAGS.exp_name + '_extracted_sents.html'), 'wb')
    if FLAGS.kaiqiang:
        kaiqiang_article_texts = []
        kaiqiang_abstract_texts = []
        util.create_dirs(kaiqiang_dir)
        kaiqiang_article_file = open(os.path.join(kaiqiang_dir, FLAGS.dataset + '_' + FLAGS.dataset_split + '_' + str(FLAGS.min_matched_tokens) + '_articles.txt'), 'wb')
        kaiqiang_abstract_file = open(os.path.join(kaiqiang_dir, FLAGS.dataset + '_' + FLAGS.dataset_split + '_' + str(FLAGS.min_matched_tokens)  + '_abstracts.txt'), 'wb')
    if FLAGS.create_lambdamart_dataset:
        lambdamart_out_dir = os.path.join(lambdamart_dir, FLAGS.dataset)
        lambdamart_out_full_dir = os.path.join(lambdamart_dir, FLAGS.dataset, 'all')
        util.create_dirs(lambdamart_out_dir)
        util.create_dirs(lambdamart_out_full_dir)
        lambdamart_writer = open(os.path.join(lambdamart_out_full_dir, FLAGS.dataset_split + '.bin'), 'wb')

    example_idx = -1
    total = len(source_files)*1000 if 'cnn' or 'newsroom' in FLAGS.dataset else len(source_files)
    for example in tqdm(example_generator, total=total):
        example_idx += 1
        if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
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
            article_sent_tokens = [write_data.process_sent(sent) for sent in raw_article_sents]
        else:
            article_text = util.to_unicode(article_text)
            # sent_pros = {'annotators': 'ssplit', 'outputFormat': 'json', 'timeout': '5000000'}
            # sents_result_dict = nlp.annotate(str(article_text), properties=sent_pros)
            # article_sent_tokens = [[token['word'] for token in sent['tokens']] for sent in sents_result_dict['sentences']]

            raw_article_sents = nltk.tokenize.sent_tokenize(article_text)
            article_sent_tokens = [write_data.process_sent(sent) for sent in raw_article_sents]
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
            doc_indices = example.features.feature['doc_indices'].bytes_list.value[0]
            if '1' in doc_indices:
                doc_indices = [int(x) for x in doc_indices.strip().split()]
                rel_sent_positions = importance_features.get_sent_indices(article_sent_tokens, doc_indices)
            else:
                num_tokens_total = sum([len(sent) for sent in article_sent_tokens])
                rel_sent_positions = [0] * num_tokens_total

        else:
            rel_sent_positions = None
        # summary_sent_tokens = limit_to_n_tokens(summary_sent_tokens, 100)


        similar_source_indices_list = []
        lcs_paths_list = []
        article_lcs_paths_list = []

        for summ_sent in summary_sent_tokens:
            remove_lcs = True
            similarities = get_sent_similarities(summ_sent, article_sent_tokens, vocab)
            if remove_lcs:
                similar_source_indices, lcs_paths, article_lcs_paths = get_similar_source_sents_by_lcs(
                    summ_sent, list(xrange(len(summ_sent))), article_sent_tokens, vocab, similarities, 0)
            else:
                similar_source_indices, start_idx, end_idx = get_similar_source_sents(summ_sent, article_sent_tokens,
                                                                                      vocab, threshold)

            if len(similar_source_indices) == 0:
                continue

            if FLAGS.save_dataset:
                merge_example = get_merge_example(similar_source_indices, article_sent_tokens, summ_sent)
                all_merge_examples.append(merge_example)

            if FLAGS.kaiqiang and len(similar_source_indices) == 1:
                kaiqiang_article, kaiqiang_abstract = get_kaiqiang_article_abstract(similar_source_indices, raw_article_sents, summ_sent)
                # kaiqiang_article_texts.append(kaiqiang_article)
                # kaiqiang_abstract_texts.append(kaiqiang_abstract)
                kaiqiang_article_file.write(kaiqiang_article + '\n')
                kaiqiang_abstract_file.write(kaiqiang_abstract + '\n')

            if FLAGS.print_output:
                if len(similar_source_indices) >= 2:
                    dist = get_shortest_distance(similar_source_indices[0], similar_source_indices[1], False, rel_sent_positions)
                    distances.append(dist)
                    if 'doc_indices' in example.features.feature:
                        rel_dist = get_shortest_distance(similar_source_indices[0], similar_source_indices[1], True, rel_sent_positions)
                        relative_distances.append(rel_dist)

                out_str = ''
                for token_idx, token in enumerate(summ_sent):
                    insert_string = token
                    for source_indices_idx, source_indices in enumerate(similar_source_indices):
                        if source_indices_idx >= len(colors) or source_indices_idx >= FLAGS.sentence_limit:
                            insert_string = token
                            break
                        if token_idx in lcs_paths[source_indices_idx]:
                            insert_string = colors[source_indices_idx] + token + bcolors.ENDC
                            break
                    out_str += insert_string + ' '
                out_str += '\n\n'
                for source_indices_idx, source_indices in enumerate(similar_source_indices):
                    if source_indices_idx >= len(colors) or source_indices_idx >= FLAGS.sentence_limit:
                        break
                    source_sentence = article_sent_tokens[source_indices[0]]
                    for token_idx, token in enumerate(source_sentence):
                        insert_string = token
                        if token_idx in article_lcs_paths[source_indices_idx]:
                            insert_string = colors[source_indices_idx] + token + bcolors.ENDC
                        out_str += insert_string + ' '
                    out_str += '\n\n'
                out_str += '-----------------------------------'
                tqdm.write(out_str)
                a=0

                html_str += html_friendly_fusions(summ_sent, similar_source_indices, lcs_paths, article_lcs_paths, article_sent_tokens)
                a=0


            # if len(similar_source_indices) == 2:
            #     dist = get_shortest_distance(similar_source_indices[0], similar_source_indices[1], False, rel_sent_positions)
            #     distances.append(dist)
            #     rel_dist = get_shortest_distance(similar_source_indices[0], similar_source_indices[1], True, rel_sent_positions)
            #     relative_distances.append(rel_dist)
            #     out_str = ''
            #     if remove_lcs:
            #         for token_idx, token in enumerate(summ_sent):
            #             insert_string = token
            #             if token_idx in lcs_paths:
            #                 insert_string = bcolors.OKBLUE + token + bcolors.ENDC
            #             out_str += insert_string + ' '
            #     else:
            #         for token_idx, token in enumerate(summ_sent):
            #             if token_idx == start_idx:
            #                 out_str += bcolors.OKBLUE
            #             if token_idx == end_idx:
            #                 out_str += bcolors.ENDC
            #             out_str += token + ' '
            #     out_str += '\n'
            #     tqdm.write(out_str)
            #     a=0
            #     # print 'Dist', dist
            num_extracted_list.append(len(similar_source_indices))
            similar_source_indices_list.append(similar_source_indices)
            lcs_paths_list.append(lcs_paths)
            article_lcs_paths_list.append(article_lcs_paths)

        simple_similar_source_indices = [ [s[0] for s in sim_source_ind] for sim_source_ind in similar_source_indices_list]
        if FLAGS.create_lambdamart_dataset:
            write_lambdamart_example(simple_similar_source_indices, raw_article_sents, summary_text, lambdamart_writer)


        if FLAGS.highlight_sents_in_article:
            extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, simple_similar_source_indices,
                                                                              article_sent_tokens, doc_indices,
                                                                              lcs_paths_list, article_lcs_paths_list)
            extracted_sents_in_article_html_file.write(extracted_sents_in_article_html)
        a=0


    if FLAGS.create_lambdamart_dataset:
        lambdamart_writer.close()
        util.chunk_file(FLAGS.dataset_split, lambdamart_out_full_dir, lambdamart_out_dir)

    if FLAGS.save_dataset:
        out_dir = os.path.join(data_dir, FLAGS.dataset + '_merge')
        if FLAGS.coreference_replacement:
            out_dir += '_coref'
        if FLAGS.top_n_sents != -1:
            out_dir += '_n=' + str(FLAGS.top_n_sents)
        write_data.write_with_generator(iter(all_merge_examples), len(all_merge_examples), out_dir, FLAGS.dataset_split)

    if FLAGS.print_output:
        html_str = FLAGS.dataset + ' | ' + FLAGS.exp_name + '<br><br><br>' + html_str
        save_fusions_to_file(html_str)

        num_extracted_2_or_more = np.sum([1 for num_extracted in num_extracted_list if num_extracted >= 2])
        print 'Percentage of summary sentences that are extracted from 2 or more source sentences %.03f' % (num_extracted_2_or_more*1./len(num_extracted_list))
        print 'Average distance between those 2 sentences: %.02f' % (np.mean(distances))
        print 'Average document-level distance between those 2 sentences: %.02f' % (np.mean(relative_distances))

    if FLAGS.kaiqiang:
        # kaiqiang_article_file.write('\n'.join(kaiqiang_article_texts))
        # kaiqiang_abstract_file.write('\n'.join(kaiqiang_abstract_texts))
        kaiqiang_article_file.close()
        kaiqiang_abstract_file.close()
    if FLAGS.highlight_sents_in_article:
        extracted_sents_in_article_html.close()
    a=0


if __name__ == '__main__':
    flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
                               ' If you want to run on human summaries, then enter "reference".')
    flags.DEFINE_string('dataset', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
    flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
    flags.DEFINE_integer('num_instances', -1, 'Number of instances to run for before stopping. Use -1 to run on all instances.')
    flags.DEFINE_string('vocab_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab', 'Path expression to text vocabulary file.')
    flags.DEFINE_boolean('only_rouge_l', False, 'Whether to use only R-L in calculating similarity or whether to average over R-1, R-2, and R-L.')
    flags.DEFINE_boolean('print_output', True, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('save_dataset', False, 'Whether to save the merged sentences as a dataset.')
    flags.DEFINE_boolean('coreference_replacement', False, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('kaiqiang', False, 'Whether to save the single sentences as a dataset for Kaiqiang.')
    flags.DEFINE_boolean('highlight_sents_in_article', False, 'Whether to save an html file that shows the selected sentences as highlighted in the article.')
    flags.DEFINE_boolean('create_lambdamart_dataset', True, 'Whether to save features as a dataset that will be used to predict which sentences should be merged, using the LambdaMART system.')
    flags.DEFINE_integer('top_n_sents', 10, 'Number of sentences to take from the beginning of the article. Use -1 to run on entire article.')
    flags.DEFINE_integer('min_matched_tokens', 2, 'Number of tokens required that still counts a source sentence as matching a summary sentence.')

    app.run(main)

















