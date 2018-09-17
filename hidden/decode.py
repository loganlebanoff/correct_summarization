# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""
import glob
import os
import time

import cPickle
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
from sumy.nlp.tokenizers import Tokenizer
import numpy as np
import itertools
from tqdm import tqdm
import warnings
from absl import flags
from absl import logging
import logging as log

import importance_features

FLAGS = flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60	# max number of seconds before loading new checkpoint
threshold = 0.5
prob_to_keep = 0.33
svm_prob_to_keep = 0.1


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab):
        """Initialize decoder.

        Args:
            model: a Seq2SeqAttentionModel object.
            batcher: a Batcher object.
            vocab: Vocabulary object
        """
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
        self._sess = tf.Session(config=util.get_config())

        # Load an initial checkpoint to use for decoding
        ckpt_path = util.load_ckpt(self._saver, self._sess)

        if FLAGS.single_pass:
            # Make a descriptive decode directory name
            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
            self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
# 			if os.path.exists(self._decode_dir):
# 				raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        else: # Generic decode dir name
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

        if FLAGS.single_pass:
            # Make the dirs to contain output written in the correct format for pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
            self._inspection_dir = os.path.join(self._decode_dir, "human_friendly")
            if not os.path.exists(self._inspection_dir): os.mkdir(self._inspection_dir)


    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        t0 = time.time()
        counter = 0
        while True:
            # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
            # print("Results_dict: ", results_dict)
            batch = self._batcher.next_batch()	# 1 example repeated across batch
            # if counter != 21:
            #     counter += 1
            #     continue
            if batch is None: # finished decoding dataset in single_pass mode
                assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                logging.info("Decoder has finished reading dataset for single_pass.")
                logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
                results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                # print("Results_dict: ", results_dict)
                rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]	# string
            original_abstract = batch.original_abstracts[0]	# string
            original_abstract_sents = batch.original_abstracts_sents[0]	# list of strings
            all_original_abstract_sents = batch.all_original_abstracts_sents[0]
            raw_article_sents = batch.raw_article_sents[0]

            article_withunks = data.show_art_oovs(original_article, self._vocab) # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

            # Only for UPitt
            specific_max_dec_steps = None
            doc_name = None
            if FLAGS.upitt:
                specific_max_dec_steps = int(original_abstract)
                doc_name = all_original_abstract_sents[1][0]

            # Run beam search to get best Hypothesis
            best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch, counter, self._batcher._hps, specific_max_dec_steps=specific_max_dec_steps)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_output = ' '.join(decoded_words) # single string

            if FLAGS.single_pass:
                self.write_for_rouge(all_original_abstract_sents, decoded_words, counter, doc_name=doc_name) # write ref summary and decoded summary to file, to eval with pyrouge later
                # self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool
                self.write_for_inspection(all_original_abstract_sents, decoded_words, raw_article_sents, counter)
                counter += 1 # this is how many examples we've decoded
            else:
                print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
                t1 = time.time()
                if t1-t0 > SECS_UNTIL_NEW_CKPT:
                    logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()

    def write_for_rouge(self, all_reference_sents, decoded_words, ex_index, doc_name=None):
        """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

        Args:
            all_reference_sents: list of list of strings
            decoded_words: list of strings
            ex_index: int, the index with which to label the files
        """
        # First, divide decoded output into sentences
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError: # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx+1:] # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        all_reference_sents = [[make_html_safe(w) for w in abstract] for abstract in all_reference_sents]

        # Write to file
        if doc_name is not None:
            decoded_file = os.path.join(self._rouge_dec_dir, doc_name)      # Only for UPitt
        else:
            decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

        for abs_idx, abs in enumerate(all_reference_sents):
            ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.%s.txt" % (
                ex_index, chr(ord('A') + abs_idx)))
            with open(ref_file, "w") as f:
                for idx,sent in enumerate(abs):
                    f.write(sent+"\n")
                    # f.write(sent) if idx==len(abs)-1 else f.write(sent+"\n")
        with open(decoded_file, "w") as f:
            for idx,sent in enumerate(decoded_sents):
                f.write(sent+"\n")
                # f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

        logging.info("Wrote example %i to file" % ex_index)

    def write_for_inspection(self, all_original_abstract_sents, decoded_words, raw_article_sents, ex_index):
        # First, divide decoded output into sentences
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError: # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx+1:] # everything else
            decoded_sents.append(' '.join(sent))

        out_str = 'Article:\n\n'
        for sent in raw_article_sents:
            out_str += sent + '\n'
        out_str += '\n\nGold Standard Summary:\n\n'
        for abstract in all_original_abstract_sents:
            for sent in abstract:
                out_str += sent + '\n'
            out_str += '\n'
        out_str += '\nSystem Summary:\n\n'
        for sent in decoded_sents:
            out_str += sent + '\n'
        decoded_file = os.path.join(self._inspection_dir, "%06d_decoded.txt" % ex_index)
        with open(decoded_file, "w") as f:
            f.write(out_str)



    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        """Write some data to json file, which can be read into the in-browser attention visualizer tool:
            https://github.com/abisee/attn_vis

        Args:
            article: The original article string.
            abstract: The human (correct) abstract string.
            attn_dists: List of arrays; the attention distributions.
            decoded_words: List of strings; the words of the generated summary.
            p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
        """
        article_lst = article.split() # list of words
        decoded_lst = decoded_words # list of decoded words
        to_write = {
                'article_lst': [make_html_safe(t) for t in article_lst],
                'decoded_lst': [make_html_safe(t) for t in decoded_lst],
                'abstract_str': make_html_safe(abstract),
                'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        logging.info('Wrote visualization data to %s', output_fname)

    def calc_importance_features(self, data_path, hps, model_save_path, docs_desired):
        """Calculate sentence-level features and save as a dataset"""
        data_path_filter_name = os.path.basename(data_path)
        if 'train' in data_path_filter_name:
            data_split = 'train'
        elif 'val' in data_path_filter_name:
            data_split = 'val'
        elif 'test' in data_path_filter_name:
            data_split = 'test'
        else:
            data_split = 'feats'
        if 'cnn-dailymail' in data_path:
            inst_per_file = 1000
        else:
            inst_per_file = 1
        filelist = glob.glob(data_path)
        num_documents_desired = docs_desired
        pbar = tqdm(initial=0, total=num_documents_desired)

        t0 = time.time()
        instances = []
        sentences = []
        counter = 0
        doc_counter = 0
        file_counter = 0
        while True:
            # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
            # print("Results_dict: ", results_dict)
            # print "Batcher num batches", self._batcher._batch_queue.qsize()
            batch = self._batcher.next_batch()	# 1 example repeated across batch
            # if counter != 21:
            #     counter += 1
            #     continue
            if doc_counter >= num_documents_desired:
                # instances = np.stack(instances)
                # instances = instances[:10000]
                # save_path = os.path.join(FLAGS.save_path, data_split + '_%06d'%file_counter)
                # np.savez_compressed(os.path.join(save_path), instances)
                save_path = os.path.join(model_save_path, data_split + '_%06d'%file_counter)
                with open(save_path, 'wb') as f:
                    cPickle.dump(instances, f)
                print('Saved features at %s' % save_path)
                return
                # instances = []
                # counter = 0
                # file_counter += 1

            if batch is None: # finished decoding dataset in single_pass mode
                raise Exception('We havent reached the num docs desired (%d), instead we reached (%d)' % (num_documents_desired, doc_counter))
                # assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                # logging.info("Decoder has finished reading dataset for single_pass.")
                # pbar.close()
                # if len(instances) > 0:
                #     instances = np.stack(instances)
                #     save_path = os.path.join(model_save_path, data_split + '_%06d'%file_counter)
                #     np.savez_compressed(os.path.join(save_path), instances)
                #     tqdm.write('Saved features at %s' % save_path)
                # return


            batch_enc_states, _ = self._model.run_encoder(self._sess, batch)
            for batch_idx, enc_states in enumerate(batch_enc_states):
                art_oovs = batch.art_oovs[batch_idx]
                all_original_abstracts_sents = batch.all_original_abstracts_sents[batch_idx]

                tokenizer = Tokenizer('english')
                # List of lists of words
                enc_sentences, enc_tokens = batch.tokenized_sents[batch_idx], batch.word_ids_sents[batch_idx]
                enc_sent_indices = importance_features.get_sent_indices(enc_sentences, batch.doc_indices[batch_idx])
                enc_sentences_str = [' '.join(sent) for sent in enc_sentences]

                sent_representations_separate = importance_features.get_separate_enc_states(self._model, self._sess, enc_sentences, self._vocab, hps)
                # enc_sentences, enc_tokens, enc_sent_indices = importance_features.get_enc_sents_and_tokens_with_cutoff_length(
                #     batch.enc_batch_extend_vocab[batch_idx], tokenizer, art_oovs, self._vocab, batch.doc_indices[batch_idx],
                #     True, FLAGS.chunk_size)
                # if enc_sentences is None and enc_tokens is None:    # indicates there was a problem getting the sentences
                #     continue

                sent_indices = enc_sent_indices
                sent_reps = importance_features.get_importance_features_for_article(
                    enc_states, enc_sentences, sent_indices, tokenizer, sent_representations_separate, use_cluster_dist=FLAGS.use_cluster_dist)
                y, y_hat = importance_features.get_ROUGE_Ls(art_oovs, all_original_abstracts_sents, self._vocab, enc_tokens)
                binary_y = importance_features.get_best_ROUGE_L_for_each_abs_sent(art_oovs, all_original_abstracts_sents, self._vocab, enc_tokens)
                for rep_idx, rep in enumerate(sent_reps):
                    rep.y = y[rep_idx]
                    rep.binary_y = binary_y[rep_idx]

                for rep_idx, rep in enumerate(sent_reps):
                    # if FLAGS.use_cluster_dist:
                    #     cluster_rep_i = [cluster_rep[i]]
                    # else:
                    #     cluster_rep_i = cluster_rep
                    # Keep all sentences with importance above threshold. All others will be kept with a probability of prob_to_keep
                    if FLAGS.importance_fn == 'svr':
                        if FLAGS.no_balancing or rep.y >= threshold or np.random.random() <= prob_to_keep:
                            # inst = np.concatenate([[abs_sent_indices[i]], [rel_sent_indices[i]], [sent_lens[i]],
                            #                        [lexrank_score[i]], sent_reps[i], cluster_rep_i, [y[i]]])
                            instances.append(rep)
                            sentences.append(sentences)
                    elif FLAGS.importance_fn == 'svm':
                        if FLAGS.svm_no_balancing or rep.binary_y >= threshold or np.random.random() <= svm_prob_to_keep:
                            # inst = np.concatenate([[abs_sent_indices[i]], [rel_sent_indices[i]], [sent_lens[i]],
                            #                        [lexrank_score[i]], sent_reps[i], cluster_rep_i, [y[i]]])
                            instances.append(rep)
                            sentences.append(sentences)
                # for inst in x_y
                # self.write_for_rouge
                # print 'Example %d features processed' % counter
                        counter += 1 # this is how many examples we've decoded
            doc_counter += len(batch_enc_states)
            pbar.update(len(batch_enc_states))




def print_results(article, abstract, decoded_output):
    """Prints the article, the reference summmary and the decoded summary to screen"""
    print ""
    logging.info('ARTICLE:	%s', article)
    logging.info('REFERENCE SUMMARY: %s', abstract)
    logging.info('GENERATED SUMMARY: %s', decoded_output)
    print ""


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
#   r.model_filename_pattern = '#ID#_reference.txt'
    r.model_filename_pattern = '#ID#_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    log.getLogger('global').setLevel(log.WARNING) # silence pyrouge logging
    rouge_args = ['-e', '/home/logan/ROUGE/RELEASE-1.5.5/data',
         '-c',
         '95',
         '-2', '4',        # This is the only one we changed (changed the max skip from -1 to 4)
         '-U',
         '-r', '1000',
         '-n', '4',
         '-w', '1.2',
         '-a',
         '-l', '100']
    rouge_args = ' '.join(rouge_args)
    rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1","2","l","s4","su4"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    logging.info(log_str) # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)

    print "\nROUGE-1, ROUGE-2, ROUGE-SU4 (PRF):\n"
    sheets_str = ""
    for x in ["1", "2", "su4"]:
        for y in ["precision", "recall", "f_score"]:
            key = "rouge_%s_%s" % (x, y)
            val = results_dict[key]
            sheets_str += "%.4f\t" % (val)
    sheets_str += "\n"
    print sheets_str
    sheets_results_file = os.path.join(dir_to_write, "sheets_results.txt")
    with open(sheets_results_file, "w") as f:
        f.write(sheets_str)


def get_decode_dir_name(ckpt_name):
    """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

    if "train" in FLAGS.data_path: dataset = "train"
    elif "val" in FLAGS.data_path: dataset = "val"
    elif "test" in FLAGS.data_path: dataset = "test"
    else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
