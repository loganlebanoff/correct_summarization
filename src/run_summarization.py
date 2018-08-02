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

"""This is the top-level file to train, evaluate or test your summarization model"""

import os
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
import sys
import time
import gpu_util
best_gpu = str(gpu_util.pick_gpu_lowest_memory())
if best_gpu != 'None':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_util.pick_gpu_lowest_memory())
    calc_features_batch_size = 100
else:
    calc_features_batch_size = 10
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
from tqdm import trange
import write_data
import importance_features
import run_importance
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill
from absl import app
from absl import flags
from absl import logging
import random

random.seed(222)
FLAGS = flags.FLAGS

# # Where to find data
# flags.DEFINE_string('data_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/chunked/train_*', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
# flags.DEFINE_string('vocab_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab', 'Path expression to text vocabulary file.')
# 
# # Important settings
# flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
# flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
# 
# # Where to save output
# flags.DEFINE_string('log_root', '/home/logan/data/multidoc_summarization/logs', 'Root directory for all logging.')
# flags.DEFINE_string('exp_name', 'myexperiment', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Where to find data
flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
flags.DEFINE_string('mode', '', 'must be one of train/eval/decode/calc_features')
flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
flags.DEFINE_string('actual_log_root', '', 'Root directory for all logging.')
flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
flags.DEFINE_integer('batch_size', 16, 'minibatch size')
flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
flags.DEFINE_float('lr', 0.15, 'learning rate')
flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

# Pointer-generator or sentence coverage model
flags.DEFINE_boolean('logan_coverage', False, 'If True, use logan\'s coverage to weight sentences.')

# Pointer-generator or sentence importance model
flags.DEFINE_boolean('logan_importance', False, 'If True, use logan\'s importance to weight sentences.')
flags.DEFINE_boolean('logan_beta', False, 'Set to true if using logan_coverage or logan_importance.')
flags.DEFINE_float('logan_coverage_tau', 1.0, 'Tau factor to skew the coverage distribution. Set to 1.0 to turn off.')
flags.DEFINE_float('logan_importance_tau', 1.0, 'Tau factor to skew the importance distribution. Set to 1.0 to turn off.')
flags.DEFINE_float('logan_beta_tau', 1.0, 'Tau factor to skew the combined beta distribution. Set to 1.0 to turn off.')
flags.DEFINE_integer('chunk_size', -1, 'How large the sentence chunks should be. Set to -1 to turn off.')
flags.DEFINE_integer('num_iterations', 60000, 'How many iterations to run. Set to -1 to run indefinitely.')
flags.DEFINE_boolean('coverage_optimization', True, 'If true, only recalculates coverage when necessary.')
flags.DEFINE_boolean('logan_reservoir', False, 'If true, use the paradigm of importance being a reservoir that keeps\
                            being reduced by the similarity to the summary sentences.')
flags.DEFINE_integer('mute_k', -1, 'Pick top k sentences to select and mute all others. Set to -1 to turn off.')
flags.DEFINE_boolean('save_distributions', False, 'If true, save plots of each distribution.')
flags.DEFINE_string('similarity_fn', 'rouge_l', 'Which similarity function to use when calculating\
                            sentence similarity or coverage. Must be one of {rouge_l, tokenwise_sentence_similarity\
                            , ngram_similarity, cosine_similarity')
flags.DEFINE_boolean('always_squash', False, 'Only used if using logan_reservoir. If true, then squash every time beta is recalculated.')
flags.DEFINE_boolean('retain_beta_values', False, 'Only used if using mute mode. If true, then the beta being\
                                                         multiplied by alpha will not be a 0/1 mask, but instead keeps their values.')
flags.DEFINE_boolean('dont_renormalize', False, 'Dont renormalize the alpha values after multiplying by beta')
flags.DEFINE_string('save_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/svr', 'Path expression to save importances features.')
flags.DEFINE_integer('svr_num_documents', 10000, 'How many iterations to run. Set to -1 to run indefinitely.')
flags.DEFINE_boolean('no_balancing', True, 'Only if in calc_features mode. If False, then perform balancing based \
                                                   on how many sentences have R-L greater than 0.5.')
flags.DEFINE_string('importance_model_name', 'svr', 'Name of importance prediction model, which is used to find the model file.')
flags.DEFINE_string('importance_fn', 'svr', 'Which model to use for calculating importance. Must be one of {svr, lex_rank, tfidf, oracle}.')
flags.DEFINE_boolean('use_cluster_dist', False, 'Only if in calc_features mode. If True, then use the cluster distance as the cluster representation')
flags.DEFINE_string('sent_vec_feature_method', 'separate', 'Which method to use for calculating the sentence vector feature. Must be one of {fw_bw, average, separate}')
flags.DEFINE_boolean('normalize_features', False, 'If True, then normalize the simple features (sent_len, sent_position) to be between [0,1].')
flags.DEFINE_boolean('lexrank_as_feature', False, 'If True, then include lexrank as a feature when predicting importance using SVR.')
flags.DEFINE_boolean('subtract_from_original_importance', True, 'If True, then dont keep a running importance value. Instead, substract similarity from the original importance for each sentence.')
flags.DEFINE_boolean('rouge_l_prec_rec', True, 'If True, then dont use F-score. Instead, use precision for calculating similarity, and recall for calculating groundtruth importance.')
flags.DEFINE_boolean('train_on_val', True, 'If True, then train SVR on validation set.')
flags.DEFINE_boolean('both_cnn_dm', False, 'If True, then train SVR on both CNN and DailyMail, rather than just CNN.')
flags.DEFINE_string('dataset_name', 'tac_2011', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm}')
flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val, test}')
flags.DEFINE_string('data_root', '/home/logan/data/multidoc_summarization/tf_examples', 'Path to root directory for all datasets.')
flags.DEFINE_float('lambda_val', 0.5, 'Lambda factor to reduce similarity amount to subtract from importance. Set to 0.5 to make importance and similarity have equal weight.')
flags.DEFINE_string('svm_model_name', 'svm_10000', 'Name of importance prediction model, which is used to find the model file.')
flags.DEFINE_string('svm_save_path', '/home/logan/data/multidoc_summarization/cnn-dailymail/svm_data_10000', 'Path expression to save importances features.')
flags.DEFINE_boolean('svm_no_balancing', False, 'Only if in calc_features mode. If False, then perform balancing based \
                                                   on how many sentences have R-L greater than 0.5.')
flags.DEFINE_boolean('randomize_sent_order', False, 'If True, then randomize the sentence order when loading the TF examples.')
flags.DEFINE_string('query', 'family', 'If True, then randomize the sentence order when loading the TF examples.')


# If use a pretrained model
flags.DEFINE_boolean('use_pretrained', True, 'If True, use pretrained model in the path FLAGS.pretrained_path.')
flags.DEFINE_string('pretrained_path', '/home/logan/data/multidoc_summarization/logs/pretrained_model/train', 'Root directory for all logging.')

# Pointer-generator or baseline model
flags.DEFINE_boolean('upitt', False, 'Set to true if working on UPitt data.')



def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        loss: loss on the most recent eval step
        running_avg_loss: running_avg_loss so far
        summary_writer: FileWriter object to write for tensorboard
        step: training iteration step
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:	# on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)	# clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print "Initializing all variables..."
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print "Restoring all non-adagrad variables from best model in eval dir..."
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print "Restored %s." % curr_ckpt

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    if FLAGS.use_pretrained:
        new_fname = os.path.join(FLAGS.pretrained_path, new_model_name)
    else:
        new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print "Saving model to %s..." % (new_fname)
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print "Saved."
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print "initializing everything..."
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print "restoring non-coverage variables..."
    curr_ckpt = util.load_ckpt(saver, sess)
    print "restored."

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print "saving model to %s..." % (new_fname)
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print "saved."
    exit()

def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph() # build the graph
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(logdir=train_dir,
                                         is_chief=True,
                                         saver=saver,
                                         summary_op=None,
                                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                                         save_model_secs=60, # checkpoint every 60 secs
                                         global_step=model.global_step)
    summary_writer = sv.summary_writer
    logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    logging.info("Created session.")
    try:
        run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    logging.info("starting run_training")
    with sess_context_manager as sess:
        if FLAGS.debug: # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        if FLAGS.num_iterations == -1:
            while True: # repeats until interrupted
                run_training_iteration(model, batcher, summary_writer, sess)
        else:
            initial_iter = model.global_step.eval(sess)
            pbar = tqdm(initial=initial_iter, total=FLAGS.num_iterations)
            print("Starting at iteration %d" % initial_iter)
            for iter_idx in range(initial_iter, FLAGS.num_iterations):
                run_training_iteration(model, batcher, summary_writer, sess)
                pbar.update(1)
            pbar.close()

def run_training_iteration(model, batcher, summary_writer, sess):
    batch = batcher.next_batch()

    # tqdm.write('running training step...')
    t0=time.time()
    results = model.run_train_step(sess, batch)
    t1=time.time()
    # tqdm.write('seconds for training step: %.3f' % (t1-t0))

    loss = results['loss']
    tqdm.write('loss: %f' % loss) # print the loss to screen

    if not np.isfinite(loss):
        raise util.InfinityValueError("Loss is not finite. Stopping.")

    if FLAGS.coverage:
        coverage_loss = results['coverage_loss']
        tqdm.write("coverage_loss: %f" % coverage_loss) # print the coverage loss to screen

    # get the summaries and iteration number so we can write summaries to tensorboard
    summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
    train_step = results['global_step'] # we need this to update our running average loss

    summary_writer.add_summary(summaries, train_step) # write the summaries
    if train_step % 100 == 0: # flush the summary writer every so often
        summary_writer.flush()

def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None	# will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess) # load a new checkpoint
        batch = batcher.next_batch() # get the next batch

        # run eval on the batch
        t0=time.time()
        results = model.run_eval_step(sess, batch)
        t1=time.time()
        logging.info('seconds for batch: %.2f', t1-t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        logging.info('loss: %f', loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()

def calc_features(cnn_dm_train_data_path, hps, vocab, batcher, save_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    decode_model_hps = hps  # This will be the hyperparameters for the decoder model
    model = SummarizationModel(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    decoder.calc_importance_features(cnn_dm_train_data_path, hps, save_path, FLAGS.svr_num_documents)



def main(unused_argv):
    start_time = time.time()
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    if FLAGS.logan_coverage and FLAGS.logan_reservoir:
        raise Exception("Logan's coverage and reservoir options cannot be used simultaneously. Please pick one or neither.")
    if FLAGS.logan_reservoir:
        FLAGS.logan_importance = True
        FLAGS.logan_beta = True
    if FLAGS.dataset_name != "":
        FLAGS.data_path = os.path.join(FLAGS.data_root, FLAGS.dataset_name, FLAGS.dataset_split + '*')
    if not os.path.exists(os.path.join(FLAGS.data_root, FLAGS.dataset_name)) or len(os.listdir(os.path.join(FLAGS.data_root, FLAGS.dataset_name))) == 0:
        print('No TF example data found at %s so creating it from raw data.' % os.path.join(FLAGS.data_root, FLAGS.dataset_name))
        write_data.process_dataset(FLAGS.dataset_name)

    logging.set_verbosity(logging.INFO) # choose what level of logging you want
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.actual_log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode=="train":
            os.makedirs(FLAGS.log_root)
        else:
            if not FLAGS.use_pretrained:
                raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size
    # if FLAGS.mode == 'calc_features':
    #     FLAGS.batch_size = 100

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and (FLAGS.mode!='decode' and FLAGS.mode!='calc_features'):
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
                   'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps',
                   'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'randomize_sent_order']
    hps_dict = {}
    for key,val in FLAGS.__flags.iteritems(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val.value # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    if FLAGS.logan_reservoir:

        if FLAGS.importance_fn == 'tfidf':
            tfidf_model_path = os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer', FLAGS.dataset_name + '.dill')
            if not os.path.exists(tfidf_model_path):
                print('No TFIDF vectorizer model file found at %s, so fitting the model now.' % tfidf_model_path)

                if not os.path.exists(os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer')):
                    os.makedirs(os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer'))

                decode_model_hps = hps	# This will be the hyperparameters for the decoder model
                decode_model_hps = hps._replace(max_dec_steps=1, batch_size=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries

                batcher = Batcher(FLAGS.data_path, vocab, decode_model_hps, single_pass=FLAGS.single_pass)
                all_sentences = []
                while True:
                    batch = batcher.next_batch()	# 1 example repeated across batch
                    if batch is None: # finished decoding dataset in single_pass mode
                        break
                    all_sentences.extend(batch.raw_article_sents[0])

                sent_term_matrix = util.get_tfidf_matrix(all_sentences)
                with open(tfidf_model_path, 'wb') as f:
                    dill.dump(util.tfidf_vectorizer, f)     # WARNING: This may cause problems now, not sure. Because tfidf_vectorizer is part of the util module now.


        if FLAGS.importance_fn == 'svr' or FLAGS.importance_fn == 'svm':
            if FLAGS.importance_fn == 'svr':
                save_path = FLAGS.save_path
                importance_model_name = FLAGS.importance_model_name
            else:
                save_path = FLAGS.svm_save_path
                importance_model_name = FLAGS.svm_model_name
            if FLAGS.both_cnn_dm:
                save_path = save_path + '_both'
                importance_model_name = importance_model_name + '_both'
            else:
                save_path = save_path + '_' + str(FLAGS.svr_num_documents)
                importance_model_name = importance_model_name + '_' + str(FLAGS.svr_num_documents)

            dataset_split = 'val' if FLAGS.train_on_val else 'train'
            if not os.path.exists(save_path) or len(os.listdir(save_path)) == 0:
                print('No importance_feature instances found at %s so creating it from raw data.' % save_path)
                decode_model_hps = hps._replace(
                    max_dec_steps=1, batch_size=calc_features_batch_size, mode='calc_features')  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
                if FLAGS.both_cnn_dm:
                    cnn_dm_train_data_path = os.path.join(FLAGS.data_root, 'cnn_500_dm_500', dataset_split + '*')
                else:
                    cnn_dm_train_data_path = os.path.join(FLAGS.data_root, 'cnn_dm', dataset_split + '*')
                batcher = Batcher(cnn_dm_train_data_path, vocab, decode_model_hps, single_pass=FLAGS.single_pass, cnn_500_dm_500=FLAGS.both_cnn_dm)
                calc_features(cnn_dm_train_data_path, decode_model_hps, vocab, batcher, save_path)
            importance_model_path = os.path.join(FLAGS.actual_log_root, importance_model_name + '.pickle')
            if not os.path.exists(importance_model_path):
                print('No importance_feature SVR model found at %s so training it now.' % importance_model_path)
                features_list = importance_features.get_features_list(True)
                sent_reps = run_importance.load_data(os.path.join(save_path, dataset_split + '*'), FLAGS.svr_num_documents)
                print 'Loaded %d sentences representations' % len(sent_reps)
                x_y = importance_features.features_to_array(sent_reps, features_list)
                train_x, train_y = x_y[:,:-1], x_y[:,-1]
                svr_model = run_importance.run_training(train_x, train_y)
                with open(importance_model_path, 'wb') as f:
                    cPickle.dump(svr_model, f)



    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111) # a seed value for randomness

    if hps.mode == 'train':
        print "creating model..."
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps	# This will be the hyperparameters for the decoder model
        decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)




        # import struct
        # from tensorflow.core.example import example_pb2
        # from gensim.models import KeyedVectors
        #
        # embedding_file = '/home/logan/data/multidoc_summarization/GoogleNews-vectors-negative300.bin'
        # input_file = '/home/logan/data/multidoc_summarization/TAC_Data/logans_test/test_001.bin'
        #
        # # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
        # model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        #
        # dog = model['dog']
        # print(dog.shape)
        # print(dog[:10])
        #
        # reader = open(input_file, 'rb')
        # while True:
        #     len_bytes = reader.read(8)
        #     if not len_bytes: break  # finished reading this file
        #     str_len = struct.unpack('q', len_bytes)[0]
        #     example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        #     example = example_pb2.Example.FromString(example_str)







        decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
    elif hps.mode == 'calc_features':
        decode_model_hps = hps._replace(
            max_dec_steps=1, batch_size=calc_features_batch_size)
        calc_features(FLAGS.save_path, decode_model_hps, vocab, batcher)

    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Finished at: ", localtime)
    time_taken = time.time() - start_time
    if time_taken < 60:
        print('Execution time: ', time_taken, ' sec')
    elif time_taken < 3600:
        print('Execution time: ', time_taken/60., ' min')
    else:
        print('Execution time: ', time_taken/3600., ' hr')

if __name__ == '__main__':
    try:
        app.run(main)
    except util.InfinityValueError as e:
        sys.exit(100)
    except:
        raise
