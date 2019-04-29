import pickle

import util
import os
import sys
import numpy as np
import json
import glob
from absl import app, flags, logging
import tensorflow as tf
from model import SummarizationModel
from collections import namedtuple
from data import Vocab
import random
from tqdm import tqdm
import nltk
from ssi_functions import html_highlight_sents_in_article, get_simple_source_indices_list


random.seed(222)

FLAGS = flags.FLAGS


random_seed = 123
# attn_dir = 'logs/cnn_dm_pretrained/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/attn_vis_data'
# html_dir = 'logs/cnn_dm_pretrained/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/extr_vis'
gradient_starts = [255, 147]
gradient_ends = [147, 255]
gradient_positions = [1, 0]
original_color = 'ffff93'

def create_gradient(match_indices):
    min_=147
    max_=255
    decimals = np.linspace(min_,max_,len(match_indices))
    hexs = [hex(int(dec))[2:] for dec in decimals]

    '''NEED TO REPLACE THIS PART WITH SOMETHING THAT MAKES THE POSITION DYNAMIC'''
    gradients = ['ff' + h + '93' for h in hexs]
    gradients = [g if match_indices[g_idx] is not None else '' for g_idx, g in enumerate(gradients)]
    return gradients

def create_html(article_lst, match_indices, decoded_lst, abstract_lst, file_idx, ssi, lcs_paths_list, article_lcs_paths_list, summary_sent_tokens, article_sent_tokens):
    colors = create_gradient(match_indices)
    '''<script>document.body.addEventListener("keydown", function (event) {
    if (event.keyCode === 39) {
        window.location.replace("%06d.html");
    }
});</script>'''

    html = '''
    
<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d.html";
    };
    
    document.addEventListener("keyup",function(e){
   var key = e.which||e.keyCode;
   switch(key){
      //left arrow
      case 37:
         document.getElementById("btnPrev").click();
      break;
      //right arrow
      case 39:
         document.getElementById("btnNext").click();
      break;
   }
});
</script>

''' % (file_idx-1, file_idx+1)
    for dec_idx, dec in enumerate(decoded_lst):
        html += '<span style="background-color:#%s;">%s </span>' % (colors[dec_idx], dec)
        if dec == '.' and dec_idx < len(decoded_lst) - 2:
            html += '<br>'

    html += '<br><br>'

    for art_idx, art in enumerate(article_lst):
        if art_idx in match_indices:
            dec_idx = match_indices.index(art_idx)
            color = colors[dec_idx]
        else:
            color = ''
        style = 'style="background-color:#%s;"' % color if color != '' else ''
        html += '<span %s>%s </span>' % (style, art)
        if art == '.' and art_idx < len(article_lst) - 2:
            html += '<br>'

    html += '<br><br><br>'

    extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, ssi,
                                                                      article_sent_tokens,
                                                                      lcs_paths_list=lcs_paths_list, article_lcs_paths_list=article_lcs_paths_list)
    if 'israel' in extracted_sents_in_article_html:
        a=0
    for ch_idx, ch in reversed(list(enumerate(extracted_sents_in_article_html))):
        if ch == '.':
            a=0
        if ch == '.' and (extracted_sents_in_article_html[ch_idx-1] == '>' or extracted_sents_in_article_html[ch_idx-1] == ' ') \
                and extracted_sents_in_article_html[ch_idx+1] == ' ' \
                and not (extracted_sents_in_article_html[ch_idx+2:] == '</mark><br><br>------------------------------------------------------<br><br>'
                    or extracted_sents_in_article_html[ch_idx+2:] == '<br><br>------------------------------------------------------<br><br>'):
            extracted_sents_in_article_html = '%s%s%s' % (extracted_sents_in_article_html[:ch_idx], '. </mark><br>', extracted_sents_in_article_html[ch_idx+9:])
            break
    html += extracted_sents_in_article_html


    return html

def process_attn_selections(attn_dir, decode_dir, vocab, extraction_eval=False):

    html_dir = os.path.join(decode_dir, 'extr_vis')
    util.create_dirs(html_dir)
    file_names = sorted(glob.glob(os.path.join(attn_dir, '*')))

    if extraction_eval:
        ssi_dir = os.path.join('data/ssi', FLAGS.dataset_name, 'test_ssi.pkl')
        with open(ssi_dir) as f:
            ssi_list = pickle.load(f)
        if len(ssi_list) != len(file_names):
            raise Exception('len of ssi_list does not equal len file_names: ', len(ssi_list), len(file_names))
    triplet_ssi_list = []
    for file_idx, file_name in enumerate(tqdm(file_names)):
        with open(file_name) as f:
            data = json.load(f)
        p_gens = util.flatten_list_of_lists(data['p_gens'])
        article_lst = data['article_lst']
        abstract_lst = data['abstract_str'].strip().split()
        decoded_lst = data['decoded_lst']
        attn_dists = np.array(data['attn_dists'])

        article_lst = [art_word.replace('__', '') for art_word in article_lst]
        decoded_lst = [dec_word.replace('__', '') for dec_word in decoded_lst]
        abstract_lst = [abs_word.replace('__', '') for abs_word in abstract_lst]

        min_matched_tokens = 2
        if 'singles' in FLAGS.exp_name:
            sentence_limit = 1
        else:
            sentence_limit = 2
        summary_sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(' '.join(abstract_lst))]
        decoded_sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(' '.join(decoded_lst))]
        article_sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(' '.join(article_lst))]
        gt_ssi_list, lcs_paths_list, article_lcs_paths_list = get_simple_source_indices_list(summary_sent_tokens, article_sent_tokens, vocab, sentence_limit,
                                       min_matched_tokens)
        sys_ssi_list, _, _ = get_simple_source_indices_list(decoded_sent_tokens, article_sent_tokens, vocab, sentence_limit,
                                       min_matched_tokens)


        match_indices = []
        for dec_idx, dec in enumerate(decoded_lst):
            art_match_indices = [art_idx for art_idx, art_word in enumerate(article_lst) if art_word.replace('__', '') == dec or art_word == dec]
            if len(art_match_indices) == 0:
                match_indices.append(None)
            else:
                art_attns = [attn_dists[dec_idx, art_idx] for art_idx in art_match_indices]
                best_match_idx = art_match_indices[np.argmax(art_attns)]
                match_indices.append(best_match_idx)

        html = create_html(article_lst, match_indices, decoded_lst, [abstract_lst], file_idx, gt_ssi_list, lcs_paths_list, article_lcs_paths_list, summary_sent_tokens, article_sent_tokens)
        with open(os.path.join(html_dir, '%06d.html' % file_idx), 'wb') as f:
            f.write(html)

        if extraction_eval:
            triplet_ssi_list.append((ssi_list[file_idx], sys_ssi_list, -1))

    if extraction_eval:
        print('Evaluating Lambdamart model F1 score...')
        suffix = util.all_sent_selection_eval(triplet_ssi_list)
        print(suffix)
        with open(os.path.join(decode_dir, 'extraction_results.txt'), 'wb') as f:
            f.write(suffix)


    a=0

def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags inc
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.dataset_name != "":
        FLAGS.data_path = os.path.join(FLAGS.data_root, FLAGS.dataset_name, FLAGS.dataset_split + '*')

    logging.set_verbosity(logging.INFO) # choose what level of logging you want
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.exp_name = FLAGS.exp_name if FLAGS.exp_name != '' else FLAGS.dataset_name
    FLAGS.actual_log_root = FLAGS.log_root
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    original_dataset_name = 'xsum' if 'xsum' in FLAGS.dataset_name else 'cnn_dm' if 'cnn_dm' in FLAGS.dataset_name or 'duc_2004' in FLAGS.dataset_name else ''
    vocab = Vocab(FLAGS.vocab_path + '_' + original_dataset_name, FLAGS.vocab_size) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
                   'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps',
                   'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'lambdamart_input']
    hps_dict = {}
    for key,val in FLAGS.__flags.items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val.value # add it to the dict
    hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

    tf.set_random_seed(113) # a seed value for randomness

    decode_model_hps = hps._replace(
        max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    np.random.seed(random_seed)
    # batcher = Batcher(None, vocab, hps, single_pass=FLAGS.single_pass)
    # model = SummarizationModel(decode_model_hps, vocab)
    # decoder = BeamSearchDecoder(model, None, vocab)

    # decode_dir = decoder._decode_dir
    ckpt_folder = util.find_largest_ckpt_folder(FLAGS.log_root)
    decode_dir = os.path.join(FLAGS.log_root, ckpt_folder)
    print(decode_dir)
    attn_dir = os.path.join(decode_dir, 'attn_vis_data')

    process_attn_selections(attn_dir, decode_dir, vocab, extraction_eval=True)


if __name__ == '__main__':
    # Where to find data
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Makes a log dir based on name.\
                                                    Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm} or a custom dataset name')
    flags.DEFINE_string('data_root', os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi',
                        'Path to root directory for all datasets (already converted to TensorFlow examples).')
    flags.DEFINE_string('vocab_path', 'logs/vocab', 'Path expression to text vocabulary file.')
    flags.DEFINE_string('pretrained_path', 'logs/cnn_dm_sent',
                        'Directory of pretrained model for PG trained on singles or pairs of sentences.')
    flags.DEFINE_boolean('use_pretrained', True, 'If True, use pretrained model in the path FLAGS.pretrained_path.')

    # Where to save output
    flags.DEFINE_string('log_root', 'logs', 'Root directory for all logging.')
    flags.DEFINE_string('exp_name', 'pg_lambdamart',
                        'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

    # Don't change these settings
    flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
    flags.DEFINE_boolean('single_pass', True,
                         'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
    flags.DEFINE_string('data_path', '',
                        'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
    flags.DEFINE_string('actual_log_root', '',
                        'Dont use this setting, only for internal use. Root directory for all logging.')
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val, test}')

    # Hyperparameters
    flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
    flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
    flags.DEFINE_integer('batch_size', 16, 'minibatch size')
    flags.DEFINE_integer('max_enc_steps', 100, 'max timesteps of encoder (max source text tokens)')
    flags.DEFINE_integer('max_dec_steps', 30, 'max timesteps of decoder (max summary tokens)')
    flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
    flags.DEFINE_integer('min_dec_steps', 10,
                         'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
    flags.DEFINE_integer('vocab_size', 50000,
                         'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
    flags.DEFINE_float('lr', 0.15, 'learning rate')
    flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
    flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
    flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
    flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

    # Pointer-generator or baseline model
    flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

    # Coverage hyperparameters
    flags.DEFINE_boolean('coverage', True,
                         'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
    flags.DEFINE_float('cov_loss_wt', 1.0,
                       'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

    # Utility flags, for restoring and changing checkpoints
    flags.DEFINE_boolean('convert_to_coverage_model', False,
                         'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
    flags.DEFINE_boolean('restore_best_model', False,
                         'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

    # Debugging. See https://www.tensorflow.org/programmers_guide/debugger
    flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

    # PG-MMR settings
    flags.DEFINE_boolean('pg_mmr', False, 'If true, use the PG-MMR model.')
    flags.DEFINE_string('importance_fn', 'tfidf',
                        'Which model to use for calculating importance. Must be one of {svr, tfidf, oracle}.')
    flags.DEFINE_float('lambda_val', 0.6,
                       'Lambda factor to reduce similarity amount to subtract from importance. Set to 0.5 to make importance and similarity have equal weight.')
    flags.DEFINE_integer('mute_k', 7, 'Pick top k sentences to select and mute all others. Set to -1 to turn off.')
    flags.DEFINE_boolean('retain_mmr_values', False, 'Only used if using mute mode. If true, then the mmr being\
                                multiplied by alpha will not be a 0/1 mask, but instead keeps their values.')
    flags.DEFINE_string('similarity_fn', 'rouge_l', 'Which similarity function to use when calculating\
                                sentence similarity or coverage. Must be one of {rouge_l, ngram_similarity}')
    flags.DEFINE_boolean('plot_distributions', False,
                         'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')

    app.run(main)

















