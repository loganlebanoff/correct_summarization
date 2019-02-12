import dill
from tqdm import tqdm
from absl import flags
from absl import app
import time
import subprocess
import itertools
import glob
import numpy as np
from . import data
import os
from collections import defaultdict
from . import util
from .preprocess_for_lambdamart_no_flags import get_features, get_single_sent_features, get_pair_sent_features, \
    Lambdamart_Instance, format_to_lambdamart
from scipy import sparse
from .count_merged import html_highlight_sents_in_article, get_simple_source_indices_list
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics


exp_name = 'lambdamart_singles_lr'
dataset = 'cnn_dm_singles_lr'
dataset_split = 'test'
model = 'lr'
importance = True
filter_sentences = True
num_instances = -1
random_seed = 123
max_sent_len_feat = 20
sentence_limit = 2
min_matched_tokens = 2
only_pairs = True

data_dir = 'data/to_lambdamart'
model_dir = 'data/lambdamart_models'

names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'), ('summary_text', 'string')]




def load_data():
    with open(os.path.join(data_dir, dataset, 'train.txt')) as f:
        train_data = np.load(f)
    with open(os.path.join(data_dir, dataset, 'val.txt')) as f:
        val_data = np.load(f)
    with open(os.path.join(data_dir, dataset, 'test.txt')) as f:
        test_data = np.load(f)
    train_x, train_y = train_data[:,:-1], train_data[:,-1]
    val_x, val_y = val_data[:,:-1], val_data[:,-1]
    test_x, test_y = test_data[:,:-1], test_data[:,-1]

    train_x, train_y = util.shuffle(train_x, train_y)
    val_x, val_y = util.shuffle(val_x, val_y)
    return train_x, train_y, val_x, val_y, test_x, test_y


def main(unused_argv):
    print('Running statistics on %s' % exp_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    start_time = time.time()
    np.random.seed(random_seed)

    train_x, train_y, val_x, val_y, test_x, test_y = load_data()

    lr = LogisticRegressionCV()
    lr.fit(train_x, train_y)
    train_acc = lr.score(train_x, train_y)
    print(train_acc)
    test_acc = lr.score(test_x, test_y)
    print(test_acc)
    train_y_pred = lr.predict(train_x)
    y_pred = lr.predict(test_x)

    print('Training eval')
    print(metrics.classification_report(train_y, train_y_pred))

    print('Testing eval')
    print('-----------------------------------------------')
    print(metrics.classification_report(test_y, y_pred))

    with open(os.path.join(model_dir, dataset + '.pkl'), 'wb') as f:
        dill.dump(lr, f)

    util.print_execution_time(start_time)

if __name__ == '__main__':

    app.run(main)





























