import glob
import numpy as np
import data
import os


def read_lambdamart_scores(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data = [[float(x) for x in line.split('\t')] for line in lines]
    data = np.array(data)
    return data

def get_best_lambdamart_scores(data):
    qid_to_inst_id = {}
    for row in data:
        qid, inst_id, score = row
        if qid


def get_qid_source_indices(line):
    items = line.split(' ')
    for item in items:
        if 'qid' in item:
            qid = int(item.split(':')[1])
            break
    comment = line.strip().split('#')
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
        out_dict[(qid,inst_id)] = source_indices
    return out_dict

def read_articles_abstracts(source_dir, dataset_split):
    source_dir = os.path.join(data_dir, dataset)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)

    print 'Creating list'
    ex_list = [ex for ex in example_generator]
    a=0

def get_source_indices_for_qid

def get_source_indices(example_idx, all_queries):
    similar_source_indices = []
    possible_qids = [example_idx*10 + item for item in list(xrange(10))]
    for qid in possible_qids:
        for inst_id in xrange









































