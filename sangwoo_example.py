from tqdm import tqdm
import glob
from data import example_generator    # The module "data" is from Abigail See's code
import json



dataset_split = 'test'
source_dir = '/home/logan/discourse/tf_data/with_coref_and_ssi/cnn_dm'

names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'), ('summary_text', 'string'), ('corefs', 'json')]

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
        return text.split(' ')
    def get_delimited_list_of_lists(name):
        text = get_string(name)
        return [[int(i) for i in (l.split(' ') if l != '' else [])] for l in text.split(';')]
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
            raise Exception('%s is not a feature of TF Example' % name)
        res.append(func[type](name))
    return res



source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

total = len(source_files) * 1000
example_generator = example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)
for example in tqdm(example_generator, total=total):
    raw_article_sents, similar_source_indices_list, summary_text, corefs = unpack_tf_example(example, names_to_types)
    print similar_source_indices_list