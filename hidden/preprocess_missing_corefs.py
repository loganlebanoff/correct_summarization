import os
import hashlib
import util
from tqdm import tqdm
import json

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9001')

def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]

data_path = '/home/logan/data'
processed_dir = '/home/logan/data/corenlp_corefs/processed/cnn_dm'


cnn_tokenized_stories_dir = '/home/logan/data/cnn_stories_tokenized'
dm_tokenized_stories_dir = '/home/logan/data/dm_stories_tokenized'
cnn_out_dir = '/home/logan/data/corenlp_corefs/cnn'
dm_out_dir = '/home/logan/data/corenlp_corefs/dm'

all_train_urls = "/home/logan/data/url_lists/all_train.txt"
all_val_urls = "/home/logan/data/url_lists/all_val.txt"
all_test_urls = "/home/logan/data/url_lists/all_test.txt"

missing_dir = '/home/logan/data/corenlp_lists/missing'
missing_train_urls = "/home/logan/data/corenlp_lists/missing/all_train.txt"
missing_val_urls = "/home/logan/data/corenlp_lists/missing/all_val.txt"
missing_test_urls = "/home/logan/data/corenlp_lists/missing/all_test.txt"

dataset_splits = ['train', 'val', 'test']

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506


def fix_single_quote(article):
    tokens = article.split(' ')
    new_tokens = []
    is_start_quote = True
    for token in tokens:
        if token == "'":
            if is_start_quote:
                to_add = "``"
            else:
                to_add = "''"
        else:
            to_add = token
        new_tokens.append(to_add)
    return ' '.join(new_tokens).strip()

for dataset_split in dataset_splits:

    url_file = os.path.join(data_path, 'url_lists', 'all_' + dataset_split + '.txt')
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s + ".story" for s in url_hashes]

    missing = []

    for idx, s in enumerate(tqdm(story_fnames)):
        name = s.split('.')[0]

        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
            out_dir = os.path.join(cnn_out_dir, name + '.article')
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
            story_file = os.path.join(dm_tokenized_stories_dir, s)
            out_dir = os.path.join(dm_out_dir, name + '.article')
        else:
            print "Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (
            s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
            # Check again if tokenized stories directories contain correct number of files
            print "Checking that the tokenized stories directories %s and %s contain correct number of files..." % (
            cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
            check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
            check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
            raise Exception(
                "Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (
                cnn_tokenized_stories_dir, dm_tokenized_stories_dir, s))

        if not os.path.exists(os.path.join(processed_dir, dataset_split, name + '.article.json')):
            missing.append(out_dir)


    for missing_file in missing:
        with open(missing_file) as f:
            article = f.read()
        name = os.path.basename(missing_file).split('.')[0]
        # print article + '\n\n'
        # fixed_article = fix_single_quote(article)
        # print fixed_article + '\n\n'

        processed_path = os.path.join(processed_dir, dataset_split, name + '.article.json')

        pros = {'annotators': 'coref', 'outputFormat': 'json', 'timeout': '5000000'}
        ann = nlp.annotate(article, pros)
        # json_text = json.dumps(ann)
        with open(processed_path, 'wb') as f:
            json.dump(ann, f, indent=2)


        # with open(missing_file, 'wb') as f:
        #     f.write(fixed_article)


    util.create_dirs(missing_dir)
    with open(os.path.join(missing_dir, 'all_' + dataset_split + '.txt'), 'w') as f:
        f.write('\n'.join(missing))

