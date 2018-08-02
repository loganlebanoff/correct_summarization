import util
from absl import flags
from absl import app
import sys
import os
import hashlib
import struct
import subprocess
import collections

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "/home/logan/data/url_lists/all_train.txt"
all_val_urls = "/home/logan/data/url_lists/all_val.txt"
all_test_urls = "/home/logan/data/url_lists/all_test.txt"

data_path = '/home/logan/data'


cnn_tokenized_stories_dir = '/home/logan/data/cnn_stories_tokenized'
dm_tokenized_stories_dir = '/home/logan/data/dm_stories_tokenized'
cnn_out_dir = '/home/logan/data/corenlp_corefs/cnn'
dm_out_dir = '/home/logan/data/corenlp_corefs/dm'


# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

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


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  # lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))



def write_to_bin(url_file, cnn_out_dir, dm_out_dir, dataset_split):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print "Making bin file for URLs listed in %s..." % url_file
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)
  corenlp_paths = []

  util.create_dirs(cnn_out_dir)
  util.create_dirs(dm_out_dir)

  for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
          print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))
      name = s.split('.')[0]

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
        story_file = os.path.join(cnn_tokenized_stories_dir, s)
        out_dir = os.path.join(cnn_out_dir, name + '.article')
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_tokenized_stories_dir, s)
        out_dir = os.path.join(dm_out_dir, name + '.article')
      else:
        print "Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        # Check again if tokenized stories directories contain correct number of files
        print "Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
        check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, s))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      with open(out_dir, 'w') as f:
          f.write(article)

      corenlp_paths.append(out_dir)

      a=0
      # # Write to tf.Example
      # tf_example = example_pb2.Example()
      # tf_example.features.feature['article'].bytes_list.value.extend([article])
      # tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      # tf_example_str = tf_example.SerializeToString()
      # str_len = len(tf_example_str)
      # writer.write(struct.pack('q', str_len))
      # writer.write(struct.pack('%ds' % str_len, tf_example_str))


  util.create_dirs(os.path.join(data_path, 'corenlp_lists'))
  with open(os.path.join(data_path, 'corenlp_lists', 'all_' + dataset_split + '.txt'), 'w') as f:
      f.write('\n'.join(corenlp_paths))

  print "Finished writing file %s\n" % url_file


def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    write_to_bin(all_test_urls, cnn_out_dir, dm_out_dir, 'test')
    write_to_bin(all_val_urls, cnn_out_dir, dm_out_dir, 'val')
    write_to_bin(all_train_urls, cnn_out_dir, dm_out_dir, 'train')


if __name__ == '__main__':
    app.run(main)











































