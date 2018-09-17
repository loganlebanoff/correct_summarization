import util
import os

data_dir = '/home/logan/data/multidoc_summarization/tf_examples/newsroom'

file_names = os.listdir(data_dir)
splits = ['test', 'train']

for split in splits:
    test_files = [file for file in file_names if split in file]
    test_files = sorted(test_files)
    first_idx = util.extract_digits(test_files[0])
    difference = first_idx - 1
    for file in test_files:
        file_path = os.path.join(data_dir, file)
        idx = util.extract_digits(file)
        new_name = os.path.join(data_dir, split + '_{:05d}.bin'.format(idx - difference))
        os.rename(file_path, new_name)