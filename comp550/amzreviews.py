# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import json
import tensorflow as tf

dataset_urls = ["http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz",
                    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz",
                  "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz",
                  "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
                  ]


def _build_vocab(filename, vocab_dir, vocab_name):
    """Reads a file to build a vocabulary.
 
    Args:
      filename: file to read list of words from.
      vocab_dir: directory where to save the vocabulary.
      vocab_name: vocab file name.
 
    Returns:
      text encoder.
    """
    vocab_path = os.path.join(vocab_dir, vocab_name)
    if not tf.gfile.Exists(vocab_path):
        data=[]
        with tf.gfile.GFile(filename, "r") as f:
            for line in f:
                r=json.loads(line)
                data.extend(r["reviewText"].split()) 
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        encoder = text_encoder.TokenTextEncoder(None, vocab_list=words)
        encoder.store_to_file(vocab_path)
    else:
        encoder = text_encoder.TokenTextEncoder(vocab_path)
    return encoder


def _maybe_download_corpus(tmp_dir, vocab_type,dataset_url,dir_name):
    """Download and unpack the corpus.

    Args:
      tmp_dir: directory containing dataset.
      vocab_type: which vocabulary are we using.

    Returns:
      The list of names of files.
    """
#     if vocab_type == text_problems.VocabType.CHARACTER:
# 
#         dataset_url = ("https://s3.amazonaws.com/research.metamind.io/wikitext"
#                        "/wikitext-103-raw-v1.zip")
#         dir_name = "wikitext-103-raw"
#     else:
#         dataset_url = ("https://s3.amazonaws.com/research.metamind.io/wikitext"
#                        "/wikitext-103-v1.zip")
#         dir_name = "wikitext-103"

    fname = os.path.basename(dataset_url)
    compressed_filepath = generator_utils.maybe_download(tmp_dir, fname,
                                                         dataset_url)
    
    unpacked_dir = os.path.join(tmp_dir, dir_name)
    if not tf.gfile.Exists(unpacked_dir):
        tf.gfile.MakeDirs(unpacked_dir)
    unpacked_file=os.path.join(compressed_filepath , unpacked_dir + "/" + os.path.splitext(fname)[0])
    generator_utils.gunzip_file(compressed_filepath , unpacked_file)
    txt=os.path.splitext(unpacked_file)[0]+".txt"
    if not tf.gfile.Exists(txt):
        with open(unpacked_file,"rb") as jf, open(txt,"w") as wf:
            for line in jf:
                wf.write(json.loads(line)["reviewText"]+"\n")
    files = os.path.join(tmp_dir, dir_name, "*.txt")
    train_file, valid_file, test_file = None, None, None
    for f in tf.gfile.Glob(files):
#         fname = os.path.basename(f)
#         if "train" in fname:
            train_file = f
#         elif "valid" in fname:
#             valid_file = f
#         elif "test" in fname:
#             test_file = f

#     assert train_file, "Training file not found"
#     assert valid_file, "Validation file not found"
#     assert test_file, "Testing file not found"

    return train_file  # , valid_file, test_file


@registry.register_problem
class Amzlm(text_problems.Text2SelfProblem):
    """amz dataset token-level."""
    def __init__(self, *args, **kwargs):
        super(Amzlm, self).__init__(*args, **kwargs)
        self.dataset_url = dataset_urls[0]
        self.dir_name = "amzlm_videos"
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 8,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]
    def dataset_filename(self):
        return self.dir_name
    @property
    def is_generate_per_split(self):
        # If we have pre-existing data splits for (train, eval, test) then we set
        # this to True, which will have generate_samples be called for each of the
        # dataset_splits.
        #
        # If we do not have pre-existing data splits, we set this to False, which
        # will have generate_samples be called just once and the Problem will
        # automatically partition the data into dataset_splits.
        return False

    @property
    def vocab_type(self):
#         return text_problems.VocabType.TOKEN
        return text_problems.VocabType.SUBWORD
    @property
    def approx_vocab_size(self):
        return 2**15  # 32768
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del dataset_split
        train_file = _maybe_download_corpus(
            tmp_dir, self.vocab_type,self.dataset_url,self.dir_name)

        filepath = train_file
        if self.vocab_type==text_problems.VocabType.TOKEN:
            _build_vocab(train_file, data_dir, self.vocab_filename)



        def _generate_samples():
            with tf.gfile.GFile(filepath, "r") as f:
                for line in f:
                    line = " ".join(line.strip().split())
                    if line:
#                         yield {"targets": json.loads(line)["reviewText"]}
                        yield {"targets": line.lower()}

        return _generate_samples()

@registry.register_problem
class AmzlmBook(Amzlm):
    def __init__(self, *args, **kwargs):
        super(AmzlmBook, self).__init__(*args, **kwargs)
        self.dataset_url = dataset_urls[1]
        self.dir_name = "amzlm_books"
    def dataset_filename(self):
        return self.dir_name

@registry.register_problem
class AmzlmMovie(Amzlm):
    def __init__(self, *args, **kwargs):
        super(AmzlmMovie, self).__init__(*args, **kwargs)
        self.dataset_url = dataset_urls[2]
        self.dir_name = "amzlm_movies"
    def dataset_filename(self):
        return self.dir_name
@registry.register_problem
class AmzlmElec(Amzlm):
    def __init__(self, *args, **kwargs):
        super(AmzlmElec, self).__init__(*args, **kwargs)
        self.dataset_url = dataset_urls[3]
        self.dir_name = "amzlm_elecs"
    def dataset_filename(self):
        return self.dir_name