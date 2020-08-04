

import os
import sys
import json
from pathlib import Path
import tensorflow as tf
import functools
from collections import Counter
import numpy as np
from data.bert_formatter.classify import convert_single_instance

MINCOUNT = 1

def parse_line(s):
    content = json.loads(s.strip())
    sentence = content['sentence']
    kwords = content['keywords']
    label = content.get('label_desc', 'UNK')
    return sentence, kwords, label

def parse_fn(line, encode=True, with_char=False, bert_out=False, bert_proj_path=None, bert_config_json=None, max_seq_len=512, extend=False):
    # Encode in Bytes for TF
    sentence, kwords, label = parse_line(line)
    sentence_words = []
    if extend:
        sentence_words.append('<SOS>')

    sentence_words.extend([w.encode() if encode and not bert_out else w for w in sentence])

    if extend:
        sentence_words.append('<SOS>')

    n_sentence_words = len(sentence_words)

    words = sentence_words, n_sentence_words

    if bert_out:
        assert bert_proj_path, 'bert_proj_path must not be None'
        sys.path.append(os.path.expanduser(bert_proj_path))
        from bert.tokenization import FullTokenizer
        tokenizer = FullTokenizer(vocab_file=bert_config_json['vocab_file'], do_lower_case=bert_config_json['do_lower_case'])
        sentence_word_ids, input_mask, segment_ids, n_words = convert_single_instance(sentence_words, max_seq_len, tokenizer)
        words = sentence_word_ids, input_mask, segment_ids

    label = label.encode() if encode else label
    if not with_char:
        return words, label
    else:
        # Chars
        # lengths = [len(c) for c in chars]
        # max_len = max(lengths)
        # chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
        raise NotImplementedError('with_char=True error')
        return ((words, n_words), (chars, lengths)), label

def generator_fn(fname, encode=True, with_char=False, bert_out=False, bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    with Path(fname).expanduser().open('r') as fid:
        for line in fid:
            _, label = parse_fn(line, encode=encode, with_char=with_char,
                               bert_out=bert_out, bert_proj_path=bert_proj_path, bert_config_json=bert_config_json, max_seq_len=max_seq_len)
            yield _, label

def input_fn(file, params=None, shuffle_and_repeat=False, with_char=False, bert_out=False,
             bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    params = params if params is not None else {}
    if bert_out:
        if not with_char:
            shapes = (([None], [None], [None]), ())
            types = ((tf.int32, tf.int32, tf.int32), tf.string)
            defaults = ((0, 0, 0), 'UNK')
        else:
            shapes = (
                (
                    (([None], [None], [None]), ()),
                    ([None, None], [None])
                ),  # (chars, nchars)
                [None]
            )  # tags
            types = (
                (
                    ((tf.int32, tf.int32, tf.int32), tf.int32),
                    (tf.string, tf.int32)
                ),
                tf.string
            )
            defaults = (
                (
                    ((0, 0, 0), 0),
                    ('<pad>', 0)
                ),
                'O'
            )
            import tensorflow.python.keras.backend as K
            K.int_shape()
    else:
        if not with_char:
            shapes = (([None], ()), ([None], ()), ()), ()
            types = ((tf.string, tf.int32), (tf.string, tf.int32), tf.string), tf.string
            defaults = (('<pad>', 0), ('<pad>', 0), 'empty'), 'neutral'
        else:
            shapes = (
                (
                    ([None], ()),  # (words, nwords)
                    ([None, None], [None])
                ),  # (chars, nchars)
                [None]
            )  # tags
            types = (
                (
                    (tf.string, tf.int32),
                    (tf.string, tf.int32)
                ),
                tf.string
            )
            defaults = (
                (
                    ('<pad>', 0),
                    ('<pad>', 0)
                ),
                'O'
            )

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, file, with_char=with_char, bert_out=bert_out,
                          bert_proj_path=bert_proj_path, bert_config_json=bert_config_json, max_seq_len=max_seq_len),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset

def build_labels(files, output_dir, min_count=MINCOUNT, force_build=False, encode=False, top_freq_words=None):
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save

    labels_path = '{}/vocab.labels.txt'.format(output_dir)

    if not force_build:
        if Path(labels_path).expanduser().exists():
            print('vocab already build, pass. {}'.format(labels_path))
            return labels_path

    print('Build vocab words/tags (may take a while)')
    counter_words = Counter()
    vocab_labels = set()
    for file in files:
        for _ in generator_fn(file, encode=encode):
            _, label = _
            vocab_labels.add(label)


    with Path(labels_path).expanduser().open('w') as f:
        for t in sorted(list(vocab_labels)):
            f.write('{}\n'.format(t))
    print('- done. Found {} labels.'.format(len(vocab_labels)))

    return labels_path


def build_glove(words_file='vocab.words.txt', output_path='glove.npz', glove_path='glove.840B.300d.txt', force_build=False):

    if not force_build:
        if Path(output_path).expanduser().exists():
            print('glove already build, pass. {}'.format(output_path))
            return output_path

    with Path(words_file).expanduser().open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.random.randn(size_vocab, 300) * 0.01

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path(glove_path).expanduser().open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(str(Path(output_path).expanduser()), embeddings=embeddings)

    return output_path

def serving_input_receiver_fn(): # https://www.jianshu.com/p/7662a939d68e
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
    receiver_tensors = {'word_ids': word_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
    features = {'word_ids': word_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)