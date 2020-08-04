
import os
import logging
import json
import sys
import functools
from pathlib import Path
import shutil
import tensorflow as tf
from bert_theseus.optimizer import create_optimizer
import bert_theseus.modeling as theseus_modeling
from data.CLUE.tnews_input import input_fn, generator_fn, build_labels

def init_log(log_path):
    Path(log_path.rsplit('/', 1)[0]).mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

def load_bert(bert_config_file, init_checkpoint, training, input_ids, input_mask, segment_ids):
    bert_config = theseus_modeling.BertConfig.from_json_file(bert_config_file)
    bert_config.update(succ_num_hidden_layers=6, replace_rate_prob=0.5)
    bert_model = theseus_modeling.BertModel(
        config=bert_config,
        is_training=training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        mode='succ_train',
    )
    tvars = tf.trainable_variables()
    # 加载BERT模型
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            theseus_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    return bert_model, bert_config


def model_fun(features, labels, mode, params):

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    with open(params['labels'], 'r') as fid:
        num_classes = 0
        for line in fid:
            num_classes += 1 if line.strip() else 0
    if tf.estimator.ModeKeys.TRAIN == mode or tf.estimator.ModeKeys.EVAL == mode:
        vocab_labels = tf.contrib.lookup.index_table_from_file(params['labels'])
        labels = vocab_labels.lookup(labels)

    sentence_word_ids, input_mask, segment_ids = features

    # Bert Embeddings
    bert_model, bert_config = load_bert(params['bert_config_file'],
        params.get('step1_init_checkpoint', False),
        training,
        sentence_word_ids,
        input_mask,
        segment_ids
    )

    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embeddings = bert_model.get_sequence_output()

    # dropout
    embeddings = tf.layers.dropout(embeddings, rate=0.1, training=training)

    # representation
    embeddings = tf.layers.dense(embeddings, bert_config.hidden_size, activation='relu')

    # pooler
    embeddings = tf.reduce_mean(embeddings, axis=1)

    # classify
    logits = tf.layers.dense(embeddings, num_classes, activation=None)

    if tf.estimator.ModeKeys.TRAIN == mode or tf.estimator.ModeKeys.EVAL == mode:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)


    if tf.estimator.ModeKeys.PREDICT == mode or tf.estimator.ModeKeys.EVAL == mode:
        pred_ids = tf.argmax(logits, axis=1)
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['labels'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'labels': pred_strings
        }


    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_optimizer(loss=loss, learning_rate=params['optimizer']['learning_rate'], trainable_regexps=['succ',])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if tf.estimator.ModeKeys.PREDICT == mode:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if tf.estimator.ModeKeys.EVAL == mode:
        weights = input_mask
        metrics = {
            'acc': tf.metrics.accuracy(labels, pred_ids)
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)


def main():

    # Params
    params = {
        'max_seq_len': 10,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1, # ？
        'epochs': 25,
        'batch_size': 16,
        'buffer': 1500, # ？
        'rand_replace_steps': 2500,
        'force_build_vocab': False,
        'vocab_dir': './',
        'files': [
            '~/.datasets/CLUE/tnews_public/train.json',
            '~/.datasets/CLUE/tnews_public/dev.json',
        ],
        'bert_project_path': '~/Documents/',
        'bert_init_checkpoint': '/data/models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/bert_model.ckpt',
        'step1_init_checkpoint': './results_theseus/model/',
        'bert_config_file': '/data/models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/bert_config.json',
        'bert_config': {
            'vocab_file': '/data/models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/vocab.txt',
            'do_lower_case': False
        },
        'optimizer': {
            'learning_rate': 1e-5,
        },
        'RESULT_PATH': './results_theseus_step2/',
        'DATADIR': '~/.datasets/CLUE/tnews_public/',
        'log_path': 'results_theseus_step2/main.log'
    }

    init_log(params['log_path'])

    if not os.path.exists(params['RESULT_PATH']):
        os.mkdir(params['RESULT_PATH'])

    with Path('{}/params.json'.format(params['RESULT_PATH'])).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fname(name):
        return str(Path('{}/{}.json'.format(params['DATADIR'], name)).expanduser())

    params['labels'] = build_labels(params['files'],
            params['vocab_dir'],
            force_build=params['force_build_vocab']
    )

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fname('train'), params, shuffle_and_repeat=True,
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))
    eval_inpf = functools.partial(input_fn, fname('dev'),
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True,
    )
    session_config.gpu_options.allow_growth = True

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120, session_config=session_config)
    model_path = '{}/model'.format(params['RESULT_PATH'])
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    estimator = tf.estimator.Estimator(model_fun, model_path, cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator, 'loss', 500, min_steps=8000, run_every_secs=600)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('{}/score'.format(params['RESULT_PATH'])).mkdir(parents=True, exist_ok=True)
        with Path('{}/score/{}.preds.txt'.format(params['RESULT_PATH'], name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fname(name),
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))
            golds_gen = generator_fn(fname(name),
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                _, label = golds
                f.write(b' '.join([label, preds['labels']]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)

if __name__ == '__main__':
    main()