
import re
import tensorflow as tf

def create_optimizer(loss, learning_rate, adam_epsilon=1e-9, clip_norm=10, trainable_regexps=None, rand_replace_steps=-1):
    '''
        re.search(r"suc_layer_\d+", v.name)
    :param learning_rate:
    :param loss:
    :param adam_epsilon:
    :param clip_norm:
    :param trainable_regexps:
    :return:
    '''
    global_step = tf.train.get_or_create_global_step()

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        epsilon=adam_epsilon)

    # 变量
    tvars = tf.trainable_variables()
    if trainable_regexps:
        tvars = [v for v in tvars if any([re.search(_, v.name) for _ in trainable_regexps])]

    # 梯度/裁剪
    gvars = tf.gradients(loss, tvars)
    if clip_norm:
        gvars, _ = tf.clip_by_global_norm(gvars, clip_norm=clip_norm)

    # train op
    train_op = optimizer.apply_gradients(zip(gvars, tvars), global_step=global_step)

    # 组合 train_op + new_global_step
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    return train_op