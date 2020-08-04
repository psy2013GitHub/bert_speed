


def convert_single_instance(words, max_seq_length, tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    tokens = ["[CLS]", ]
    for i, word in enumerate(words):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)

    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志

    tokens.append("[SEP]")

    segment_ids = [0, ] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1, ] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用

    n_tokens = len(input_ids)
    if n_tokens < max_seq_length:
        gap = max_seq_length - n_tokens
        input_ids.extend([0,] * gap)
        input_mask.extend([0, ] * gap)
        segment_ids.extend([0, ] * gap)
        # we don't concerned about it!

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, n_tokens


if __name__ == '__main__':
    pass