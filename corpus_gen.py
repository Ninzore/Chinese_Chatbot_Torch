import csv
import itertools
import torch

preprocess_path = 'D:/preprocess.csv'
corpus_paris_path = 'D:/corpus_paris.csv'

test_batch_size = 10
PAD_token = 0


def zeroPadding(seq_list, fillvalue=0):
    '''
    输入一个长度为batch_size的sequence list，
    使用zip_longest将其padding，padding长度为max_seq_len
    变为矩阵并转置，矩阵大小为batch_size, max_seq_len，
    目的是保持seq的长度一致，因为模型无法接受变长的sequence

    Example:
        input:              ouput:
        [[1,2,3,4],        [[1,2,3,4],
        [1,2],             [1,2,0,0],
        [1,2,3],           [1,2,3,0],
        [1]],              [1,0,0,0]]

    Args:
        seq_list: list， 内部储存sequence_list
        fillvalue: str, 用作padding的字符串，默认为0

    Return:
        list: 大小为batch_size, max_seq_len的一个矩阵
    '''
    return list(itertools.zip_longest(*seq_list, fillvalue=fillvalue))


def binaryMatrix(pad_seq):
    '''
    输入经过zeropadding后的矩阵pad_seq，输出mask矩阵，大小和pad_seq一致
    其中未pad部分不论大小皆为1，pad为0

    Example：
        输入                    输出
        [[1,2,3,4],             [[1,1,1,1], 
        [1,2,0,0],              [1,1,0,0],
        [1,2,3,0],              [1,1,1,0],
        [1,0,0,0]]              [1,0,0,0]]

    Args:
        pad_seq: list，大小为batch_size, max_seq_len的矩阵

    Return:
        mask: list, 大小为batch_size, max_seq_len的矩阵
    '''
    mask = []
    for index, seq in enumerate(pad_seq):
        mask.append([])
        for token in seq:
            if token == PAD_token:
                mask[index].append(0)
            else:
                mask[index].append(1)
    return mask

# pairsGen(preprocess_path, corpus_paris_path)

# with open(corpus_paris_path, 'r', encoding='utf-8') as corpus_paris:
#     corpus = csv.reader(corpus_paris, delimiter="\t")
#     header = next(corpus)
#     qes = []
#     ans = []
#     for row in corpus:
#         qes.append(list(map(int, row[0].split(","))))
#         ans.append(list(map(int, row[1].split(","))))
#     # print(qes[0])
#     # print(ans[0])

# in_batch = qes[:test_batch_size]
# target_batch = ans[:test_batch_size]


def batchGen(batch):
    '''
    in_batch: 包含所有
    '''
    batch.sort(key=lambda batch: len(batch[0]), reverse=True)
    in_batch, target_batch = zip(*batch)
    
    len_tensor = torch.tensor([len(seq) for seq in in_batch])
    target_len = max([len(seq) for seq in target_batch])

    in_tensor = torch.LongTensor(zeroPadding(in_batch))
    target_tensor = torch.LongTensor(zeroPadding(target_batch))
    mask = torch.BoolTensor(binaryMatrix(zeroPadding(target_batch)))

    return in_tensor, target_tensor, mask, len_tensor, target_len

# import random
# batch_size = 5
# with open(corpus_paris_path, 'r', encoding='utf-8') as corpus_paris:
#     corpus = csv.reader(corpus_paris, delimiter="\t")
#     header = next(corpus)
#     all_in = []
#     all_target = []
#     for row in corpus:
#         all_in.append(list(map(int, row[0].split(","))))
#         all_target.append(list(map(int, row[1].split(","))))

# # 填充batch
# # batch_list: shape [[in_batch_1, target_batch_1], [in_batch_2, target_batch_2],.....]
# batch_list = [[all_in[i], all_target[i]] for i in range(len(all_in))]

# i = 0
# for _ in range(1,5):
#     # 把batch扔给batchGen拿到需要的数据
#     single_batch = []
#     for _ in range(batch_size):
#         single_batch.append(batch_list[i])
#         i += 1
#         if i >= len(batch_list):
#             i = 1
#     # single_batch = [random.choice(batch_list) for _ in range(batch_size)]
#     in_tensor, target_tensor, mask, len_tensor, target_len = batchGen(single_batch)
#     print(in_tensor)
#     print(target_tensor)
#     print(target_len)
#     print(mask)
#     print(mask.sum())