import csv
import jieba

prepreprocess_path = 'D:/prepreprocess.csv'
preprocess_path = 'D:/preprocess.csv'
corpus_paris_path = 'D:/corpus_paris.csv'
voc_index_path = 'D:/voc_index.txt'
costom_dict_path = 'D:/costom_dict.txt'


MAX_LEN = 20
MIN_FREQ = 10
MAX_DICT_LEN = 20000
PAD_token = "</PAD>"
SOS_token = "</SOS>"
EOS_token = "</EOS>"
UNKNOWEN_token = "</SOMETHING>"

def wordCount(MAX_LEN=50):
    try: 
        jieba.load_userdict(costom_dict_path)
    except FileNotFoundError: 
        pass
    voc_index = {}
    with open (prepreprocess_path, 'r',encoding='utf-8') as prepreprocess_data:
        prepreprocess = csv.DictReader(prepreprocess_data, delimiter='\t')
        # 记录词频
        sentences = iter([row['sentence'] for row in prepreprocess])
        for sentence in sentences:
            cutted = jieba.lcut(sentence, cut_all=False)[:MAX_LEN]
            for word in cutted:
                if word not in voc_index:
                    voc_index[word] = 1
                else:
                    voc_index[word] += 1

    #删除小于最小词频的词
    voc_dict_filt = {}
    n = 0
    for voc in voc_index.items():
        if voc[1] > MIN_FREQ and n<MAX_DICT_LEN:
            voc_dict_filt[voc[0]] = voc[1]
            n += 1
            if n%20 == 0: print(len(voc_dict_filt))
        else: 
            pass
    # voc_dict_filt.update({PAD_token:0, UNKNOWEN_token:1, SOS_token:2, EOS_token:3})
    voc_list = sorted(voc_dict_filt.items(), key=lambda voc_dict_filt: voc_dict_filt[1], reverse=True) #按照词频排序
    # voc_list =  [PAD_token, UNKNOWEN_token, SOS_token, EOS_token] + [voc[0] for voc in voc_list]
    voc_list =  [PAD_token, UNKNOWEN_token, SOS_token, EOS_token] + [voc[0] for voc in voc_list]
    index2voc = {index : voc for index, voc in  enumerate(voc_list, 0)}  #得出序列:词语字典
    voc2index = {voc : index for index, voc in index2voc.items()}  #得到词语:序列字典

    # print(index2voc)
    # print(voc2index)
    return index2voc, voc2index

def wordToIndex(sentence, voc2index):
    sequence = []
    cutted = jieba.lcut(sentence, cut_all=False)[:MAX_LEN-1] + [EOS_token]
    for voc in cutted:
        sequence.append(int(voc2index.get(voc, 1)))
    return sequence

def saveWordToIndexLog(index2voc, voc2index):
    # 把句子转换为词语的index
    with open (prepreprocess_path, 'r',encoding='utf-8') as prepreprocess_data:
        prepreprocess = csv.DictReader(prepreprocess_data, delimiter='\t')
        sentences = iter([row['sentence'] for row in prepreprocess])
        sequence = []
        with open (preprocess_path, 'w', encoding='utf-8', newline="") as preprocess_data:
            preprocess = csv.writer(preprocess_data)
            for sentence in sentences:
                #跳过复读
                # try:
                #     if sentence == sentences.__next__(): continue
                #     else: 
                #         sequence = wordToIndex(sentence, index2voc, voc2index)
                # except StopIteration:
                #     pass
                # print(cutted)
                # print(sequence)

                # if sentence == sentences.__next__()[:len(sentence)]: continue
                # else: sequence = wordToIndex(sentence, voc2index)
                sequence = wordToIndex(sentence, voc2index)
                preprocess.writerow(sequence)
                sequence = []

def save(index2voc, voc2index):
    with open (voc_index_path, 'w', encoding='utf-8') as voc_index_save:
        voc_index_save.write(str(index2voc))
        voc_index_save.write(str("\n"))
        voc_index_save.write(str(voc2index))

def load(path):
    '''
    return voc2index, index2voc
    '''
    with open (path, 'r', encoding='utf-8') as voc_index_save:
        index2voc = eval(voc_index_save.readline())
        voc2index = eval(voc_index_save.readline())
        # print(index2voc)
        # print(voc2index)
        return voc2index, index2voc


def pairsGen(preprocess_path, corpus_paris_path):
    '''
    输入单列储存的sequence，转为首尾相连的双列sequence并保存
    例如从[1,2,3,4]变为[1,2],[2,3],[3,4]
    输出文件格式为csv, 分为两排，qes和ans，一问一答
    Args:
        preprocess_path: str, 已经过预处理的语料库的文件路径
        corpus_paris_path: str, 保存处理过的语料库的路径

    Return:
        None
    '''
    with open(preprocess_path, 'r', encoding='utf-8') as preprocess_data:
        preprocess = csv.reader(preprocess_data)
        with open(corpus_paris_path, 'w', encoding='utf-8', newline="") as corpus_paris:
            pairs = csv.writer(corpus_paris, delimiter="\t")
            pairs.writerow(["qes", "ans"])
            left = preprocess.__next__()
            right = []
            if left == right[:len(left)]: pass
            else:
                for seq in preprocess:
                    right = seq
                    pairs.writerow([",".join(left), ",".join(right)])
                    left = right

# index2voc, voc2index = wordCount()
# saveWordToIndexLog(index2voc, voc2index)
# save(index2voc, voc2index)
# pairsGen(preprocess_path, corpus_paris_path)

# load()


# with open (preprocess_path, 'w', encoding='utf-8') as preprocess_data:
#     preprocess = csv.writer(preprocess_data, delimiter='\t')
#     preprocess.writerow(["left", "right"])
#     #把句子从[1,2,3,4]变成[1,2],[2,3],[3,4]
#     left = jieba.lcut(sentence.__next__(), cut_all=False)[:MAX_LEN]
#     right = []
#     for content in sentence:
#         right = jieba.lcut(content, cut_all=False)[:MAX_LEN]
#         #略过复读
#         if (left != right):
#             preprocess.writerow(["/".join(left), "/".join(right)])
#             #print("left = " + "/".join(left) +"right = " + "/".join(right))
#         left = right

