import torch
import random
import csv
import preprocess
import rnn_model
import corpus_gen

def maskedLoss(input_seq, target, mask, device):
    '''
    对比解码器的输出和binary mask来计算损失
    损失计算对为binary mask张量中那些1的词语的对应词序求平均值的负对数  
    Args:
        input_seq: tensor 解码器输出
        target: tensor target_tensor
        mask: booltensor 经过binaryMask后的target_seq
    '''
    num_voc = mask.sum()
    cross_entropy = -torch.log(torch.gather(input_seq, 1, target.view(-1, 1)).squeeze(1))
    mask_loss = cross_entropy.masked_select(mask).mean()
    mask_loss = mask_loss.to(device)
    return mask_loss, num_voc.item()


def trainOnce(in_tensor, target_tensor, mask, len_tensor, target_len, batch_size, clip,
              encoder, decoder, encoder_optimizer, decoder_optimizer, device):
    '''
    一次训练  
    Args:
        in_tensor: tensor 经过zeropadding后的input seq
        target_tensor tensor 经过zeropadding后的target seq
        mask: booltensor 经过binaryMask后的target seq
        len_tensor: tensor 表示in_tensor的每列的长度
        target_len: int 表示target seq的长度
        batch_size: 单个batch的大小
        clip: 裁剪率
    '''

    # 清零梯度
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 选择设备
    in_tensor = in_tensor.to(device)
    target_tensor = target_tensor.to(device)
    len_tensor = len_tensor.to(device)
    mask = mask.to(device)

    #初始化变量
    teacher_force_ratio = 1
    loss = 0
    loss_list = []
    n = 0

    # 初始化编码器前向传播
    encoder_out, encoder_hidden = encoder(in_tensor, len_tensor)

    # 初始化解码器
    # 在句子开头加入SOS_token
    decoder_in = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_in = decoder_in.to(device)
    #初始化解码器隐藏层状态 = 编码器最终隐藏层状态
    decoder_hidden = encoder_hidden[:decoder.num_layers]

    # 随机是否使用teacher force
    use_teacher_force = True if random.random() < teacher_force_ratio else False

    # 开始
    for t in range(target_len):
        decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_out)
        if use_teacher_force:
            # 使用teacher force， 让下一个输入为当前的目标输出
            # 使用view进行reshape，自适应形状为[1, batch_size]
            decoder_in = target_tensor[t].view(1, -1)

        else:
            # 不使用teacher force
            # 解码器的下一个输入是当前解码器的输出
            _, topk = decoder_out.topk(1)
            decoder_in = torch.LongTensor([[topk[i][0] for i in range(batch_size)]])
            decoder_in = decoder_in.to(device)

        # 计算损失
        mask_loss, num_voc = maskedLoss(decoder_out, target_tensor[t], mask[t], device)
        loss += mask_loss
        loss_list.append(mask_loss.item() * num_voc)
        n += num_voc

    # 计算反向传播
    loss.backward()
    # 进行梯度剪裁
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # 更新权重
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 返回在当前batch中的平均损失
    return sum(loss_list) / n


def train(batch_list, batch_size, learning_rate, total_gen, encoder, decoder, device, checkpoint=False):
    '''
    迭代训练  
    Args:
        batch_list: list [[input_seq],[target_seq]]....
        batch_size: int 决定单个batch的大小
        learning_rate: int encoder的学习率
        total_gen: int 总迭代次数
        checkpoint: dict 模型参数储存字典
    '''
    decoder_lr_ratio = 5
    encoder_lr = learning_rate
    decoder_lr = learning_rate * decoder_lr_ratio
    clip = 50
    total_loss = 0
    print_loss = 0
    train_save_path = 'D:/2000.pth'
    startIter = 1

    if checkpoint:
        startIter =2000 + 1
    print(startIter)

    encoder.train()
    decoder.train()

    # 定义优化器
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_lr)

    if load:
        encoder_optimizer.load_state_dict(checkpoint["en_opt"])
        decoder_optimizer.load_state_dict(checkpoint["de_opt"])

    i = 0
    for g in range(startIter, total_gen+1):
        # 把batch扔给batchGen拿到需要的数据
        # single_batch = []
        # for _ in range(batch_size):
        #     single_batch.append(batch_list[i])
        #     i += 1
        #     if i >= len(batch_list):
        #         i = 1
        single_batch = [random.choice(batch_list) for _ in range(batch_size)]
        in_tensor, target_tensor, mask, len_tensor, target_len = corpus_gen.batchGen(single_batch)
        # print(in_tensor)
        # print(target_tensor)
        # print(target_len)
        #计算当前损失和总损失
        loss = trainOnce(in_tensor, target_tensor, mask, len_tensor, target_len, batch_size, clip, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, device)
        
        print("generation:{}, total loss:{:.3f}, percentage:{:.2f}%".format(g, loss, 100*g/total_gen))
        
        if g%1000 == 0:
            torch.save({
                "en" : encoder.state_dict(),
                "de" : decoder.state_dict(),
                "en_opt" : encoder_optimizer.state_dict(),
                "de_opt" : decoder_optimizer.state_dict(),
                "loss" : loss,
                "embedding" : embedding.state_dict()
            }, train_save_path)
            print("saved")



def test(checkpoint):
    # 读取语料库
    with open(corpus_paris_path, 'r', encoding='utf-8') as corpus_paris:
        corpus = csv.reader(corpus_paris, delimiter="\t")
        header = next(corpus)
        all_in = []
        all_target = []
        for row in corpus:
            all_in.append(list(map(int, row[0].split(","))))
            all_target.append(list(map(int, row[1].split(","))))

    # 填充batch
    # batch_list: shape [[in_batch_1, target_batch_1], [in_batch_2, target_batch_2],.....]
    batch_list = [[all_in[i], all_target[i]] for i in range(len(all_in))]

    if checkpoint:
        embedding.load_state_dict(checkpoint["embedding"])

    # 定义编码器和解码器
    encoder = rnn_model.encoderRNN(embedding, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
    decoder = rnn_model.decoderRNN(embedding, hidden_size, output_size, num_layers, dropout=dropout)

    if checkpoint:
        encoder.load_state_dict(checkpoint["en"])
        decoder.load_state_dict(checkpoint["de"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    train(batch_list, batch_size, learning_rate, total_gen, encoder, decoder, device, checkpoint)

class GreedySearchDecoder(torch.nn.Module):
    '''
    贪婪解码
    '''
    def __init__(self, encoder, decoder):
        torch.nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, seq_in, seq_len, max_len):
        #获得初始encoder输出和隐藏层状态
        encoder_out, encoder_hidden = self.encoder(seq_in, seq_len)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        #产生一个全1的列表以保存之后拿到的token，填入句子起始符作为输出
        decoder_in = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        #用于保存token和score
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_len):
            #对每一个位置都进行一次解码
            decoder_out, decoder_hidden = self.decoder(decoder_in, decoder_hidden, encoder_out)
            decoder_score, decoder_in = torch.max(decoder_out, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_in), dim=0)
            all_scores = torch.cat((all_scores, decoder_score), dim=0)
            decoder_in = torch.unsqueeze(decoder_in, 0)
        return all_tokens, all_scores


def evaluate(sentence, word2index, index2word, encoder, decoder, GreedySearchDecoder, device, max_len):
    '''
    接收输入的句子，转换为一个batch，经过贪婪解码后获取每个位置的token，查找index2word字典转换为文字输出  
    Args:
        sentence: str 输入的句子
        word2index: dict 字符转index字典
        index2word：dict index转字符字典
        encoder: 编码器
        decoder: 解码器
        GreedySearchDecoder: 贪婪解码器
        device: 使用的硬件
        max_len: int 解码器输出字符串最长长度
    
    return:
        sentence_out: str 输出字符串
    '''
    
    sequence = [preprocess.wordToIndex(sentence, word2index)]
    # print("here is sequence")
    # print(sequence)
    len_tensor = torch.tensor([len(seq) for seq in sequence])
    batch = torch.LongTensor(sequence).transpose(0,1)
    batch = batch.to(device)
    len_tensor = len_tensor.to(device)
    tokens, scores = GreedySearchDecoder(batch, len_tensor, max_len)
    print(tokens)
    word_out = [index2word[token.item()] for token in tokens if token.item() != EOS_token]
    #print(word_list)
    sentence_out = "".join(word_out)
    return sentence_out

def chat(encoder, decoder, word2index, index2word, GreedySearchDecoder, device, max_len):
    while True:
        try:
            sentence_in = input(">")
            if sentence_in == "88" or sentence_in == "quit": break
            sentence_out = evaluate(sentence_in, word2index, index2word,
                                    encoder, decoder, GreedySearchDecoder, device, max_len)
        print("Bot" + sentence_out)
        except:
            print("Error")
        

def chatBegin(){
    #存放用于测试聊天用的代码
    SOS_token = 2
    EOS_token = 3
    train_save_path = 'D:/1000.pth'
    # corpus_paris_path = 'D:/corpus_paris.csv'
    corpus_paris_path = "D:/clean_chat_corpus/xiaohuangji_processed.tsv"
    dict_path = "D:/clean_chat_corpus/xiaohuangji_dict.tsv"
    word2index = []
    index2word = []
    word2index, index2word = preprocess.load(dict_path)
    num_words = len(index2word)

    num_layers = 2
    dropout = 0.1
    hidden_size = 256
    output_size = num_words
    embedding = torch.nn.Embedding(num_words, hidden_size)
    learning_rate = 0.0001
    decoder_lr_ratio = 5
    encoder_lr = learning_rate
    decoder_lr = learning_rate * decoder_lr_ratio
    total_gen = 5000
    batch_size = 1024


    encoder = rnn_model.encoderRNN(embedding, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
    decoder = rnn_model.decoderRNN(embedding, hidden_size, output_size, num_layers, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    encoder.to(device)
    decoder.to(device)
    checkpoint = torch.load(train_save_path)
    encoder.load_state_dict(checkpoint["en"])
    decoder.load_state_dict(checkpoint["de"])
    encoder.eval()
    decoder.eval()
    GreedySearchDecoder = GreedySearchDecoder(encoder, decoder)
    chat(encoder, decoder, word2index, index2word, GreedySearchDecoder, device, 20)
}


def trainBegin(load=False, load_dict=True){
    #存放用于训练用的代码
    
    checkpoint = False

    if load:
        checkpoint = torch.load(train_save_path)

    if load_dict:
        word2index, index2word = preprocess.load(dict_path)
        num_words = len(index2word)
    else:
        word2index, index2word = preprocess.wordCount()
        num_words = len(index2word)

    test(checkpoint)    
}

trainBegin(load=False, load_dict=True)
chatBegin()
