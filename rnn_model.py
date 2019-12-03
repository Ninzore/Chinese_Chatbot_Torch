import torch.nn


class encoderRNN(torch.nn.Module):
    '''
    创建RNN编码器
    '''

    def __init__(self, embedding, num_layers=1, hidden_size=1, dropout=0):
        torch.nn.Module.__init__(self)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = 0 if num_layers == 1 else dropout
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seq, input_len, hidden=None):
        '''
        Args:
            inputs_seq:  输入的sequence, shape = (input_len, batch_size)
            input_len: 输入句子的长度
            hidden: 初始隐藏状态

        Output:
            bidir_out: 最后一道隐藏层的双向输出特征之和
            hidden: 最后的隐藏层状态
        '''
        embedding = self.embedding(input_seq)
        pack_pad = torch.nn.utils.rnn.pack_padded_sequence(embedding, input_len)
        outputs, hidden = self.gru(pack_pad, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        bidir_out = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return bidir_out, hidden


class attention(torch.nn.Module):
    '''
    '''

    def __init__(self, hidden_size):
        '''
        '''
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.mode = "dot"

        if self.mode == 'general':
            # general attention配置
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)

        elif self.mode == "concat":
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot(self, hidden, encoder_out):
        return torch.sum(hidden * encoder_out, dim=2)

    def general(self, hidden, encoder_out):
        attn_energy = self.attn(encoder_out)
        return torch.sum(hidden * energy, dim=2)

    def concat(self, hidden, encoder_out):
        attn_energy = self.attn(torch.cat((hidden.expand(encoder_out.size(0), -1, -1), encoder_out), dim=2)).tanh()
        return torch.sum(self.v * attn_energy, dim=2)

    def forward(self, hidden, encoder_out):
        '''
        '''
        if self.mode == "general":
        # general attention配置
            attn_energy = self.general(hidden, encoder_out)
        elif self.mode == "dot":
            attn_energy = self.dot(hidden, encoder_out)
        elif self.mode == "concat":
            attn_energy = self.concat(hidden, encoder_out)
        
        attn_energy = attn_energy.t()
        return torch.nn.functional.softmax(attn_energy, dim=1).unsqueeze(1)


class decoderRNN(torch.nn.Module):
    '''
    '''

    def __init__(self, embedding, hidden_size, output_size, num_layers=1, dropout=0.1):
        torch.nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = (0 if num_layers == 1 else dropout)

        self.embedding = embedding
        self.embed_dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=dropout)
        self.concat = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)
        self.attention = attention(hidden_size)

    def forward(self, input_step, last_hidden, encoder_out):
        # 获取当前输出词的embedding
        curr_embedding = self.embedding(input_step)
        curr_embedding = self.embed_dropout(curr_embedding)
        # 通过GRU，获取输出和隐藏层状态
        rnn_out, hidden = self.gru(curr_embedding, last_hidden)
        # 通过当前GRU和解码器的输出，计算attention的权重
        attn_weight = self.attention(rnn_out, encoder_out)
        # 将attention的权重和解码器输出相乘可以获得词向量
        context_vec = attn_weight.bmm(encoder_out.transpose(0, 1))
        # 压缩维度
        rnn_out = rnn_out.squeeze(0)
        context_vec = context_vec.squeeze(1)

        # 将词向量和GRU输出在第一维使用concat进行连接
        concat_in = torch.cat((rnn_out, context_vec), 1)
        concat_out = torch.tanh(self.concat(concat_in))
        # 使用全连接
        voc_pred = self.out(concat_out)
        # 使用softmax计算概率
        voc_out = torch.nn.functional.softmax(voc_pred, dim=1)
        # 返回预测结果和隐藏层状态
        return voc_out, hidden

# num_words = 300
# num_layers = 2
# dropout = 0.1
# hidden_size = 500
# output_size = num_words
# learning_rate = 10e-3
# decoder_lr_ratio = 5
# encoder_lr = learning_rate
# decoder_lr = learning_rate * decoder_lr_ratio
# total_gen = 10
# batch_size = 10
# clip = 50
# MAX_LEN = 20
# import csv
# import random
# import corpus_gen
# corpus_paris_path = 'D:/corpus_paris.csv'

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
# single_batch = [random.choice(batch_list) for _ in range(batch_size)]
# in_tensor, target_tensor, mask, len_tensor, target_len = corpus_gen.batchGen(single_batch)

# encoder = encoderRNN(num_words, num_layers, hidden_size, dropout)
# decoder = decoderRNN(num_words, hidden_size, output_size, num_layers, dropout=dropout)
# encoder.train()
# decoder.train()
# encoder_out, encoder_hidden = encoder(in_tensor, len_tensor)
# decoder_hidden = encoder_hidden[:decoder.num_layers]
# decoder_in = torch.LongTensor([[2 for _ in range(batch_size)]])
# decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_out)