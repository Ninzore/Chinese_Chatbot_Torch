# Chinese_Chatbot_Torch

requirements: PyTorch, Jieba  

使用QQ的聊天记录作为语料库，需要先把QQ聊天记录提取为txt格式才能使用  
每个模块在干什么，在代码注释里写的还是比较清楚的  

程序运行顺序
```graph
    格式化聊天记录-->预处理-->生成语料库-->训练模型-->聊天测试;
```

运行程序
```graph
    chatlog-->preprocess-->train;
```

###  chatlog:
  会根据QQ聊天记录生成一个csv文件，作为之后的材料  
  
###  preprocess：会在这个csv的基础上生成
  1.  所需要的字典index2voc和voc2index
  2.  根据voc2index字典生成的index句子对，句子对文件分为2行，左边是input，右边是target
  
###  corpus_gen:
  根据输入的句子对生成一个batch，这个batch会作为一次训练的素材
  
###  rnn_model: 
  训练用的神经网络，模型为RNN+Attention，其中RNN作为双向编码还有解码，Attention作为解码的一部分
  
###  train:
  训练和测试，使用trainBegin函数进行训练，使用chatBegin开始试着聊天
  在实际进行聊天时使用了贪婪算法，在GreedySearchDecoder当中可以调参数
  默认为每1000次迭代保存一次模型参数，初始时不加载模型参数(从0开始)，字典默认是加  载的以便加快速度(重新生成字典很慢的)
