### 改编自lightNLP项目，地址：https://github.com/smilelight/lightNLP，用于自己的学习与实践，侵权必删。

### NER
本项目使用
+ python 3.6
+ pytorch 3.6.4
+ 语料库 https://share.weiyun.com/5ssrD5S

### 本文目的：目前对LSTM处理序列标注经过了几次debug，对代码有了一定的了解，但害怕我的认识仅限于方法表面功能，所以此次深度解析所以然。

### 项目结构
+ example   
   + ner_saves  模型保存文件夹
   + .xls   模型结果训练过程的P、R、F1、loss、acc
   + test_ner.py    训练与测试的入口
+ lightnlp
   + base   各种基础方法，后面模型都重写这些方法
   + sl     序列标注问题
        + ner   
            + utils 暂时未看
            + config 配置信息
            + model 定义网络
                + class Config(BaseConfig):加载BaseConfig和新传入的config
                + class BiLstmCrf(BaseModel):加载config与定义初始化网络（包括损失函数和前向传播）
                + 注意：Bi-LSTM输出的结果为：out,h,c
                    + out :tensor{len(sentence)，batchsize,hidden}(hidden:若网络中为hidden_lstm=hidden//2，则此hidden为hidden_lstm；若为双向，则此hidden为2*hidden_lstm),注意：若每个词进行标注，则取网络的out，若为一句话进行标注分类，则去下面的h。
                    + h ：tensor{direction*layer,batchsize,hidden}(hidden:若网络中为hidden_lstm=hidden//2，则此hidden为hidden_lstm；若为双向，则此hidden为hidden_lstm，这一点与上面不同)。
                    + c ：c相比于out是不同的，c经过一个输出们才得到的out，所以一般不会使用c。
                + 具体的细节结合项目研究
            + module 训练与预测   
                + def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):训练：zero_grad()->loss()->backward()->step()
                + def predict(self, text):预测
                + def load(self, save_path=DEFAULT_CONFIG['save_path']):加载模型
                + def test(self, test_path):
                + def _validate(self, dev_dataset, sheet=None, x=None, bool=False):获得P,R,F1,acc,loss
            + tool 数据相关处理
                + TEXT = Field(sequential=True, tokenize=light_tokenize, include_lengths=True)
                + SequenceTaggingDataset(path, fields=fields, separator=separator)
                + BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key,sort_within_batch=sort_within_batch)
                + TEXT.build_vocab(*dataset)
   + utils  各种工具方法
+ text 语料库

### 方法粗略讲解（部分表面难以深刻理解，必须实践结合代码）
+ Filed
    + 官网定义：Every dataset consists of one or more types of data. For instance, a text classification dataset contains sentences and their classes
    + 重要参数：
        + sequential：Whether the datatype represents sequential data. If False, no tokenization is applied. Default: True.
        + use_vocab：Whether to use a Vocab object. If False, the data in this field should already be numerical. Default: True.
        + init_token：A token that will be prepended to every example using this field, or None for no initial token. Default: None.
        + eos_token：A token that will be appended to every example using this field, or None for no end-of-sentence token. Default: None.
        + fix_length：A fixed length that all examples using this field will be padded to, or None for flexible sequence lengths. Default: None.
        + dtype：The torch.dtype class that represents a batch of examples of this kind of data. Default: torch.long.
        + preprocessing:在分词之后和数值化之前使用的管道 默认值: None.
        + lower：Whether to lowercase the text in this field. Default: False.
        + tokenize – The function used to tokenize strings using this field into sequential examples. If “spacy”, the SpaCy tokenizer is used. If a non-serializable function is passed as an argument, the field will not be able to be serialized. Default: string.split.分词函数. 默认值: str.split.
        + include_lengths：Whether to return a tuple of a padded minibatch and a list containing the lengths of each examples, or just a padded minibatch. Default: False.
        + batch_first：Whether to produce tensors with the batch dimension first. Default: False.
        + pad_token：The string token used as padding. Default: “<pad>”.
        + unk_token：The string token used to represent OOV words. Default: “<unk>”.
        + pad_first：Do the padding of the sequence at the beginning. Default: False.
    + 几个重要的方法
        + pad(minibatch): 在一个batch对齐每条数据
        + build_vocab(): 建立词典（对数据已经填充）
        + numericalize(): 把文本数据数值化，返回tensor
      
+ Dataset
    + 官网定义：Defines a dataset composed of Examples along with its Fields.
    + 理解：
        + torchtext的Dataset是继承自pytorch的Dataset，提供了一个可以下载压缩数据并解压的方法（支持.zip, .gz, .tgz）
        + splits方法可以同时读取训练集，验证集，测试集
    + 重要参数：
        + ...


+ Iterator
    + 官网定义：Defines an iterator that loads batches of data from a Dataset.
    + 理解：
        + Iterator是torchtext到模型的输出，它提供了我们对数据的一般处理方式，比如打乱，排序，等等，可以动态修改batch大小，这里也有splits方法 可以同时输出训练集，验证集，测试集
    + 重要参数：
        + dataset – The Dataset object to load Examples from.
        + batch_size – Batch size.
        + sort_key – A key to use for sorting examples in order to batch together examples with similar lengths and minimize padding. eg：sort_key=lambda x: len(x.text)：此处x是一个二维list(第几句话,这句话的所有单词按照),所以这个就是按照len(x.text)进行排序的
        + train – Whether the iterator represents a train set.
        + repeat – Whether to repeat the iterator for multiple epochs. Default: False.（不是太理解）
        + repeat: 是否在不同epoch中重复迭代（中文解释）
        + shuffle – Whether to shuffle examples between epochs.
        + sort – Whether to sort examples according to self.sort_key. Note that shuffle and sort default to train and (not train).
        + sort_within_batch – Whether to sort (in descending order according to self.sort_key) within each batch. If None, defaults to self.sort. If self.sort is True and this is False, the batch is left in the original (ascending（升序）) sorted order.
        + device: 建立batch的设备 -1:CPU ；0,1 ...：对应的GPU



     
+ 此项目中的Dataset：SequenceTaggingDataset(path, fields=fields, separator=separator)
    + 官网定义：Defines a dataset for sequence tagging. Examples in this dataset contain paired lists – paired list of words and tags.
    + path：文件路径
    + fileds：按照fields定义的规则加载
    + separator：语料中的分隔符
    
    
+ 此项目中的Filed：Filed(对象).build_vocab  定义：Construct the Vocab object for this field from one or more datasets.
    + eg:TEXT.build_vocab(*dataset) ：可以接收一个或多个Dataset对象，多个存入一个list传入
    + build_vocab方法返回的对象具有三个重要的属性：freqs,itos,stoi
        + freqs：按照语料库字符(作为key)传入顺序建立词典，value记录每个字符出现的次数
        + itos:建立 数字表示->字符 的字典  作用：可以将句子的数字表示转成字符化
        + stoi：建立 字符->数字表示 的字典，作用：将句子转为tensor的基础，可以说这就是调用build_vocab方法的作用（常用）
        
        
+ 此项目中的Iterator：BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key,sort_within_batch=sort_within_batch)：
    + 官网定义：Defines an iterator that batches examples of similar lengths together.
    + Iterator重要参数：
        + sort_key：A key to use for sorting examples in order to batch together examples with similar lengths and minimize padding.
        + 具体参数上面阐述过。需要注意的是vocab、Iterator之后都是padded的，这点需要注意，在CRF那一层中（未确认）去掉了pad，代码如下
        + CRF代码
        ```
        mask = torch.ne(x, self.pad_index)  # 此处记录了哪些是pad
        emissions = self.lstm_forward(x, poses, sent_lengths)
        a = self.crflayer.decode(emissions, mask=mask)
        return self.crflayer.decode(emissions, mask=mask)   # 此处疑似忽略pad，达到去掉的效果
         ```
### 总结：
+ 很难凭借这个介绍对Bi-LSTM有很深的理解，关键还要结合代码，可以尝试一步步的debug进行观察数据的变化，刚开始可以先看方法的作用，想把数据处理成什么样，然后进一步看方法是如何处理的。
