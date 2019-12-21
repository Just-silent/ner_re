import nltk
import pycrfsuite

# 1.read th corpus
def read():         ##1.读取中文数据，读入到一个列表中，并且每个单词以一个二元组的形式存储
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    # print(train_sents)
    return [train_sents,test_sents]

# 2.feature
def word2features(sent,i):          ##2.特征处理：单词和单词的前后单词，并且一个单词的特征以一个列表的形式展现，所有单词也包含在一个列表中
    word = sent[i][0]   #第i个单词
    # postag = sent[i][1]     #第i个单词的词性
    features = [
        'bias',
        'word.lower=' + word.lower(),  # 当前词的小写格式
        'word[-3:]=' + word[-3:],  # 单词后三位
        'word[-2:]=' + word[-2:],  # 单词后两位
        'word.isupper=%s' % word.isupper(),  # 当前词是否全大写 isupper
        'word.istitle=%s' % word.istitle(),  # 当前词的首字母大写，其他字母小写判断 istitle
        'word.isdigit=%s' % word.isdigit(),  # 当前词是否为数字 isdigit
        # 'postag=' + postag,  # 当前词的词性
        # 'postag[:2]=' + postag[:2],  # 当前词的词性前缀
    ]  # 还有就是与之前后相关联的词的上述特征（类似于特征模板的定义）
    if i > 0:
        word1 = sent[i - 1][0]
        # postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=%s' % word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.issupper=%s' % word1.isupper(),
            # '-1:postag=%s' % postag1,
            # '-1:postag[:2]=%s' % postag1[:2],
        ])
    else:
        features.append('BOS')  # 可能是句子的开头

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        # postag1 = sent[i + 1][1]
        features.extend([  # extend接收一个list，并把list中的元素添加到原列表中
            '+1:word.lower=%s' % word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.issupper=%s' % word1.isupper(),
            # '+1:postag=%s' % postag1,
            # '+1:postag[:2]=%s' % postag1[:2],
        ])
    else:
        features.append('EOS')  # append是直接添加一个元素到列表的末尾
    # print(features)
    return features

#完成特征转化
def sent2features(sent):
    #将一句话中每个词的特征转化结果存储到一个列表中
    return [word2features(sent,i) for i in range(len(sent))]

#获取类别，即标签
def sent2labels(sent):
    return [label for token,postag,label in sent]

#获取词
def sent2tokens(sent):
    return [token for token,postag,label in sent]

# get the x_train and y_train from data,and to get features from x_train
def getTrainAndTest():          ##3.x_train and y_train要以列表的形式展现
    list = read()
    train_sents = list[0]
    test_sents = list[1]
    X_train = [sent2features(s) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    Y_test = [sent2labels(s) for s in test_sents]
    return [X_train,Y_train,X_test,Y_test]

def train():
    # Train the model, create pycrfsuite. Trainer, load the training data and call 'train' method.
    trainer = pycrfsuite.Trainer(verbose=False)

    #load X_train
    list = getTrainAndTest()
    X_train = list[0]
    Y_train = list[1]
    for xseq, yseq in zip(X_train, Y_train):
        # print('xseq        ',xseq,'\n','yseq      ',yseq)
        trainer.append(xseq, yseq)          ##4.以列表的加载进去

    #set params
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    # Train the model
    trainer.train('./model/conll2002-esp.crfsuite')

    # Make predictions
    tagger = pycrfsuite.Tagger()
    tagger.open('./model/conll2002-esp.crfsuite')

    return tagger

def NER(example_sent):
    tagger = train()
    # print(sent2features(example_sent))
    list_babel = tagger.tag(sent2features(example_sent))            ##5.最终特征转换后的数据以列表的形式输入最终返回结果
    list_token = example_sent
    print((tagger.tag(sent2features(example_sent))))
    result = []
    for i in range(0,len(list_babel)):
        ch = list_token[i]
        label = list_babel[i]
        if label == 'B-PER' and list_babel[i + 1] == 'O':
            word = ch
            result.append(word)
        elif label == 'B-PER' and list_babel[i + 1] != 'O':
            word = ch
        elif label == 'I-PER' and list_babel[i+1] == 'I-PER':
            word += ch
        elif label == 'I-PER' and list_babel[i+1] == 'O':
            word += ch
            result.append(word)
        if label == 'B-ORG' and list_babel[i + 1] == 'O':
            word = ch
            result.append(word)
        elif label == 'B-ORG' and list_babel[i + 1] != 'O':
            word = ch
        elif label == 'I-ORG' and list_babel[i+1] == 'I-ORG':
            word += ch
        elif label == 'I-ORG' and list_babel[i+1] == 'O':
            word += ch
            result.append(word)
        if label == 'B-LOC'and list_babel[i+1] == 'O':
            word = ch
            result.append(word)
        elif label == 'B-LOC' and list_babel[i+1] != 'O':
            word = ch
        elif label == 'I-LOC' and list_babel[i+1] == 'I-LOC':
            word += ch
        elif label == 'I-LOC' and list_babel[i+1] == 'O':
            word += ch
            result.append(word)
    return ' '.join(result)
    # return ("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))

if __name__ == '__main__':
    # example_sent = read()[1][2]
    # print(sent2features(example_sent))
    example_sent = ['Las', 'reservas', '"','on', 'line',
     '"', 'de', 'billetes', 'aéreos', 'a',
     'través', 'de', 'Internet', 'aumentaron',
     'en', 'España', 'un', '300', 'por',
     'ciento', 'en', 'el', 'primer', 'trimestre',
     'de', 'este', 'año', 'con', 'respecto',
     'al', 'mismo', 'período', 'de', '1999',
     ',', 'aseguró', 'hoy', 'Iñigo', 'García',
     'Aranda', ',', 'responsable', 'de',
     'comunicación', 'de', 'Savia', 'Amadeus',
     '.']
    result = NER(example_sent)
    print(result)