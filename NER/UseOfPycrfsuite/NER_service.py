import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from itertools import chain

tagger = '' # 加载model
train_sents = ''
test_sents = ''

# 1.read th data
def read(str):
    if str == 'train':
        f = open('./data/MSRA_BI_220w/MSRA_train.txt', encoding='utf-8')
    elif str == 'test':
        f = open('./data/MSRA_BI_220w/MSRA_test.txt', encoding='utf-8')
    lines = f.readlines()
    list1 = []
    list2 = []
    for line in lines:
        if line != '\n':
            tuple = (line[0], line[2:-1])
            list1.append(tuple)
        elif line == '\n':
            list2.append(list1)
            list1 = []
    return list2

# 2.feature
def word2features(sent,i):
    word = sent[i][0]   #第i个单词
    # postag = sent[i][1]     #第i个单词的词性
    features = [
        'word.lower=' + word # 当前词的小写格式
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        features.append('-1:word=%s' % word1)
    else:
        features.append('BOS')  # 可能是句子的开头

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        # postag1 = sent[i + 1][1]
        features.append('+1:word=%s' % word1)
    else:
        features.append('EOS')  # append是直接添加一个元素到列表的末尾
    return features

#完成特征转化
def sent2features(sent):
    #将一句话中每个词的特征转化结果存储到一个列表中
    return [word2features(sent,i) for i in range(len(sent))]

#获取类别，即标签
def sent2labels(sent):
    return [label for token,label in sent]

#获取词
def sent2tokens(sent):
    return [token for token,label in sent]

def getTrainAndTest():          ##3.x_train and y_train要以列表的形式展现
    global train_sents
    global test_sents
    if train_sents == '':
        train_sents = read('train')
        test_sents = read('test')
    X_train = [sent2features(s) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    Y_test = [sent2labels(s) for s in test_sents]
    return [X_train,Y_train,X_test,Y_test]

def train():
    global tagger
    # Train the model, create pycrfsuite. Trainer, load the training data and call 'train' method.
    trainer = pycrfsuite.Trainer(verbose=False)

    #load X_train
    list = getTrainAndTest()
    X_train = list[0]
    Y_train = list[1]
    for xseq, yseq in zip(X_train, Y_train):
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
    trainer.train('./model/tain.crfsuite')
    # Make predictions
    tagger = pycrfsuite.Tagger()
    tagger.open('./model/tain.crfsuite')

    return tagger

def get_BIO_Result(list_token, list_babel):
    result = []
    for i in range(0, len(list_babel)):
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
    return ''.join(result)

def get_BME_Result(list_token, list_babel):
    result = []
    for i in range(0, len(list_babel)):
        ch = list_token[i]
        label = list_babel[i]
        if list_babel[i] == 'B_PERSON' or list_babel[i] == 'M_PERSON' or list_babel[i] == 'E_PERSON':
            word = ch
            result.append(word)
        if list_babel[i] == 'B_TIME' or list_babel[i] == 'M_TIME' or list_babel[i] == 'E_TIME':
            word = ch
            result.append(word)
            if list_babel[i] == 'E_TIME':
                result.append(' ')
        if list_babel[i] == 'B_LOCATION' or list_babel[i] == 'M_LOCATION' or list_babel[i] == 'E_LOCATION':
            word = ch
            result.append(word)
            if list_babel[i] == 'E_LOCATION':
                result.append(' ')
        if list_babel[i] == 'B_ORGANIZATION' or list_babel[i] == 'M_ORGANIZATION' or list_babel[i] == 'E_ORGANIZATION':
            word = ch
            result.append(word)
            if list_babel[i] == 'E_ORGANIZATION':
                result.append(' ')
    return ''.join(result)

def NER(example_sent):
    tagger = pycrfsuite.Tagger()
    tagger.open('./model/tain.crfsuite')
    list_babel = tagger.tag(sent2features(example_sent))
    list_token = example_sent
    print(list_babel)
    result1 = get_BIO_Result(list_token, list_babel)
    result2 = get_BME_Result(list_token, list_babel)
    if result1 != None:
        return result1
    else:
        return result2

def bio_classification_report():
    tagger = pycrfsuite.Tagger()
    tagger.open('./model/tain.crfsuite')
    X_test = getTrainAndTest()[2]
    y_true = getTrainAndTest()[3]
    # Predict entity labels for all sequences in our testing set
    # 标注所有的信息
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    # 使用sklearn.metrics中的方法计算NER的p、r、acc、r、f1
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def _validate():
    tagger = pycrfsuite.Tagger()
    tagger.open('./model/tain.crfsuite')
    test_x = getTrainAndTest()[2]
    test_y = getTrainAndTest()[3]
    list_y1 = []
    list_y2 = []
    s1 = []
    s2 = []
    # 使用sklearn.metrics中的方法计算NER的p、r、acc、r、f1
    for i in range(len(test_y)):
        predict_y = tagger.tag(test_x[i])
        true_y = test_y[i]
        list_y1.extend(true_y)
        list_y2.extend(predict_y)
        s1.append(true_y)
        s2.append(predict_y)

    labels = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER']
    p = precision_score(list_y1, list_y2, average='macro', labels=labels)
    r = recall_score(list_y1, list_y2, average='macro', labels=labels)
    acc = accuracy_score(list_y1, list_y2)
    f1 = f1_score(list_y1, list_y2, average='weighted', labels=labels)

    # 传进来的参数：
    # acc：list_y1, list_y2
    # p、r、f1：list_y1, list_y2, average='weighted', labels=labels
    print(classification_report(list_y1, list_y2, labels=labels))

    # average各个参数的作用：
    #   None：返回每一类各自的f1_score，得到一个array
    #   binary: 只对二分类问题有效，返回由pos_label指定的类的f1_score
    #   micro：所有类别放在一起计算平均
    #   macro：先计算每个类别的，在计算平均
    #   weighted：先计算每个类别的，在乘与每个样本的占比
    #
    # p: 0.8555820239182169
    # r: 0.7763942668666806
    # acc: 0.9667926858206566
    # f1: 0.8132489413652524
    print("p:{}   r:{}   acc:{}   f1:{}   ".format(p, r, acc, f1))



if __name__ == '__main__':
    while True:
        print('请输入你想要的提取实体的句子')
        str = input()+'。'
        example_sent = list(str)
        result = NER(example_sent)
        print('提取结果：',result)
        print('是否打印出评测报告（y/n）')
        str1 = input()
        if str1=='y':
            # _validate()
            print(bio_classification_report())