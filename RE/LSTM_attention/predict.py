import torch
import pickle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model/train_0_1500___test_1500_1800_changed/model_epoch60.pkl').to(DEVICE)
entity1 = "邢朋举"
entity2 = "刘研"
sentence = "邢朋举和刘研是师生关系。"
sentences = sentence

with open('./data/people_relation/people_relation_train.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)

max_len = 50
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["UNKNOW"])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([word2id["BLANK"]] * (max_len - len(ids)))
    return ids

def pos(num):
    if num < -40:
        return 0
    if num >= -40 and num <= 40:
        return num + 40
    if num > 40:
        return 80

def position_padding(words):
    words = [pos(i) for i in words]
    if len(words) >= max_len:
        return words[:max_len]
    words.extend([81] * (max_len - len(words)))
    return words

position1 = []
position2 = []
index1 = sentence.index(entity1[0])
index2 = sentence.index(entity2[0])
for i, word in enumerate(sentence):
    position1.append(i - index1)
    position2.append(i - index2)
    i += 1

sentence = torch.LongTensor(X_padding(sentence)).unsqueeze(0).expand(128,50).to(DEVICE)
position1 = torch.LongTensor(position_padding(position1)).unsqueeze(0).expand(128,50).to(DEVICE)
position2 = torch.LongTensor(position_padding(position2)).unsqueeze(0).expand(128,50).to(DEVICE)

y = model(sentence, position1, position2)[0].cpu()

index = torch.argmax(y)

print('sentence：',sentences)
print('关系：{}'.format(list(relation2id)[index]))
print('准确率：{}%'.format(y[index].item()))
