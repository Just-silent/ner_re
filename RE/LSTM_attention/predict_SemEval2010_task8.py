import torch
import pickle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ridge	surge	Other	A misty ridge uprises from the surge.
# women	accident	Cause-Effect	The women that caused the accident was on the cell phone and ran thru the intersection without pausing on the median.
# master	stick	Instrument-Agency	The school master teaches the lesson with a stick.
# tail	shark	Component-Whole	We noted the tail of the shark curved to its left side (away from the camera) and moving to its right side (towards the camera).
# generator	principle	Instrument-Agency	The generator creates electricity using much the same principle as the alternator on your car (depending on the turbine type).
model = torch.load('model/SemEval2010_task8/model_epoch80.pkl').to(DEVICE)
entity1 = "generator"
entity2 = "principle"
sentence = "The generator creates electricity using much the same principle as the alternator on your car (depending on the turbine type)."
sentences = sentence

with open('./data/SemEval2010_task8/people_relation_train.pkl', 'rb') as inp:
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
x = sentence.split()
x1 = entity1.split()[0]
x2 = entity2.split()[0]
if x1 in x:
    index1 = x.index(x1)  # the index of entity1
else:
    for i in range(len(x)):
        if x1 in x[i]:
            index1 = i
if x2 in x:
    index2 = x.index(x2)  # the index of entity1
else:
    for i in range(len(x)):
        if x2 in x[i]:
            index2 = i

for i, word in enumerate(sentence.split()):
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
