from Frame.tc.re.tool import re_tool

train_path = '../data/train.txt'
dev_path = '../data/test.txt'
vec_path = '../data/vec.txt'

from Frame.tc import RE

re = RE()

re.train(train_path, dev_path=train_path, vectors_path=vec_path, save_path='./re_saves')

re.load('./re_saves')
re.test(dev_path)
# 钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
print(re.predict('邢朋举', '刘研', '邢朋举和刘研是校友。'))
