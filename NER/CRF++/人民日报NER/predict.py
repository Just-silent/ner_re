def f1(path):
    with open(path,encoding='utf-8') as f:
        all_tag = 0  # 记录所有的标记数
        loc_tag = 0  # 记录真实的地理位置标记数
        pred_loc_tag = 0  # 记录预测的地理位置标记数
        correct_tag = 0  # 记录正确的标记数
        correct_loc_tag = 0  # 记录正确的地理位置标记数
        # 地理命名实体标记
        states = ['B', 'M', 'E', 'S']
        # i=0
        for line in f:
            # i=i+1
            line = line.strip()
            if line == '': continue
            _, r, p = line.split()
            # print(_, r, p)
            all_tag += 1
            if r == p:
                correct_tag += 1
                if r in states:
                    correct_loc_tag += 1
            if r in states:
                loc_tag += 1
            if p in states:
                pred_loc_tag += 1

            # if i==50: break  # 测试用

        loc_P = 1.0 * correct_loc_tag / pred_loc_tag
        loc_R = 1.0 * correct_loc_tag / loc_tag
        print('loc_P:{0}, loc_R:{1}, loc_F1:{2}'.format(loc_P, loc_R, (2 * loc_P * loc_R) / (loc_P + loc_R)))


def load_model(path):
    import os, CRFPP
    # -v 3: access deep information like alpha,beta,prob
    # -nN: enable nbest output. N should be >= 2
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    return None

def locationNER(text):
    tagger = load_model('./resources/model')
    # 利用训练好的模型标记每个字
    for c in text:
        tagger.add(c)
    result = []
    # parse and change internal stated as 'parsed'
    tagger.parse()
    # print(tagger)
    word = ''
    x = tagger.size()
    y = tagger.xsize()

    for i in range(0, tagger.size()):  # tagger.size：要预测的句子的字数
        for j in range(0, tagger.xsize()):  # tagger.xsize：特征列的个数
            ch = tagger.x(i, j)
            tag = tagger.y2(i)
            # print(ch,tag)
            if tag == 'B':
                word = ch
            elif tag == 'M':
                word += ch
            elif tag == 'E':
                word += ch
                result.append(word)
            elif tag == 'S':
                word = ch
                result.append(word)
    return result

if __name__ == '__main__':
    f1('./resources/test.rst')

    # 测试
    text = '我中午要去北京饭店，下午去中山公园，晚上回亚运村。'
    print(text, locationNER(text), sep='==> ')

    while True:
        print('请输入：(input：1,over!)',end='')
        text = input()+'。' # 此处粗略解决标签提取对应字符的问题
        if(text.__eq__("1。")):
            break
        print(locationNER(text),sep='')
