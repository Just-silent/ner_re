def locationNER(text):
    tagger = load_model('./resources/model')
    # 利用训练好的模型标记每个字
    for c in text:
        tagger.add(c)

    result = []
    # parse and change internal stated as 'parsed'
    tagger.parse()
    # print(tagger)
    word1 = ''
    predict = []
    for i in range(0, tagger.size()):  # tagger.size：要预测的句子的字数
        for j in range(0, tagger.xsize()):  # tagger.xsize：特征列的个数
            ch = tagger.x(i, j)
            tag = tagger.y2(i)
            predict.append(tag)
            #print(ch,tag)
            if tag == 'B-PER':
                word1 = ch
            elif tag == 'I-PER' and tagger.y2(i+1) == 'I-PER':
                word1 += ch
            elif tag == 'I-PER' and tagger.y2(i + 1) == 'O':
                word1 += ch
                result.append(word1)

            if tag == 'B-ORG':
                word2 = ch
            elif tag == 'I-ORG' and tagger.y2(i+1) == 'I-ORG':
                word2 += ch
            elif tag == 'I-ORG' and tagger.y2(i + 1) == 'O':
                word2 += ch
                result.append(word2)

            if tag == 'B-LOC':
                word3 = ch
            elif tag == 'I-LOC' and tagger.y2(i+1) == 'I-LOC':
                word3 += ch
            elif tag == 'I-LOC' and tagger.y2(i + 1) == 'O':
                word3 += ch
                result.append(word3)
    # print(predict)
    return result

def load_model(path):
    import os, CRFPP
    # -v 3: access deep information like alpha,beta,prob
    # -nN: enable nbest output. N should be >= 2
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    return None

if __name__ == '__main__':

    # predict
    text = 'eg：刘燕上午要去北京饭店，下午去中山公园，晚上回亚运村。'
    print(text, locationNER(text), sep='==> ')

    while True:
        print('请输入：(input：1,over!)',end='')
        text = input()+'。' # 此处粗略解决标签提取对应字符的问题
        if(text.__eq__("1。")):
            break
        print(locationNER(text),sep='')
