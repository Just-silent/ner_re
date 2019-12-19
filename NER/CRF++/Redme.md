# ChineseNRE

本项目使用
+ python 3.6
+ pytorch 3.6.4
+ CRF++0.58工具包  微云：https://share.weiyun.com/5gClqOJ

## 数据
```
胡	B-PER
久	I-PER
根	I-PER
6	O
岁	O
丧	O
父	O
，	O
跟	O
着	O
母	O
...
```

## 结构
- template为特征模版；
- test.data为测试数据；train.data为训练数据
- crf_learn.exe：CRF++的训练程序；
- crf_test.exe：CRF++的预测程序；
- libcrfpp.dll：训练程序和预测程序需要使用的静态链接库。

## resources目录结构
- conlleval.pl ：测评脚本
- crf_learn.exe ：训练程序
- crf_test.exe ：测试程序
- libcrfpp.dll ：类库
- model ：模型
- train.txt ：训练预料
- test.txt ：测试语料
- output.txt： test->output 三列式
- template ：模板

## predict
- 调用模型，返回预测实体。

## 训练
- cmd
- cd进入该文件夹
- crf_learn template train.data model   训练数据    (出错的话：clr_learn -f 3 -c 1.5 tempalte train.data model)
- 在这里有四个参数可以调整：
```
    -a CRF-L2 or CRF-L1
 规范化算法的选择。默认是CRF-L2。一般来说L2算法效果要比L1算法稍微好一点，虽然L1算法中非零特征的数值要比L2中大幅度的小。

    -c float
    这个参数设置CRF的hyper-parameter。c的数值越大，CRF拟合训练数据的程度越高。这个参数可以调整过拟合和不拟合之间的平衡度。这个参数可以通过交叉验证等方法寻找较优的参数。

    -f NUM
    这个参数设置特征的cut-off threshold。CRF++使用训练数据中至少出现NUM次的 特征。默认值为1。当使用CRF++到大规模数据的时候，只出现一次的特征可能会有百万个，这个选项就会在这样的情况下起作用了。

    -p NUM
    如果电脑有多个CPU ,那么可以通过多线程提升训练速度。NUM是线程数量。

    举一个带参数的命令例子：
    clr_learn -f 3 -c 1.5 tempalte train.data model##过滤掉了频数低于3的特征，并且设超参数为1.5
```

- crf_test -m model test.data >output.txt   测试数据
- perl conlleval.pl < output.txt   评估效果
```
1. perl安装  官网：https://www.perl.org/get.html
2. https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt下载conlleval.txt并改名为conlleval.pl
3. 注：使用评测工具前要将评测文件中的所有制表位转换成空格，否则评测工具会出错。
```

## 准确率
processed 434061 tokens with 12339 phrases; found: 11107 phrases; correct: 9973.
accuracy:  98.09%; precision:  89.79%; recall:  80.83%; FB1:  85.07
              LOC: precision:  90.88%; recall:  85.08%; FB1:  87.88  5263
              ORG: precision:  83.96%; recall:  75.94%; FB1:  79.75  2962
              PER: precision:  93.79%; recall:  78.53%; FB1:  85.48  2882



## 遇到的难题
- Windows下：improt CRFPP失败
- https://blog.csdn.net/likianta/article/details/86318565(solution method)


## 总结
- 人民日报NER相同的方法
- template模板需要了解
- 这只是一个CRF++0.58工具包的使用，具体CRF的用法很那看出，此项目初步了解NER的处理流程，继续了解需要进一步学习其他实际的模型。






