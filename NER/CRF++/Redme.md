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

## 训练
- cmd
- cd进入该文件夹
- crf_learn template train.data model   训练数据
- crf_test -m model test.data >output.txt   测试数据
- conlleval.pl < output.txt   评估效果

## 结构
- template为特征模版；
- test.data为测试数据；train.data为训练数据
- crf_learn.exe：CRF++的训练程序；
- crf_test.exe：CRF++的预测程序；
- libcrfpp.dll：训练程序和预测程序需要使用的静态链接库。


## 准确率


## 遇到的难题
- Windows下：improt CRFPP失败
- https://blog.csdn.net/likianta/article/details/86318565(solution method)


