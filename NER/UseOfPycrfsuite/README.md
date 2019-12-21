# NER

本项目使用
+ python 3.6
+ pytorch 3.6.4
+ CRF++0.58工具包  微云：https://share.weiyun.com/5eGvgxC

## 数据
```
在 O
西 B-PERSON
沃 I-PERSON
特 I-PERSON
戴 I-PERSON
维 I-PERSON
宣 O
读 O
的 O
书 O
...
```
### 训练
直接运行NER_service
```
- trainer = pycrfsuite.Trainer(verbose=False)
- trainer.append(xseq, yseq)  加载数据
- trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
```

### 训练结果p、r、f1、acc
- 在MSRA中文NER语料库上进行训练
- 结果（不包括类别O）：
```
                precision    recall  f1-score   support

       B-LOC       0.87      0.77      0.82      5622
       I-LOC       0.81      0.74      0.78      7661
       B-ORG       0.73      0.65      0.69      3274
       I-ORG       0.78      0.74      0.76     14583
       B-PER       0.92      0.75      0.83      3442
       I-PER       0.90      0.79      0.84      6037

   micro avg       0.82      0.75      0.78     40619
   macro avg       0.83      0.74      0.78     40619
weighted avg       0.82      0.75      0.78     40619
 samples avg       0.07      0.07      0.07     40619
 ```
 
 ### 难点
 - 学习与理解CRF在NER上如何进行train