from lightnlp.sl import NER

ner_model = NER()

train_path = '../text/onto_train.txt'
dev_path = '../text/onto_test.txt'
vec_path = '../text/token_vec_300.bin'

ner_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./ner_saves/finance')

ner_model.load('./ner_saves/finance')

from pprint import pprint

ner_model.test(train_path)

pprint(ner_model.predict('趣步公司可靠吗？'))
