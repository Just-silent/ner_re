import torch
import xlwt
from tqdm import tqdm

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import Config, BiLstmCrf
from .tool import ner_tool
from .utils.convert import iob_ranges

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class NER(Module):
    """
    """
    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None
    
    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = ner_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ner_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = ner_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = ner_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        train_iter = ner_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        workbook = xlwt.Workbook(encoding='utf-8')
        sheet = workbook.add_sheet('prf1')
        sheet.write(0, 0, 'acc_loss')
        sheet.write(0, 1, 'precision')
        sheet.write(0, 2, 'recall')
        sheet.write(0, 3, 'accuracy')
        sheet.write(0, 4, 'f1')
        sheet.write(0, 5, 'micro_f1')
        sheet.write(0, 6, 'macro_f1')
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = (-bilstmcrf.loss(item_text_sentences, item_text_lengths, item.tag)) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            sheet.write(epoch + 1, 0, acc_loss)
            if dev_path:
                dev_score = self._validate(dev_dataset, sheet, epoch + 1, True)
                logger.info('dev score:{}'.format(dev_score))

            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        workbook.save('Finance_BiLSTM_CRF.xls')
        config.save()
        bilstmcrf.save()

    def predict(self, text):
        self._model.eval()
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text])
        len_text = torch.tensor([len(vec_text)]).to(DEVICE)
        vec_predict = self._model(vec_text.view(-1, 1).to(DEVICE), len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        print(tag_predict, vec_predict)
        return iob_ranges([x for x in text], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
    
    def test(self, test_path):
        test_dataset = ner_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))
    
    def _validate(self, dev_dataset, sheet=None, x=None, bool=False):
        self._model.eval()
        dev_score_list = []
        p_list = []
        r_list = []
        acc_list = []
        f1_list = []
        micro_f1_list = []
        macro_f1_list = []

        for dev_item in tqdm(dev_dataset):
            item_score, p, r, acc, f1, micro_f1, macro_f1= ner_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab, self._tag_vocab)
            dev_score_list.append(item_score)
            p_list.append(p)
            r_list.append(r)
            acc_list.append(acc)
            f1_list.append(f1)
            micro_f1_list.append(micro_f1)
            macro_f1_list.append(macro_f1)
        p = sum(p_list) / len(p_list)
        r = sum(r_list) / len(r_list)
        acc = sum(acc_list) / len(acc_list)
        f1 = sum(f1_list) / len(f1_list)
        micro_f1 = sum(micro_f1_list) / len(micro_f1_list)
        macro_f1 = sum(macro_f1_list) / len(macro_f1_list)
        logger.info('precision score:{}'.format(p))
        logger.info('recall score:{}'.format(r))
        logger.info('accuracy score:{}'.format(acc))
        logger.info('f1 score:{}'.format(f1))
        logger.info('micro_f1 score:{}'.format(micro_f1))
        logger.info('macro_f1 score:{}'.format(macro_f1))
        if bool:
            sheet.write(x, 1, p)
            sheet.write(x, 2, r)
            sheet.write(x, 3, acc)
            sheet.write(x, 4, f1)
            sheet.write(x, 5, micro_f1)
            sheet.write(x, 6, macro_f1)
        return sum(dev_score_list) / len(dev_score_list)
