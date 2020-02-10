import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vectors

from ...utils.log import logger
from ...base.model import BaseConfig, BaseModel

from .config import DEVICE, DEFAULT_CONFIG


class Config(BaseConfig):
    def __init__(self, word_vocab, label_vocab, vector_path, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.class_num = len(self.label_vocab)
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class TextCNN(BaseModel):
    def __init__(self, args):
        super(TextCNN, self).__init__(args)

        self.class_num = args.class_num
        self.chanel_num = 1
        self.filter_num = args.filter_num
        self.filter_sizes = args.filter_sizes

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.embedding_dimension).from_pretrained(
                args.vectors).to(DEVICE)
            self.chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.chanel_num, self.filter_num, (size, self.embedding_dimension)) for size in
             self.filter_sizes]).to(DEVICE)
        self.dropout = nn.Dropout(args.dropout).to(DEVICE)
        self.fc = nn.Linear(len(self.filter_sizes) * self.filter_num, self.class_num).to(DEVICE)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack((self.embedding(x), self.embedding2(x)), dim=1).to(DEVICE)
        else:
            x = self.embedding(x).to(DEVICE)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

