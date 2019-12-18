#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

class BiLSTM_ATT(nn.Module):
    def __init__(self,config,embedding_pre):
        super(BiLSTM_ATT,self).__init__()
        self.batch = config['BATCH']
        
        self.embedding_size = config['EMBEDDING_SIZE']
        self.embedding_dim = config['EMBEDDING_DIM']    # 100
        
        self.hidden_dim = config['HIDDEN_DIM']  # 200
        self.tag_size = config['TAG_SIZE']
        
        self.pos_size = config['POS_SIZE'] # 82
        self.pos_dim = config['POS_DIM'] # 25
        
        self.pretrained = config['pretrained'] # false
        if self.pretrained:
            #self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_pre))
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre),freeze=False).to(DEVICE)
        else:
            self.word_embeds = nn.Embedding(self.embedding_size,self.embedding_dim).to(DEVICE) # (4409,100)
        
        self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim).to(DEVICE) # (82,25)
        self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim).to(DEVICE) # (82,25)
        self.relation_embeds = nn.Embedding(self.tag_size,self.hidden_dim).to(DEVICE) # (12,200)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True).to(DEVICE)     # n*layer*bidirect  (150, 200)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.tag_size).to(DEVICE)  # (200, 12)
        
        self.dropout_emb=nn.Dropout(p=0.5).to(DEVICE)
        self.dropout_lstm=nn.Dropout(p=0.5).to(DEVICE)
        self.dropout_att=nn.Dropout(p=0.5).to(DEVICE)
        
        self.hidden = self.init_hidden()
        
        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim)).to(DEVICE) # (batch, 1, 200)
        self.relation_bias = nn.Parameter(torch.randn(self.batch,self.tag_size,1)).to(DEVICE)
        
    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2).to(DEVICE)    # (layer*bidirect, batch, hidden_dim//2)
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(DEVICE),
                torch.randn(2, self.batch, self.hidden_dim // 2).to(DEVICE))
                
    def attention(self,H):
        M = F.tanh(H)
        a1 = F.softmax(torch.bmm(self.att_weight,M),2)
        a = torch.transpose(a1,1,2)
        return torch.bmm(H,a)
        
    
                
    def forward(self,sentence,pos1,pos2):

        self.hidden = self.init_hidden_lstm()

        embeds = torch.cat((self.word_embeds(sentence),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),2)
        
        embeds = torch.transpose(embeds,0,1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_out = torch.transpose(lstm_out,0,1)
        lstm_out = torch.transpose(lstm_out,1,2)
        
        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))
        #att_out = self.dropout_att(att_out)
        
        relation = torch.tensor([i for i in range(self.tag_size)],dtype = torch.long).repeat(self.batch, 1).to(DEVICE)

        relation = self.relation_embeds(relation)
        
        res = torch.add(torch.bmm(relation,att_out),self.relation_bias)
        
        res = F.softmax(res,1)

        return res.view(self.batch,-1).to(DEVICE)

#    attention：lstm->out   out->tanh->*weight->结果1    结果1*out->结果2   结果2*relation+bias-softmax->最终结果
