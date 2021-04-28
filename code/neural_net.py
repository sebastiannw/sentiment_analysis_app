#Torch Packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocabulary, n_output, n_embedding, n_hidden, n_layers, drop=0.2):
        super().__init__()
        
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        self.embedding = nn.Embedding(vocabulary, n_embedding)
        self.lstm      = nn.LSTM(n_embedding, n_hidden, n_layers,
                                 dropout=drop, batch_first=True)
        
        self.linear  = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid() 
    
    def forward(self, x, hidden):
        #pdb.set_trace()
        batch = x.size(0)
        embed = self.embedding(x)
        
        lstm_out, hidden = self.lstm(embed, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)
        
        output = self.linear(lstm_out)
        output = self.sigmoid(output)
        output = output.view(batch, -1)[:, -1]
        
        return output, hidden
    
    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch, self.n_hidden).zero_())
        
        return hidden