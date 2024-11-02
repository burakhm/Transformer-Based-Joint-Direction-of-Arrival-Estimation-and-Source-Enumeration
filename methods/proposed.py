import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_size, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=15)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src)
        src = src.swapaxes(0,1)
        src = self.pos_encoder(src)
        src = src.swapaxes(0,1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output
        
class Covariance_Reconstructer(nn.Module):
    def __init__(self):
        super(Covariance_Reconstructer, self).__init__()
        self.transformer2 = TransformerEncoder(num_layers=12, input_size=2, d_model=128, n_heads=8, dim_feedforward=128, dropout=0.1)
        self.fcl = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self, x, mask):
        x = self.transformer2(x, mask)
        x = self.fcl(x)
        return x

class DOA_Estimator(nn.Module):
    def __init__(self):
        super(DOA_Estimator, self).__init__()
        self.transformer2 = TransformerEncoder(num_layers=6, input_size=4, d_model=32, n_heads=4, dim_feedforward=64, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(32,256),
            nn.ReLU(),
            nn.Linear(256,121)
        )

    def forward(self, x, mask):
        x = self.transformer2(x, mask)
        x = x * (~mask.unsqueeze(2).repeat(1,1,x.shape[2]))
        x1 = x.mean(dim=1)
        x2 = self.classifier(x1)
        return x1, x2
    
class Source_Number_Estimator(nn.Module):
    def __init__(self):
        super(Source_Number_Estimator, self).__init__()
        self.number_estimator = nn.Sequential(
            nn.Linear(32,256),
            nn.ReLU(),
            nn.Linear(256,4)
        )

    def forward(self, x):
        x = self.number_estimator(x)
        return x