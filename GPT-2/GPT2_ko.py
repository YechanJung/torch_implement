import torch
from torch import nn, optim
from kobert_tokenizer import KoBERTTokenizer
import pandas as pd
from tqdm import tqdm
import math, random
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Tokenizer
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
pad_idx = tokenizer.pad_token_id
mask_idx = tokenizer.mask_token_id
sep_idx = tokenizer.sep_token_id
vocab_size = tokenizer.vocab_size


# Hyperparameters
BATCH_SIZE = 64
LAMBDA = 0
EPOCH = 15
max_len = 100
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
warmup_steps = 2000
LR_peak = 2.5e-4
total_steps = EPOCH * math.ceil(97000/BATCH_SIZE)

def lr_lambda(step):
    return min(step / warmup_steps, (0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))))

save_model_path = '/results/GPT_small.pt'
save_history_path = '/results/GPT_small_history.pt'
n_layers = 12
d_model = 768
d_ff = d_model * 4
n_heads = 12
drop_p = 0.25 

# Data
data = pd.read_excel('대화체.xlsx')
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, '원문']

custom_DS = CustomDataset(data)

train_DS, val_DS, test_DS, _ = torch.utils.data.random_split(custom_DS, [97000, 2000, 1000, len(custom_DS)-97000-2000-1000])

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)


# Model
class MHA(nn.Module):
    def __init__(self, d_model, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_p)
        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, x, mask=None):
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = rearrange(Q, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        K = rearrange(K, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        V = rearrange(V, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)

        attention_score = Q @ K.transpose(-2,-1) / self.scale

        if mask is not None:
            attention_score[mask] = -1e10
        attention_weights = torch.softmax(attention_score, dim=-1)

        attention_weights = self.dropout(attention_weights)

        attention = attention_weights @ V

        x = rearrange(attention, '개 헤 단 차 -> 개 단 (헤 차)')
        x = self.fc_o(x)

        return x, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.GELU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.linear(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten_LN = nn.LayerNorm(d_model)
        self.self_atten = MHA(d_model, n_heads, drop_p)

        self.FF_LN = nn.LayerNorm(d_model)
        self.FF = FeedForward(d_model, d_ff, drop_p)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, dec_mask):

        residual = self.self_atten_LN(x)
        residual, atten_dec = self.self_atten(residual, dec_mask)
        residual = self.dropout(residual)
        x = x + residual

        residual = self.FF_LN(x)
        residual = self.FF(residual)
        residual = self.dropout(residual)
        x = x + residual

        return x, atten_dec

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len=3660):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)

        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))

        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    def forward(self, x):
        B, S, d_model = x.shape
        
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        
        return x
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos_encoding = PositionalEncoder(d_model, drop_p, max_len)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])

        self.LN_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, dec_mask, atten_map_save = False):

        # pos = torch.arange(x.shape[1]).expand_as(x).to(device)

        # x = self.input_embedding(x) + self.pos_embedding(pos)
        x = self.input_embedding(x) + self.pos_encoding(x)
        x = self.dropout(x)

        atten_decs = torch.tensor([]).to(device)
        for layer in self.layers:
            x, atten_dec = layer(x, dec_mask)
            if atten_map_save is True:
                atten_decs = torch.cat([atten_decs , atten_dec[0].unsqueeze(0)], dim=0)

        x = self.LN_out(x)
        x = self.fc_out(x)

        return x, atten_decs

class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.decoder = Decoder(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p)

        self.n_heads = n_heads

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                m.weight.data *= 1/torch.sqrt(torch.tensor(n_layers*2))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        nn.init.normal_(self.decoder.fc_out.weight, mean=0, std=0.02)

    def make_dec_mask(self, x):

        pad_mask = (x == pad_idx).unsqueeze(1).unsqueeze(2)
        pad_mask = pad_mask.expand(x.shape[0], self.n_heads, x.shape[1], x.shape[1])
        future_mask = torch.tril(torch.ones(x.shape[0], self.n_heads, x.shape[1], x.shape[1]))==0
        future_mask = future_mask.to(device)
        dec_mask = pad_mask | future_mask 

        return dec_mask

    def forward(self, x, atten_map_save = False):

        dec_mask = self.make_dec_mask(x)
        out, atten_decs = self.decoder(x, dec_mask, atten_map_save = atten_map_save)

        return out, atten_decs
    
model = GPT(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p).to(device)