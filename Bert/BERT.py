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

# Hyperparameters
BATCH_SIZE = 128 
LAMBDA = 0 
EPOCH = 15 
max_len = 100
criterion = nn.CrossEntropyLoss(ignore_index = -100) 
warmup_steps = 1000 
LR_peak = 5e-4 
save_model_path = '/results/BERT_base.pt'
save_history_path = '/results/BERT_base_history.pt'

vocab_size = tokenizer.vocab_size
n_layers = 12
d_model = 768
d_ff = d_model * 4
n_heads = 12
drop_p = 0.1

# Data
data = pd.read_excel('대화체.xlsx')
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def mask_tokens(self, sentence):
        MARK_PROB = 0.15

        input_ids = tokenizer.encode(sentence, truncation=True, max_length = max_len, add_special_tokens=False)

        segment_ids = [] 
        labels = []
        is_second_sentence = False  
        for i, token in enumerate(input_ids):
            if token == sep_idx:
                is_second_sentence = True  
            segment_ids.append(0 if not is_second_sentence else 1)  

            val = random.random()
            if val <= MARK_PROB and token not in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
                labels.append(token) 
                val = random.random() 
                if val < 0.8: 
                    input_ids[i] = mask_idx
                elif 0.8 <= val < 0.9: 
                    input_ids[i] = random.choice(list(tokenizer.get_vocab().values()))
            else:
                labels.append(-100)

        return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(segment_ids)

    def __getitem__(self, idx):
        sentence1 = self.data.loc[idx, '원문']

        random_idx = random.randint(0, len(self.data) - 1)
        sentence2 = self.data.loc[random_idx, '원문']

        nsp_label = torch.tensor(0) # 이어지는 문장 데이터셋이 아님

        combined_sentence = '[CLS]' +  sentence1 +  '[SEP]' + sentence2 + '[SEP]'

        input_ids, mtp_label, segment_ids = self.mask_tokens(combined_sentence)

        return input_ids, mtp_label, nsp_label, segment_ids

def custom_collate_fn(batch): 

    input_ids = [item[0] for item in batch]
    mtp_labels = [item[1] for item in batch]
    nsp_labels = [item[2] for item in batch]
    segment_ids = [item[3] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    mtp_labels = pad_sequence(mtp_labels, batch_first=True, padding_value=-100)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=1)

    nsp_labels = torch.stack(nsp_labels)

    return input_ids, mtp_labels, nsp_labels, segment_ids

custom_DS = CustomDataset(data)

train_DS, val_DS, test_DS, _ = torch.utils.data.random_split(custom_DS, [97000, 2000, 1000, len(custom_DS)-97000-2000-1000])

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)


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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten_LN = nn.LayerNorm(d_model)
        self.self_atten = MHA(d_model, n_heads, drop_p)

        self.FF_LN = nn.LayerNorm(d_model)
        self.FF = FeedForward(d_model, d_ff, drop_p)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):

        residual = self.self_atten_LN(x)
        residual, atten_enc = self.self_atten(residual, enc_mask)
        residual = self.dropout(residual) 
        x = x + residual                  

        residual = self.FF_LN(x)
        residual = self.FF(residual)
        residual = self.dropout(residual)
        x = x + residual

        return x, atten_enc

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.seg_embedding = nn.Embedding(2, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])

        self.LN_out = nn.LayerNorm(d_model)

    def forward(self, x, seg, enc_mask, atten_map_save = False): 

        pos = torch.arange(x.shape[1]).expand_as(x).to(device) 

        x = self.token_embedding(x) + self.pos_embedding(pos) + self.seg_embedding(seg) 
        x = self.dropout(x)

        atten_encs = torch.tensor([]).to(device)
        for layer in self.layers:
            x, atten_enc = layer(x, enc_mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs , atten_enc[0].unsqueeze(0)], dim=0) 

        x = self.LN_out(x) 

        return x, atten_encs

class BERT(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.encoder = Encoder(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p)

        self.n_heads = n_heads

        # 초기화 기법은 GPT-2 참고해서 만듦
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02) 
                m.weight.data *= 1/torch.sqrt(torch.tensor(n_layers*2)) 
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02) 

    def make_enc_mask(self, x): 

        enc_mask = (x == pad_idx).unsqueeze(1).unsqueeze(2) 
        enc_mask = enc_mask.expand(x.shape[0], self.n_heads, x.shape[1], x.shape[1]) 
        return enc_mask

    def forward(self, x, seg, atten_map_save = False):

        enc_mask = self.make_enc_mask(x)

        out, atten_encs = self.encoder(x, seg, enc_mask, atten_map_save = atten_map_save)

        return out, atten_encs

class BERTLM(nn.Module): 
    def __init__(self, bert, vocab_size, d_model):
        super().__init__()

        self.bert = bert

        self.nsp = nn.Linear(d_model, 2) 
        self.mtp = nn.Linear(d_model, vocab_size) 

        nn.init.normal_(self.nsp.weight, mean=0, std=0.02) 
        nn.init.normal_(self.mtp.weight, mean=0, std=0.02)

    def forward(self, x, seg, atten_map_save = False):

        x, atten_encs = self.bert(x, seg, atten_map_save)

        return self.nsp(x[:,0]), self.mtp(x), atten_encs
    

bert = BERT(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p).to(device)
model = BERTLM(bert, vocab_size, d_model).to(device)