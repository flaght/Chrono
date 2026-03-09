import torch,pdb
import torch.nn as nn
from .config import ModelConfig
from .ops import OPS_CONFIG


class AlphaGPT(nn.Module):
    def __init__(self, features_list, ops_list, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        self.features_list = list(features_list)
        self.ops_list = list(ops_list)
        
        self.vocab = self.features_list + self.ops_list
        self.vocab_size = len(self.vocab)
        self.n_features = len(self.features_list)
        
        # ---------- layers ----------
        # Embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ModelConfig.MAX_FORMULA_LEN + 1, self.d_model))
        
        # Transformer Decoder
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, 
                                           dim_feedforward=dim_feedforward)
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Output Heads
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head_actor = nn.Linear(self.d_model, self.vocab_size)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=idx.device), diagonal=1)
        
        # TransformerEncoder 默认 seq-first: (T, B, E)，前后转置兼容所有版本
        x = self.blocks(x.transpose(0, 1), mask=mask).transpose(0, 1)
        x = self.ln_f(x)
        
        last_emb = x[:, -1, :]
        logits = self.head_actor(last_emb)
        value = self.head_critic(last_emb)
        
        return logits, value