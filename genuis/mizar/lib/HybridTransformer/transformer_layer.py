import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, d_model, ff_dim, dropout, activation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=ff_dim,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim,
                               out_channels=d_model,
                               kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x


class MultiheadFeedForward(nn.Module):

    def __init__(self, d_model, n_heads, ff_dim, dropout, activation):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mhfw = nn.ModuleList([
            FeedForward(d_model=self.head_dim,
                        ff_dim=ff_dim,
                        dropout=dropout,
                        activation=activation) for i in range(self.n_heads)
        ])

    def forward(self, x):  # [bs, seq_len, d_model]
        bs = x.shape[0]
        input = x.reshape(bs, -1, self.n_heads,
                          self.head_dim)  # [bs, seq_len, n_heads, head_dim]
        outputs = []
        for i in range(self.n_heads):
            outputs.append(self.mhfw[i](
                input[:, :, i, :]))  # [bs, seq_len, head_dim]
        outputs = torch.stack(outputs, dim=-2).reshape(
            bs, -1, self.d_model)  # [bs, seq_len, n_heads, head_dim]
        return outputs


class EncoderLayer(nn.Module):

    def __init__(self,
                 attention,
                 d_model,
                 d_ff,
                 n_heads=8,
                 dropout=0.1,
                 activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.mhfw = MultiheadFeedForward(d_model=d_model,
                                         n_heads=n_heads,
                                         ff_dim=d_ff,
                                         dropout=dropout,
                                         activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Post-LN: 先计算，再残差，最后 LayerNorm
        #new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        #x = x + self.dropout(new_x)
        #y = x = self.norm1(x)
        #y = self.mhfw(y)

        # Pre-LN: 先 LayerNorm，再计算，最后残差
        norm_x = self.norm1(x)   # ← LayerNorm 在子层之前
        new_x, attn = self.attention(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout(new_x) 

        # FFN 子层
        norm_x = self.norm2(x)   # ← LayerNorm 在子层之前
        y = self.mhfw(norm_x)
        x = x + y

        return x, attn  # ← Pre-LN: 直接返回，final norm 由 Encoder 处理


class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers,
                                              self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):

    def __init__(self,
                 self_attention,
                 cross_attention,
                 d_model,
                 d_ff,
                 n_heads=8,
                 dropout=0.1,
                 activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.mhfw = MultiheadFeedForward(d_model=d_model,
                                         n_heads=n_heads,
                                         ff_dim=d_ff,
                                         dropout=dropout,
                                         activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        '''
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x) # ← Post-LN

        # Cross-Attention
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])

        y = x = self.norm2(x) # ← Post-LN
        y = self.mhfw(y)

        return self.norm3(x + y)  # ← Post-LN
        '''
        # Self-Attention (Pre-LN)
        norm_x = self.norm1(x)   # ← LayerNorm 在子层之前
        self_attn_out = self.self_attention(norm_x, norm_x, norm_x, attn_mask=x_mask)[0]
        x = x + self.dropout(self_attn_out)

        # Cross-Attention (Pre-LN)
        norm_x = self.norm2(x)   # ← LayerNorm 在子层之前
        cross_attn_out = self.cross_attention(norm_x, cross, cross, attn_mask=cross_mask)[0]
        x = x + self.dropout(cross_attn_out)

        # FFN (Pre-LN)
        norm_x = self.norm3(x)   # ← LayerNorm 在子层之前
        y = self.mhfw(norm_x)
        x = x + y

        return x



class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class DecoderOnlyLayer(nn.Module):
    """
    Decoder-Only 层: 只有因果 Self-Attention，没有 Cross-Attention

    对比 DecoderLayer:
    - DecoderLayer: Self-Attn + Cross-Attn + FFN (3 个子层)
    - DecoderOnlyLayer: Self-Attn + FFN (2 个子层)

    注意力类型:
    - Self-Attention 使用因果掩码 (mask=True)
    - 每个位置只能看到自己和之前的位置
    """
    def __init__(self,
                 self_attention,
                 d_model,
                 d_ff,
                 n_heads=8,
                 dropout=0.1,
                 activation="relu"):
        super(DecoderOnlyLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.mhfw = MultiheadFeedForward(d_model=d_model,
                                         n_heads=n_heads,
                                         ff_dim=d_ff,
                                         dropout=dropout,
                                         activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: 注意力掩码 (可选，因果掩码在 FullAttention 内部处理)

        Returns:
            x: [batch, seq_len, d_model]
        """
        # Self-Attention (Pre-LN)
        norm_x = self.norm1(x)
        self_attn_out, _ = self.self_attention(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout(self_attn_out)

        # FFN (Pre-LN)
        norm_x = self.norm2(x)
        y = self.mhfw(norm_x)
        x = x + y

        return x

class DecoderOnly(nn.Module):
    """
    Decoder-Only 容器: 堆叠多个 DecoderOnlyLayer

    对比 Decoder:
    - Decoder: 需要 cross (enc_out) 参数
    - DecoderOnly: 不需要 cross 参数
    """
    def __init__(self, layers, norm_layer=None):
        super(DecoderOnly, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: 注意力掩码 (可选)

        Returns:
            x: [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x