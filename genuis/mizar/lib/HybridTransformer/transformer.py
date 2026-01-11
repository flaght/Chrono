#from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import sys

from .transformer_layer import EncoderLayer, DecoderLayer, Encoder, Decoder
from .attn import FullAttention, AttentionLayer
from .embed import DataEmbedding


class Transformer_base(nn.Module):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=False):
        super(Transformer_base, self).__init__()

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.encoder = Encoder([
            EncoderLayer(AttentionLayer(
                FullAttention(False,
                              attention_dropout=dropout,
                              output_attention=output_attention), d_model,
                n_heads),
                         d_model,
                         d_ff,
                         n_heads,
                         dropout=dropout,
                         activation=activation) for l in range(e_layers)
        ],
                               norm_layer=torch.nn.LayerNorm(d_model))
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True,
                                      attention_dropout=dropout,
                                      output_attention=False), d_model,
                        n_heads),
                    AttentionLayer(
                        FullAttention(False,
                                      attention_dropout=dropout,
                                      output_attention=False), d_model,
                        n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection_decoder = nn.Linear(d_model, c_out, bias=True)
        self.c_out = c_out

    def forward(self,
                x_enc,
                x_dec,
                enc_self_mask=None,
                dec_self_mask=None,
                dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out,
                               enc_out,
                               x_mask=dec_self_mask,
                               cross_mask=dec_enc_mask)

        output = self.projection_decoder(dec_out)

        return enc_out, dec_out, output


class PlaneTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=False,
                 dim_num=None):
        super(PlaneTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.dim_num = dim_num

    def hidden_size(self):
        return self.d_model

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        bs, ticker_num = inputs.shape[0], inputs.shape[1]
        mask = torch.ones_like(inputs)
        rand_indices = torch.rand(bs, ticker_num).argsort(dim=-1)
        mask_indices = rand_indices[:, :int(ticker_num / 2) + 1]
        batch_range = torch.arange(bs)[:, None]
        mask[batch_range, mask_indices, self.dim_num:] = 0
        enc_inp = mask * inputs
        enc_out, dec_out, output = super(PlaneTransformer,
                                         self).forward(enc_inp, enc_inp)
        return enc_out.transpose(1, 2), dec_out.transpose(
            1, 2), output[:, -1:, :].squeeze(-1)
        #return enc_out, dec_out[:, -1:, :], output.squeeze(-1)


class PlanarTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dim_num=None,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=False,
                 is_align=False):
        super(PlanarTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.dim_num = dim_num
        self.is_align = is_align

    def hidden_size(self):
        return self.d_model

    ## inputs: [bs, ticker, feature]
    def forward(self, inputs):
        bs, ticker_num = inputs.shape[0], inputs.shape[1]
        mask = torch.ones_like(inputs)
        rand_indices = torch.rand(bs, ticker_num).argsort(dim=-1)
        mask_indices = rand_indices[:, :int(ticker_num / 2) + 1]
        batch_range = torch.arange(bs)[:, None]
        mask[batch_range, mask_indices, self.dim_num:] = 0
        enc_inp = mask * inputs
        enc_out, dec_out, output = super(PlanarTransformer,
                                         self).forward(enc_inp, enc_inp)

        return enc_out, dec_out, output
        #return enc_out, dec_out, output if not self.is_align else enc_out.transpose(
        #    1, 2), dec_out.transpose(1, 2), output[:, -1:, :].squeeze(-1)
        #return enc_out, dec_out[:, -1:, :], output.squeeze(-1)


class TemporalTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=-1,
                 output_attention=False):
        super(TemporalTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim

    def hidden_size(self):
        return self.d_model  # * self.c_out

    # enc_out: 需进行 enc_out.transpose(1, 2), 时间维度和特征交换，特征维度:self.d_model
    # dec_out:  需进行 enc_out.transpose(1, 2), 时间维度和特征交换，特征维度:self.d_model
    def forward(self, inputs):
        enc_inp = inputs.transpose(1, 2)
        if self.denc_dim > 0:
            dec_inp = enc_inp[:, -self.denc_dim:, :]
        else:
            dec_inp = enc_inp
        enc_out, dec_out, output = super(TemporalTransformer,
                                         self).forward(enc_inp, dec_inp)
        return enc_out.transpose(1, 2), dec_out.transpose(
            1, 2), output[:, -self.c_out:, :].squeeze(-1)


class TransientTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=-1,
                 output_attention=False):
        super(TransientTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim

    def hidden_size(self):
        return self.d_model  # * self.c_out

    def forward(self, inputs):
        enc_inp = inputs
        if self.denc_dim > 0:
            dec_inp = inputs[:, :, -self.denc_dim:, :]
        else:
            dec_inp = inputs
        bs, ticker_num = enc_inp.shape[0], enc_inp.shape[1]
        enc_inp = enc_inp.reshape(-1, enc_inp.shape[-2],
                                  enc_inp.shape[-1]).float().to(enc_inp.device)
        dec_inp = dec_inp.reshape(-1, dec_inp.shape[-2],
                                  dec_inp.shape[-1]).float().to(dec_inp.device)

        enc_out, dec_out, output = super(TransientTransformer,
                                         self).forward(enc_inp, dec_inp)

        # 基类的输出形状为 [bs*ticker_num, seq_len, c_out]。
        # 只需要最后一个时间步的预测。
        output = output[:, -1, :]
        # 输出形状为 [bs*ticker_num, c_out]。
        # 由于 c_out=1，形状为 [400, 1]。

        enc_out = enc_out.reshape(bs, ticker_num, enc_out.shape[-2],
                                  enc_out.shape[-1])
        dec_out = dec_out.reshape(bs, ticker_num, dec_out.shape[-2],
                                  dec_out.shape[-1])
        output = output.reshape(bs, ticker_num)
        return enc_out, dec_out, output

        #return enc_out.transpose(1, 2), dec_out.transpose(
        #    1, 2), output[:, -self.c_out:, :].squeeze(-1)


class TransitoryTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=-1,
                 softmax_output=False,
                 output_attention=False):
        super(TransitoryTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim
        self.softmax_output = softmax_output

    def hidden_size(self):
        return self.d_model  # * self.c_out

    def forward(self, inputs):
        enc_inp = inputs
        if self.denc_dim > 0:
            dec_inp = inputs[:, :, -self.denc_dim:, :]
        else:
            dec_inp = inputs
        bs, ticker_num = enc_inp.shape[0], enc_inp.shape[1]
        enc_inp = enc_inp.reshape(-1, enc_inp.shape[-2],
                                  enc_inp.shape[-1]).float().to(enc_inp.device)
        dec_inp = dec_inp.reshape(-1, dec_inp.shape[-2],
                                  dec_inp.shape[-1]).float().to(dec_inp.device)

        enc_out, dec_out, output = super(TransitoryTransformer,
                                         self).forward(enc_inp, dec_inp)
        output = output[:, -1, :]
        enc_out = enc_out.reshape(bs, ticker_num, enc_out.shape[-2],
                                  enc_out.shape[-1])
        dec_out = dec_out.reshape(bs, ticker_num, dec_out.shape[-2],
                                  dec_out.shape[-1])
        output = output.reshape(bs, ticker_num, self.c_out)
        if self.softmax_output:
            output = torch.softmax(output, axis=-1)
        return enc_out, dec_out, output


class SequentialTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=-1,
                 output_attention=False):
        super(SequentialTransformer, self).__init__(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=1,
            #c_out=c_out,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim

    def hidden_size(self):
        return self.d_model  # * self.c_out

    def forward(self, inputs):
        enc_inp = inputs
        if self.denc_dim > 0:
            dec_inp = inputs[:, -self.
                             denc_dim:, :]  #inputs[:, :, -self.denc_dim:, :]
        else:
            dec_inp = inputs
        enc_out, dec_out, output = super(SequentialTransformer,
                                         self).forward(enc_inp, dec_inp)
        output = output[:, -self.c_out:, :].squeeze(-1)
        # 确保输出层的处理不会导致所有值相同
        # 可以考虑增加一个非线性激活函数或其他处理来增加输出的多样性
        if self.c_out != 1: 
            output = torch.sigmoid(output)  # 例如使用sigmoid激活函数, 用于回归任务
        return enc_out, dec_out, output


class TemporientTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 d_model=128,
                 n_heads=8,
                 e_layers=3,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.1,
                 activation='gelu',
                 masking_ratio=0.15):
        dec_in = enc_in
        c_out = enc_in
        self.masking_ratio = masking_ratio

        super(TemporientTransformer, self).__init__(enc_in=enc_in,
                                                    dec_in=dec_in,
                                                    c_out=c_out,
                                                    d_model=d_model,
                                                    n_heads=n_heads,
                                                    e_layers=e_layers,
                                                    d_layers=d_layers,
                                                    d_ff=d_ff,
                                                    dropout=dropout,
                                                    activation=activation)

    def forward(self, inputs, masking_ratio=None):
        masking_ratio = self.masking_ratio if masking_ratio is None else masking_ratio
        if masking_ratio == 0:
            enc_inp = inputs
            dec_inp = inputs
        else:
            ## 训练时：创建并应用随机时间步遮盖
            bs, seq_len, _ = inputs.shape

            ## 计算要遮盖的索引数量
            num_masked_steps = int(seq_len * self.masking_ratio)

            # 为每个样本随机生成要遮盖的索引
            # randperm 生成一个随机排列, 取前 num_masked_steps 个
            masked_indices = torch.cat([
                torch.randperm(seq_len)[:num_masked_steps].unsqueeze(0)
                for _ in range(bs)
            ]).to(inputs.device)

            # 创建一个副本用于制作编码器输入
            enc_inp = inputs.clone()

            # 将被选中的时间步的所有特征置为0 (或一个可学习的掩码token)
            batch_indices = torch.arange(bs).unsqueeze(1).to(inputs.device)
            enc_inp[batch_indices, masked_indices, :] = 0.0

            # 解码器的输入就是被遮盖的序列，它需要学会重建
            dec_inp = enc_inp

        # 调用父类的标准 Encoder-Decoder 流程
        enc_out, dec_out, output = super(TemporientTransformer,
                                         self).forward(enc_inp, dec_inp)

        return enc_out, dec_out, output

class SequentialNLLTransformer(Transformer_base):
    """
    支持高斯NLL损失的时序Transformer模型。
    输出包含预测的均值(mean)和方差(variance)。
    """
    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=-1,
                 output_attention=False,
                 output_variance=True,
                 var_min=1e-6,
                 var_max=1e-4,
                 bias=0.05):
        super(SequentialNLLTransformer, self).__init__(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim
        self.output_variance = output_variance

        self.var_min = var_min
        self.var_max = var_max
        self.bias = bias

        # 均值预测头初始化 (防止预测偏差)
        # 基类的 projection_decoder 用于均值预测，确保其初始化正确
        nn.init.constant_(self.projection_decoder.bias, 0.0)
        # 使用更小的初始化范围，提高训练稳定性
        nn.init.uniform_(self.projection_decoder.weight, -0.01, 0.01)

        if self.output_variance:
            # 方差预测头：线性层将隐层特征映射到标量方差
            self.variance_head = nn.Linear(d_model, 1)
            # Softplus激活函数确保方差始终为正数
            self.softplus = nn.Softplus()
            # 初始化偏置为正值(0.5)，防止训练初期方差过小导致NLL Loss爆炸
            nn.init.constant_(self.variance_head.bias, 0.5)

    def hidden_size(self):
        return self.d_model
    
    def forward(self, inputs):
        """
        前向传播
        Returns:
            enc_out: 编码器输出
            dec_out: 解码器输出
            output: [batch, 2] 如果 output_variance=True, 包含 [mean, var]
                    [batch, 1] 如果 output_variance=False, 仅包含 [mean]
        """
        enc_inp = inputs
        if self.denc_dim > 0:
            dec_inp = inputs[:, -self.denc_dim:, :]
        else:
            dec_inp = inputs

        # 调用基类的前向传播
        enc_out, dec_out, output_mean = super(SequentialNLLTransformer,
                                         self).forward(enc_inp, dec_inp)
        # 均值头输出处理
        output_mean = output_mean[:, -self.c_out:, :].squeeze(-1)
        # 添加 Tanh 约束，强制输出在合理范围内 (解决预测偏差问题)
        # 限制在 [-0.05, 0.05] 范围，防止极端预测偏差
        output_mean = torch.tanh(output_mean) * self.bias

        if self.output_variance:
            # 方差头输出处理
            # 使用解码器最后一个时间步的输出特征
            dec_out_last = dec_out[:, -1, :] # [batch, d_model]
            
            # 使用 Sigmoid 缩放方法 (解决 Softplus 导致的方差坍缩问题)
            # 步骤: Linear -> Sigmoid -> 线性缩放到 [var_min, var_max]
            output_var_logits = self.variance_head(dec_out_last)
            output_var_normalized = torch.sigmoid(output_var_logits)  # 映射到 [0, 1]
            
            # 线性缩放到目标范围 [var_min, var_max]
            output_var = self.var_min + output_var_normalized * (self.var_max - self.var_min)
            
            # 不需要 clamp (Sigmoid 自动保证范围)
            # 堆叠均值和方差: [batch, 2]
            # 确保 output_mean 维度正确以便堆叠
            if output_mean.dim() == 1:
                output_mean = output_mean.unsqueeze(-1)
            
            return enc_out, dec_out, torch.cat([output_mean, output_var], dim=-1)
        else:
            return enc_out, dec_out, output_mean