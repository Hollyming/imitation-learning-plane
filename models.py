import torch
import torch.nn as nn
# TransformerModel类
   
class TransformerModel(nn.Module):
    # 类的定义和实现
    def __init__(self, input_dim, action_dim, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, dim_feedforward))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(dim_feedforward, action_dim)

    def forward(self, x):
        print("x",x.shape)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)  # 将输入展平以适应嵌入层
        x = self.embedding(x)  # 应用嵌入层
        x = x.view(batch_size, seq_len, -1)  # 重塑回 (batch_size, seq_len, dim_feedforward)
        x = x + self.positional_encoding[:, :seq_len, :]  # 加上位置编码
        x = self.transformer_encoder(x)  # 应用Transformer编码器
        x = self.fc_out(x).view(batch_size, seq_len, 3)  # 最终输出重塑
        return x