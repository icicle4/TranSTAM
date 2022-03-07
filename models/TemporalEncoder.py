import torch
import torch.nn as nn
import math

from models.EdgeTransformer import EdgeTransformerEncoderLayer, TransformerEncoderEdge


class TemporalTransformerEncoderLayer(nn.Module):
    def __init__(self, track_history_len=5, input_feature_dim=256, head_num=8, num_layers=2, dropout=0.0,
                 dim_feedforward=1024):
        super(TemporalTransformerEncoderLayer, self).__init__()
        self.T = track_history_len
        self.input_feature_dim = input_feature_dim
        print(input_feature_dim, head_num)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_feature_dim, nhead=head_num, dropout=dropout,
                                                        dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, Fs, track_mask=None):
        return self.encoder(Fs, mask=track_mask)


class TemporalTransformerEncoderLayerEdge(nn.Module):
    def __init__(self, input_feature_dim=256, head_num=8, num_layers=2, dropout=0.0,
                 dim_feedforward=1024, norm=None):
        super(TemporalTransformerEncoderLayerEdge, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.encoder_layer = EdgeTransformerEncoderLayer(d_model=input_feature_dim, nhead=head_num, dropout=dropout,
                                                         dim_feedforward=dim_feedforward)
        self.encoder = TransformerEncoderEdge(self.encoder_layer, num_layers=num_layers, norm=norm)

    def forward(self, Fs, track_mask=None, edge_score_matrix=None, relative_style="add"):
        return self.encoder(Fs, mask=track_mask, src_relative_matrix=edge_score_matrix, relative_style=relative_style)
