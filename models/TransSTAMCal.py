import torch

from models.TemporalEncoder import TemporalTransformerEncoderLayerEdge
from models.EdgeTransformer import EdgeTransformerDecoderLayer, TransformerDecoderEdge

import torch.nn as nn
from torch.nn.init import xavier_uniform_

from models.utils import MLP
from models.help import *


class PEMOT(nn.Module):
    def __init__(self, history_window_size, track_valid_size, appearance_feature_dim=512, pos_feature_dim=2,
                 hidden_dim=512, n_heads=8, dim_feedforward=2048, dropout=0.1, enc_layer_num=6, dec_layer_num=6,
                 sample_valid_frame_method="last", with_abs_pe=True, with_relative_pe=True, with_assignment_pe=True,
                 aspe_style="diff"
                 ):

        super(PEMOT, self).__init__()

        self.track_history_len = track_valid_size
        self.stage = "train"

        self.appearance_feature_dim = appearance_feature_dim
        self.pos_feature_dim = pos_feature_dim
        
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.temporal_enhance = TemporalTransformerEncoderLayerEdge(hidden_dim, head_num=n_heads,
                                                                    num_layers=enc_layer_num, dropout=dropout,
                                                                    dim_feedforward=dim_feedforward,
                                                                    norm=self.encoder_norm)

        self.simple_embed = nn.Linear(self.appearance_feature_dim, hidden_dim)

        self.cross_attn = EdgeTransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout)
        self.aspe_style = aspe_style
        self.with_abs_pe = with_abs_pe
        self.with_relative_pe = with_relative_pe
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoderEdge(self.cross_attn, num_layers=dec_layer_num, norm=self.decoder_norm)

        num_classes = 1
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        
        self.spatial_embed = MLP(pos_feature_dim, hidden_dim, hidden_dim, 3)
        
        self.drop_concat_layer = nn.Linear(2 * hidden_dim, hidden_dim)

        self.head_num = n_heads
    
        self.track_edge_bbox_weight = nn.Parameter(torch.Tensor(self.head_num, 5))
        self.det_edge_bbox_weight = nn.Parameter(torch.Tensor(self.head_num, 5))
        self.det_track_edge_bbox_weight = nn.Parameter(torch.Tensor(self.head_num, 5))
        
        self.history_window_size = history_window_size
        self.sample_valid_frame_method = sample_valid_frame_method

        xavier_uniform_(self.track_edge_bbox_weight)
        xavier_uniform_(self.det_edge_bbox_weight)
        xavier_uniform_(self.det_track_edge_bbox_weight)

    def forward(self, batch_track_app, batch_track_bbox, batch_track_frames, batch_detection_app, batch_detection_bbox):
        
        """
        :param batch_track_app: B, M, T, C
        :param batch_track_bbox: B, M, T, 4
        :param batch_track_frames: B, M, T
        :param batch_detection_app: B, N, C
        :param batch_detection_bbox: B, N, 4
        :return:
        """

        B, M, T, C = batch_track_app.size()
        B, N, C = batch_detection_app.size()

        batch_tracks_app_emb = self.simple_embed(torch.flatten(batch_track_app, end_dim=-2))
        batch_tracks_app_emb = batch_tracks_app_emb.view(B, M, T, -1).permute(2, 0, 1, 3).contiguous()

        batch_tracks_spatial_emb = self.spatial_embed(torch.flatten(batch_track_bbox, end_dim=-2))
        # T, B, M, -1
        batch_tracks_spatial_emb = batch_tracks_spatial_emb.view(B, M, T, -1).permute(2, 0, 1, 3).contiguous()

        batch_det_emb = self.simple_embed(torch.flatten(batch_detection_app, end_dim=1)).view(B, N, -1)

        if self.with_abs_pe:
            batch_tracks_emb = batch_tracks_app_emb + batch_tracks_spatial_emb
            batch_det_spatial_emb = self.spatial_embed(torch.flatten(batch_detection_bbox, end_dim=1)).view(B, N, -1)
            batch_det_emb = batch_det_emb + batch_det_spatial_emb
        else:
            batch_tracks_emb = batch_tracks_app_emb
        
        if self.with_relative_pe:
            track_edge_fea_spt = track_edge_matrix_by_spt(batch_track_bbox, batch_track_frames, self.history_window_size)
            
            # B, M, T, T
            track_edge_score = torch.einsum("bmtdc,hc->bmhtd",
                                            track_edge_fea_spt,
                                            self.track_edge_bbox_weight).reshape(B * M * self.head_num, T, T)
        else:
            track_edge_score = None
            
        # T, B * M, D
        temporal_tracks_fea = batch_tracks_emb.view(T, B * M, -1)
        
        least_tracks_spatial_emb = least_tracklet_features(batch_tracks_spatial_emb.reshape(T, B * M, -1)).view(B, M, -1)
        
        avg_tracks_app_emb = mean_tracklet_features(batch_tracks_app_emb.reshape(T, B * M, -1)).view(B, M, -1)

        least_batch_track_bbox = least_tracklet_features(batch_track_bbox.permute(2, 0, 1, 3).view(T, B * M, -1)).view(B, M, -1)

        enhanced_tracks_fea = self.temporal_enhance(temporal_tracks_fea, None, track_edge_score).view(T, B,M, -1)
        least_enhance_tracks_fea = least_tracklet_features(enhanced_tracks_fea.reshape(T, B * M, -1)).view(B, M, -1)

        least_batch_track_frame = least_tracklet_features(batch_track_frames.permute(2, 0, 1).view(T, B * M),
                                                          temporal_tracks_mask).view(B, M)

        # B * M, D
        if self.with_abs_pe:
            dup_avg_tracks_emb = (least_tracks_spatial_emb + avg_tracks_app_emb).unsqueeze(dim=1).expand(-1, N, -1, -1).contiguous()
        else:
            dup_avg_tracks_emb = avg_tracks_app_emb.unsqueeze(dim=1).expand(-1, N, -1, -1).contiguous()
            
        # B X N, M, D
        duplicate_detection_emb = batch_det_emb.unsqueeze(dim=2).expand(-1, -1, M, -1)
        
        if self.aspe_style == "diff":
            mix_feature = duplicate_detection_emb.reshape(B, N, M, -1) - dup_avg_tracks_emb.reshape(B, N, M, -1)
        elif self.aspe_style == "add":
            mix_feature = duplicate_detection_emb.reshape(B, N, M, -1) + dup_avg_tracks_emb.reshape(B, N, M, -1)
        elif self.aspe_style == "residual":
            mix_feature = duplicate_detection_emb.reshape(B, N, M, -1) + (duplicate_detection_emb.reshape(B, N, M, -1) - dup_avg_tracks_emb.reshape(B, N, M, -1))
        elif self.aspe_style == "concat":
            concated_fea = torch.cat([duplicate_detection_emb.reshape(B, N, M, -1), dup_avg_tracks_emb.reshape(B, N, M, -1)], dim=-1)
            mix_feature = self.drop_concat_layer(torch.flatten(concated_fea, end_dim=-2)).reshape(B, N, M, -1)
        elif self.aspe_style == "none":
            mix_feature = duplicate_detection_emb.reshape(B, N, M, -1)
            
        # N * M, B, D
        tgt = mix_feature.reshape(B, N, M, -1)
        # T, M, B, D
        memory = least_enhance_tracks_fea

        det_bbox = batch_detection_bbox[0]
        b_tgt = tgt[0].reshape(M * N, -1)

        dup_det_bbox = det_bbox[:, None, :].expand(-1, M, -1).reshape(M * N, 4)

        possible_dup_det_bbox = dup_det_bbox
        least_track_bbox = least_batch_track_bbox[0]
        least_track_frame = least_batch_track_frame[0]
        
        det_track_edge_fea_spt = det_track_edge_matrix_by_spt_single_batch(possible_dup_det_bbox,
                                                                           least_track_bbox,
                                                                           least_track_frame,
                                                                           self.history_window_size
                                                                           )
        
        det_track_edge_score = torch.einsum("nmc,hc->hnm",
                                            det_track_edge_fea_spt,
                                            self.track_edge_bbox_weight).reshape(self.head_num,
                                                                                 -1, M)

        det_edge_fea_spt = det_edge_matrix_by_spt_single_batch(possible_dup_det_bbox)

        det_edge_score = torch.einsum("nmc,hc->hnm",
                                      det_edge_fea_spt,
                                      self.det_edge_bbox_weight).reshape(self.head_num,
                                                                         M * N,
                                                                         M * N)
        
        sb_tgt = b_tgt[:, None, :]
        sb_memory = memory[:, None, :]
        
        cross_relation_feature = self.decoder(sb_tgt, sb_memory,
                                              self_relative_matrix=det_edge_score,
                                              cross_relative_matrix=det_track_edge_score)
        outputs_class = self.class_embed(cross_relation_feature.flatten(2)).view(M * N, 1, -1)
        outputs_class = outputs_class.view(N, M, -1)
        
        return outputs_class