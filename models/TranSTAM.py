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

    def forward_batch(self, batch_track_app, batch_track_bbox, batch_track_mask, batch_track_frames,
                            batch_detection_app, batch_detection_bbox, batch_detection_mask,
                            batch_track_num, batch_det_num,
                            impossible_mask=None):
        
        """
        :param batch_tracks_inp_fea: B, M, T, C
        :param batch_det_inp_fea: B, N, C
        :param batch_track_mask:  B, M, T
        :param batch_detection_mask: B, N
        :param batch_track_num: B, 2
        :param batch_det_num: B, 2
        :param impossible_mask: B, det_num, track_num
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

        temporal_tracks_mask = batch_track_mask.view(B * M, T)
        
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
        
        least_tracks_spatial_emb = least_tracklet_features(batch_tracks_spatial_emb.reshape(T, B * M, -1),
                                                         temporal_tracks_mask).view(B, M, -1)
        
        avg_tracks_app_emb = mean_tracklet_features(batch_tracks_app_emb.reshape(T, B * M, -1),
                                                    temporal_tracks_mask).view(B, M, -1)

        least_batch_track_bbox = least_tracklet_features(batch_track_bbox.permute(2, 0, 1, 3).view(T, B * M, -1),
                                                         temporal_tracks_mask).view(B, M, -1)

        # B * M, T
        track_attn_mask = build_attn_mask(temporal_tracks_mask, T, self.head_num)
        # T, B, M, D
        
        enhanced_tracks_fea = self.temporal_enhance(temporal_tracks_fea, track_attn_mask, track_edge_score).view(T, B,
                                                                                                                 M, -1)
        least_enhance_tracks_fea = least_tracklet_features(enhanced_tracks_fea.reshape(T, B * M, -1),
                                                           temporal_tracks_mask).view(B, M, -1)

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

        outputs_class = []
        
        for i in range(B):
            
            det_bbox = batch_detection_bbox[i]
            track_num = batch_track_num[i]
            det_num = batch_det_num[i]
            
            b_tgt = tgt[i][:det_num, :track_num].reshape(det_num * track_num, -1)

            dup_det_bbox = det_bbox[:, None, :].expand(-1, M, -1)[:det_num, :track_num].reshape(det_num * track_num, 4)
            single_batch_impossible_mask = torch.flatten(impossible_mask[i][:det_num, :track_num], end_dim=-1)
            select_index = torch.where(single_batch_impossible_mask)
            
            sb_outputs_class = torch.zeros((det_num * track_num, 1, 2)).cuda().float()
            sb_outputs_class[..., 0] = 10.0
            sb_outputs_class[..., 1] = -10.0
            
            if select_index[0].size()[0] > 0:
                possible_dup_det_bbox = dup_det_bbox[select_index]
                least_track_bbox = least_batch_track_bbox[i, :track_num]
                least_track_frame = least_batch_track_frame[i, :track_num]
                valid_pair_num = torch.sum(single_batch_impossible_mask)
                
                if self.with_relative_pe:
                    det_track_edge_fea_spt = det_track_edge_matrix_by_spt_single_batch(possible_dup_det_bbox,
                                                                                       least_track_bbox,
                                                                                       least_track_frame,
                                                                                       self.history_window_size
                                                                                       )
                    
                    det_track_edge_score = torch.einsum("nmc,hc->hnm",
                                                        det_track_edge_fea_spt,
                                                        self.track_edge_bbox_weight).reshape(self.head_num,
                                                                                             valid_pair_num, track_num)
        
                    det_edge_fea_spt = det_edge_matrix_by_spt_single_batch(possible_dup_det_bbox)
        
                    det_edge_score = torch.einsum("nmc,hc->hnm",
                                                  det_edge_fea_spt,
                                                  self.det_edge_bbox_weight).reshape(self.head_num,
                                                                                     valid_pair_num,
                                                                                     valid_pair_num)
                else:
                    det_edge_score = None
                    det_track_edge_score = None
                    
                sb_tgt = b_tgt[select_index[0]][:, None, :]
                sb_memory = memory[i, :track_num][:, None, :]
                
                cross_relation_feature = self.decoder(sb_tgt, sb_memory,
                                                      self_relative_matrix=det_edge_score,
                                                      cross_relative_matrix=det_track_edge_score)
                sb_outputs_class[select_index[0]] = self.class_embed(cross_relation_feature.flatten(2)).view(valid_pair_num, 1, -1)
                
            sb_outputs_class = sb_outputs_class.view(det_num, track_num, -1)
            outputs_class.append(sb_outputs_class)
        
        return outputs_class
        
    def forward(self, samples, stage="train", register_hook=False):
        # tracklets (B, T, M, C)
        # detections (B, N, C)
    
        outputs = warp_samples_to_equal_tensor(samples)

        batch_track_app, batch_track_bbox, batch_track_mask, batch_track_num, batch_track_frames, \
            batch_detection_app, batch_detection_bbox, batch_detection_mask, batch_det_num, impossible_mask = outputs
            
        match_matrix = self.forward_batch(batch_track_app, batch_track_bbox, batch_track_mask, batch_track_frames,
                                      batch_detection_app, batch_detection_bbox, batch_detection_mask,
                                      batch_track_num, batch_det_num,
                                      impossible_mask)
        return {
            "match_matrix": match_matrix,
            "track_num": batch_track_num,
            "det_num": batch_det_num,
            "impossible_mask": impossible_mask
        }

