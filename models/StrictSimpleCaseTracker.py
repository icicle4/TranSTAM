import torch
from torch import Tensor


class ImpossibleTracker:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def xywhtoxyxy(self, bbox: Tensor):
        new_bbox = bbox
        new_bbox[:, 2] += bbox[:, 0]
        new_bbox[:, 3] += bbox[:, 1]
        return new_bbox

    def forward(self, tracker_tensors, detections_tensor, sample_window_size):
        # M, T, C
        track_frames, track_apps, track_masks, track_bboxs = tracker_tensors["frames"], tracker_tensors["apps"], \
                                                             tracker_tensors["masks"], tracker_tensors["bboxs"]
        # N, C
        detection_apps, detection_bboxs = detections_tensor["app"], detections_tensor["bbox"]
    
        track_num = track_frames.size()[0]
        
        frame_gap = sample_window_size - track_frames[torch.arange(track_num), torch.sum(track_masks, dim=1) - 1]
    
        track_fv_bboxs = track_bboxs[torch.arange(track_bboxs.size()[0]), torch.sum(track_masks, dim=1) - 1, :]
        
        det_radius = torch.sqrt(detection_bboxs[..., 2] * detection_bboxs[..., 3])
        
        det_track_elu_dis_matrix = torch.cdist(detection_bboxs[None, ..., :2] + detection_bboxs[None, ..., 2:]/2,
                                               track_fv_bboxs[None, ..., :2] + track_fv_bboxs[None, ..., 2:] / 2,
                                               p=2) / det_radius[None, :, None] / frame_gap[None, None, :]

        return det_track_elu_dis_matrix < self.threshold
