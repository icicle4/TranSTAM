import torch
from collections import defaultdict
import numpy as np
import copy


def merge_tracklet_into_tensors(tracklets):
    res = defaultdict(list)
    for tracklet in tracklets:
        for k in tracklet.keys():
            res[k].append(tracklet[k][None, ...])
    return {k: torch.cat(res[k], dim=0) for k in res.keys()}


def convert_detections_to_tensor(detections):

    list_format_detections = defaultdict(list)
    for detection in detections:
        for k in detection.keys():
            list_format_detections[k].append(detection[k])

    return {
            k: torch.from_numpy(np.stack(list_format_detections[k], axis=0))
            for k in list_format_detections.keys()
            }


def convet_tracklet_to_tensor(tracklet, frames, remain_length: int, track_id: int):
    tracklet_frames = []
    tracklet_apps = []
    tracklet_bboxs = []

    start_frame = min(frames)
    for frame in frames:
        if tracklet.get(frame) is not None:
            tracklet_frames.append(frame)
            
            tracklet_apps.append(tracklet[frame]["app"])
            tracklet_bboxs.append(tracklet[frame]["bbox"])

    tracklet_frames = tracklet_frames[-remain_length:]
    tracklet_apps = tracklet_apps[-remain_length:]
    tracklet_bboxs = tracklet_bboxs[-remain_length:]

    valid_frame_num = min(len(tracklet_frames), remain_length)

    tracklet_frames = torch.from_numpy(np.stack(tracklet_frames, axis=0)) - start_frame
    tracklet_apps = torch.from_numpy(np.stack(tracklet_apps, axis=0))
    tracklet_bboxs = torch.from_numpy(np.stack(tracklet_bboxs, axis=0))
    
    _, Ac = tracklet_apps.size()
    _, Bc = tracklet_bboxs.size()

    tracklet_mask = torch.zeros(remain_length, dtype=torch.long)

    if valid_frame_num < remain_length:
        tracklet_frames = torch.cat([tracklet_frames, torch.zeros(remain_length - valid_frame_num,
                                                                 dtype=torch.long)], dim=0)
        tracklet_apps = torch.cat([tracklet_apps, torch.zeros((remain_length - valid_frame_num,
                                                               Ac), dtype=torch.float32)], dim=0)
        tracklet_bboxs = torch.cat([tracklet_bboxs, torch.zeros((remain_length - valid_frame_num,
                                                                Bc), dtype=torch.float32)], dim=0)

    tracklet_mask[:valid_frame_num] = 1

    tracklet = {
            "frames": tracklet_frames,
            "apps": tracklet_apps,
            "masks": tracklet_mask,
            "bboxs": tracklet_bboxs
    }
    return tracklet

def merge_batch_tracklets_tensors(batch, max_track_num):
    res = defaultdict(list)
    for item in batch:
        track_tensors = item["tracks"]

        for k in track_tensors.keys():
            if len(track_tensors[k].size()) == 2:
                M, T = track_tensors[k].size()
                if M < max_track_num:
                    res[k].append(
                        torch.cat([track_tensors[k], torch.zeros((max_track_num - M, T), dtype=track_tensors[k].dtype)])[None, ...]
                    )
                else:
                    res[k].append(
                        track_tensors[k][None, ...]
                    )
            elif len(track_tensors[k].size()) == 3:
                M, T, D = track_tensors[k].size()

                if M < max_track_num:
                    res[k].append(
                        torch.cat(
                            [track_tensors[k], torch.zeros((max_track_num - M, T, D), dtype=track_tensors[k].dtype)]
                        )[None, ...]
                    )
                else:
                    res[k].append(
                        track_tensors[k][None, ...]
                    )
    return {"track_{}".format(k): torch.cat(res[k], dim=0) for k in res.keys()}


def merge_batch_detections_tensors(batch, max_det_num):
    res = defaultdict(list)
    for item in batch:
        detections_tensor = item["detections"]

        for k in detections_tensor.keys():
            if len(detections_tensor[k].size()) == 2:
                N, D = detections_tensor[k].size()
                if N < max_det_num:
                    res[k].append(
                        torch.cat(
                            [detections_tensor[k], torch.zeros((max_det_num - N, D), dtype=detections_tensor[k].dtype)]
                        )[None, :]
                    )
                else:
                    res[k].append(detections_tensor[k][None, :])
            elif len(detections_tensor[k].size()) == 1:
                N = detections_tensor[k].size()[0]
                if N < max_det_num:
                    res[k].append(
                        torch.cat(
                            [detections_tensor[k], torch.zeros((max_det_num - N, ), dtype=detections_tensor[k].dtype)]
                        )[None, ...]
                    )
                else:
                    res[k].append(detections_tensor[k][None, ...])

    return {"det_{}".format(k): torch.cat(res[k], dim=0) for k in res.keys()}


def normalize_det_xywh_with_video_wh(detection, video_width, video_height):
    norm_det = copy.deepcopy(detection)
    x, y, w, h = norm_det["bbox"]
    norm_det["bbox"] = np.asarray([x / video_width, y / video_height, w / video_width, h / video_height],
                                  dtype=np.float32)
    return norm_det
