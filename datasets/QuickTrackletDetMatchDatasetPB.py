import torch
from torch.utils.data import Dataset, DataLoader
import random

import os
import sys
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))


from datasets.memmory_bank_pb import MemoryBank
import tqdm
from datasets.warp_tracklet_and_detection_tensor import *
from models.StrictSimpleCaseTracker import StrictSimpleCaseTracker, ImpossibleTracker
from utils.metrics import AvgMetric


def collate_fn(batch):
    batch_track_nums = [item["track_num"] for item in batch]
    batch_det_nums = [item["det_num"] for item in batch]

    max_track_num = max(batch_track_nums)
    max_det_num = max(batch_det_nums)
    batch_tracks = merge_batch_tracklets_tensors(batch, max_track_num)
    batch_dets = merge_batch_detections_tensors(batch, max_det_num)

    batch_det_mask = torch.zeros((len(batch), max_det_num), dtype=torch.long)

    for i, det_num in enumerate(batch_det_nums):
        batch_det_mask[i, :det_num] = 1

    labels = [item["labels"] for item in batch]
    final_res = {}

    final_res.update(batch_tracks)
    final_res.update(batch_dets)
    final_res.update(
                        {
                          "labels": labels,
                          "det_mask": batch_det_mask,
                          "track_num": torch.from_numpy(np.asarray(batch_track_nums, dtype=np.int32)),
                          "det_num": torch.from_numpy(np.asarray(batch_det_nums, dtype=np.int32))
                        }
                    )
    return final_res


class TrackDetMatchDatasetPublic(Dataset):
    def __init__(self, stage="train", root_dir=".", tracklet_sample_region_length=5, drop_simple_case=False,
                 cache_window=20, threshold=3.0):
        
        self.drop_simple_case = False
        
        self.cache_window = cache_window
        if drop_simple_case:
            self.drop_simple_case = True
            self.impossible_tracker = ImpossibleTracker(threshold)
        else:
            self.impossible_tracker = None
            
        super(TrackDetMatchDatasetPublic, self).__init__()

        self.memory_bank = MemoryBank(root_dir)

        self.stage = stage

        self.memory_bank.load_pbs_to_memory_bank(phase=stage)
        self.memory_bank.build_memmap_memory()

        self.tracklet_sample_region_length = tracklet_sample_region_length

        self.sample_start_frames = []

        for video_id in self.memory_bank.video_names:
            video_frames = self.memory_bank.video_frames[video_id]
            sorted_video_frames = sorted(list(video_frames))

            for video_frame in sorted_video_frames[:-tracklet_sample_region_length]:
                end_frame = video_frame + self.tracklet_sample_region_length
                track_ids = self.memory_bank.get_track_ids_in_frame(video_id, end_frame)

                if len(track_ids) > 0:
                    self.sample_start_frames.append([video_id, video_frame])

        print("build dataset done, total sample num is : {}".format(len(self.sample_start_frames)))

    def __len__(self):
        return len(self.sample_start_frames)

    def __getitem__(self, item):
        video_id, start_frame = self.sample_start_frames[item]
        end_frame = start_frame + self.tracklet_sample_region_length
        sampled_frames = list(range(start_frame, end_frame))
        video_width, video_height = self.memory_bank.get_video_width_and_height(video_id)

        tracklets = defaultdict(dict)
        for f in sampled_frames:
            track_and_det_ids = self.memory_bank.get_track_ids_in_frame(video_id, f)
            for track_id, det_idx in track_and_det_ids:
                detection = self.memory_bank.query_detection(video_id, track_id, det_idx)
                detection = normalize_det_xywh_with_video_wh(detection, video_width, video_height)
                tracklets[track_id][f] = detection

        max_cache_track_num = 150
        total_tracklets_ids = list(tracklets.keys())

        if len(tracklets) > max_cache_track_num:
            tracklet_ed_frames = {track_id: max(tracklets[track_id].keys()) for track_id in tracklets.keys()}
            sorted_track_ids = [k for k, v in sorted(tracklet_ed_frames.items(), key=lambda item: item[1])]
            remain_track_ids = sorted_track_ids[-max_cache_track_num:]
            total_tracklets_ids = remain_track_ids
        
        sample_detection_frame = end_frame

        detections = defaultdict()

        track_and_det_ids = self.memory_bank.get_track_ids_in_frame(video_id, sample_detection_frame)

        for track_id, det_idx in track_and_det_ids:
            detection = self.memory_bank.query_detection(video_id, track_id, det_idx)
            detection["frame"] = sample_detection_frame - start_frame
            detection = normalize_det_xywh_with_video_wh(detection, video_width, video_height)
            detections[track_id] = detection

        filled_tracklets = []

        random.shuffle(total_tracklets_ids)

        for track_id in total_tracklets_ids:
            filled_tracklets.append(
                convet_tracklet_to_tensor(tracklets[track_id], sampled_frames, self.cache_window, track_id=track_id)
            )
        tracklets_tensors = merge_tracklet_into_tensors(filled_tracklets)
        
        track_num = len(total_tracklets_ids)
        detection_ids = list(detections.keys())
        det_num = len(detection_ids)

        random.shuffle(detection_ids)
        detections = [detections[id] for id in detection_ids]

        detections_tensor = convert_detections_to_tensor(detections)
        
        if self.drop_simple_case:
            impossible_mask = self.impossible_tracker.forward(tracklets_tensors, detections_tensor,
                                                              sample_window_size=self.tracklet_sample_region_length)
            
            label_mask = torch.zeros_like(impossible_mask)
            for i, detection_id in enumerate(detection_ids):
                if detection_id in total_tracklets_ids:
                    j = total_tracklets_ids.index(detection_id)
                    label_mask[0, i, j] = True
            
            impossible_mask = torch.logical_or(label_mask, impossible_mask)
            print('ratio', torch.sum(impossible_mask) / torch.sum(torch.ones_like(impossible_mask)))
        else:
            impossible_mask = None
            
        labels = []
        for detection_id in detection_ids:
            if detection_id in total_tracklets_ids:
                label = total_tracklets_ids.index(detection_id)
                labels.append(label)
            else:
                labels.append(-1)

        res = {
            "track_num": track_num,
            "det_num": det_num,
            "tracks": tracklets_tensors,
            "detections": detections_tensor,
            "labels": labels,
            "impossible_mask": impossible_mask
        }
        return res


def build_dataset(stage, args):
    dataset = TrackDetMatchDatasetPublic(stage=stage,
                                         root_dir=args.dataset_dir,
                                         tracklet_sample_region_length=args.track_history_len
                                         )
    return dataset


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)

    args = parser.parse_args()

    time1 = time.time()

    simple_dataset = TrackDetMatchDatasetPublic(stage="train",
                                                root_dir=args.dataset_dir,
                                                tracklet_sample_region_length=20,
                                                drop_simple_case=True)
    
    tracklet_num_avg_metric = AvgMetric()
    det_num_avg_metric = AvgMetric()
    
    raw_track_num_avg_metric = AvgMetric()
    raw_det_num_avg_metric = AvgMetric()
    
    tp_num_avg_metric = AvgMetric()
    fp_num_avg_metric = AvgMetric()

    track_mul_det_num_avg_metric = AvgMetric()
    
    for i, sample in enumerate(simple_dataset):
        
        tracklet_num_avg_metric.update(sample['track_num'])
        det_num_avg_metric.update(sample['det_num'])
        track_mul_det_num_avg_metric.update(sample['track_num'] * sample['det_num'])
        
        tp_num_avg_metric.update(sample["tp_num"])
        fp_num_avg_metric.update(sample["fp_num"])
        raw_track_num_avg_metric.update(sample["raw_track_num"])
        raw_det_num_avg_metric.update(sample["raw_det_num"])
        
        if i % 1000 == 0:
            print('total track num: {}, avg track num per graph: {}, max track num: {}'.format(tracklet_num_avg_metric.total,
                                                                                               tracklet_num_avg_metric.avg,
                                                                                               tracklet_num_avg_metric.max))
            
            print('total det num: {}, avg det num per graph: {}, max det num: {}'.format(det_num_avg_metric.total,
                                                                                         det_num_avg_metric.avg,
                                                                                         det_num_avg_metric.max))

            print('total tp num: {}, avg tp num per graph: {}'.format(tp_num_avg_metric.total, tp_num_avg_metric.avg))
            print('total fp num: {}, avg fp num per graph: {}'.format(fp_num_avg_metric.total, fp_num_avg_metric.avg))

            print('raw track num: {}, avg track num per graph: {}, max track num: {}'.format(
                    raw_track_num_avg_metric.total,
                    raw_track_num_avg_metric.avg,
                    raw_track_num_avg_metric.max))
            
            print('raw det num: {}, avg det num per graph: {}, max det num: {}'.format(raw_det_num_avg_metric.total,
                                                                                       raw_det_num_avg_metric.avg,
                                                                                       raw_det_num_avg_metric.max))

            print('track_mul_det num: {}, avg  per graph: {}, max  num: {}'.format(track_mul_det_num_avg_metric.total,
                                                                                   track_mul_det_num_avg_metric.avg,
                                                                                   track_mul_det_num_avg_metric.max))

    time2 = time.time()

    print("Build dataset and loop one epoch cost time: {}".format(time2 - time1))

    print('total track num: {}, avg track num per graph: {}, max track num: {}'.format(tracklet_num_avg_metric.total,
                                                                    tracklet_num_avg_metric.avg, tracklet_num_avg_metric.max))
    
    print('total det num: {}, avg det num per graph: {}, max det num: {}'.format(det_num_avg_metric.total,
                                                                det_num_avg_metric.avg, det_num_avg_metric.max))

    print('raw track num: {}, avg track num per graph: {}, max track num: {}'.format(
        raw_track_num_avg_metric.total,
        raw_track_num_avg_metric.avg,
        raw_track_num_avg_metric.max))

    print('raw det num: {}, avg det num per graph: {}, max det num: {}'.format(raw_det_num_avg_metric.total,
                                                                               raw_det_num_avg_metric.avg,
                                                                               raw_det_num_avg_metric.max))

    print('track_mul_det num: {}, avg  per graph: {}, max  num: {}'.format(track_mul_det_num_avg_metric.total,
                                                                           track_mul_det_num_avg_metric.avg,
                                                                           track_mul_det_num_avg_metric.max))
    
    print('Simple Tracker Performance, precison: {}, recall: {}'.format(
        tp_num_avg_metric.total / (tp_num_avg_metric.total + fp_num_avg_metric.total),
        (tp_num_avg_metric.total + fp_num_avg_metric.total) / (tp_num_avg_metric.total + fp_num_avg_metric.total + det_num_avg_metric.total)
    ))
