from torch.utils.data import Dataset, DataLoader


import os
import sys
import pdb
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))

import numpy as np
import torch
import copy
import tqdm

from datasets.warp_tracklet_and_detection_tensor import merge_batch_tracklets_tensors, merge_batch_detections_tensors


def collate_fn(batch):
    batch_track_nums = [item["track_num"] for item in batch]
    batch_det_nums = [item["det_num"] for item in batch]
    batch_impossible_mask = [item["impossible_mask"] for item in batch]

    max_track_num = max(batch_track_nums)
    max_det_num = max(batch_det_nums)
    
    impossible_mask = torch.ones((len(batch), max_det_num, max_track_num), dtype=torch.bool)
    batch_tracks = merge_batch_tracklets_tensors(batch, max_track_num)
    batch_dets = merge_batch_detections_tensors(batch, max_det_num)

    batch_det_mask = torch.zeros((len(batch), max_det_num), dtype=torch.long)

    for i, det_num in enumerate(batch_det_nums):
        batch_det_mask[i, :det_num] = 1
        track_num = batch_track_nums[i]
        if batch_impossible_mask[i] is not None:
            impossible_mask[i, :det_num, :track_num] = batch_impossible_mask[i]

    labels = [item["labels"] for item in batch]

    final_res = {}

    final_res.update(batch_tracks)
    final_res.update(batch_dets)
    final_res.update(
                        {
                          "labels": labels,
                          "det_mask": batch_det_mask,
                          "track_num": torch.from_numpy(np.asarray(batch_track_nums, dtype=np.int32)),
                          "det_num": torch.from_numpy(np.asarray(batch_det_nums, dtype=np.int32)),
                          "impossible_mask": impossible_mask
                        }
                    )
    return final_res


def normalize_det_xywh_with_video_wh(detection, video_width, video_height):
    norm_det = copy.deepcopy(detection)
    x, y, w, h = norm_det["bbox"]
    norm_det["bbox"] = np.asarray([x / video_width, y / video_height, w / video_width, h / video_height],
                                  dtype=np.float32)
    return norm_det


class StaticDataset(Dataset):
    def __init__(self, stage="train", hdf5_file=""):

        super(StaticDataset, self).__init__()

        f = h5py.File(hdf5_file, mode="r")
        self.grp = f["static_dataset"]
        self.keys = list(self.grp.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        sub_grp = self.grp[self.keys[item]]
        sample = {}

        detections = {}
        tracklets = {}

        for key in sub_grp.keys():
            value = sub_grp[key][()]

            if isinstance(value, np.ndarray):
                value = torch.from_numpy(sub_grp[key][()])

            if "det" in key and "num" not in key:
                detections[key] = value
            elif "track" in key and "num" not in key:
                tracklets[key] = value
            else:
                sample[key] = value

        sample["detections"] = detections
        sample["tracks"] = tracklets

        return sample


def build_dataset(stage, args):
    if os.path.isdir(args.root_dir):
        dataset = StaticDataset(stage=stage,
                                hdf5_file= os.path.join(args.root_dir, "{}.hdf5".format(stage))
                                )
    else:
        dataset = StaticDataset(stage=stage, hdf5_file=args.root_dir)

    return dataset


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()

    time1 = time.time()

    simple_dataset = build_dataset("train", args)
    dataloader = DataLoader(simple_dataset, batch_size=4, collate_fn=collate_fn)

    print(len(dataloader))

    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        pass

    time2 = time.time()

    print("Build dataset and loop one epoch cost time: {}".format(time2 - time1))
