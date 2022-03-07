import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))
from datasets.QuickTrackletDetMatchDatasetPB import TrackDetMatchDatasetPublic, collate_fn

from collections import defaultdict
import h5py

from utils.metrics import AvgMetric


def convert_detections_to_tensor(detections):

    list_format_detections = defaultdict(list)
    for detection in detections:
        for k in detection.keys():
            list_format_detections[k].append(detection[k])

    return {k: np.stack(list_format_detections[k], axis=0) for k in list_format_detections.keys()}


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--hdf5_path", type=str, required=True)
    
    parser.add_argument('--sample_window', type=int, default=50)
    parser.add_argument('--cache_window', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=3.0)
    args = parser.parse_args()

    time1 = time.time()

    simple_dataset = TrackDetMatchDatasetPublic(stage="train",
                                                root_dir=args.dataset_dir,
                                                tracklet_sample_region_length=args.sample_window,
                                                drop_simple_case=True,
                                                cache_window=args.cache_window,
                                                threshold=args.threshold
                                                )

    # input_names = ["track_apps", "track_bboxs", "track_masks", "track_frames",
    #                "det_app", "det_bbox", "det_mask", "det_frame", "track_num", "det_num",
    #                "labels", "impossible_mask"]

    f = h5py.File(args.hdf5_path, mode="w")

    grp = f.create_group("static_dataset")

    tracklet_num_avg_metric = AvgMetric()
    det_num_avg_metric = AvgMetric()

    cumsum = 0

    for (i, sample) in tqdm.tqdm(enumerate(simple_dataset)):
        track_num = sample["track_num"]
        det_num = sample["det_num"]

        if det_num < 1:
            continue

        tracklet_num_avg_metric.update(track_num)
        det_num_avg_metric.update(det_num)

        sub_grp = f.create_group("sub{}".format(cumsum))
        sub_grp["track_num"] = sample["track_num"]
        sub_grp["det_num"] = sample["det_num"]
        sub_grp["labels"] = np.asarray(sample["labels"], dtype=np.int32)
        sub_grp["impossible_mask"] = np.asarray(sample["impossible_mask"], dtype=np.bool)

        for key in sample["tracks"]:
            sub_grp["track_{}".format(key)] = sample["tracks"][key]

        for key in sample["detections"]:
            sub_grp["det_{}".format(key)] = sample["detections"][key]
        grp["{}".format(cumsum)] = sub_grp

        cumsum += 1

    print('total track num: {}, avg track num per graph: {}, max track num: {}'.format(tracklet_num_avg_metric.total,
                                                                                       tracklet_num_avg_metric.avg,
                                                                                       tracklet_num_avg_metric.max))

    print('total det num: {}, avg det num per graph: {}, max det num: {}'.format(det_num_avg_metric.total,
                                                                                 det_num_avg_metric.avg,
                                                                                 det_num_avg_metric.max))
