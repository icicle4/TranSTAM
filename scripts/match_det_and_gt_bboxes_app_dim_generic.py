import argparse

import os
import sys


sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../proto/"))

import detection_results_pb2

from utils.matcher import HungarianMatcher
from bisect import bisect_left
import copy
import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_box_dir", type=str, help="dir of det bboxes pb files")
    parser.add_argument("--dataset_dir", type=str, help="dir of MOT dataset")
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str, help="dir of res bboxes pb files")
    return parser

def find_video_name(pb_file_name):
    return os.path.basename(pb_file_name)[:8]


def find_part_pbs(video_name, pb_file_names):
    part_pb_names = [pb_file_name for pb_file_name in pb_file_names if video_name in pb_file_name]
    return sorted(part_pb_names)


def load_pb(det_bbox_pb_file):
    detections_pb = detection_results_pb2.Detections()
    with open(det_bbox_pb_file, 'rb') as f:
        detections_pb.ParseFromString(f.read())
    return detections_pb


def load_gt_txt(txt_file):
    res = {}
    with open(txt_file, 'r+') as f1:
        for line in f1:
            infos = line.rstrip()
            infos = infos.split(',')
            frame = int(infos[0])
            track_id = int(infos[1])
            bbx = [float(infos[2]), float(infos[3]), float(infos[4]), float(infos[5])]
            res.setdefault(frame, []).append(
                {
                    "bbox": bbx,
                    "track_id": track_id
                }
            )
    return res


def match_track_id_by_iou(det_res, gt_res, matcher, part_detection_pbs):
    
    detection_nums = [len(part_detection_pb.tracked_detections) for part_detection_pb in part_detection_pbs]
    
    cumsums = [-1, ]
    
    for detection_num in detection_nums:
        cumsums.append(detection_num + cumsums[-1])
    
    for frame in tqdm.tqdm(det_res.keys()):
        det_bboxes = det_res[frame]
        gt_bboxes = gt_res[frame]
        matched_pair = matcher(gt_bboxes, det_bboxes)

        for gt_index, det_index in matched_pair:
            print('match det-{} & gt-{} in frame {}, det bbox is {}, gt bbox is {}'.format(
                det_bboxes[det_index]['detection_id'],
                gt_bboxes[gt_index]['track_id'],
                frame,
                det_bboxes[det_index]['bbox'],
                gt_bboxes[gt_index]['bbox']
            ))
            det_bboxes[det_index]["track_id"] = gt_bboxes[gt_index]["track_id"]

        for det in det_bboxes:
            detect_index = det["detect_index"]
            idx = bisect_left(cumsums, detect_index) - 1
            part_detection_pbs[idx].tracked_detections[detect_index - cumsums[idx] - 1].detection_id = det["track_id"]
            
    return part_detection_pbs


def load_det_res(part_detection_files):
    det_res = {}
    cumsum = 0
    
    part_detection_pbs = []
    
    for part_detection_file in part_detection_files:
        part_detection_pb = load_pb(part_detection_file)
        part_detection_pbs.append(copy.deepcopy(part_detection_pb))
        
        for i, detection in enumerate(part_detection_pb.tracked_detections):
            det_res.setdefault(detection.frame_index, []).append(
                {
                    "bbox": [detection.box_x, detection.box_y, detection.box_width, detection.box_height],
                    "track_id": 0,
                    "detect_index": cumsum,
                    "detection_id": detection.detection_id
                }
            )
            cumsum += 1

    return part_detection_pbs, det_res


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    iou_matcher = HungarianMatcher(iou_threshold=args.iou_threshold)

    print("output dir", args.output_dir)
    if not os.path.exists(args.output_dir):
        print("Making output dir", args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        
    det_box_dir = args.det_box_dir
    
    pb_files = []
    video_names = []
    for pb_file in os.listdir(det_box_dir):
        if pb_file.endswith(".pb"):
            pb_files.append(os.path.join(det_box_dir, pb_file))
            video_names.append(find_video_name(pb_file))
    
    video_names = list(set(video_names))
    
    for video_name in video_names:
        sub_pb_files = find_part_pbs(video_name, pb_files)
        gt_box_file = os.path.join(args.dataset_dir, "train", video_name, 'gt/gt.txt')
    
        if not os.path.exists(gt_box_file):
            print("not exist {}, so not matched these pbs: {}".format(gt_box_file, sub_pb_files))
            continue
            
        part_detection_pbs, det_res = load_det_res(sub_pb_files)
        gt_res = load_gt_txt(gt_box_file)

        part_track_pbs = match_track_id_by_iou(det_res, gt_res,
                                               matcher=iou_matcher, part_detection_pbs=part_detection_pbs)
        
        for sub_pb_file, part_track_pb in zip(sub_pb_files, part_track_pbs):
            with open(os.path.join(args.output_dir, os.path.basename(sub_pb_file)), 'wb') as f:
                f.write(part_track_pb.SerializeToString())
