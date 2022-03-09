import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../proto/"))

import detection_results_pb2

import argparse
import json
from imutils.paths import list_files


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_box_dir", type=str, help="dir of det bboxes pb files")
    parser.add_argument("--output_dir", type=str, help="dir of generate txt files dir")
    return parser


def load_pb(det_bbox_pb_file):
    detections_pb = detection_results_pb2.Detections()
    with open(det_bbox_pb_file, 'rb') as f:
        detections_pb.ParseFromString(f.read())
    return detections_pb


def convert_detections_to_json(detections_pb, json_file):
    
    res = {}

    txt_to_write = []
    for detection in detections_pb.tracked_detections:
        frame_index = detection.frame_index
        track_id = detection.detection_id

        x = detection.box_x
        y = detection.box_y
        h = detection.box_height
        w = detection.box_width
        bbx = [float(x), float(y), float(w), float(h)]
        res.setdefault(frame_index, []).append([bbx, track_id])
        txt_to_write.append("{},{},{},{},{},{},-1,-1,-1,-1\n".format(frame_index, track_id, x, y, w, h))
        
    with open(json_file, 'w') as f2:
        f2.writelines(txt_to_write)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    pb_files = list(list_files(args.det_box_dir, validExts=".pb"))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for pb_file in pb_files:
        detections_pb = load_pb(pb_file)
        pb_name = os.path.basename(pb_file).split('.')[0]
        json_file = os.path.join(args.output_dir, "{}.txt".format(pb_name))
        convert_detections_to_json(detections_pb, json_file)
