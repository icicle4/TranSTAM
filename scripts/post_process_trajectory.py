import argparse
import os

import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from shutil import copy

def post_process_track_and_save(txt_file, post_processed_txt_file,  min_track_length_filter=2):
    track_dict = defaultdict(list)
    with open(txt_file, 'r+') as f1:
        for line in f1:
            infos = line.rstrip()
            infos = infos.split(',')
            frame = int(infos[0])
            tid = int(round(float(infos[1])))
            bbx = [float(infos[2]), float(infos[3]), float(infos[4]), float(infos[5])]

            track_dict[tid].append(
                [frame, bbx]
            )

    smoothed_tracklet = post_processing_trajectory_smoothness(track_dict, min_track_length_filter=min_track_length_filter)

    track_infos = []
    for frame, bbxs in smoothed_tracklet.items():
        bbxs = sorted(bbxs, key=lambda x: x[1])
        for bbx in bbxs:
            track_id = bbx[1]
            bb_left = bbx[0][0]
            bb_top = bbx[0][1]
            bb_width = bbx[0][2]
            bb_height = bbx[0][3]
            track_infos.append(
                f"{frame},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},-1,-1,-1,-1\n"
            )

    with open(post_processed_txt_file, 'w') as f1:
        f1.writelines(track_infos)

def post_processing_trajectory_smoothness(track_res, min_track_length_filter=2):
    def interpolate_boxes(frames, boxes, target_frames):
        uniq_idx = [0] + [i for i in range(1, len(frames)) if frames[i] > frames[i - 1]]
        uniq_frames = [frames[i] for i in uniq_idx]
        if len(uniq_frames) == 1:
            return target_frames, boxes[uniq_idx]

        ret = []
        for i in range(boxes.shape[1]):
            f = interp1d(uniq_frames, boxes[uniq_idx, i], assume_sorted=True)
            ret.append(f(target_frames))
        return target_frames, np.array(ret).T

    track_res_stats = {}
    for track_id, item in track_res.items():
        frames = []
        bboxes = []

        for frame_bbox in item:
            frame, bbox = frame_bbox

            frames.append(frame)
            bboxes.append(bbox)

        track_res_stats.setdefault(track_id, {})
        track_res_stats[track_id]["frames"] = frames
        track_res_stats[track_id]["boxes"] = bboxes

    track_res_smoothness = {}
    for track_id in track_res_stats:
        frames = track_res_stats[track_id]['frames']
        assert sorted(frames) == frames
        if len(frames) < min_track_length_filter:
            continue
        boxes = track_res_stats[track_id]['boxes']
        sampled_frames = [ii for ii in range(min(frames), max(frames) + 1)]
        tframes, tboxes = interpolate_boxes(np.array(frames), np.array(boxes), sampled_frames)
        valid = [i for i, box in enumerate(tboxes) if not np.any(np.isnan(box))]
        tframes = [tframes[i] for i in valid]
        tboxes = [tboxes[i].tolist() for i in valid]
        if len(tframes) < min_track_length_filter:
            continue
        for ii, frame in enumerate(tframes):
            track_res_smoothness.setdefault(frame, []).append([tboxes[ii], track_id])
    return track_res_smoothness


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_dir", type=str, required=True, help="predict directory")
    parser.add_argument('--out_path', type=str, required=True, help="output directory")
    parser.add_argument("--min_track_length_filter", type=int, default=2, help="maximum tracklet frame gap can be tolerate")
    parser.add_argument("--not_include_move_camera", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    detectors = ['SDP', 'DPM', 'FRCNN']
    static_camera_ids = ['01', '02', '03', '04', '08', '09']
    move_camera_ids = ['05', '06', '07', '10', '11', '12', '13', '14']

    static_cameras = ['MOT17-{}-{}'.format(static_camera_id, detector) for detector in detectors for static_camera_id in
                      static_camera_ids]
    move_cameras = ['MOT17-{}-{}'.format(move_camera_id, detector) for detector in detectors for move_camera_id in
                    move_camera_ids]

    for result_txt in os.listdir(args.predict_dir):
        if result_txt.endswith(".txt"):
            video_name = os.path.basename(result_txt).split('.')[0]

            output_txt_file = os.path.join(args.out_path, result_txt)
            predict_txt_file = os.path.join(args.predict_dir, result_txt)

            if video_name in move_cameras and args.not_include_move_camera == 1:
                copy(predict_txt_file, output_txt_file)
            elif "MOT20" in video_name:
                post_process_track_and_save(predict_txt_file, output_txt_file, args.min_track_length_filter)
            else:
                post_process_track_and_save(predict_txt_file, output_txt_file, args.min_track_length_filter)
