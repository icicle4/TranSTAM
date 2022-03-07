from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./../'))

import numpy as np
from imutils.paths import list_files
from datasets.LRU import LRUCache
from datasets.video_info import get_video_width_and_height, find_video_name


sys.path.append(os.path.join(os.path.dirname(__file__), "../proto/"))
import detection_results_pb2


def load_pb(det_bbox_pb_file):
    detections_pb = detection_results_pb2.Detections()
    with open(det_bbox_pb_file, 'rb') as f:
        detections_pb.ParseFromString(f.read())
    return detections_pb


def load_ith_det_fea_and_bbox(detections_pb, idx):
    if isinstance(detections_pb, list):
        for sub_detections_pb in detections_pb:
            if idx >= len(sub_detections_pb.tracked_detections):
                idx -= len(sub_detections_pb.tracked_detections)
                continue
            else:
                detection = sub_detections_pb.tracked_detections[idx]
                break
    else:
        detection = detections_pb.tracked_detections[idx]

    feat = np.asarray([d for d in detection.features.features[0].feats], dtype=np.float32)
    x = detection.box_x
    y = detection.box_y
    w = detection.box_width
    h = detection.box_height
    bbox = np.array([x, y, w, h], dtype=np.float32)
    return feat, bbox


class MemoryBank:
    def __init__(self, root_dir_or_single_pb_file):
        self.memory = defaultdict(dict)

        self.root_dir = root_dir_or_single_pb_file

        self.track_id_in_same_frame = defaultdict(list)
        self.track_ids = []
        self.video_names = set()
        self.video_frames = defaultdict(set)
        self.video_infos = {}

        self.memmap_dict = {}
        self.obj_match_index_dict = {}

        self.buff_memmap_dict = LRUCache(capacity=10)

        
        if isinstance(self.root_dir, list):
            self.load_pbs_to_memory_bank(phase="inference")
            self.build_memmap_memory()
        elif os.path.isfile(self.root_dir):
            self.load_pbs_to_memory_bank(phase="inference")
            self.build_memmap_memory()

    def build_memmap_memory(self):
        
        
        if isinstance(self.root_dir, list):
            sub_pb_files = self.root_dir
            if "_" in os.path.basename(sub_pb_files[0]):
                video_name = os.path.basename(sub_pb_files[0])[:8]
            else:
                video_name = os.path.basename(sub_pb_files[0]).split('.')[0]

            detections_pbs = []
            for sub_pb_file in sub_pb_files:
                detections_pb = load_pb(sub_pb_file)
                detections_pbs.append(detections_pb)
            self.memmap_dict[video_name] = detections_pbs
            
        elif os.path.isdir(self.root_dir):
            for video_id in self.video_names:
                pb_file = os.path.join(self.root_dir, "{}.pb".format(video_id))
                self.memmap_dict[video_id] = pb_file
                
        elif os.path.isfile(self.root_dir):
            video_id = os.path.basename(self.root_dir).split('.')[0]
            pb_file = self.root_dir
            detections_pb = load_pb(pb_file)
            self.memmap_dict[video_id] = detections_pb
    def save_in(self, video_id, track_id, image_id, index):
        self.video_names.add(video_id)
        self.video_frames[video_id].add(image_id)

        self.track_id_in_same_frame["{}_{}".format(video_id, image_id)].append([track_id, index])
        self.track_ids.append((video_id, track_id, image_id))

    def query_detection(self, video_id, track_id, index):
        if isinstance(self.memmap_dict.get(video_id), str):
            pb_file_path = self.memmap_dict[video_id]

            if self.buff_memmap_dict.get(pb_file_path) is not None:
                detections_pb = self.buff_memmap_dict.get(pb_file_path)
            else:
                detections_pb = load_pb(pb_file_path)
                self.buff_memmap_dict.put(pb_file_path, detections_pb)
        else:
            detections_pb = self.memmap_dict[video_id]

        fea, bbox = load_ith_det_fea_and_bbox(
            detections_pb, index
        )

        return {
            "app": fea,
            "bbox": bbox
        }

    def get_track_ids_in_frame(self, video_id, image_id):
        return self.track_id_in_same_frame["{}_{}".format(video_id, image_id)]

    # def save_video_info(self, video_name, video_width, video_height):
    #     self.video_infos[video_name] = (video_width, video_height)

    def get_video_width_and_height(self, video_name):
        return get_video_width_and_height(find_video_name(video_name))

    def save_in_memory_bank(self, video_name, detection_pb, stage):
        
        if isinstance(detection_pb, list):
            index = 0
            
            for sub_detection_pb in detection_pb:
                for i, detection in enumerate(sub_detection_pb.tracked_detections):
                    frame_index = detection.frame_index
                    track_id = detection.detection_id
                    if track_id == 0 and stage != "inference":
                        continue
        
                    self.save_in(video_id=video_name, track_id=track_id, image_id=int(frame_index), index=index)
                    
                    index += 1
        else:
            for i, detection in enumerate(detection_pb.tracked_detections):
                frame_index = detection.frame_index
                track_id = detection.detection_id
                if track_id == 0 and stage != "inference":
                    continue
    
                self.save_in(video_id=video_name, track_id=track_id, image_id=int(frame_index), index=i)

    def load_pbs_to_memory_bank(self, phase):
        
        if isinstance(self.root_dir, list):
            sub_pb_files = self.root_dir
            sub_res_pbs = []

            if "_" in os.path.basename(sub_pb_files[0]):
                video_name = os.path.basename(sub_pb_files[0])[:8]
            else:
                video_name = os.path.basename(sub_pb_files[0]).split('.')[0]
            
            for i, sub_pb_file in enumerate(sub_pb_files):
                res_pb = load_pb(sub_pb_file)
                sub_res_pbs.append(res_pb)
    
            self.save_in_memory_bank(video_name, sub_res_pbs, phase)
        elif os.path.isdir(self.root_dir):
            if phase != "inference" and os.path.exists(os.path.join(self.root_dir, phase)):
                res_box_dir = os.path.join(self.root_dir, phase)
            else:
                res_box_dir = self.root_dir

            print('res_box_dir', res_box_dir)
            pb_files = list(list_files(res_box_dir, validExts=".pb"))

            print('total pb file num: {}'.format(len(pb_files)))

            for res_bbox_pb_file in pb_files:
                video_name = os.path.basename(res_bbox_pb_file).split('.')[0]
                res_pb = load_pb(res_bbox_pb_file)

                self.save_in_memory_bank(video_name, res_pb, phase)

        elif os.path.isfile(self.root_dir):
            res_bbox_pb_file = self.root_dir
            video_name = os.path.basename(res_bbox_pb_file).split('.')[0]
            res_pb = load_pb(res_bbox_pb_file)

            self.save_in_memory_bank(video_name, res_pb, phase)
