import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from datasets.warp_tracklet_and_detection_tensor import *
from models import StrictSimpleCaseTracker
import torch


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

    final_res = {}

    final_res.update(batch_tracks)
    final_res.update(batch_dets)
    final_res.update(
        {
            "det_mask": batch_det_mask,
            "track_num": torch.from_numpy(np.asarray(batch_track_nums, dtype=np.int32)),
            "det_num": torch.from_numpy(np.asarray(batch_det_nums, dtype=np.int32))
        }
    )
    return final_res


class InferenceMachine:
    def __init__(self, model, track_len, output_dir,  app_feature_dim, pos_feature_dim,
                 match_threshold=0.4, debug=False, drop_simple_case=False, cache_window=20,
                 impossible_threshold=3.0
                 ):
        
        self.model = model
        self.model.eval()
        
        self.cache_window = cache_window
        if drop_simple_case:
            print('Init impossible tracker')
            self.drop_simple_case = drop_simple_case
            self.impossible_tracker = StrictSimpleCaseTracker.ImpossibleTracker(impossible_threshold)

        self.debug = debug

        self.app_feature_dim = app_feature_dim
        self.pos_feature_dim = pos_feature_dim

        self.output_dir = output_dir
        self.track_his_len = track_len
        self.match_threshold = match_threshold

        self.track_dict = defaultdict(list)

    def query_tracked_tracklets(self, sampled_frames, cache_window=20):
        tracklets = defaultdict(dict)
        for f in sampled_frames:
            one_frame_tracks = self.track_dict[f]
            for track in one_frame_tracks:
                track_id = track["track_id"]
                tracklets[track_id][f] = track
        
        total_tracklets_ids = list(tracklets.keys())
        
        filled_tracklets = []
        for track_id in total_tracklets_ids:
            filled_tracklets.append(
                convet_tracklet_to_tensor(tracklets[track_id], sampled_frames, cache_window, track_id=track_id)
            )

        return filled_tracklets, total_tracklets_ids

    def query_detections(self, video_name, memory_bank, sampled_frame, video_width, video_height, start_frame):

        track_ids = memory_bank.get_track_ids_in_frame(video_name, sampled_frame)
        detections = defaultdict()

        for track_id, det_idx in track_ids:
            detection = memory_bank.query_detection(video_name, track_id, det_idx)
            detection["track_id"] = -1
            detection["frame"] = sampled_frame - start_frame
            detection = normalize_det_xywh_with_video_wh(detection, video_width, video_height)
            detections[track_id] = detection

        detection_ids = sorted(list(detections.keys()))
        detections = [detections[id] for id in detection_ids]
        cp_detections = copy.deepcopy(detections)
        return cp_detections, detection_ids

    def match(self, to_match_pair):

        tracked_tracklets = to_match_pair["tracks"]
        detections = to_match_pair["detections"]

        tracklets_tensors = merge_tracklet_into_tensors(tracked_tracklets)
        detections_tensor = convert_detections_to_tensor(detections)
        
        det_inds = list(range(len(detections)))
        track_inds = list(range(len(tracked_tracklets)))

        matched_det_inds = []
        matched_track_inds = []

        track_num = len(tracked_tracklets)
        det_num = len(detections)
        
        if self.drop_simple_case:
            impossible_mask = self.impossible_tracker.forward(tracklets_tensors, detections_tensor,
                                                          sample_window_size=self.track_his_len)
        else:
            impossible_mask = None
        
        sample = {
                    "track_num": track_num,
                    "det_num": det_num,
                    "tracks": tracklets_tensors,
                    "detections": detections_tensor
                }
        
        to_match_pair = collate_fn([sample])
        
        to_match_pair.update(
            {
                "impossible_mask": impossible_mask
            }
        )
        
        res = self.model(to_match_pair, stage="test")
        match_matrix = res["match_matrix"][0]
        det_num = res["det_num"][0]
        track_num = res["track_num"][0]
        match_matrix = match_matrix[:det_num, :track_num]
        match_matrix = match_matrix.flatten(end_dim=1)
        final_match_matrix = torch.softmax(match_matrix, dim=1)[:, 1].reshape(det_num, track_num).detach().cpu().numpy()
        
        cost_matrix = 1 - final_match_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for row, col in zip(row_ind, col_ind):
            if final_match_matrix[row, col] > self.match_threshold:
                matched_det_inds.append(det_inds[row])
                matched_track_inds.append(track_inds[col])
        return matched_det_inds, matched_track_inds

    @torch.no_grad()
    def track_one_video(self, video_name, memory_bank, debug_start_frame=None):
        self.track_dict.clear()

        sorted_video_frames = sorted(list(memory_bank.video_frames[video_name]))

        global_track_id = 1
        video_width, video_height = memory_bank.get_video_width_and_height(video_name)

        start_track_idx = 0
        tracked_tracklets = []
        tracked_track_ids = []
        
        for f in tqdm(sorted_video_frames[start_track_idx:]):
            detections, detection_ids = self.query_detections(video_name, memory_bank, f, video_width, video_height,
                                                              f - self.track_his_len)
            
            if len(tracked_tracklets) == 0 or len(detections) == 0:
                pass
            else:
                to_match_pair = {"tracks": tracked_tracklets, "detections": detections}
                matched_det_inds, matched_track_inds = self.match(to_match_pair)
                
                for det_ind, track_ind in zip(matched_det_inds, matched_track_inds):
                    matched_track_id = tracked_track_ids[track_ind]
                    detections[det_ind]["track_id"] = matched_track_id

            for detection in detections:
                if detection["track_id"] == -1:
                    detection["track_id"] = global_track_id
                    global_track_id += 1
                self.track_dict[f].append(detection)

            sampled_frames = list(range(f + 1 - self.track_his_len, f + 1))
            tracked_tracklets, tracked_track_ids = self.query_tracked_tracklets(sampled_frames,
                                                                                cache_window=self.cache_window)

        output_json_file = os.path.join(self.output_dir, "{}.txt".format(video_name))
        self.save_track_res_in_txt(output_json_file, video_width, video_height)
        self.track_dict.clear()

    def save_track_res_in_txt(self, txt_file, video_width, video_height):
        txt_to_write = []
        frames = sorted(self.track_dict.keys())

        for f in frames:
            detections = self.track_dict[f]

            for detection in detections:
                track_id = detection["track_id"]
                x, y, w, h = detection["bbox"]
                x, y, w, h = x * video_width, y * video_height, w * video_width, h * video_height
                txt_to_write.append("{},{},{},{},{},{},-1,-1,-1,-1\n".format(f, track_id, x, y, w, h))

        with open(txt_file, 'w') as f2:
            f2.writelines(txt_to_write)

    def save_track_res_in_json(self, json_file, video_width, video_height):
        res = {}
        frames = sorted(self.track_dict.keys())

        for f in frames:
            detections = self.track_dict[f]

            for detection in detections:
                track_id = detection["track_id"]
                x, y, w, h = detection["bbox"]
                x, y, w, h = x * video_width, y * video_height, w * video_width, h * video_height

                res.setdefault(f, []).append([[x, y, w, h], track_id])

        with open(json_file, 'w') as f:
            json.dump(res, f)
