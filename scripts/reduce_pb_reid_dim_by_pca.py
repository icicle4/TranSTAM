import os
import tqdm
import numpy as np
import json
import argparse
from imutils.paths import list_files
from sklearn.decomposition import PCA
from itertools import groupby

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "../proto"))

import detection_results_pb2


def load_pb(det_bbox_pb_file):
    detections_pb = detection_results_pb2.Detections()
    with open(det_bbox_pb_file, 'rb') as f:
        detections_pb.ParseFromString(f.read())
    return detections_pb


def reduce_reid_dim_multi_file(detection_pbs, out_pb_files, old_dim, new_dim):
    N = sum(len(detection_pb.tracked_detections) for detection_pb in detection_pbs)
    all_features = np.zeros((N, old_dim))
    
    accum = 0
    print("Reading all features IN")
    for detection_pb in detection_pbs:
        for i, detection in enumerate(tqdm.tqdm(detection_pb.tracked_detections)):
            feat = np.asarray([d for d in detection.features.features[0].feats], dtype=np.float32)
            all_features[accum] = feat
            accum += 1
    
    pca = PCA(n_components=new_dim, whiten=False)
    pca.fit(all_features)
    
    del all_features
    
    for detection_pb, out_pb_file in zip(detection_pbs, out_pb_files):
        new_detection_pb = detection_results_pb2.Detections()
        det_num = len(detection_pb.tracked_detections)
        old_features = np.zeros((det_num, old_dim))

        print('gen old feature')
        for i, detection in enumerate(tqdm.tqdm(detection_pb.tracked_detections)):
            feat = np.asarray([d for d in detection.features.features[0].feats], dtype=np.float32)
            old_features[i] = feat
        
        features_pca = pca.transform(old_features)
        
        print("Saving out {}".format(out_pb_file))
        for i, detection in enumerate(tqdm.tqdm(detection_pb.tracked_detections)):
            _detection = new_detection_pb.tracked_detections.add()
            _detection.frame_index = int(detection.frame_index)
            _detection.detection_id = detection.detection_id
            
            x = detection.box_x
            y = detection.box_y
            w = detection.box_width
            h = detection.box_height
            
            _detection.box_x = x
            _detection.box_y = y
            _detection.box_width = w
            _detection.box_height = h
            
            tf = _detection.features.features.add()
            
            feat = features_pca[i].tolist()
            for d in feat:
                tf.feats.append(d)
    
        with open(out_pb_file, 'wb') as f:
            f.write(new_detection_pb.SerializeToString())


def reduce_reid_dim(detection_pb, out_pb_file, old_dim, new_dim):
    N = len(detection_pb.tracked_detections)
    all_features = np.zeros((N, old_dim))

    for i, detection in enumerate(tqdm.tqdm(detection_pb.tracked_detections)):
        feat = np.asarray([d for d in detection.features.features[0].feats], dtype=np.float32)
        all_features[i] = feat

    pca = PCA(n_components=new_dim, whiten=False)
    pca.fit(all_features)
    features_pca = pca.transform(all_features)

    new_detection_pb = detection_results_pb2.Detections()

    for i, detection in enumerate(tqdm.tqdm(detection_pb.tracked_detections)):
        _detection = new_detection_pb.tracked_detections.add()
        _detection.frame_index = int(detection.frame_index)
        _detection.detection_id = detection.detection_id

        x = detection.box_x
        y = detection.box_y
        w = detection.box_width
        h = detection.box_height

        _detection.box_x = x
        _detection.box_y = y
        _detection.box_width = w
        _detection.box_height = h

        tf = _detection.features.features.add()

        feat = features_pca[i].tolist()
        for d in feat:
            tf.feats.append(d)

    with open(out_pb_file, 'wb') as f:
        f.write(new_detection_pb.SerializeToString())


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_det_box_dir", type=str, help="dir of det bboxes pb files")
    parser.add_argument("--new_det_box_dir", type=str, help="dir of det bboxes pb files")
    parser.add_argument("--old_reid_dim", type=int)
    parser.add_argument("--new_reid_dim", type=int)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.new_det_box_dir):
        os.makedirs(args.new_det_box_dir)

    pb_files = list(list_files(args.old_det_box_dir, validExts=".pb"))

    pb_files = sorted(list(list_files(pb_files)))

    for key, value in groupby(pb_files, key=lambda x: os.path.basename(x)[:8]
                                            if "_" in os.path.basename(x)
                                            else os.path.basename(x).split('.')[0]):
        
        sub_pb_files = list(value)
        
        print('files', sub_pb_files)
        new_pb_files = [
            os.path.join(args.new_det_box_dir, "{}.pb".format(
                os.path.basename(pb_file).split('.')[0]
            )) for pb_file in sub_pb_files
        ]
        
        detection_pbs = [load_pb(pb_file) for pb_file in sub_pb_files]
        
        reduce_reid_dim_multi_file(detection_pbs, new_pb_files, args.old_reid_dim, args.new_reid_dim)
