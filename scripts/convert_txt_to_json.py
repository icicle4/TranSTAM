import json
import argparse
import os


def convert_txt_to_json(txt_file, json_file):
    res = {}
    with open(txt_file, 'r+') as f1:
        for line in f1:
            infos = line.rstrip()
            infos = infos.split(',')
            frame = int(infos[0])
            tid = int(round(float(infos[1])))
            bbx = [float(infos[2]), float(infos[3]), float(infos[4]), float(infos[5])]
            res.setdefault(frame, []).append([bbx, tid])

    with open(json_file, 'w') as f2:
        json.dump(res, f2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_dir", type=str, required=True, help="predict txt directory")
    parser.add_argument('--out_path', type=str, required=True, help="output json files directory")
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    for result_txt in os.listdir(args.predict_dir):
        if result_txt.endswith(".txt"):
            video_name = os.path.basename(result_txt).split('.')[0]
            output_json_file = os.path.join(args.out_path, video_name + ".mp4.final.reduced.json")
            convert_txt_to_json(os.path.join(args.predict_dir, result_txt), output_json_file)
