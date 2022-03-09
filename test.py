import argparse
import datetime
import random
import os
import time
from pathlib import Path

import numpy as np
import torch
from itertools import groupby

from utils import misc as utils
from imutils.paths import list_files

from models import model_factory

from datasets.memmory_bank_pb import MemoryBank
from models.inference import InferenceMachine


def get_args_parser():
    parser = argparse.ArgumentParser('Temporal Enhance Temporal', add_help=False)
    
    parser.add_argument('--model', default="temporal", type=str, help="[temporal / test]")
    
    parser.add_argument('--app_dim', default=256, type=int, help="Dimension of Appearance Feature")
    parser.add_argument('--pos_dim', default=4, type=int, help="Dimension of Pos Feature")
    
    # * Transformer
    parser.add_argument('--track_history_len', default=50, type=int,
                        help="Number of max history lenght of tracklets, equal to sample tracklet region length")
    parser.add_argument('--cache_window_size', default=20, type=int)
    
    parser.add_argument('--enc_layer_num', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layer_num', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    
    # dataset parameters
    parser.add_argument("--to_inference_pb_dir", required=True, type=str, help="Dir of to inferenced pb files")
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    parser.add_argument('--num_workers', default=2, type=int)
    
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--match_threshold', default=0.4, type=float)
    
    parser.add_argument("--drop_simple_case", action="store_true", default=False)
    parser.add_argument('--with_abs_pe', type=str, default='with_abs_pe')
    parser.add_argument('--with_relative_pe', type=str, default='with_relative_pe')
    parser.add_argument('--with_assignment_pe', type=str, default='with_assignment_pe')
    parser.add_argument('--aspe_style', type=str, default="diff")

    parser.add_argument('--impossible_threshold', type=float, default=3.0)
    return parser


def main(args):
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, criterion = model_factory.build_model(args)
    model.to(device)
    
    if args.resume:
        print('args.resume', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        
        print('loading pretrained model from {}'.format(args.resume))
    
    inference_machine = InferenceMachine(model,
                                         track_len=args.track_history_len,
                                         output_dir=args.output_dir,
                                         app_feature_dim=args.app_dim,
                                         pos_feature_dim=args.pos_dim,
                                         match_threshold=args.match_threshold,
                                         drop_simple_case=args.drop_simple_case,
                                         cache_window=args.cache_window_size,
                                         impossible_threshold=args.impossible_threshold
                                         )
    
    pb_files = sorted(list(list_files(args.to_inference_pb_dir)))
    
    start_time = time.time()
    for key, value in groupby(pb_files, key=lambda x: os.path.basename(x)[:8]
                                            if "_" in os.path.basename(x)
                                            else os.path.basename(x).split('.')[0]):
        
        sub_pb_files = list(value)
        
        if "_" in os.path.basename(key):
            video_name = os.path.basename(key)[:8]
        else:
            video_name = os.path.basename(key).split('.')[0]
        
        memory_bank = MemoryBank(sub_pb_files)
        inference_machine.track_one_video(video_name, memory_bank)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('Test time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TranSTAM inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
