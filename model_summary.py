import torch
from torchvision import models
from torchsummary import summary
import argparse

from models.model_factory import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Temporal Enhance Temporal', add_help=False)
    
    parser.add_argument('--model', default="trans_stam", type=str)
    
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
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument('--with_abs_pe', type=str, default='with_abs_pe')
    parser.add_argument('--with_relative_pe', type=str, default='with_relative_pe')
    parser.add_argument('--with_assignment_pe', type=str, default='with_assignment_pe')
    parser.add_argument('--aspe_style', type=str, default="diff")
    
    return parser


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('TransSTAM inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    trans_stam_model, criterion = build_model(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trans_stam_model.to(device)

    """
            :param batch_track_app: B, M, T, C
            :param batch_track_bbox: B, M, T, 4
            :param batch_track_frames: B, M, T
            :param batch_detection_app: B, N, C
            :param batch_detection_bbox: B, N, 4
            :return:
    """
    
    B = 1
    T = 20
    track_num = 40
    det_num = 30
    C = 256

    summary(model, [(B, track_num, T, C),
                    (B, track_num, T, 4),
                    (B, track_num, T),
                    (B, det_num, C),
                    (B, det_num, 4)])
