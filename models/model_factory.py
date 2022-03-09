import torch

from models import TranSTAM, TranSTAMCal

from models.loss import SetCriterion, PosNegBalanceListCriterion


def build_model(args):
    criterion = SetCriterion()
    
    if args.model == "tran_stam":
        model = TranSTAM.PEMOT(
            history_window_size=args.track_history_len,
            track_valid_size=args.cache_window_size,
            appearance_feature_dim=args.app_dim,
            pos_feature_dim=args.pos_dim,
            n_heads=args.nheads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            enc_layer_num=args.enc_layer_num,
            dec_layer_num=args.dec_layer_num,
            with_abs_pe=args.with_abs_pe == "with_abs_pe",
            with_relative_pe=args.with_relative_pe == "with_relative_pe",
            with_assignment_pe=args.with_assignment_pe == "with_assignment_pe",
            aspe_style=args.aspe_style
        )
        criterion = PosNegBalanceListCriterion()
    elif args.model == "tran_stam_cal":
        model = TranSTAMCal.PEMOT(
            history_window_size=args.track_history_len,
            track_valid_size=args.cache_window_size,
            appearance_feature_dim=args.app_dim,
            pos_feature_dim=args.pos_dim,
            n_heads=args.nheads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            enc_layer_num=args.enc_layer_num,
            dec_layer_num=args.dec_layer_num,
            with_abs_pe=args.with_abs_pe == "with_abs_pe",
            with_relative_pe=args.with_relative_pe == "with_relative_pe",
            with_assignment_pe=args.with_assignment_pe == "with_assignment_pe",
            aspe_style=args.aspe_style
        )
    else:
        raise NotImplementedError("This {} model not implement now!".format(args.model))
    return model, criterion
