import torch
import random
import torch.nn.functional as F
from torchvision.ops import box_iou


def track_edge_matrix_by_reid(batch_track_app):
    track_reid_cosine = torch.einsum("bmxc, bmyc->bmxy",
                                     batch_track_app,
                                     batch_track_app
                                     )[..., None]
    return (track_reid_cosine + 1) / 2


def track_edge_matrix_by_spt(batch_track_bbox, batch_track_frames, history_window_size=50):
    """
    :param batch_track_bbox: B, M, T, 4 (x, y, w, h)
    :return:
    """
    B, M, T, _ = batch_track_bbox.size()
    batch_track_xy = batch_track_bbox[:, :, :, :2]
    batch_track_wh = batch_track_bbox[:, :, :, 2:]
    batch_track_t = batch_track_frames[:, :, :, None]

    batch_track_diff_t =  1 - torch.abs(batch_track_t[:, :, :, None, :].expand(-1, -1, -1, T, -1) - batch_track_t[:, :, None, :, :].expand(-1, -1, T, -1, -1)) / history_window_size
    batch_track_diff_xy = 1 - torch.abs(batch_track_xy[:, :, :, None, :].expand(-1, -1, -1, T, -1) - batch_track_xy[:, :, None, :, :].expand(-1, -1, T, -1, -1))
    batch_track_diff_wh = 1 - torch.abs(batch_track_wh[:, :, :, None, :].expand(-1, -1, -1, T, -1) - (batch_track_wh[:, :, None, :, :].expand(-1, -1, T, -1, -1)))

    # B, M, T, T, 5
    track_edge_matrix = torch.cat([batch_track_diff_t, batch_track_diff_xy, batch_track_diff_wh], dim=-1)
    return track_edge_matrix


def dup_det_edge_matrix_by_reid(dup_batch_det_app):
    B, N, M, _ = dup_batch_det_app.size()
    dup_batch_det_app = dup_batch_det_app.reshape(B, N * M, -1)
    det_reid_cosine = torch.einsum("bnc, bmc->bnm", dup_batch_det_app, dup_batch_det_app)[..., None]
    return (det_reid_cosine + 1) / 2



def dup_det_edge_matrix_by_spt(dup_batch_det_box):
    B, N, M, _ = dup_batch_det_box.size()
    dup_batch_det_box = dup_batch_det_box.reshape(B, N * M, 4)

    S = N * M

    batch_det_xy = dup_batch_det_box[:, :, :2]
    batch_det_wh = dup_batch_det_box[:, :, 2:]

    batch_track_diff_t = 1 - torch.zeros(B, S, S, 1).float().cuda()
    batch_track_diff_xy = 1 - torch.abs(batch_det_xy[:, :, None, :].expand(-1, -1, S, -1) - batch_det_xy[:, None, :, :].expand(-1, S, -1, -1))
    batch_track_diff_wh = 1 - torch.abs(batch_det_wh[:, :, None, :].expand(-1, -1, S, -1) - (batch_det_wh[:, None, :, :].expand(-1,S,-1,-1)))
    # B, S, S, 5
    det_edge_matrix = torch.cat([batch_track_diff_t, batch_track_diff_xy, batch_track_diff_wh], dim=-1)
    return det_edge_matrix


def det_edge_matrix_by_spt_single_batch(det_bbox):
    '''
    :param det_bbox: N, 4
    :param track_bbox: M, 4
    :param track_frame: M
    :param history_window_size: N, M, 5
    :return:
    '''
    track_bbox = det_bbox
    N, _ = det_bbox.size()
    M, _ = track_bbox.size()
    
    det_xy = det_bbox[:, :2]
    det_wh = det_bbox[:, 2:]
    
    track_xy = track_bbox[:, :2]
    track_wh = track_bbox[:, 2:]
    
    det_track_diff_xy = 1 - torch.abs(det_xy[:, None, :].expand(-1, M, -1) - track_xy[None, :, :].expand(N, -1, -1))
    det_track_diff_wh = 1 - torch.abs(det_wh[:, None, :].expand(-1, M, -1) - track_wh[None, :, :].expand(N, -1, -1))
    det_track_diff_t = torch.ones_like(det_track_diff_xy)[..., :1]
    
    det_track_edge_matrix = torch.cat([det_track_diff_t, det_track_diff_xy, det_track_diff_wh], dim=-1)
    return det_track_edge_matrix


def det_track_edge_matrix_by_spt_single_batch(det_bbox, track_bbox, track_frame, history_window_size=50):
    '''
    :param det_bbox: N, 4
    :param track_bbox: M, 4
    :param track_frame: M
    :param history_window_size: N, M, 5
    :return:
    '''
    N, _ = det_bbox.size()
    M, _ = track_bbox.size()
    
    det_xy = det_bbox[:, :2]
    det_wh = det_bbox[:, 2:]
    
    track_xy = track_bbox[:, :2]
    track_wh = track_bbox[:, 2:]
    
    det_track_diff_xy = 1 - torch.abs(det_xy[:, None, :].expand(-1, M, -1) - track_xy[None, :, :].expand(N, -1, -1))
    det_track_diff_wh = 1 - torch.abs(det_wh[:, None, :].expand(-1, M, -1) - track_wh[None, :, :].expand(N, -1, -1))
    det_track_diff_t = track_frame[None, :, None].expand(N, -1, -1) / history_window_size
    
    det_track_edge_matrix = torch.cat([det_track_diff_t, det_track_diff_xy, det_track_diff_wh], dim=-1)
    return det_track_edge_matrix
    

def det_track_edge_matrix_by_spt(dup_batch_det_bbox, batch_track_bbox, batch_track_frames, history_window_size=50):
    """
    :param batch_det_bbox: B, N * M, 4 (x, y, w, h)
    :param batch_track_bbox: B, M, T, 4 (x, y, w, h)

    :return:
    """
    B, N, _ = dup_batch_det_bbox.size()
    B, M, T, _ = batch_track_bbox.size()

    batch_track_xy = batch_track_bbox[:, :, :, :2].reshape(B, M * T, 2)
    batch_track_wh = batch_track_bbox[:, :, :, 2:].reshape(B, M * T, 2)
    batch_track_t = batch_track_frames[:, :, :, None].reshape(B, M*T, -1).cuda()

    batch_det_xy = dup_batch_det_bbox[:, :, :2]
    batch_det_wh = dup_batch_det_bbox[:, :, 2:]
    batch_det_t = (torch.ones(1) * T)[None, None, :].expand(B, N, -1).cuda()

    batch_det_track_diff_t = 1 - torch.abs(batch_det_t[:, :, None, :].expand(-1, -1, M*T, -1) - batch_track_t[:, None, :, :].expand(-1, N, -1, -1)) / history_window_size
    batch_det_track_diff_xy = 1 - torch.abs(batch_det_xy[:, :, None, :].expand(-1, -1, M*T, -1) - batch_track_xy[:, None, :, :].expand(-1, N, -1, -1))
    batch_det_track_diff_wh = 1 - torch.abs(batch_det_wh[:, :, None, :].expand(-1, -1, M * T, -1) - (batch_track_wh[:, None, :, :].expand(-1, N, -1, -1)))

    # B, N * M, M *T, 5
    det_track_edge_matrix = torch.cat([batch_det_track_diff_t, batch_det_track_diff_xy, batch_det_track_diff_wh], dim=-1)
    return det_track_edge_matrix


def det_track_edge_matrix_by_reid(dup_batch_det_app, batch_track_app):

    """
    :param dup_batch_det_app: B, N * M, C
    :param batch_track_app: B, M, T, C

    :return:
    """
    B, N, _ = dup_batch_det_app.size()
    B, M, T, _ = batch_track_app.size()

    batch_track_app = batch_track_app.reshape(B, M*T, -1)
    det_track_cosine = torch.einsum("bnc, bmc->bnm", dup_batch_det_app, batch_track_app)[..., None]

    return (det_track_cosine + 1) / 2


def det_track_edge_matrix_by_iou(dup_batch_det_bbox, batch_track_bbox):
    
    B, N, _ = dup_batch_det_bbox.size()
    B, M, T, _ = batch_track_bbox.size()
    
    batch_track_bbox[..., 2:] += batch_track_bbox[..., :2]
    dup_batch_det_bbox[..., 2:] += dup_batch_det_bbox[..., :2]
    
    batch_track_bbox = batch_track_bbox.view(B, M * T, -1)
    
    iou_matrixs = []
    for i in range(B):
        iou_matrix = box_iou(dup_batch_det_bbox[i], batch_track_bbox[i])
        iou_matrixs.append(iou_matrix)
    return iou_matrixs


def det_track_edge_matrix_by_cdist(dup_batch_det_bbox, batch_track_bbox):
    B, N, _ = dup_batch_det_bbox.size()
    B, M, T, _ = batch_track_bbox.size()

    batch_track_center = batch_track_bbox[..., 2:] + batch_track_bbox[..., :2] / 2
    dup_batch_det_center = dup_batch_det_bbox[..., 2:] + dup_batch_det_bbox[..., :2] / 2
    batch_track_center = batch_track_center.view(B, M * T, -1)
    return torch.cdist(dup_batch_det_center, batch_track_center, p=2)


def warp_samples_to_equal_tensor(samples):
    """
    :param samples:
    :param device: "cuda"
    :return:
    """
    if 'track_track_apps' in samples.keys():
        keys = ['track_track_apps', 'track_track_bboxs', 'track_track_masks', 'track_num', 'track_track_frames',
                'det_det_app', 'det_det_bbox', 'det_mask', 'det_num', "impossible_mask"]
    else:
        keys = ['track_apps', 'track_bboxs', 'track_masks',   'track_num', 'track_frames',
                'det_app', 'det_bbox', 'det_mask', 'det_num', "impossible_mask"]

    batch_track_app, batch_track_bbox, batch_track_mask, batch_track_num, batch_track_frames, \
        batch_detection_app, batch_detection_bbox, batch_detection_mask, batch_det_num, impossible_mask = [samples[k].cuda() for k in keys]

    return batch_track_app, batch_track_bbox, batch_track_mask, batch_track_num, batch_track_frames, \
           batch_detection_app, batch_detection_bbox, batch_detection_mask, batch_det_num, impossible_mask


def build_attn_mask(source_mask: torch.Tensor, s: int, head_num:int):
    """
    :param source_mask: B, t
    :param s: source size
    :param head_num: multi head attention's head num
    :return: atten_mask: (B * head_num, s, t)

    attn_mask 的 第 i 个位置表示 source 和 target 是否要进行attention操作，对于没有定义的source，只要空值部分不影响loss，
    可以进行attention操作。因此，只需要将 target 中没有定义的部分不与source 进行attention操作即可。
    为了防止 source 对应的 target 全无定义（此时 MHA在生成mask时会产生全为 -inf 的向量，在经过softmax层后会发生NaN值的传染),
    且在我们的情况中 tracklet self atten 时才会有这种情况，在这种情况下，由于source也全无定义，
    因此将无定义行中的target任意一个mask设置为False即可。
    """
    B, t = source_mask.size()
    attn_mask = source_mask.clone().unsqueeze(dim=1).expand(-1, s, -1)
    attn_mask = torch.where(attn_mask > 0, False, True)

    attn_mask = attn_mask.view(B * s, t)
    attn_mask[torch.nonzero(torch.eq(torch.sum(attn_mask, dim=1), t)), 0] = False

    attn_mask = attn_mask.unsqueeze(dim=1).expand(-1, head_num, -1)
    attn_mask = attn_mask.reshape(B * head_num, s, t)
    return attn_mask


def least_tracklet_features(batch_tracks_emb, batch_track_mask, normalize=False):
    """
        :param batch_tracks_emb: (T, B * M, D)
        :param batch_track_mask: (B * M, T)
        :param output: B * M, D
        :return:
    """
    T, valid_num, C = batch_tracks_emb.size()
    
    if batch_track_mask is None:
        batch_track_mask = torch.ones((valid_num, T), device='cuda').long()
    
    tmp_mask = batch_track_mask.permute(1, 0).contiguous()
    
    batch_last_idx = torch.sum(tmp_mask, dim=0) - 1
    ret = batch_tracks_emb[batch_last_idx, torch.arange(batch_tracks_emb.size(1))]
    return ret


def mean_tracklet_features(batch_tracks_emb, batch_track_mask, normalize=False):
    """
    :param batch_tracks_emb: (T, B * M, D)
    :param batch_track_mask: (B * M, T)
    :param output: B * M, D
    :return:
    """
    
    T, valid_num, C = batch_tracks_emb.size()
    if batch_track_mask is None:
        batch_track_mask = torch.ones((valid_num, T), device='cuda').long()
    
    tmp_mask = batch_track_mask.permute(1, 0).contiguous()
    masked_batch_tracks_emb = torch.einsum("tnd,tn->tnd", batch_tracks_emb, tmp_mask)
    valid_num = torch.sum(tmp_mask, dim=0).unsqueeze(dim=1)
    total_track_meb = torch.sum(masked_batch_tracks_emb, dim=0)
    mean_track_emb = total_track_meb / (valid_num + 1e-9)

    if normalize:
        mean_track_emb = F.normalize(mean_track_emb, dim=1)
    return mean_track_emb