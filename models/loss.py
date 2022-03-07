# copy from torchreid


from __future__ import division, absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from collections import defaultdict



@torch.no_grad()
def top1_correct_num(output, target, topk=(1,)):
    if target.numel() == 0:
        return torch.zeros([], device=output.device)
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0)
    return correct_k

@torch.no_grad()
def cal_precison_and_recall(predict, label):
    label_non_neg_index = torch.nonzero(label != -1)
    predict_non_neg_index = torch.nonzero(predict != -1)
    recall = torch.sum(torch.eq(label[label_non_neg_index], predict[label_non_neg_index])) / (len(label_non_neg_index) + 1e-5)
    precision = torch.sum(torch.eq(label[predict_non_neg_index], predict[predict_non_neg_index])) / (len(predict_non_neg_index) + 1e-5)
    return precision, recall


@torch.no_grad()
def hungarian_correct_match_num(match_matrix, labels):
    match_matrix = match_matrix.detach().clone().cpu().numpy()
    cost_matrix = 1 - match_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    diff_threshold = {}

    for threshold in [0, 2, 4, 6]:
        predict_labels = -torch.ones(len(labels)).cuda()

        for pair in zip(row_ind, col_ind):
            row, col = pair

            if match_matrix[row, col] > (threshold * 0.1):
                predict_labels[row] = col

        precision, recall = cal_precison_and_recall(predict_labels, labels.cuda())

        diff_threshold['@{}_recall'.format(threshold)] = recall
        diff_threshold['@{}_precision'.format(threshold)] = precision
    return diff_threshold


def remove_cuda(to_remove_tensor):
    if to_remove_tensor is not None:
        return to_remove_tensor.detach().cpu().numpy()
    else:
        return None



class PosNegBalanceListCriterion(nn.Module):
    def __init__(self):
        super(PosNegBalanceListCriterion, self).__init__()
        self.num_classes = 1
        
        # empty_weight = torch.ones(self.num_classes + 1, device="cuda")
        # empty_weight[0] = 0.1
        # self.register_buffer('empty_weight', empty_weight)
        
    
    def forward(self, outputs, targets):
        match_matrixs = outputs['match_matrix']
        impossible_mask = outputs["impossible_mask"]
        batch_size = len(match_matrixs)
        
        batch_diff_threshold = defaultdict(int)
        
        all_batch_predicts = []
        all_batch_labels = []
        
        for b in range(batch_size):
            output = match_matrixs[b]
            cur_det_num, cur_track_num = output.size()[:2]
            
            bs_detection_labels = targets[b]
            target_heatmap = np.zeros((cur_det_num, cur_track_num), dtype=np.int)
            
            for i, label in enumerate(bs_detection_labels):
                if label != -1:
                    target_heatmap[i, label] = 1

            b_impossible_mask = torch.flatten(impossible_mask[b][:cur_det_num, :cur_track_num])
            select_index = torch.where(b_impossible_mask)
            sampled_features = torch.flatten(output, end_dim=-2)[select_index[0]]
            sampled_labels = torch.flatten(torch.from_numpy(target_heatmap).cuda(), end_dim=-1)[select_index[0]]
            
            all_batch_predicts.append(sampled_features)
            all_batch_labels.append(sampled_labels)
        
        all_batch_predicts = torch.cat(all_batch_predicts, dim=0)
        all_batch_labels = torch.cat(all_batch_labels, dim=0)
        
        pos_index = torch.where(all_batch_labels == 1)
        neg_index = torch.where(all_batch_labels == 0)

        if len(pos_index[0]) > len(neg_index[0]):
            neg_batch_predicts = all_batch_predicts[neg_index[0]]
            pos_batch_predicts = all_batch_predicts[pos_index[0][:max(len(neg_index[0]), 1)]]
            neg_labels = all_batch_labels[neg_index[0]]
            pos_labels = all_batch_labels[pos_index[0][:max(len(neg_index[0]), 1)]]
        else:
            neg_batch_predicts = all_batch_predicts[neg_index[0][:max(len(pos_index[0]), 1)]]
            pos_batch_predicts = all_batch_predicts[pos_index[0]]
            neg_labels = all_batch_labels[neg_index[0][:max(len(pos_index[0]), 1)]]
            pos_labels = all_batch_labels[pos_index[0]]

        all_batch_predicts = torch.cat([pos_batch_predicts, neg_batch_predicts], dim=0)
        all_batch_labels = torch.cat([pos_labels, neg_labels], dim=0)

        correct_num = top1_correct_num(torch.softmax(all_batch_predicts, dim=1), all_batch_labels)
        
        loss = F.cross_entropy(all_batch_predicts, all_batch_labels)
        
        losses = {'loss_ce': loss / batch_size}
        
        accuracy_dict = {
            "precision": correct_num / (len(all_batch_labels) + 1e-5),
        }
        
        for key in batch_diff_threshold:
            accuracy_dict[key] = batch_diff_threshold[key] / batch_size
        
        return losses, accuracy_dict


class SetCriterion(nn.Module):
    def __init__(self):
        super(SetCriterion, self).__init__()

        self.num_classes = 1
        empty_weight = torch.ones(self.num_classes + 1, device="cuda")
        empty_weight[0] = 0.05
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        match_matrix = outputs["match_matrix"]
        batch_track_num = outputs["track_num"]
        batch_det_num = outputs["det_num"]
        batch_size = match_matrix.size()[0]

        loss = torch.FloatTensor([0.]).cuda()[0]
        correct_num = torch.FloatTensor([0.]).cuda()[0]
        detection_mum = torch.FloatTensor([0.]).cuda()[0]

        batch_diff_threshold = defaultdict(int)
        real_det_num = 0.
        real_track_num = 0.

        valid_batch_size = 0

        for b in range(batch_size):
            output = match_matrix[b]
            cur_det_num = batch_det_num[b]
            cur_track_num = batch_track_num[b]

            bs_detection_labels = targets[b]
            target_heatmap = torch.zeros((cur_det_num, cur_track_num), device="cuda", dtype=torch.long)

            for i, label in enumerate(bs_detection_labels):
                if label != -1:
                    target_heatmap[i, label] = 1

            valid_match_matirx = output[:cur_det_num, :cur_track_num]

            target_heatmap = target_heatmap.view(-1)
            valid_match_matirx = valid_match_matirx.flatten(end_dim=1)

            soft_match_matrix = torch.softmax(valid_match_matirx, dim=1)

            reshaped_soft_match_matrix = soft_match_matrix.clone().reshape((cur_det_num, cur_track_num, -1))
            diff_threshold = hungarian_correct_match_num(reshaped_soft_match_matrix[:, :, 1], bs_detection_labels)
            for key in diff_threshold:
                batch_diff_threshold[key] += diff_threshold[key]

            detection_mum += cur_det_num * cur_track_num
            real_det_num += cur_det_num
            real_track_num += cur_track_num

            correct_num += top1_correct_num(soft_match_matrix, target_heatmap)

            if target_heatmap.numel() == 0:
                pass
            else:
                loss += F.cross_entropy(valid_match_matirx, target_heatmap.flatten(), weight=self.empty_weight)
                valid_batch_size += 1

        valid_batch_size = max(valid_batch_size, 1)
        losses = {'loss_ce': loss / valid_batch_size}

        accuracy_dict = {
            "precision": correct_num / (detection_mum + 1e-5),
        }

        for key in batch_diff_threshold:
            accuracy_dict[key] = batch_diff_threshold[key] / valid_batch_size

        return losses, accuracy_dict

