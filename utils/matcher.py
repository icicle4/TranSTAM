import numpy as np

from scipy.optimize import linear_sum_assignment


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y + h]


def single_batch_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    w1, h1 = xmax1 - xmin1, ymax1 - ymin1
    w2, h2 = xmax2 - xmin2, ymax2 - ymin2

    cross_x = max((w1 + w2) - (max(xmax2, xmax1) - min(xmin1, xmin2)) , 0)
    cross_y = max((h1 + h2) - (max(ymax2, ymax1) - min(ymin1, ymin2)) , 0)
    inter = cross_x * cross_y
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / (union + 1e-5)


def iou(A,B):
    M, N = len(A), len(B)
    iou_matrix = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = single_batch_iou(A[i], B[j])
    return iou_matrix


class HungarianMatcher:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, gts, detects):
        gt_boxs = [xywh2xyxy(gt["bbox"]) for gt in gts]
        det_boxs = [xywh2xyxy(det["bbox"]) for det in detects]
        iou_matrix = iou(np.asarray(gt_boxs, dtype=np.float32),
                         np.asarray(det_boxs, dtype=np.float32))

        match_cost_matrix = 1 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(match_cost_matrix)

        match_paris = []
        for row, col in zip(row_ind, col_ind):
            if iou_matrix[row, col] > self.iou_threshold:
                match_paris.append(
                    (row, col)
                )

        return match_paris


if __name__ == '__main__':
    A = [[1, 1, 3, 3], [0, 0, 2, 2]]
    B = [[1, 1, 3, 3], [0, 0, 2, 2], [-1, -1, 1, 1]]

    print(iou(A, B))





