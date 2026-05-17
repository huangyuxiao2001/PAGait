import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


from utils import is_tensor


class DynamicWEUC(nn.Module):
    """
    可学习 alpha 的融合距离模块。
    - alpha 为标量可学习参数（可以扩展为 per-dim/per-bin 可学习）
    - forward 接受 x: [N, D], y: [M, D]，返回 [N, M] 距离矩阵（带梯度）
    """
    def __init__(self, init_alpha=0.5):
        super(DynamicWEUC, self).__init__()
        # 可改造成 per-dim 或 小网络输出 alpha(x,y)，这里先用 scalar parameter
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x, y):
        # x: [N, D], y: [M, D]
        # 先归一化（可选）
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        # 欧氏距离
        # 使用 cdist 更高效
        euc = torch.cdist(x, y, p=2)  # [N, M]

        # 余弦距离
        cos_sim = torch.matmul(x_norm, y_norm.t())  # [N, M]
        cos_dist = 1.0 - cos_sim

        alpha = torch.sigmoid(self.alpha)  # 保证 (0,1)
        dist = alpha * euc + (1.0 - alpha) * cos_dist
        return dist


def cuda_dist(x, y, metric='euc', alpha=0.7, hidden_dim=128, dweuc_module=None, detach_result=True):
    """
    兼容接口：可以在训练时传入 dweuc_module 使 alpha 被训练。
    参数:
      - x, y: np.ndarray 或 torch.Tensor，shape=[n, c, num_bin]
      - metric: 'euc'|'cos'|'weuc'|'dweuc'
      - alpha: 固定 weuc 的权重
      - dweuc_module: DynamicWEUC 的实例（如果使用 'dweuc'，传入它以启用学习）
      - detach_result: 是否返回 detach() 的 tensor（评估时 True，训练时 False）
    返回:
      - dist: [n_x, n_y] 的 tensor（位于 x.device）
    """
    # numpy 转 torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to('cuda')
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float().to('cuda')

    device = x.device
    # 对于 cos 相关的 metric 先做归一化（按 channel）
    if metric in ['cos', 'weuc', 'dweuc']:
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
    else:
        x_norm = x
        y_norm = y

    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)

    dist = torch.zeros(n_x, n_y, device=device)

    for i in range(num_bin):
        _x = x[:, :, i]       # [n_x, c]
        _y = y[:, :, i]       # [n_y, c]
        _x_norm = x_norm[:, :, i]
        _y_norm = y_norm[:, :, i]

        if metric == 'cos':
            sim = torch.matmul(_x_norm, _y_norm.t())
            dist += (1.0 - sim)

        elif metric == 'euc':
            # stable euclidean
            _dist = torch.sum(_x**2, dim=1).unsqueeze(1) + torch.sum(_y**2, dim=1).unsqueeze(0) - 2.0 * torch.matmul(_x, _y.t())
            dist += torch.sqrt(F.relu(_dist) + 1e-12)

        elif metric == 'weuc':
            _dist = torch.sum(_x**2, dim=1).unsqueeze(1) + torch.sum(_y**2, dim=1).unsqueeze(0) - 2.0 * torch.matmul(_x, _y.t())
            euc_dist = torch.sqrt(F.relu(_dist) + 1e-12)
            cos_sim = torch.matmul(_x_norm, _y_norm.t())
            fused = alpha * euc_dist + (1.0 - alpha) * (1.0 - cos_sim)
            dist += fused

        elif metric == 'dweuc':
            # 若用户传入了 dweuc_module，就用它（可学习）；否则内部创建一个临时 module（不建议）
            if dweuc_module is None:
                # 将 _x/_y 展平（如果希望 per-bin concat 可修改）
                _x_flat = _x
                _y_flat = _y
                tmp = DynamicWEUC().to(device)
                dist += tmp(_x_flat, _y_flat)
            else:
                # Ensure module on same device
                dweuc_module.to(device)
                _x_flat = _x
                _y_flat = _y
                # dweuc_module expects [N, C] and [M, C]
                dist += dweuc_module(_x_flat, _y_flat)

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    dist = dist / float(num_bin)
    if detach_result:
        return dist.detach()
    else:
        return dist


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = torch.from_numpy(msk1).cuda()
    if not is_tensor(msk2):
        msk2 = torch.from_numpy(msk2).cuda()
    n = msk1.size(0)
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).float()
    miou = inter.view(n, -1).sum(-1) / (union.view(n, -1).sum(-1) + eps)
    return miou


def compute_ACC_mAP(distmat, q_pids, g_pids, q_views=None, g_views=None, rank=1):
    num_q, _ = distmat.shape
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if q_views is not None and g_views is not None:
            q_idx_mask = np.isin(g_views, q_views[q_idx], invert=True) | np.isin(
                g_pids, q_pids[q_idx], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]

        assert(len(q_idx_glabels) >
               0), "No gallery after excluding identical-view cases!"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices]
                         == q_pids[q_idx]).astype(np.int32)

        # binary vector, positions with value 1 are correct matches
        # orig_cmc = matches[q_idx]
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank-1])

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    # all_ACC = np.asarray(all_ACC).astype(np.float32)
    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP


def evaluate_rank(distmat, p_lbls, g_lbls, max_rank=50):
    '''
    Copy from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/utils/rank.py#L12-L63
    '''
    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_many(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # 对应位置变成从小到大的序号
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(
        np.int32)  # 根据indices调整顺序 g_pids[indices]
    # print(matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP

'''import torch
import numpy as np
import torch.nn.functional as F

from utils import is_tensor


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = torch.from_numpy(msk1).cuda()
    if not is_tensor(msk2):
        msk2 = torch.from_numpy(msk2).cuda()
    n = msk1.size(0)
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).float()
    miou = inter.view(n, -1).sum(-1) / (union.view(n, -1).sum(-1) + eps)
    return miou


def compute_ACC_mAP(distmat, q_pids, g_pids, q_views=None, g_views=None, rank=1):
    num_q, _ = distmat.shape
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if q_views is not None and g_views is not None:
            q_idx_mask = np.isin(g_views, q_views[q_idx], invert=True) | np.isin(
                g_pids, q_pids[q_idx], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]

        assert(len(q_idx_glabels) >
               0), "No gallery after excluding identical-view cases!"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices]
                         == q_pids[q_idx]).astype(np.int32)

        # binary vector, positions with value 1 are correct matches
        # orig_cmc = matches[q_idx]
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank-1])

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    # all_ACC = np.asarray(all_ACC).astype(np.float32)
    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP


def evaluate_rank(distmat, p_lbls, g_lbls, max_rank=50):

    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_many(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # 对应位置变成从小到大的序号
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(
        np.int32)  # 根据indices调整顺序 g_pids[indices]
    # print(matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP
'''