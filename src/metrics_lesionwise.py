# src/metrics.py
"""
Lesion-wise Dice and HD95 evaluation. Exposes:
- evaluate_case_lesionwise(gt_path, pred_path, spacing=None)
Returns dict per class: lists of dice and hd95 values.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label, binary_erosion
from scipy.spatial.distance import cdist

# medpy.hd95 is convenient; fallback to custom if unavailable
try:
    from medpy.metric.binary import hd95 as medpy_hd95
    _have_medpy = True
except Exception:
    _have_medpy = False

def _components(mask):
    if mask.max() == 0:
        return []
    lab, n = cc_label(mask)
    comps = []
    for i in range(1, n+1):
        comps.append(lab == i)
    return comps

def dice_binary(a: np.ndarray, b: np.ndarray):
    a = a.astype(bool); b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0: return 1.0
    inter = (a & b).sum()
    return 2.0 * inter / denom

def _surface_voxels(m):
    return m & ~binary_erosion(m, iterations=1)

def _compute_hd95(a, b, spacing=(1,1,1)):
    a = a.astype(bool); b = b.astype(bool)
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if _have_medpy:
        try:
            return float(medpy_hd95(a, b, voxelspacing=spacing))
        except Exception:
            pass
    # fallback: compute distances between surface voxels
    A = np.argwhere(_surface_voxels(a))
    B = np.argwhere(_surface_voxels(b))
    if A.size == 0 or B.size == 0:
        return float("inf")
    # scale by spacing
    A = A * np.array(spacing)
    B = B * np.array(spacing)
    d = cdist(A, B)
    # symmetric: take 95th percentile of combined minima
    d1 = d.min(axis=1)
    d2 = d.min(axis=0)
    combo = np.hstack([d1, d2])
    return float(np.percentile(combo, 95))

CLASS_MAP = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

def evaluate_case_lesionwise(gt_path, pred_path, spacing=None):
    gt = nib.load(gt_path); gtl = gt.get_fdata().astype(np.int16)
    pr = nib.load(pred_path); prl = pr.get_fdata().astype(np.int16)
    if spacing is None:
        spacing = gt.header.get_zooms()[:3]
    out = {name: {"dice": [], "hd95": []} for name in CLASS_MAP.values()}
    for cls, name in CLASS_MAP.items():
        gmask = (gtl == cls).astype(np.uint8)
        pmask = (prl == cls).astype(np.uint8)
        g_comps = _components(gmask)
        p_comps = _components(pmask)
        used_p = set()
        # match GT comps to predicted comps by overlap (greedy)
        for gi, gcomp in enumerate(g_comps):
            overlaps = []
            for pi, pcomp in enumerate(p_comps):
                if pi in used_p: continue
                ov = (gcomp & pcomp).sum()
                overlaps.append((pi, ov))
            if overlaps:
                pi, ov = max(overlaps, key=lambda x: x[1])
                if ov > 0:
                    used_p.add(pi)
                    dice_v = dice_binary(gcomp, p_comps[pi])
                    hd_v = _compute_hd95(gcomp, p_comps[pi], spacing)
                    out[name]["dice"].append(dice_v)
                    out[name]["hd95"].append(hd_v)
                    continue
            # no match -> FN lesion
            out[name]["dice"].append(0.0)
            out[name]["hd95"].append(_compute_hd95(gcomp, np.zeros_like(gcomp), spacing))
        # leftover predicted comps -> FPs (count as dice=0)
        for pi, pcomp in enumerate(p_comps):
            if pi in used_p: continue
            out[name]["dice"].append(0.0)
            out[name]["hd95"].append(_compute_hd95(np.zeros_like(pcomp), pcomp, spacing))
    # also compute derived WT/TC voxel-wise (non-lesion)
    derived = {}
    WT_gt = ( (gtl==1) | (gtl==2) | (gtl==3) )
    WT_pr = ( (prl==1) | (prl==2) | (prl==3) )
    TC_gt = ( (gtl==1) | (gtl==3) )
    TC_pr = ( (prl==1) | (prl==3) )
    derived["WT"] = {"dice": [dice_binary(WT_gt, WT_pr)], "hd95": [_compute_hd95(WT_gt, WT_pr, spacing)]}
    derived["TC"] = {"dice": [dice_binary(TC_gt, TC_pr)], "hd95": [_compute_hd95(TC_gt, TC_pr, spacing)]}
    return out, derived
