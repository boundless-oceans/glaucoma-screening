"""CPU-only CAM sensitivity summaries; never change the selected ROI."""
import math
import numpy as np


def cam_peak_summary(cam,valid_mask):
    """Mirror archived attention_box's eligible mask, report its closest rival.

    relative_gap is descriptive, not an acceptance threshold or a new ROI rule.
    Ties follow the original flattened argmax (first eligible pixel).
    """
    cam=np.asarray(cam,dtype=np.float64);valid=np.asarray(valid_mask,dtype=bool).copy()
    if cam.ndim!=2 or cam.shape!=valid.shape or not np.isfinite(cam).all():
        raise ValueError('Finite aligned CAM/mask required')
    h,w=cam.shape;yy,xx=np.mgrid[:h,:w]
    valid &= ((xx+.5-w/2)/(w*.46))**2+((yy+.5-h/2)/(h*.46))**2<1
    ids=np.flatnonzero(valid.ravel())
    values=np.abs(cam).ravel()
    order=ids[np.argsort(-values[ids],kind='stable')][:2]
    peaks=[dict(y=int(i//w),x=int(i%w),absolute_value=float(values[i])) for i in order]
    gap=float(values[order[0]]-values[order[1]]) if len(order)==2 else None
    relative_gap=gap/max(float(values[order[0]]),1e-12) if gap is not None else None
    return dict(eligible_pixels=int(valid.sum()),peaks=peaks,absolute_gap=gap,relative_gap=relative_gap,
                roi_policy_changed=False)


def audit_decision(actual,reference,guard,original_metrics,reference_metrics):
    """A finished audit can fail reproduction; it must never unlock TTA."""
    actual=np.asarray(actual,dtype=float);reference=np.asarray(reference,dtype=float)
    if (actual.ndim!=1 or not actual.size or actual.shape!=reference.shape
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()
            or ((actual<0)|(actual>1)|(reference<0)|(reference>1)).any()):
        raise ValueError('Aligned finite probabilities required')
    deltas=np.abs(actual-reference)
    metric_deltas={key:float(original_metrics[key]-reference_metrics[key]) for key in
                   ('auc_probability','ap_probability','prauc_trapezoid_probability')}
    if not all(math.isfinite(x) for x in metric_deltas.values()):raise ValueError('Nonfinite metrics')
    outliers=int((deltas>guard['max_abs_probability_delta']).sum())
    reproduced=(outliers==0 and abs(metric_deltas['auc_probability'])<=guard['max_abs_auc_delta']
                and abs(metric_deltas['ap_probability'])<=guard['max_abs_ap_delta'])
    return dict(baseline_reproduced=bool(reproduced),outlier_count=outliers,
                max_abs_probability_delta=float(deltas.max()),mean_abs_probability_delta=float(deltas.mean()),
                hard_label_changes=int(((actual>=.5)!=(reference>=.5)).sum()),metric_deltas=metric_deltas,
                tta_unlocked=False,guard_changed=False)
