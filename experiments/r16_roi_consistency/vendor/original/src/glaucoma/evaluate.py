import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def _validate(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Nonempty one-dimensional aligned labels/scores required')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be 0/1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and within [0,1]')
    return y.astype(int), p


def evaluate_predictions(manifest, predictions, threshold=0.5):
    for df in [manifest, predictions]:
        if df.image_id.isna().any() or df.image_id.duplicated().any():
            raise ValueError('Null or duplicate image ID')
    if set(manifest.image_id) != set(predictions.image_id):
        raise ValueError('Prediction and manifest IDs differ')
    joined = manifest[['image_id', 'label']].merge(
        predictions[['image_id','probability']], on='image_id', validate='one_to_one')
    y,p = _validate(joined.label.to_numpy(), joined.probability.to_numpy())
    if not np.isfinite(threshold):
        raise ValueError('Finite threshold required')
    z = (p >= threshold).astype(int)
    tp,fn = int(((y == 1)&(z == 1)).sum()),int(((y == 1)&(z == 0)).sum())
    fp,tn = int(((y == 0)&(z == 1)).sum()),int(((y == 0)&(z == 0)).sum())
    tpr,tnr = tp/(tp+fn) if tp+fn else None,tn/(tn+fp) if tn+fp else None
    result = dict(n=len(y),n_positive=int(y.sum()),n_negative=int((y==0).sum()),
                  threshold=float(threshold),tp=tp,fn=fn,fp=fp,tn=tn,tpr=tpr,tnr=tnr,
                  precision=tp/(tp+fp) if tp+fp else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                  predicted_positive_rate=float(z.mean()),status='ok')
    if len(np.unique(y)) < 2:
        result.update(status='single_class',auc_probability=None,auc_label=None,
                      balanced_accuracy=None,ap_probability=None,
                      prauc_trapezoid_probability=None,ap_label=None,prauc_trapezoid_label=None)
        return result
    precision,recall,_ = precision_recall_curve(y,p)
    lp,lr,_ = precision_recall_curve(y,z)
    result.update(auc_probability=float(roc_auc_score(y,p)),
                  auc_label=float(roc_auc_score(y,z)),balanced_accuracy=(tpr+tnr)/2,
                  ap_probability=float(average_precision_score(y,p)),
                  prauc_trapezoid_probability=float(auc(recall,precision)),
                  ap_label=float(average_precision_score(y,z)),
                  prauc_trapezoid_label=float(auc(lr,lp)))
    return result


def select_threshold(y, p):
    y,p = _validate(y,p)
    if len(np.unique(y)) < 2:
        raise ValueError('Threshold fitting needs both classes')
    order=np.argsort(p,kind='stable'); sp,sy=p[order],y[order]
    unique=np.unique(sp)
    edges=np.unique(np.r_[0.,unique,1.])
    low,high=edges[:-1],edges[1:]
    mids=low+(high-low)/2
    mids=np.where(mids<=low,np.nextafter(low,high),mids)
    candidates=np.r_[0.,mids,np.nextafter(1.,2.)]
    widths=np.r_[0.,high-low,0.]
    removed=np.searchsorted(sp,candidates,side='left')
    prefix=np.r_[0,np.cumsum(sy)]; npos=int(y.sum()); nneg=len(y)-npos
    tpr=(npos-prefix[removed])/npos
    tnr=(removed-prefix[removed])/nneg
    ba=(tpr+tnr)/2
    best=float(ba.max())
    if len(unique)==1 or best<=.5+1e-12:
        return dict(threshold=.5,comparison='>=',balanced_accuracy=.5,
                    degenerate=True,source='dev',policy='youden_recall_plateau')
    eligible=np.flatnonzero(np.abs(ba-best)<=1e-12)
    i=max(eligible,key=lambda j:(float(tpr[j]),float(widths[j]),-float(candidates[j])))
    return dict(threshold=float(candidates[i]),comparison='>=',balanced_accuracy=best,
                tpr=float(tpr[i]),tnr=float(tnr[i]),plateau_width=float(widths[i]),
                degenerate=False,source='dev',policy='youden_recall_plateau')


def compare_predictions(manifest,a,b,tau_a,tau_b,seed=20261015,n_bootstrap=1000):
    evaluate_predictions(manifest,a,tau_a)
    evaluate_predictions(manifest,b,tau_b)
    if not isinstance(n_bootstrap,int) or n_bootstrap<1:raise ValueError('Positive integer resample count required')
    base=manifest[['image_id','label']].copy()
    base=base.merge(a[['image_id','probability']].rename(columns={'probability':'a'}),on='image_id',validate='one_to_one')
    base=base.merge(b[['image_id','probability']].rename(columns={'probability':'b'}),on='image_id',validate='one_to_one')
    if len(base)!=len(manifest): raise ValueError('Bootstrap ID mismatch')
    rng=np.random.default_rng(seed); values=[];auc_values=[]
    group_col='patient_id' if 'patient_id' in manifest and manifest.patient_id.notna().all() else None
    groups=None
    if group_col:
        base=base.merge(manifest[['image_id',group_col]],on='image_id')
        groups=[x.index.to_numpy() for _,x in base.groupby(group_col)]
    else:
        strata=[np.flatnonzero(base.label.to_numpy()==v) for v in [0,1]]
        if any(len(x)==0 for x in strata): raise ValueError('Both classes required')
    for _ in range(n_bootstrap):
        if groups:
            idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        else: idx=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        y=base.label.to_numpy()[idx]
        if len(np.unique(y))<2: continue
        scores=[]
        for col,tau in [('a',tau_a),('b',tau_b)]:
            z=base[col].to_numpy()[idx]>=tau
            scores.append((z[y==1].mean()+(~z[y==0]).mean())/2)
        values.append(scores[1]-scores[0])
        auc_values.append(roc_auc_score(y,base.b.to_numpy()[idx])-roc_auc_score(y,base.a.to_numpy()[idx]))
    return dict(valid_resamples=len(values),unit='patient' if groups else 'class_stratified_image',
                delta_b_minus_a_ci95=np.quantile(values,[.025,.975]).tolist() if values else None,
                delta_probability_auc_ci95=np.quantile(auc_values,[.025,.975]).tolist() if auc_values else None,
                reliable_resampling=len(values)>=.9*n_bootstrap)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True)
    ap.add_argument('--predictions',required=True); ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--fit-threshold',action='store_true'); ap.add_argument('--output',required=True)
    args=ap.parse_args(); m=pd.read_csv(args.manifest); p=pd.read_csv(args.predictions)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    result={'fixed':evaluate_predictions(m,p,args.threshold)}
    if args.fit_threshold:
        if 'role' not in m or set(m.role)!= {'dev'}: raise ValueError('Only dev may fit thresholds')
        j=m[['image_id','label']].merge(p[['image_id','probability']],on='image_id',validate='one_to_one')
        threshold=select_threshold(j.label.to_numpy(),j.probability.to_numpy())
        result['selected']=evaluate_predictions(m,p,threshold['threshold'])
        (out/'threshold.json').write_text(json.dumps(threshold,indent=2))
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
