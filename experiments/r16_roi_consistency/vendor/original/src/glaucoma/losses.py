def positive_weight_from_labels(labels,policy='moderate'):
    import math
    import numpy as np
    y=np.asarray(labels)
    if y.ndim!=1 or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Binary fit labels required')
    positives=int((y==1).sum());negatives=int((y==0).sum())
    if not positives or not negatives:raise ValueError('Both classes required')
    ratio=negatives/positives
    if policy=='moderate':return min(20.,math.sqrt(ratio))
    if policy=='balanced':return ratio
    raise ValueError(f'Unknown positive weight policy: {policy}')
