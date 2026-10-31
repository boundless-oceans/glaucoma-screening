"""Stdlib-only identities, validation alignment and fixed probability policy."""
import csv
import hashlib
import math
from pathlib import Path, PurePosixPath
import random

WEIGHTS = {
    'global': 'b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2',
    'local': '1eb7f8627be0c341ed1ac426e8d4ce463475134edca8542cc201679841914175',
}
REFERENCE_SHA = '40aa3754c94895aaf0f4d67e60a1f8eab206e2cd7562017d3b2ec792c9c05296'
POLICY = dict(variant='original_r16_hflip_tta_v1',views=['original','horizontal_flip'],
              view_fusion='arithmetic_mean_of_probabilities',view_weights=[0.5,0.5],
              within_view_fusion='equal_raw_logit_mean',fallback='per_view_global',
              global_size=512,local_size=384,jpeg_draft_size=768,batch_size=1,
              flip_stage='oriented_decoded_RGB_before_fundus_crop',
              cam='recomputed_independently_per_view',precision='fp16',
              threshold=0.5,use_age=False,training=False,test_export=False)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def identity(name):
    name=str(name)
    if '/' in name or '\\' in name or Path(name).suffix.casefold() not in ('.jpg','.jpeg'):
        raise ValueError('Expected an original JPEG basename: '+name)
    return name.rsplit('.',1)[0].casefold()


def binary(value):
    if str(value) not in ('0','1'):raise ValueError('Expected binary integer label')
    return int(value)


def probability(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid probability')
    return value


def average_views(original,flipped):
    # R16 branches fuse logits WITHIN each view; TTA averages probabilities ACROSS views.
    return 0.5*(probability(original)+probability(flipped))


def validation_rows(manifest,official,reference):
    """Use fixed R16 valid order; official labels and archived baseline must agree."""
    records=read_csv(manifest);official=read_csv(official);reference=read_csv(reference)
    def unique(rows,key):
        result={identity(row[key]):row for row in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate image identity')
        return result
    left=unique(records,'image_id');right=unique(official,'Image_name');ref=unique(reference,'Image_name')
    if len(records)!=5849 or set(left)!=set(right) or set(left)!=set(ref):
        raise ValueError('Official valid/reference coverage mismatch')
    result=[]
    for key,row in left.items():
        label=binary(row['label'])
        if label!=binary(right[key]['True_Label']) or label!=binary(ref[key]['Label']):
            raise ValueError('Validation labels differ: '+key)
        path=PurePosixPath(row['relative_path'])
        if (path.is_absolute() or len(path.parts)!=2 or path.parts[0]!='valid'
                or '\\' in str(path) or identity(path.name)!=key
                or row['official_split']!='val' or row['role']!='r16_official_valid'):
            raise ValueError('Invalid valid image path/role: '+key)
        p=probability(ref[key]['Probability'])
        if binary(ref[key]['Pred_Label'])!=int(p>=0.5):raise ValueError('Reference hard label differs')
        result.append(dict(image_id=row['image_id'],relative_path=str(path),label=label,reference_probability=p))
    if sum(row['label'] for row in result)!=191:raise ValueError('Unexpected valid positive count')
    return result


def check_sample(rows):
    rng=random.Random(20261030)
    return [row for label in (0,1) for row in rng.sample([r for r in rows if r['label']==label],16)]


def baseline_deltas(actual,reference):
    actual=[probability(x) for x in actual];reference=[probability(x) for x in reference]
    if not actual or len(actual)!=len(reference):raise ValueError('Unaligned baseline probabilities')
    deltas=[abs(a-b) for a,b in zip(actual,reference)]
    return dict(max_abs_probability_delta=max(deltas),mean_abs_probability_delta=sum(deltas)/len(deltas),
                hard_label_changes=sum((a>=0.5)!=(b>=0.5) for a,b in zip(actual,reference)))
