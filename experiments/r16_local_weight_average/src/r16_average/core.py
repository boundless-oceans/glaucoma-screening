"""Stdlib contracts. Array arithmetic also accepts NumPy in CPU-only tests."""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random

WEIGHTS = {
    'global': 'b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2',
    'original': '1eb7f8627be0c341ed1ac426e8d4ce463475134edca8542cc201679841914175',
    'ema': '98841189b6d165360cbeb93eb96f7c15dea8f42784ac925251817f60945d33dd',
}
INITIALIZER = 'f22f8850dcded245f4ae3bc402bbde9c6f5e4e23e76dfaea3aadb75aa9d3705a'
VALID_SHA = 'edda257a801a1036ea2b1a9ffe983a1f9fea73c3f89fd6ac0f6ab6a4da355f2a'
REFERENCES = {
    'original': ('448f12d3139dd8958ae2efbadfa3bc1c99d27492dcf547c8d29e7c115d820627', 'original_probability'),
    'ema': ('0874dfa72876eb6f22a64a27fe57b3aedafcbe7b7ed3a54c3176e552cae134d6', 'probability'),
}
POLICY = dict(local_coefficients=[.5,.5], parameters='FP32', buffers='identical_copy',
              global_size=512, local_size=384, jpeg_draft_size=768, view='original',
              fusion='equal_raw_logit_mean', fallback='global', precision='cuda_fp16_autocast',
              threshold=.5, training=False, test_export=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def probability(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid probability')
    return value


def validation_rows(data):
    rows=csv_rows(Path(data)/'valid.csv')
    names=[r['Image_name'] for r in rows]
    labels=[int(r['True_Label']) for r in rows]
    if len(rows)!=5849 or len({n.casefold() for n in names})!=5849 or set(labels)!={0,1} or sum(labels)!=191:
        raise ValueError('Official valid coverage/labels differ')
    return names,labels


def reference_values(path,names,column):
    rows=csv_rows(path)
    if [r['image_id'] for r in rows]!=names:raise ValueError('Reference order differs')
    return [probability(r[column]) for r in rows]


def validate_project(root):
    root=Path(root).resolve();meta=read(root/'PREPARATION.json')
    if meta['schema']!='r16-local-weight-average-v1' or meta['policy']!=POLICY:
        raise ValueError('Prepared policy changed')
    if set(meta['weights'])!=set(WEIGHTS) or set(meta['references'])!=set(REFERENCES):raise ValueError('Input roles differ')
    required={'run.py','src/r16_average/core.py','src/r16_average/runtime.py',
              'src/glaucoma/models.py','src/glaucoma/r08_export.py','src/r16_deploy/core.py'}
    if not required<=set(meta['sources']):raise ValueError('Incomplete source inventory')
    for name,digest in meta['sources'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:raise ValueError('Unsafe source path')
        p=(root/name).resolve()
        if not p.is_relative_to(root) or sha(p)!=digest:raise ValueError('Prepared source changed: '+name)
    for role,item in meta['weights'].items():
        p=Path(item['path'])
        if item['sha256']!=WEIGHTS[role] or sha(p)!=WEIGHTS[role] or p.stat().st_size!=item['bytes']:
            raise ValueError('Pinned input weight changed: '+role)
    if sha(Path(meta['data_root'])/'valid.csv')!=VALID_SHA:raise ValueError('Official valid CSV changed')
    names,labels=validation_rows(meta['data_root'])
    for role,item in meta['references'].items():
        digest,column=REFERENCES[role]
        if item['sha256']!=digest or item['column']!=column or sha(item['path'])!=digest:
            raise ValueError('Reference changed: '+role)
        reference_values(item['path'],names,column)
    return meta,names,labels


def validate_parent_configs(original,ema):
    for key,value in dict(model='convnext_tiny',image_size=384,use_age=False,seed=20261026,initializer_sha256=INITIALIZER).items():
        if original.get(key)!=value or ema.get(key)!=value:raise ValueError('Local config incompatible: '+key)
    for config in (original,ema):
        if config.get('local_region',{}).get('teacher_sha256')!=WEIGHTS['global']:
            raise ValueError('Frozen global differs')


def average_parameters_(target,left,right,parameters,*,equal,finite):
    """Reject ALL incompatibilities before modifying target; never change parents."""
    keys=set(left)
    if keys!=set(right) or keys!=set(target) or not parameters or not set(parameters)<=keys:
        raise ValueError('State/parameter keys differ')
    for name in left:
        a,b,dst=left[name],right[name],target[name]
        if a.shape!=b.shape or a.shape!=dst.shape or a.dtype!=b.dtype or a.dtype!=dst.dtype:
            raise ValueError('State shape/dtype differs: '+name)
        if not finite(a) or not finite(b):raise ValueError('Nonfinite parent: '+name)
        if name in parameters:
            if str(a.dtype) not in ('float32','torch.float32'):raise ValueError('FP32 parameter required: '+name)
        elif not equal(a,b):raise ValueError('Nonparameter state differs: '+name)
    for name,dst in target.items():
        dst[...] = left[name]*.5+right[name]*.5 if name in parameters else left[name]
        if not finite(dst):raise ValueError('Nonfinite candidate: '+name)


def check_indices(names,labels):
    rng=random.Random(20261031)
    indices=sorted(rng.sample([i for i,y in enumerate(labels) if y==0],16)+
                   rng.sample([i for i,y in enumerate(labels) if y==1],16))
    for name in ('valid_03118.jpg','valid_00296.jpg','valid_01329.jpg'):
        if name not in names:raise ValueError('Known CAM-boundary sample missing')
        i=names.index(name)
        if i not in indices:indices.append(i)
    return indices


def compare_reference(actual,expected,tolerance=.001):
    if not actual or len(actual)!=len(expected):raise ValueError('Reference coverage differs')
    actual=[probability(v) for v in actual];expected=[probability(v) for v in expected]
    deltas=[abs(a-b) for a,b in zip(actual,expected)]
    labels=sum((a>=.5)!=(b>=.5) for a,b in zip(actual,expected))
    return dict(rows=len(actual),max_abs_delta=max(deltas),mean_abs_delta=sum(deltas)/len(deltas),
                hard_label_changes=labels,passed=max(deltas)<=tolerance and labels==0,tolerance=tolerance)


def complete(folder,root,files):
    dump(Path(folder)/'complete.json',dict(preparation_sha256=sha(Path(root)/'PREPARATION.json'),
         files={name:sha(Path(folder)/name) for name in files}))


def verify_complete(folder,root,required):
    folder=Path(folder);m=read(folder/'complete.json')
    if m['preparation_sha256']!=sha(Path(root)/'PREPARATION.json') or not set(required)<=set(m['files']):
        raise ValueError('Completed stage contract differs')
    for name,digest in m['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p)!=digest:raise ValueError('Completed output changed: '+name)
    return m
