"""Stdlib-only bundle, input and measurement contracts. No model imports."""
import csv
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import random
import re
import time

WEIGHTS={'global':'b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2',
         'local':'1eb7f8627be0c341ed1ac426e8d4ce463475134edca8542cc201679841914175'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def validate_bundle(root):
    root=Path(root).resolve();m=json.loads((root/'bundle.json').read_bytes())
    if m['schema']!='r16-inference-v1' or set(m['weights'])!=set(WEIGHTS):raise ValueError('Bundle schema/weights differ')
    if not {'run.py','src/r16_deploy/runtime.py','src/r16_tta/pipeline.py','src/glaucoma/models.py'}<=set(m['sources']):
        raise ValueError('Incomplete inference source inventory')
    for name,digest in m['sources'].items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:raise ValueError('Unsafe source path')
        if not (root/name).resolve().is_relative_to(root) or sha(root/name)!=digest:raise ValueError('Bundle source changed: '+name)
    for name,digest in WEIGHTS.items():
        item=m['weights'][name];p=root/'model'/(name+'.pt')
        if (not p.resolve().is_relative_to(root) or item['sha256']!=digest or sha(p)!=digest
                or p.stat().st_size!=item['bytes']):raise ValueError('Original weight changed: '+name)
    if sum(i['bytes'] for i in m['weights'].values())>500_000_000:raise ValueError('Weight budget exceeded')
    return m


def inputs(images,metadata,expected=None):
    images=Path(images).resolve()
    with Path(metadata).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if 'Image_name' not in (reader.fieldnames or []):raise ValueError('Image_name is required')
        names=[r['Image_name'] for r in reader]
    def key(name):
        if not name or name!=name.strip() or any(c in name for c in '/\\:') or Path(name).suffix.casefold() not in ('.jpg','.jpeg','.png'):
            raise ValueError('Original JPG/PNG basename required')
        return name.casefold()
    keys=[key(n) for n in names]
    if not names or len(set(keys))!=len(keys) or (expected is not None and len(names)!=expected):raise ValueError('Input count/identity mismatch')
    files={}
    for p in images.iterdir():
        if not p.is_file() or p.suffix.casefold() not in ('.jpg','.jpeg','.png'):continue
        k=key(p.name)
        if k in files or not p.resolve().is_relative_to(images):raise ValueError('Ambiguous/escaping JPEG')
        files[k]=p
    if not set(keys)<=set(files):raise ValueError('Missing metadata image')
    return [(n,files[k]) for n,k in zip(names,keys)]


def select(rows,limit,seed=20261031):
    if limit<0:raise ValueError('Limit cannot be negative')
    return rows if not limit or limit>=len(rows) else random.Random(seed).sample(rows,limit)


def validate_probability(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid prediction probability')
    return value


def summary(values,gpu,full_coverage):
    values=sorted(float(x) for x in values)
    if not values or any(not math.isfinite(x) or x<0 for x in values):raise ValueError('Nonnegative finite measurements required')
    def q(f):
        p=(len(values)-1)*f;i=math.floor(p);j=math.ceil(p)
        return values[i]+(values[j]-values[i])*(p-i)
    return dict(n=len(values),mean_ms=sum(values)/len(values),p50_ms=q(.5),p95_ms=q(.95),p99_ms=q(.99),max_ms=max(values),
                over_100ms=sum(x>100 for x in values),all_measured_under_or_equal_100ms=max(values)<=100,
                hardware_is_t4=bool(re.search(r'\bT4\b',gpu,re.I)),full_coverage=full_coverage,official_acceptance=False)


def timed_call(predict,path,synchronize,clock=time.perf_counter):
    synchronize();start=clock();details=predict(path);synchronize();elapsed=1000*(clock()-start)
    validate_probability(details['probability'])
    return details,elapsed
