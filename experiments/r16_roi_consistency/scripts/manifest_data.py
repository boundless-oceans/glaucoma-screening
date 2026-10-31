"""Generate original R16 role manifests from user-supplied official CSVs.

Stdlib only. No images, models, downloads, or bundled real sample rows.
"""
import csv
import hashlib
import io
import math
from pathlib import Path
import re

COUNTS={'train':(51396,1657),'valid':(5849,191)}
EXPECTED_SHA={'train':'8a532cedf9e7944b8e6ee2c8581877a82770e0235a1e0642af2c8b9a7d385515',
              'valid':'a7054e8c7860bbe2b2b30149b011aca6df4f8470e4413af31b0a52a668aa345c'}
FIELDS=['image_id','relative_path','official_split','role','label','age','patient_id']


def render_manifest(data,split):
    """Canonical LF/UTF8 serialization reproduces the original R16 snapshot."""
    if split not in COUNTS:raise ValueError('Only train/valid role manifests are supported')
    with (Path(data)/(split+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'Image_name','Age','True_Label'}<=set(reader.fieldnames or []):raise ValueError('Official CSV columns differ')
        rows=list(reader)
    n,positive=COUNTS[split];seen=set();prepared=[]
    if len(rows)!=n:raise ValueError('Official row count differs: '+split)
    for row in rows:
        name=row['Image_name']
        if not re.fullmatch(split+r'_\d{5}\.jpg',name):raise ValueError('Unexpected original R16 JPEG name')
        if name in seen:raise ValueError('Duplicate official image identity')
        seen.add(name)
        if row['True_Label'] not in ('0','1'):raise ValueError('Expected binary official label')
        try:age=float(row['Age']) if row['Age'].strip() else float('nan')
        except (ValueError,TypeError):raise ValueError('Invalid official age') from None
        if math.isinf(age):raise ValueError('Infinite official age')
        prepared.append([name,f'{split}/{name[:-4]}.JPG','val' if split=='valid' else 'train',
                         'r16_official_'+split,int(row['True_Label']),'' if math.isnan(age) else str(age),''])
    if sum(r[4] for r in prepared)!=positive:raise ValueError('Official positive count differs: '+split)
    output=io.StringIO(newline='');writer=csv.writer(output,lineterminator='\n');writer.writerow(FIELDS)
    writer.writerows(sorted(prepared,key=lambda r:r[0]))
    return output.getvalue().encode('utf-8')


def manifest_payloads(data,splits=('train','valid')):
    payloads={split:render_manifest(data,split) for split in splits}
    for split,payload in payloads.items():
        if hashlib.sha256(payload).hexdigest()!=EXPECTED_SHA[split]:
            raise ValueError('Official data does not reproduce original R16 manifest: '+split)
    return payloads


def write_manifests(target,payloads):
    target=Path(target);paths={s:target/(s+'.csv') for s in payloads}
    if any(p.exists() for p in paths.values()):raise FileExistsError('Refusing to replace generated manifests')
    target.mkdir(parents=True,exist_ok=True)
    for split,path in paths.items():path.write_bytes(payloads[split])
    return paths
