"""Single-flip policy and ordered, unlabelled submission contracts; no framework."""
import csv
import json
from pathlib import Path

from .contracts import POLICY,identity,probability,read_csv,sha

SINGLE_POLICY={**POLICY,'variant':'original_r16_single_flip_v1','views':['horizontal_flip'],
               'view_weights':[1.0],'view_fusion':'single_view_probability','test_export':'explicit_single-test_only'}


def test_rows(data,expected_sha):
    data=Path(data)
    if not expected_sha or sha(data/'test.csv')!=expected_sha:raise ValueError('Prepare again with unchanged official test.csv')
    rows=read_csv(data/'test.csv');names=[r['Image_name'] for r in rows]
    keys=[identity(n) for n in names]
    if len(rows)!=5855 or len(set(keys))!=5855:raise ValueError('Expected 5855 unique official test identities')
    if any(r.get('True_Label','').strip() for r in rows):raise ValueError('Test labels must be unavailable')
    files={}
    for path in (data/'test').iterdir():
        if not path.is_file() or path.suffix.casefold() not in ('.jpg','.jpeg'):continue
        key=identity(path.name)
        if key in files:raise ValueError('Duplicate test image identity')
        if not path.resolve().is_relative_to(data.resolve()):raise ValueError('Test image escapes root')
        files[key]=path
    if not set(keys)<=set(files):raise ValueError('Missing official test JPEG')
    return [dict(image_id=name,relative_path=files[key].relative_to(data).as_posix()) for name,key in zip(names,keys)]


def write_submission(path,rows,probabilities):
    if len(rows)!=5855 or len(probabilities)!=len(rows):raise ValueError('Incomplete test predictions')
    names=[r['image_id'] for r in rows]
    if len({identity(n) for n in names})!=5855:raise ValueError('Duplicate test identities')
    values=[probability(p) for p in probabilities]
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','Probability','Pred_Label'])
        writer.writerows((n,format(p,'.17g'),int(p>=.5)) for n,p in zip(names,values))
    restored=read_csv(temporary)
    if ([r['Image_name'] for r in restored]!=names or
            any(float(r['Probability'])!=p or int(r['Pred_Label'])!=int(p>=.5) for r,p in zip(restored,values))):
        raise ValueError('Submission serialization/order mismatch')
    temporary.replace(path)


def verify_stage_marker(root,name,fingerprint,control=None):
    marker=json.loads((Path(root)/name).read_bytes())
    if marker['fingerprint']!=fingerprint:raise ValueError('Run prerequisite in the same project/runtime first')
    if control is not None and marker['control']!=control:raise ValueError('Prerequisite used another baseline')
    if sha(marker['result_path'])!=marker['result_sha256']:raise ValueError('Prerequisite report changed')
    report=json.loads(Path(marker['result_path']).read_bytes())
    expected_mode='single-check' if name=='single_check_passed.json' else 'single-valid'
    if report['status']!='completed' or report['mode']!=expected_mode or report['n']!=(32 if expected_mode=='single-check' else 5849):
        raise ValueError('Incomplete prerequisite report')
