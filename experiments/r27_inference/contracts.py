"""R27 portable package identity; no torch imports."""
import json
from pathlib import Path,PurePosixPath
from io_contracts import sha

DETAIL_SHA='3172d1858e4382019d5abb705aad618c7c5e6a0c06d123e74bfa103932e30c52'
MODES=('captain-original','stable-flip384','detail-flip512')
PARENT_WEIGHTS={'global':'b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2',
                'local':'76aa0e1701a392b624d228fc5a896e89a05df3371d6af584f3bd86c3601cf390',
                'locator':'a50079a469eadbdf2cb371886984541739510b0874aeb81ddc4944398f77a4dc'}


def verify(root):
    root=Path(root).resolve();m=json.loads((root/'bundle.json').read_text())
    if m['schema']!='r27-portable-v1':raise ValueError('Wrong bundle schema')
    required={'run.py','contracts.py','io_contracts.py','inference/predict.py','inference/config.json','inference/manifest.json','adapter/adapter.py','adapter/imaging.py','adapter/stable_precision.py'}
    if m['detail']:required|={'adapter/parallel_decode.py','adapter/parallel_detail_adapter.py','adapter/detail_crop.py','model/detail512.pt'}
    required|={'inference/weights/'+name+'.pt' for name in PARENT_WEIGHTS}
    if not required<=set(m['files']):raise ValueError('Incomplete bundle')
    for name,digest in PARENT_WEIGHTS.items():
        if m['files']['inference/weights/'+name+'.pt']['sha256']!=digest:raise ValueError('Wrong R27 parent weight: '+name)
    for name,item in m['files'].items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:raise ValueError('Unsafe bundle path')
        file=(root/name).resolve()
        if not file.is_relative_to(root) or file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:raise ValueError('Bundle changed: '+name)
    weights=[v['bytes'] for n,v in m['files'].items() if n.endswith('.pt')]
    if not weights or sum(weights)>500_000_000 or sum(weights)!=m['weight_bytes']:raise ValueError('Model size budget/identity mismatch')
    if m['detail'] and m['files']['model/detail512.pt']['sha256']!=DETAIL_SHA:raise ValueError('Wrong detail checkpoint')
    return m
