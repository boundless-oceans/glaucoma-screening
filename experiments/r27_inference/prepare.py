"""Create private standalone R27 inference bundle by hash-checked file copies."""
import argparse,json,shutil
from pathlib import Path
from io_contracts import sha
from contracts import DETAIL_SHA,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--materials-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--detail-checkpoint',type=Path)
    a=p.parse_args();here=Path(__file__).resolve().parent;experiment=here.parent/'r27_single_flip'
    pins=json.loads((experiment/'materials_sha256.json').read_text())
    selected={n:v for n,v in pins.items() if n.startswith('inference/')}
    if not selected:raise ValueError('Inference pins required')
    for n,v in selected.items():
        file=(a.materials_root/n).resolve()
        if not file.is_relative_to(a.materials_root.resolve()) or file.stat().st_size!=v['bytes'] or sha(file)!=v['sha256']:raise ValueError('Captain material changed: '+n)
    if a.detail_checkpoint is not None and sha(a.detail_checkpoint)!=DETAIL_SHA:raise ValueError('Fixed epoch3 detail checkpoint required')
    adapters=['adapter.py','imaging.py','stable_precision.py']
    if a.detail_checkpoint is not None:adapters+=['parallel_decode.py','parallel_detail_adapter.py','detail_crop.py']
    a.output.mkdir(parents=True,exist_ok=False)
    def copy(source,name):
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for n in selected:copy(a.materials_root/n,n)
    for n in adapters:copy(experiment/n,'adapter/'+n)
    for n in ('run.py','contracts.py','io_contracts.py','environment.yml','README.md'):copy(here/n,n)
    if a.detail_checkpoint is not None:copy(a.detail_checkpoint,'model/detail512.pt')
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.output.rglob('*')) if p.is_file()}
    model_bytes=sum(v['bytes'] for n,v in files.items() if n.endswith('.pt'))
    manifest=dict(schema='r27-portable-v1',detail=a.detail_checkpoint is not None,files=files,weight_bytes=model_bytes,official_acceptance=False,note='Private weights/code only; no images, metadata, historical predictions or training directories. CPU packaging does not prove GPU portability.')
    (a.output/'bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify(a.output);print('Prepared and verified without model loading:',a.output,'weight_bytes=',model_bytes)

if __name__=='__main__':main()
