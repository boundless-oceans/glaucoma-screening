"""Prepare a NEW portable R16 bundle using file copies and hashes only."""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from r16_deploy.core import WEIGHTS,sha,validate_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','global-checkpoint','local-checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a NEW bundle directory')
    weights={n:getattr(a,n+'_checkpoint').resolve() for n in WEIGHTS}
    for n,path in weights.items():
        if sha(path)!=WEIGHTS[n]:raise ValueError('Requires original R16 '+n+' checkpoint')
    vendor=HERE.parent/'r16_roi_consistency/vendor'
    source=json.loads((vendor/'SOURCE_SHA256.json').read_bytes())
    for name,digest in source.items():
        if sha(vendor/name)!=digest:raise ValueError('Fixed vendor changed: '+name)
    out.mkdir();(out/'model').mkdir();(out/'src/glaucoma').mkdir(parents=True)
    for name in ('__init__.py','models.py','data.py','local_region.py','r08.py','r08_export.py'):
        shutil.copyfile(vendor/'original/src/glaucoma'/name,out/'src/glaucoma'/name)
    shutil.copytree(HERE/'src/r16_deploy',out/'src/r16_deploy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'src/r16_tta').mkdir();(out/'src/r16_tta/__init__.py').write_text('',encoding='utf-8')
    for name in ('imaging.py','pipeline.py'):
        shutil.copyfile(HERE.parent/'r16_hflip_tta/src/r16_tta'/name,out/'src/r16_tta'/name)
    shutil.copyfile(HERE/'scripts/run.py',out/'run.py')
    shutil.copyfile(HERE/'README.md',out/'README.md')
    shutil.copyfile(vendor/'original/pretrained/LICENSE.ConvNeXt',out/'LICENSE.ConvNeXt')
    for n,path in weights.items():shutil.copyfile(path,out/'model'/(n+'.pt'))
    manifest=dict(schema='r16-inference-v1',weights={n:dict(sha256=WEIGHTS[n],bytes=(out/'model'/(n+'.pt')).stat().st_size) for n in weights},
                  sources={str(f.relative_to(out).as_posix()):sha(f) for f in out.rglob('*') if f.is_file() and 'model' not in f.relative_to(out).parts},
                  training=False,data_rows_included=False,views=['original','single-flip'],threshold=.5,
                  provenance='Captain original R16; archived glaucoma source plus unchanged previously validated flip helper')
    (out/'bundle.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    validate_bundle(out)
    print('Prepared files only:',out,'weights bytes:',sum(x['bytes'] for x in manifest['weights'].values()))


if __name__=='__main__':main()
