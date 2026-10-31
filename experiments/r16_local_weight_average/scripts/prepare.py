"""File-only preparation of a NEW fixed-weight-average experiment; no torch."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from r16_average import core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data-root','global-checkpoint','original-checkpoint','ema-checkpoint','reference-original','reference-ema'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();target=args.output.resolve()
    if target.exists():raise FileExistsError('Choose a NEW output directory')
    weights={n:getattr(args,n+'_checkpoint').resolve() for n in core.WEIGHTS}
    for name,path in weights.items():
        if core.sha(path)!=core.WEIGHTS[name]:raise ValueError('Requires pinned '+name+' checkpoint')
    data=args.data_root.resolve()
    if core.sha(data/'valid.csv')!=core.VALID_SHA:raise ValueError('Official valid CSV required')
    names,_=core.validation_rows(data)
    refs={name:getattr(args,'reference_'+name).resolve() for name in core.REFERENCES}
    for name,path in refs.items():
        digest,column=core.REFERENCES[name]
        if core.sha(path)!=digest:raise ValueError('Requires pinned full reference '+name)
        core.reference_values(path,names,column)
    vendor=HERE.parent/'r16_roi_consistency/vendor'
    for name,digest in core.read(vendor/'SOURCE_SHA256.json').items():
        if core.sha(vendor/name)!=digest:raise ValueError('Vendor changed: '+name)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.average_prepare_',dir=target.parent) as temp:
        out=Path(temp)/'project';(out/'src/glaucoma').mkdir(parents=True)
        for name in ('__init__.py','models.py','data.py','local_region.py','r08.py','r08_export.py'):
            shutil.copyfile(vendor/'original/src/glaucoma'/name,out/'src/glaucoma'/name)
        shutil.copytree(HERE/'src/r16_average',out/'src/r16_average',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (out/'src/r16_deploy').mkdir();(out/'src/r16_deploy/__init__.py').write_text('',encoding='utf-8')
        shutil.copyfile(HERE.parent/'r16_inference/src/r16_deploy/core.py',out/'src/r16_deploy/core.py')
        shutil.copyfile(HERE.parents[1]/'src/glaucoma_training/data_utils.py',out/'src/r16_average/metrics.py')
        shutil.copyfile(HERE/'scripts/run.py',out/'run.py')
        shutil.copyfile(HERE/'README.md',out/'README.md')
        shutil.copyfile(vendor/'original/pretrained/LICENSE.ConvNeXt',out/'LICENSE.ConvNeXt')
        (out/'logs').mkdir()
        meta=dict(schema='r16-local-weight-average-v1',policy=core.POLICY,data_root=str(data),
            weights={n:dict(path=str(q),sha256=core.WEIGHTS[n],bytes=q.stat().st_size) for n,q in weights.items()},
            references={n:dict(path=str(q),sha256=core.REFERENCES[n][0],column=core.REFERENCES[n][1]) for n,q in refs.items()},
            sources={q.relative_to(out).as_posix():core.sha(q) for q in out.rglob('*') if q.is_file()},
            preparation_script_sha256=core.sha(Path(__file__)),training=False,official_rows_copied=False)
        core.dump(out/'PREPARATION.json',meta)
        core.validate_project(out)
        out.rename(target)
    print('Prepared files only:',target,'; checkpoints and references remain external/read-only')


if __name__=='__main__':main()
