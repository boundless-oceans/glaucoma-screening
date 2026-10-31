"""Stdlib-only preparation of a NEW frozen original-R16 inference project."""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from r16_tta.contracts import WEIGHTS,REFERENCE_SHA,POLICY,sha,validation_rows
sys.path.insert(0,str(HERE.parent/'r16_roi_consistency/scripts'))
import manifest_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data-root','global-checkpoint','local-checkpoint','reference-predictions'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    target=args.output.resolve();data=args.data_root.resolve()
    if target.exists():raise FileExistsError('Choose a NEW output directory')
    weights={name:getattr(args,name+'_checkpoint').resolve() for name in WEIGHTS}
    for name,path in weights.items():
        if sha(path)!=WEIGHTS[name]:raise ValueError('Requires original R16 '+name+' checkpoint; do not substitute ROI-consistency weights')
    reference=args.reference_predictions.resolve()
    if sha(reference)!=REFERENCE_SHA:raise ValueError('Requires archived original R16 validation predictions')
    vendor=HERE.parent/'r16_roi_consistency/vendor'
    records=json.loads((vendor/'SOURCE_SHA256.json').read_bytes())
    for name,expected in records.items():
        if sha(vendor/name)!=expected:raise ValueError('Vendored source differs: '+name)
    payload=manifest_data.manifest_payloads(data,('valid',))['valid']
    target.mkdir(parents=True)
    shutil.copytree(vendor/'original/src',target/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(HERE/'src/r16_tta',target/'src/r16_tta',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(HERE/'scripts/run.py',target/'run.py')
    (target/'valid_manifest.csv').write_bytes(payload)
    validation_rows(target/'valid_manifest.csv',data/'valid.csv',reference)
    shutil.copyfile(reference,target/'reference_predictions.csv')
    (target/'outputs').mkdir()
    config=dict(policy=POLICY,data_root=str(data),official_valid_sha256=sha(data/'valid.csv'),
                weights={name:dict(path=str(path),sha256=WEIGHTS[name]) for name,path in weights.items()},
                reference_sha256=REFERENCE_SHA,vendor_sha256=records,
                role_manifests_generated_locally=True,manifest_generator_sha256=sha(HERE.parent/'r16_roi_consistency/scripts/manifest_data.py'),
                source_sha256={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()},
                baseline_guard=dict(max_abs_probability_delta=0.001,max_abs_auc_delta=0.0001,max_abs_ap_delta=0.0005),
                training=False,test_export=False,single_view_export_available=True,
                official_test_sha256=sha(data/'test.csv') if (data/'test.csv').is_file() else None)
    (target/'EXPERIMENT.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Prepared inference files only:',target)


if __name__=='__main__':main()
