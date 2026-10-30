"""Prepare an isolated R16 ROI-consistency project. Stdlib only; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]
GLOBAL_SHA = 'b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2'
LOCAL_INITIALIZER_SHA = 'f22f8850dcded245f4ae3bc402bbde9c6f5e4e23e76dfaea3aadb75aa9d3705a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Archived source changed; refusing ambiguous patch: '+old)
    return text.replace(old, new, 1)


def patched_sources(vendor):
    data=(vendor/'original/src/glaucoma/data.py').read_text(encoding='utf-8-sig')
    data=replace_once(data,"boxes=pd.read_csv(config['local_region']['boxes_csv'])",
                      "boxes=pd.read_csv(config['local_region']['boxes_csv'],float_precision='round_trip')")
    controller=(vendor/'original/tools/run_round_sixteen.py').read_text(encoding='utf-8-sig')
    controller=replace_once(controller,"OUT=ROOT/'reports/R16-20261026-official'", "OUT=ROOT/'reports/R16-roi-consistency'")
    controller=replace_once(controller,"[('train',fit,32),('valid',val,1)]", "[('train',fit,1),('valid',val,1)]")
    controller=replace_once(controller,"asset.to_csv(OUT/(role+'_assets.csv'),index=False)",
                            "asset.to_csv(OUT/(role+'_assets.csv'),index=False,float_format='%.17g')")
    controller=replace_once(controller,"pd.concat(frames,ignore_index=True).to_csv(OUT/'roi_boxes.csv',index=False)",
                            "pd.concat(frames,ignore_index=True).to_csv(OUT/'roi_boxes.csv',index=False,float_format='%.17g')")
    controller=replace_once(controller,"round='R16'", "round='R16-roi-consistency'")
    controller=replace_once(controller,"version='r16-official-train-only'", "version='r16-consistency-b1-roundtrip'")
    return data, controller


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--global-checkpoint',type=Path,required=True)
    parser.add_argument('--local-initializer',type=Path)
    args=parser.parse_args()
    target=args.output.resolve(); data=args.data_root.resolve(); weights=args.global_checkpoint.resolve()
    if target.exists(): raise FileExistsError('Choose a NEW output directory')
    if not data.is_dir() or not all((data/(s+'.csv')).is_file() for s in ('train','valid')):
        raise ValueError('Official extracted data root with train.csv/valid.csv required')
    if sha(weights)!=GLOBAL_SHA: raise ValueError('Requires original R16 E32 epoch7 global checkpoint')
    initializer=args.local_initializer.resolve() if args.local_initializer else None
    if initializer is not None and sha(initializer)!=LOCAL_INITIALIZER_SHA:
        raise ValueError('Requires original generic ConvNeXt 22K->1K/384 initializer')
    vendor=HERE/'vendor'; records=json.loads((vendor/'SOURCE_SHA256.json').read_bytes())
    for name,expected in records.items():
        if sha(vendor/name)!=expected: raise ValueError('Vendored file differs: '+name)
    patched_data, patched_controller=patched_sources(vendor)
    target.mkdir(parents=True)
    (target/'runs').mkdir()
    shutil.copytree(vendor/'original/src',target/'src')
    (target/'tools').mkdir()
    (target/'src/glaucoma/data.py').write_text(patched_data,encoding='utf-8',newline='\n')
    (target/'tools/run_round_sixteen.py').write_text(patched_controller,encoding='utf-8',newline='\n')
    shutil.copyfile(HERE/'scripts/run.py',target/'run.py')
    shutil.copyfile(vendor/'requirements-training.txt',target/'requirements-recorded.txt')
    config=json.loads((vendor/'original/configs/r16_registration_20261026.json').read_bytes())
    config.update(data_root=str(data),local_initializer=str(initializer) if initializer else '',
                  assets_seconds=3600,variant='r16_roi_consistency',authorized_by='explicit_user_stage_command')
    config.pop('server_port',None)
    (target/'configs').mkdir()
    (target/'configs/r16_registration_20261026.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    manifest_dir=target/'manifests';manifest_dir.mkdir()
    for split in ('train','valid'):
        shutil.copyfile(vendor/f'original/manifests/official_{split}.csv',manifest_dir/(split+'.csv'))
    pretrained=target/'pretrained/convnext22k_r10';pretrained.mkdir(parents=True)
    shutil.copyfile(vendor/'original/pretrained/local_initializer_provenance.json',pretrained/'provenance.json')
    shutil.copyfile(vendor/'original/pretrained/LICENSE.ConvNeXt',pretrained/'LICENSE')
    (target/'reports/R16-roi-consistency').mkdir(parents=True)
    # The reference remains external and read-only; the driver checks its hash at every stage.
    metadata=dict(variant='r16_roi_consistency',global_checkpoint=str(weights),global_checkpoint_sha256=GLOBAL_SHA,
        data_root=str(data),official_csv_sha256={s:sha(data/(s+'.csv')) for s in ('train','valid')},
        generic_local_initializer_sha256=LOCAL_INITIALIZER_SHA,
        vendored_source_sha256=records,
        original_source_sha256=json.loads((vendor/'UPSTREAM_SHA256.json').read_bytes()),
        source_sha256={p.relative_to(target).as_posix():sha(p) for folder in ('src','tools','configs','manifests','pretrained')
                       for p in (target/folder).rglob('*') if p.is_file()},
        runner_sha256=sha(target/'run.py'),
        changes=['ROI CSV round_trip read','ROI coordinate CSV 17 significant digits',
                 'train and valid CAM assets both B1 to match live inference',
                 'new project/report identity','local-only stages reuse frozen E32',
                 'asset budget 3600s instead of 1200s for serial B1; fail on timeout, no auto retry'],
        training_started=False,initializer_optional_for_check_and_assets=True,
        resume='Original local trainer is one-shot; no mid-batch resume or auto retraining')
    (target/'PREPARATION.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Prepared only:',target)


if __name__=='__main__': main()
