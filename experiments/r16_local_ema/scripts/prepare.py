"""Prepare a NEW local-EMA experiment using the frozen ROI-consistency recipe.

Stdlib only; no checkpoint deserialization, framework import, or training.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parent/'r16_roi_consistency'
spec = importlib.util.spec_from_file_location('ema_base_prepare',BASE/'scripts/prepare.py')
base = importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
replace = base.replace_once


def patch_controller(text):
    text = replace(text, 'from glaucoma.models import build_model',
                   'from glaucoma.models import build_model\nfrom glaucoma.local_ema import ModelEMA, DECAY')
    text = replace(text, "OUT=ROOT/'reports/R16-roi-consistency'", "OUT=ROOT/'reports/R16-local-ema'")
    text = replace(text, "ema=False,initializer_sha256=", "ema=dict(decay=DECAY,parameters='FP32',buffers='copy'),initializer_sha256=")
    text = replace(text, "    history=[];best=(-1.,-1.);updates=0;stale=0;stop='epoch_limit';start=time.monotonic()",
                   "    ema=ModelEMA(model,DECAY);ema_best=(-1.,-1.);paired_history=[]\n"
                   "    history=[];best=(-1.,-1.);updates=0;stale=0;stop='epoch_limit';start=time.monotonic()")
    text = replace(text, "    local=kind=='local';configure_runtime(True);seed()",
                   "    if kind!='local':raise ValueError('EMA experiment trains local only')\n"
                   "    if SET['ema_decay']!=DECAY:raise ValueError('EMA decay differs from registered implementation')\n"
                   "    local=True;configure_runtime(True);seed()")
    text = replace(text, "if stats['class_count']!=[49739,1657]:raise ValueError('Official train coverage changed')",
                   "if stats['class_count']!=[len(fit)-int(fit.label.sum()),int(fit.label.sum())]:raise ValueError('Train coverage changed')")
    text = replace(text, "expired,scaler,scheduler);updates+=stats['updates']",
                   "expired,scaler,scheduler,on_update=lambda:ema.update(model));updates+=stats['updates']\n"
                   "        if ema.updates!=updates:raise RuntimeError('EMA/update count differs')")
    text = replace(text, "optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict())",
                   "optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),ema=ema.state_dict())")
    text = replace(text, "        _checkpoint(run/'last.pt',payload);pd.DataFrame(history).to_csv(run/'history.csv',index=False)",
                   """        ema_pred=predict_bounded(ema.module,b,DEVICE,expired)
        if ema_pred is None:
            _checkpoint(run/'last.pt',payload);stop='ema_validation_budget';break
        ema_combined=fused(ema_pred,assets)
        ema_metrics=evaluate_predictions(val,ema_combined)
        ema_pred.to_csv(run/f'valid_ema_raw_{epoch:02d}.csv',index=False,float_format='%.17g')
        ema_combined.to_csv(run/f'valid_ema_fused_{epoch:02d}.csv',index=False,float_format='%.17g')
        if selection_key(ema_metrics)>ema_best:
            ema_best=selection_key(ema_metrics)
            _checkpoint(run/'best_ema.pt',dict(state_dict=ema.module.state_dict(),config=config,
                epoch=epoch,updates=updates,branch='ema',ema_decay=DECAY,ema_updates=ema.updates))
            dump(run/'ema_metrics.json',dict(epoch=epoch,metrics=ema_metrics))
            ema_combined.to_csv(run/'valid_ema_predictions.csv',index=False,float_format='%.17g')
        paired_history.append(dict(epoch=epoch,updates=updates,ema_updates=ema.updates,
            raw=metrics,ema=ema_metrics,seconds=time.monotonic()-start))
        dump(run/'paired_history.json',paired_history)
        _checkpoint(run/'last.pt',payload);pd.DataFrame(history).to_csv(run/'history.csv',index=False)""")
    text = replace(text, "ap=metrics['ap_probability'],seconds=time.monotonic()-start,free_bytes=free())",
                   "ap=metrics['ap_probability'],ema_auc=ema_metrics['auc_probability'],ema_ap=ema_metrics['ap_probability'],\n"
                   "            seconds=time.monotonic()-start,free_bytes=free())")
    text = replace(text, "    dump(run/'completion.json',result);dump(OUT/(kind+'_result.json'),result)",
                   """    result.update(status='completed' if stop in ('epoch_limit','auc_patience') else 'stopped_incomplete',
        ema_updates=ema.updates,paired_completed_epochs=len(paired_history),
        ema_best=json.loads((run/'ema_metrics.json').read_text()) if (run/'ema_metrics.json').exists() else None,
        ema_best_sha256=sha(run/'best_ema.pt') if (run/'best_ema.pt').exists() else None,
        comparison='Same-run matched epochs; independent best epochs are selection diagnostics',
        stop_control='Raw validation patience only; EMA does not change stopping')
    dump(run/'completion.json',result);dump(OUT/(kind+'_result.json'),result)""")
    text = replace(text, 'def final_b1(global_result,local_result,val):',
                   "def final_b1(global_result,local_result,val,checkpoint_name='best.pt',tag='raw'):")
    text = replace(text, "        ck=torch.load(Path(result['run'])/'best.pt',map_location='cpu',weights_only=True)",
                   "        path=Path(result['run'])/('best.pt' if result is global_result else checkpoint_name)\n"
                   "        if sha(path)!=result['best_sha256']:raise ValueError('Evaluation checkpoint changed')\n"
                   "        ck=torch.load(path,map_location='cpu',weights_only=True)")
    text = replace(text, "pred.to_csv(OUT/'official_valid_original_b1.csv',index=False)",
                   "pred.to_csv(OUT/f'official_valid_original_b1_{tag}.csv',index=False,float_format='%.17g')")
    text = replace(text, "weights=sum((Path(r['run'])/'best.pt').stat().st_size for r in [global_result,local_result])",
                   "weights=(Path(global_result['run'])/'best.pt').stat().st_size+(Path(local_result['run'])/checkpoint_name).stat().st_size")
    text = replace(text, "dump(OUT/'official_valid_b1_metrics.json',result)",
                   "dump(OUT/f'official_valid_b1_metrics_{tag}.json',result)")
    return text


def patch_runner(text):
    text = replace(text, 'SERVER ONLY, isolated R16 stages; --help is stdlib-only. No automatic training.',
                   'SERVER ONLY, paired local-EMA stages; --help is stdlib-only. No automatic training.')
    text = replace(text, "report=root/'reports/R16-roi-consistency'", "report=root/'reports/R16-local-ema'")
    text = replace(text, "        controller.configure_runtime(False);controller.seed()",
                   "        from glaucoma.local_ema import ModelEMA, server_self_check, server_training_smoke\n"
                   "        ema_contract=server_self_check(torch.device('cuda'))\n"
                   "        controller.configure_runtime(False);controller.seed()")
    text = replace(text, "        before=local.head[-1].weight.detach().clone()",
                   "        before=local.head[-1].weight.detach().clone()\n        ema=ModelEMA(local)")
    text = replace(text, "torch.amp.GradScaler('cuda',init_scale=16.))",
                   "torch.amp.GradScaler('cuda',init_scale=16.),on_update=lambda:ema.update(local))\n"
                   "        if ema.updates!=stats['updates']:raise RuntimeError('Real local EMA count mismatch')\n"
                   "        if any(not torch.isfinite(v).all() for v in ema.module.state_dict().values()):raise RuntimeError('Nonfinite local EMA')")
    text = replace(text, "random_initialized_local_weights_discarded=True,generic_initializer_checked=False))",
                   "random_initialized_local_weights_discarded=True,generic_initializer_checked=False,\n"
                   "                         ema_contract=ema_contract,local_ema_updates=ema.updates,\n"
                   "                         training_smoke=server_training_smoke(controller,report)))")
    text = replace(text, "print('LOCAL_TRAINING_COMPLETED; run evaluate explicitly')",
                   "print('LOCAL_TRAINING_ENDED',result['status'],result['stop'],'; run evaluate explicitly')\n"
                   "        return 0 if result['status']=='completed' else 3")
    text = replace(text, "        evaluation=controller.final_b1(reference,result,valid)\n        controller.dump(report/'results.json',dict(frozen_global=reference,local=result,evaluation=evaluation,variant=registration['variant']))",
                   """        if (report/'results.json').exists():raise FileExistsError('Evaluation already completed')
        if not result['ema_best_sha256']:raise RuntimeError('No fully validated EMA checkpoint')
        evaluation=controller.final_b1(reference,result,valid)
        ema_result=dict(result,best_sha256=result['ema_best_sha256'])
        ema_evaluation=controller.final_b1(reference,ema_result,valid,checkpoint_name='best_ema.pt',tag='ema')
        controller.dump(report/'results.json',dict(frozen_global=reference,local=result,raw=evaluation,
            ema=ema_evaluation,variant=registration['variant'],
            comparison='Independently selected best epochs; see paired_history.json for matched epochs'))""")
    text = replace(text, "if __name__=='__main__': main()", "if __name__=='__main__': raise SystemExit(main())")
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data-root','global-checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--local-initializer',type=Path)
    args=parser.parse_args()
    target=args.output.resolve()
    if target.exists():raise FileExistsError('Choose a NEW output directory')
    # Validate source-patch contracts before any preparation; original vendor is immutable.
    _,controller=base.patched_sources(BASE/'vendor')
    controller=patch_controller(controller)
    runner=patch_runner((BASE/'scripts/run.py').read_text(encoding='utf-8'))
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ema_prepare_',dir=target.parent) as temporary:
        staging=Path(temporary)/'project'
        argv=['prepare','--output',str(staging),'--data-root',str(args.data_root),
              '--global-checkpoint',str(args.global_checkpoint)]
        if args.local_initializer:argv+=['--local-initializer',str(args.local_initializer)]
        previous=sys.argv
        try:sys.argv=argv;base.main()
        finally:sys.argv=previous
        (staging/'tools/run_round_sixteen.py').write_text(controller,encoding='utf-8',newline='\n')
        (staging/'run.py').write_text(runner,encoding='utf-8',newline='\n')
        shutil.copyfile(HERE/'src/local_ema.py',staging/'src/glaucoma/local_ema.py')
        (staging/'reports/R16-roi-consistency').rename(staging/'reports/R16-local-ema')
        config=staging/'configs/r16_registration_20261026.json'
        settings=json.loads(config.read_text());settings.update(variant='r16_local_ema',local_seconds=10800,
            ema_decay=.999,ema_parameters='FP32',ema_buffers='copy',ema_stop_control='raw_patience')
        config.write_text(json.dumps(settings,indent=2),encoding='utf-8')
        meta=json.loads((staging/'PREPARATION.json').read_text())
        meta.update(variant='r16_local_ema',ema=dict(decay=.999,parameters='FP32',buffers='copy',
            updates='Successful optimizer steps only; no probe steps',stop_control='raw_patience'),
            preparation_generator_sha256=base.sha(Path(__file__)),ema_helper_sha256=base.sha(HERE/'src/local_ema.py'),
            base_preparation_generator_sha256=base.sha(BASE/'scripts/prepare.py'),
            base_runner_sha256=base.sha(BASE/'scripts/run.py'))
        meta['changes']+=['Local-only EMA paired trajectory; preserve raw best.pt',
            'Ordinary recipe patience/epochs/optimizer unchanged; local budget 10800s includes extra validation']
        meta['source_sha256']={p.relative_to(staging).as_posix():base.sha(p) for folder in ('src','tools','configs','manifests','pretrained')
                               for p in (staging/folder).rglob('*') if p.is_file()}
        meta['runner_sha256']=base.sha(staging/'run.py')
        split=staging/'reports/R16-local-ema/split_registration.json'
        record=json.loads(split.read_text());record['variant']='r16_local_ema'
        split.write_text(json.dumps(record,indent=2),encoding='utf-8')
        (staging/'PREPARATION.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        # Publish only a fully constructed experiment; no runnable base stage is exposed.
        staging.rename(target)
    print('EMA project prepared only:',target)


if __name__=='__main__':main()
