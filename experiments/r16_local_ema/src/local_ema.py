"""EMA of local parameters only. Importing this module never imports torch."""
import copy
import math

DECAY = 0.999


def validate_decay(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError('EMA decay must be finite and in [0, 1)')
    return float(value)


def average_state_(target, source, parameters, decay):
    """In-place array/tensor arithmetic; caller owns no-grad and float32 policy.

    Validate the entire structure before mutating. Buffers are copied, never
    averaged, including integer counters and floating running statistics.
    """
    decay = validate_decay(decay)
    if target.keys() != source.keys() or not set(parameters) <= set(target):
        raise ValueError('EMA state keys changed')
    for name in target:
        if target[name].shape != source[name].shape or target[name].dtype != source[name].dtype:
            raise ValueError('EMA shape/dtype changed: ' + name)
    for name, dst in target.items():
        if name in parameters:
            dst *= decay
            dst += source[name] * (1 - decay)
        else:
            dst[...] = source[name]


class ModelEMA:
    def __init__(self, model, decay=DECAY):
        self.decay = validate_decay(decay)
        self.updates = 0
        self.module = copy.deepcopy(model).float().eval().requires_grad_(False)
        self.parameters = frozenset(dict(self.module.named_parameters()))
        # AMP leaves the ordinary model in FP32. Reject other parameter types
        # rather than silently changing the declared averaging precision.
        import torch
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('EMA requires FP32 ordinary parameters (AMP is allowed)')

    def update(self, model):
        import torch
        with torch.no_grad():
            average_state_(self.module.state_dict(), model.state_dict(), self.parameters, self.decay)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, state_dict=self.module.state_dict())

    def load_state_dict(self, state):
        if state['decay'] != self.decay or type(state['updates']) is not int or state['updates'] < 0:
            raise ValueError('EMA policy/count differs')
        self.module.load_state_dict(state['state_dict'], strict=True)
        self.module.eval().requires_grad_(False)
        self.updates = state['updates']


def server_self_check(device):
    """Tiny real-tensor/AMP-skip integration. Explicit Linux server invocation only."""
    import io
    import torch
    from glaucoma.r08 import bounded_train_epoch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)
            self.register_buffer('count', torch.tensor(0, dtype=torch.int64))
            self.register_buffer('running', torch.tensor(0.))

        def forward(self, image, age):
            return self.layer(image).squeeze(1)

    model = Tiny().to(device)
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device.type == 'cuda' else None
    ema = ModelEMA(model, .5)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None: assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert all(not p.requires_grad for p in ema.module.parameters()) and not ema.module.training
    assert all(ema.module.state_dict()[k].data_ptr() != v.data_ptr() for k,v in model.state_dict().items())
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _:1.)
    batch = dict(image=torch.tensor([[1.,2.],[-1.,-2.],[2.,1.],[-2.,-1.]]),
                 age=torch.zeros(4,2), label=torch.tensor([1.,0.,1.,0.]))
    expected = {k:v.clone() for k,v in before.items()}
    def updated():
        # Change both buffer types to exercise copy rather than averaging.
        model.count.add_(1);model.running.add_(2.)
        with torch.no_grad():
            for name,p in model.state_dict().items():
                expected[name] = .5 * expected[name] + .5 * p if name in ema.parameters else p.clone()
        ordinary={k:v.detach().clone() for k,v in model.state_dict().items()}
        ema.update(model)
        assert all(torch.equal(ordinary[k],v) for k,v in model.state_dict().items())
    scaler = torch.amp.GradScaler('cuda', init_scale=16.) if device.type == 'cuda' else None
    stats = bounded_train_epoch(model, [batch]*2, opt, device, 1., lambda:False, scaler, scheduler, updated)
    assert stats['updates'] == ema.updates == 2
    assert all(torch.equal(expected[k],v) for k,v in ema.module.state_dict().items())
    assert any(not torch.equal(model.state_dict()[k],before[k]) for k in ema.parameters)
    skipped = 0
    if scaler is not None:
        snapshot = {k:v.clone() for k,v in model.state_dict().items()}
        scheduler_step = scheduler.last_epoch
        hook = model.layer.weight.register_hook(lambda g:g*float('inf'))
        failed = bounded_train_epoch(model, [batch], opt, device, 1., lambda:False, scaler, scheduler, updated)
        hook.remove()
        assert failed['updates'] == 0 and failed['skipped_updates'] == 1 and ema.updates == 2
        assert scheduler.last_epoch == scheduler_step
        assert all(torch.equal(snapshot[k],v) for k,v in model.state_dict().items())
        skipped = 1
    # The EMA checkpoint is restorable; this does not imply trainer resume.
    stream = io.BytesIO();torch.save(ema.state_dict(), stream);stream.seek(0)
    restored = ModelEMA(model,.5)
    restored.load_state_dict(torch.load(stream,weights_only=True,map_location=device))
    assert restored.updates == 2
    assert all(torch.equal(v,restored.module.state_dict()[k]) for k,v in ema.module.state_dict().items())
    # Evaluating the shadow must not change the ordinary weights/buffers or RNG.
    snapshot = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.inference_mode(): restored.module(batch['image'].to(device),batch['age'].to(device))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(snapshot[k],v) for k,v in model.state_dict().items())
    return dict(successful_updates=2, forced_amp_skips=skipped, checkpoint_round_trip=True,
                parameters_fp32=True, buffers_copied=True, ordinary_unchanged_by_ema=True)


def server_training_smoke(controller, report):
    """Run the generated train loop for two synthetic tiny epochs on the server.

    Replace only heavyweight dependencies and paths. Exercise the real loop,
    scaler/callbacks, both validations, selection, checkpoints and paired report.
    This is a wiring test, not real-data/model performance or initializer QA.
    """
    import json
    from pathlib import Path
    import tempfile
    from unittest.mock import patch
    import pandas as pd
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Module()
            self.backbone.features=torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(3,2,1)))
            self.head=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(2,1))

        def forward(self, images, age):
            return self.head(self.backbone.features(images).mean((2,3))).squeeze(1)

    class Data(torch.utils.data.Dataset):
        def __init__(self, frame, *unused):self.frame=frame
        def __len__(self):return len(self.frame)
        def __getitem__(self,index):
            row=self.frame.iloc[index]
            return dict(image=torch.full((3,8,8),.01 if row.label else -.01),age=torch.zeros(2),
                        label=torch.tensor(float(row.label)),image_id=row.image_id)

    real_loader=torch.utils.data.DataLoader
    def loader(dataset,**kwargs):
        for key in ('persistent_workers','prefetch_factor','worker_init_fn'):kwargs.pop(key,None)
        kwargs['num_workers']=0
        return real_loader(dataset,**kwargs)

    with tempfile.TemporaryDirectory(prefix='synthetic_ema_',dir=report) as temp:
        root=Path(temp);out=root/'reports';out.mkdir();(root/'runs').mkdir()
        (root/'src/glaucoma').mkdir(parents=True)
        (root/'src/glaucoma/SYNTHETIC.txt').write_text('Tiny wiring fixture; no official data')
        (out/'split_registration.json').write_text('{}');(out/'roi_boxes.csv').write_text('synthetic')
        pretrained=root/'pretrained/convnext22k_r10';pretrained.mkdir(parents=True)
        (pretrained/'provenance.json').write_text(json.dumps(dict(domain='synthetic fixture')))
        teacher=root/'teacher.pt';teacher.write_bytes(b'fixture not loaded')
        initializer=root/'initializer.pth';initializer.write_bytes(b'fixture not loaded')
        settings=dict(controller.SET,local_epochs=2,local_seconds=120,local_initializer=str(initializer),patience=5)
        sf=root/'config.json';sf.write_text(json.dumps(settings))
        fit=pd.DataFrame(dict(image_id=['synthetic_train_%02d'%i for i in range(32)],label=[1,1]+[0]*30))
        valid=pd.DataFrame(dict(image_id=['synthetic_valid_%02d'%i for i in range(4)],label=[1,0,1,0]))
        assets=pd.DataFrame(dict(image_id=valid.image_id,global_logit=[.1,-.1,.1,-.1],fallback=False))
        replacements=dict(ROOT=root,OUT=out,SF=sf,SET=settings,build_model=lambda *a:Tiny(),
            FundusDataset=Data,DataLoader=loader,free=lambda:2_000_000_000,
            load_22k_initializer=lambda *a:dict(source='synthetic fixture; no pretrained file loaded'))
        with patch.multiple(controller,**replacements):
            result=controller.train('local',fit,valid,assets,teacher)
        run=Path(result['run'])
        paired=json.loads((run/'paired_history.json').read_text())
        assert result['status']=='completed' and result['completed_epochs']==result['paired_completed_epochs']==2
        assert result['updates']==result['ema_updates'] and result['updates']>=2
        assert len(paired)==2 and all(p['updates']==p['ema_updates'] for p in paired)
        for name,branch in [('best.pt','raw'),('best_ema.pt','ema')]:
            ck=torch.load(run/name,map_location='cpu',weights_only=True)
            assert ck['branch']==branch
            assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        assert last['ema']['updates']==last['updates']==result['updates']
        assert last['ema']['decay']==DECAY
        for epoch in (1,2):
            for prefix in ('valid_raw_','valid_fused_','valid_ema_raw_','valid_ema_fused_'):
                frame=pd.read_csv(run/(prefix+'%02d.csv'%epoch))
                assert frame.image_id.tolist()==valid.image_id.tolist()
        summary=dict(synthetic_only=True,ordinary_and_ema_epochs=2,updates=result['updates'],
                     paired_reports_checked=True,both_selected_checkpoints_loaded=True,
                     last_checkpoint_ema_loaded=True,generic_initializer_tested=False,
                     formal_training_started=False)
    return summary
