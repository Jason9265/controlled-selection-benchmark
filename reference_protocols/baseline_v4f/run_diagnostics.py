"""Frozen v4f Mahé-only adaptation-stability diagnostic.

The program never opens old target results or future-confirmation files.  Pool
labels are loaded only after the preselected random indices have been frozen.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'baseline_v2'
OLD = ROOT.parent / 'baseline'
sys.path[:0] = [str(V2), str(OLD)]
import engine as numeric
from runner import (configure_backend, load_model, optimizer, sample_source,
                    source_statistics, weighted_ce)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(f'refusing overwrite: {path}')
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)
def rel_anchor(model, initial):
    terms=[]
    for name,p in model.encoder.named_parameters():
        q=initial[name]; den=q.float().square().sum()
        if den > 1e-12: terms.append((p.float()-q).square().sum()/den)
    return torch.stack(terms).mean() if terms else torch.zeros((),device='cuda')
def encoder_shift(model, initial): return float(rel_anchor(model,initial).sqrt().detach())
def lock_encoder(model):
    for p in model.encoder.parameters(): p.requires_grad_(False)
    model.encoder.eval()
def evaluate(model, source_val, dev_val, classes):
    model.eval()
    result={'source_val':numeric.evaluate(model,source_val,classes), 'development_val':numeric.evaluate(model,dev_val,classes)}
    return result
def select(pool_size, budget, query_seed):
    return np.random.default_rng(query_seed).permutation(pool_size)[:budget].astype(int).tolist()

def run_one(cfg, manifest, base_cfg, arch, train_seed, query_seed, variant, source, source_val, dev_val, stats, pools, output, smoke=False):
    run=output/'runs'/arch/f'm{train_seed}_q{query_seed}'/variant['name']
    complete=run/'complete.json'
    if complete.exists():
        return read(complete)
    if run.exists():
        raise FileExistsError(f'preserving incomplete run: {run}')
    selected=select(len(manifest['pool']),cfg['query']['budget_patches'],query_seed)
    # The manifest is in memory; only these label paths are passed to Pack.
    target=numeric.Pack([manifest['pool'][i] for i in selected],ROOT,True,base_cfg['num_classes'])
    model=load_model(V2/'runs/oem'/arch/'source'/str(train_seed)/'best.pt',arch,base_cfg['num_classes'],base_cfg)
    if variant['freeze_encoder']: lock_encoder(model)
    initial={n:p.detach().clone() for n,p in model.encoder.named_parameters()}
    opt=optimizer(model,base_cfg,arch)
    for group in opt.param_groups: group['lr']*=variant['lr_multiplier'];group['base_lr']*=variant['lr_multiplier']
    weights=torch.tensor(stats['class_weights'],device='cuda',dtype=torch.float32)
    rng=np.random.default_rng(train_seed)
    rows=[]; started=time.time(); half=base_cfg['adapt_batch']//2
    planned_steps=2 if smoke else cfg['adaptation']['steps']
    for step in range(1,planned_steps+1):
        model.train()
        if variant['freeze_encoder']: model.encoder.eval()
        ix=sample_source(rng,len(source),half,pools,.5); sx,sy=source.batch(ix); tx,ty=target.batch(rng.integers(len(target),size=half))
        x,y=numeric.augment(torch.cat([sx,tx]),torch.cat([sy,ty]),rng)
        opt.zero_grad(set_to_none=True)
        with numeric.amp():
            ce=weighted_ce(model(x)[0],y,weights); anchor=rel_anchor(model,initial); loss=ce+variant['anchor_lambda']*anchor
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step in cfg['adaptation']['checkpoints'] or (smoke and step==planned_steps):
            metrics=evaluate(model,source_val,dev_val,base_cfg['num_classes'])
            rows.append({'step':step,'ce':float(ce.detach()),'anchor':float(anchor.detach()),'loss':float(loss.detach()),'encoder_relative_shift':encoder_shift(model,initial),**metrics})
            model.train()
            if variant['freeze_encoder']: model.encoder.eval()
    if smoke:
        del target,model,opt;torch.cuda.empty_cache();return rows
    checkpoint=run/'model.pt'; run.mkdir(parents=True)
    numeric.save_model(checkpoint,model)
    receipt={'protocol':cfg['protocol'],'arch':arch,'train_seed':train_seed,'query_seed':query_seed,'variant':variant,
             'selected_indices':selected,'selected_ids':[manifest['pool'][i]['id'] for i in selected],
             'steps':cfg['adaptation']['steps'],'checkpoint_sha256':digest(checkpoint),'metrics':rows,
             'source_checkpoint_sha256':digest(V2/'runs/oem'/arch/'source'/str(train_seed)/'best.pt'),
             'manifest_sha256':digest(ROOT/cfg['manifest']),'config_sha256':digest(ROOT/'config.json'),
             'runner_sha256':digest(Path(__file__)),
             'target_label_scope':'only ten selected pool masks loaded for training; development-val labels read for diagnostics',
             'future_confirmation_accessed':False,'seconds':time.time()-started}
    write(run/'complete.json',receipt)
    del target,model,opt;torch.cuda.empty_cache();return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--arch',choices=['resnet18_unet','segformer_b0']);p.add_argument('--seed',type=int);p.add_argument('--variant')
    a=p.parse_args()
    if a.smoke==a.run: p.error('choose exactly one of --smoke / --run')
    cfg=read(ROOT/'config.json');raw_manifest=read(ROOT/cfg['manifest']);base_cfg=read(V2/'configs/oem.json')
    target=raw_manifest['targets']['mahe']
    manifest={'source_train':raw_manifest['source_train'],'source_val':raw_manifest['source_val'],
              'pool':target['pool'],'dev_val':target['test']}
    for key in ('source_train','source_val','pool','dev_val'): assert manifest[key]
    source=numeric.Pack(manifest['source_train'],ROOT,True,base_cfg['num_classes']);source_val=numeric.Pack(manifest['source_val'],ROOT,True,base_cfg['num_classes']);dev_val=numeric.Pack(manifest['dev_val'],ROOT,True,base_cfg['num_classes'])
    archs=[a.arch] if a.arch else cfg['architectures']; pairs=[x for x in cfg['seed_pairs'] if a.seed is None or x[0]==a.seed]; variants=[x for x in cfg['variants'] if a.variant is None or x['name']==a.variant]
    if not pairs or not variants:p.error('requested seed/variant absent from frozen config')
    for arch in archs:
        stats,_=source_statistics(source,base_cfg['num_classes']);saved=read(V2/'runs/oem'/arch/'source_statistics.json');assert stats==saved
        _,pools=source_statistics(source,base_cfg['num_classes'])
        for train_seed,query_seed in pairs:
            for variant in variants:
                result=run_one(cfg,manifest,base_cfg,arch,train_seed,query_seed,variant,source,source_val,dev_val,stats,pools,ROOT,smoke=a.smoke)
                print(json.dumps({'mode':'smoke' if a.smoke else 'run','arch':arch,'seed':train_seed,'variant':variant['name'],'last_step':result[-1]['step'] if a.smoke else result['metrics'][-1]['step']}),flush=True)

if __name__=='__main__':main()
