"""v4f source-model predictive-entropy confirmation control."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;V2=ROOT.parent/'baseline_v2';OLD=ROOT.parent/'baseline';sys.path[:0]=[str(V2),str(OLD)]
import engine as numeric
from runner import load_model,optimizer,sample_source,source_statistics,weighted_ce
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def anchor(model,initial):
 z=[]
 for n,p in model.encoder.named_parameters():
  d=initial[n].float().square().sum()
  if d>1e-12:z.append((p.float()-initial[n]).square().sum()/d)
 return torch.stack(z).mean()
def entropy_order(pool,arch,seed,base):
 target=numeric.Pack(pool,ROOT,False,base['num_classes'])
 model=load_model(V2/'runs/oem'/arch/'source'/str(seed)/'best.pt',arch,base['num_classes'],base)
 scores=numeric.score(model,target,base['num_classes'])
 order=np.argsort(-scores['entropy'],kind='stable').astype(int).tolist()
 del target,model;torch.cuda.empty_cache()
 return order
def adapt(cfg,region,man,base,arch,ts,qs,var,src,stats,pools):
 budget_rows=[];order=entropy_order(man['pool'],arch,ts,base)
 budgets=[max(1,int(np.ceil(len(man['pool'])*x))) for x in cfg['budget_fractions']]
 if len(set(budgets))!=len(budgets):raise RuntimeError('collapsed budgets '+region)
 for budget in budgets:
  out=ROOT/'entropy_runs'/region/arch/f'm{ts}_q{qs}'/var['name']/f'b{budget}'
  if (out/'adapt_complete.json').exists():continue
  if out.exists():raise FileExistsError(out)
  chosen=order[:budget];target=numeric.Pack([man['pool'][i] for i in chosen],ROOT,True,base['num_classes'])
  model=load_model(V2/'runs/oem'/arch/'source'/str(ts)/'best.pt',arch,base['num_classes'],base)
  if var['freeze_encoder']:
   for p in model.encoder.parameters():p.requires_grad_(False)
  initial={n:p.detach().clone() for n,p in model.encoder.named_parameters()};opt=optimizer(model,base,arch)
  for g in opt.param_groups:g['lr']*=var['lr_multiplier']
  w=torch.tensor(stats['class_weights'],device='cuda',dtype=torch.float32);rng=np.random.default_rng(ts);start=time.time()
  for step in range(1,301):
   model.train()
   if var['freeze_encoder']:model.encoder.eval()
   ii=sample_source(rng,len(src),4,pools,.5);sx,sy=src.batch(ii);tx,ty=target.batch(rng.integers(len(target),size=4));x,y=numeric.augment(torch.cat([sx,tx]),torch.cat([sy,ty]),rng)
   opt.zero_grad(set_to_none=True)
   with numeric.amp():ce=weighted_ce(model(x)[0],y,w);reg=anchor(model,initial);loss=ce+var['anchor_lambda']*reg
   if not torch.isfinite(loss):raise RuntimeError('nonfinite')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
  out.mkdir(parents=True);numeric.save_model(out/'model.pt',model)
  put(out/'adapt_complete.json',{'region':region,'arch':arch,'train_seed':ts,'query_seed':qs,'variant':var,'budget':budget,'indices':chosen,'ids':[man['pool'][i]['id'] for i in chosen],'original_groups':[man['pool'][i]['group'] for i in chosen],'steps':300,'model_sha256':sha(out/'model.pt'),'source_checkpoint_sha256':sha(V2/'runs/oem'/arch/'source'/str(ts)/'best.pt'),'config_sha256':sha(ROOT/'entropy_config.json'),'manifest_sha256':sha(ROOT/cfg['manifest']),'final_ce':float(ce.detach()),'final_anchor':float(reg.detach()),'seconds':time.time()-start,'confirmation_test_labels_read_by_training':False})
  del target,model,opt;torch.cuda.empty_cache();print(json.dumps({'phase':'adapt','region':region,'arch':arch,'seed':ts,'variant':var['name'],'budget':budget}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['adapt','evaluate']);a=p.parse_args();cfg=read(ROOT/'entropy_config.json');man=read(ROOT/cfg['manifest']);base=read(V2/'configs/oem.json');expected=36
 if a.phase=='adapt':
  src=numeric.Pack(man['source_train'],ROOT,True,base['num_classes']);stats,pools=source_statistics(src,base['num_classes']);assert stats==read(V2/'runs/oem/resnet18_unet/source_statistics.json')
  for arch in cfg['architectures']:
   for ts,qs in cfg['seed_pairs']:
    adapt_var=cfg['variants'][arch]
    for region,target in man['targets'].items():adapt(cfg,region,target,base,arch,ts,qs,adapt_var,src,stats,pools)
  put(ROOT/'entropy_runs/adaptation_complete.json',{'expected':expected,'found':len(list((ROOT/'entropy_runs').rglob('adapt_complete.json'))),'test_labels_read_by_training':False})
 else:
  gate=read(ROOT/'entropy_runs/adaptation_complete.json');assert gate['expected']==gate['found']==expected
  for region,target in man['targets'].items():
   test=numeric.Pack(target['test'],ROOT,True,base['num_classes'])
   for record in (ROOT/'entropy_runs'/region).rglob('adapt_complete.json'):
    d=record.parent;model=load_model(d/'model.pt',read(record)['arch'],base['num_classes'],base);metrics=numeric.evaluate(model,test,base['num_classes']);put(d/'evaluation.json',{'metrics':metrics,'model_sha256':sha(d/'model.pt'),'confirmation_test_labels_read_by_evaluation':True});del model;torch.cuda.empty_cache()
   del test;torch.cuda.empty_cache()
  put(ROOT/'entropy_runs/evaluation_complete.json',{'expected':expected,'found':len(list((ROOT/'entropy_runs').rglob('evaluation.json')))})
if __name__=='__main__':main()
