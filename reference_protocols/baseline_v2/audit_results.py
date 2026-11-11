"""Independent, CPU-only fail-closed audit of completed baseline_v2 runner artifacts.
Does not import the runner, deserialize models, or read image/mask pixels.
Use only after all architectures finish evaluation. File timestamps are evidence,
not proof of absent unlogged accesses. Missing/incomplete artifacts never pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

class AuditError(ValueError): pass

def require(condition, message):
    if not condition: raise AuditError(message)
def read(path):
    require(Path(path).is_file(), 'missing artifact: '+str(path))
    return json.loads(Path(path).read_text())
def digest(path):
    require(Path(path).is_file(), 'missing hashed file: '+str(path))
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def close(a,b):
    return isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-9)

def metrics(row, classes):
    cm=row['confusion_matrix']
    require(len(cm)==classes and all(len(r)==classes for r in cm),'CM dimensions')
    require(all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for r in cm for x in r),'CM integer/nonnegative')
    truth=[sum(r) for r in cm];pred=[sum(cm[i][j] for i in range(classes)) for j in range(classes)]
    union=[truth[i]+pred[i]-cm[i][i] for i in range(classes)]
    iou=[cm[i][i]/union[i] if union[i] else 0. for i in range(classes)]
    require(len(row['iou'])==classes and all(close(a,b) for a,b in zip(row['iou'],iou)),'per-class IoU mismatch')
    require(close(row['miou_fixed'],sum(iou)/classes),'fixed-class mIoU mismatch')
    require(row['empty_union']==[i for i,u in enumerate(union) if not u],'empty unions mismatch')
    support=[i for i,n in enumerate(truth) if n]
    require(row['gt_supported_classes']==support,'truth-supported classes mismatch')
    require((close(row['gt_supported_miou'],sum(iou[i] for i in support)/len(support)) if support else row['gt_supported_miou'] is None),'supported mIoU mismatch')
    require(close(row['accuracy'],sum(cm[i][i] for i in range(classes))/max(sum(truth),1)),'accuracy mismatch')
    require(row['valid_pixels']==sum(truth),'valid-pixel count mismatch')
    require(row['metric_policy']=='fixed class denominator; empty union=0','metric policy mismatch')
    return iou

def history_check(history, done, cfg):
    require(bool(history),'empty source history')
    require([r['step'] for r in history]==list(range(cfg['source_eval_every'],done['steps']+1,cfg['source_eval_every'])),'source evaluation cadence')
    require(done['steps']<=cfg['source_max_steps'] and done['steps']%cfg['source_eval_every']==0,'source step range')
    keys=[];monitor=(-1,-1.);stale=0;first_stop=None
    for row in history:
        iou=metrics(row['validation'],cfg['num_classes'])
        eligible=all(v>=cfg['minimum_class_iou'] for v in iou)
        require(row['eligible'] is eligible,'recorded eligibility mismatch')
        require(close(row['train_loss'],row['train_loss']),'nonfinite source loss')
        key=(int(eligible),row['validation']['miou_fixed']);keys.append(key)
        if key[0]>monitor[0] or (key[0]==monitor[0] and key[1]>monitor[1]+cfg['source_min_delta']):monitor=key;stale=0
        else:stale+=1
        if row['step']>=cfg['source_min_steps'] and stale>=cfg['source_patience'] and first_stop is None:first_stop=row['step']
    index=max(range(len(keys)),key=lambda i:keys[i]) # first tie, as strict > in runner
    require(done['best_step']==history[index]['step'],'lexicographic source checkpoint selection mismatch')
    require(close(done['best_source_val_miou'],keys[index][1]),'best source mIoU mismatch')
    require(done['best_source_val_iou']==history[index]['validation']['iou'],'best source class IoU mismatch')
    require(done['eligible_classes'] is bool(keys[index][0]),'best source eligibility mismatch')
    plateau=stale>=cfg['source_patience']
    require(done['convergence_diagnosed'] is plateau,'plateau diagnostic mismatch')
    require(done['steps']==(first_stop if first_stop is not None else cfg['source_max_steps']),'premature/late source stop')
    require(done['stop_reason']==('source_validation_plateau' if plateau else 'max_steps_reached_convergence_not_established'),'source stop reason')
    require(done['minimum_class_iou_threshold']==cfg['minimum_class_iou'],'source threshold mismatch')
    return keys[index]

def manifest_check(manifest):
    used_ids=set();used_groups=set();used_rgb={}
    splits=[manifest['source_train'],manifest['source_val']]+[v[k] for v in manifest['targets'].values() for k in ('pool','test')]
    require(bool(manifest['targets']),'no target regions')
    for rows in splits:
        require(bool(rows),'empty manifest split');groups={r['group'] for r in rows}
        require(not groups&used_groups,'original group overlaps splits');used_groups|=groups
        for r in rows:
            require(r['id'] not in used_ids,'duplicate patch ID');used_ids.add(r['id'])
            require(r['area_pixels']==65536,'non-256 patch budget')
            h=r.get('original_rgb_sha256')
            if h:
                require(h not in used_rgb or used_rgb[h]==r['group'],'identical RGB in distinct groups');used_rgb[h]=r['group']

def selection_check(selection,done,rows,previous,budget):
    ids=selection['indices']
    require(len(ids)==budget and len(set(ids))==budget,'selection count/duplicates')
    require(all(isinstance(i,int) and not isinstance(i,bool) and 0<=i<len(rows) for i in ids),'selection outside pool')
    require(ids[:len(previous)]==previous and selection['previous']==previous,'non-nested budgets')
    require(selection['ids']==[rows[i]['id'] for i in ids],'selection IDs not pool-derived')
    require(selection['groups']==[rows[i]['group'] for i in ids],'selection groups not pool-derived')
    require(all(done.get(k)==v for k,v in selection.items()),'complete versus selection mismatch')
    return ids

def audit(config_path,manifest_path,runs_root,source_manifest_path=None,synthetic_fixture=False):
    config_path=Path(config_path).resolve();manifest_path=Path(manifest_path).resolve();runs_root=Path(runs_root).resolve()
    cfg=read(config_path);manifest=read(manifest_path);root=Path(__file__).resolve().parent
    if synthetic_fixture:
        require(cfg.get('synthetic_fixture') is True,'synthetic mode requires explicit config flag')
        for path in (config_path,manifest_path,runs_root,*([Path(source_manifest_path).resolve()] if source_manifest_path else [])):
            require(path.is_relative_to(root/'tests') and path.relative_to(root/'tests').parts[0].startswith('fixture_runtime'),'synthetic mode restricted to tests/fixture_runtime* paths')
    else:
        require(cfg.get('synthetic_fixture') is not True,'synthetic config cannot pass as formal experiment')
        require(cfg['adapt_steps']==300,'frozen protocol requires 300 adaptation steps')
        require(cfg['budget_fractions']==[.01,.03,.05],'frozen budget fractions')
        require(cfg['seed_pairs']==[[2026,17],[2027,42],[2028,73]],'frozen paired seeds')
    require(isinstance(cfg['adapt_steps'],int) and cfg['adapt_steps']>0,'positive adaptation steps')
    require(bool(cfg['seed_pairs']) and len({tuple(x) for x in cfg['seed_pairs']})==len(cfg['seed_pairs']),'nonempty unique seed pairs')
    require(bool(cfg['budget_fractions']) and cfg['budget_fractions']==sorted(set(cfg['budget_fractions'])) and all(0<f<=1 for f in cfg['budget_fractions']),'valid ascending budget fractions')
    require(cfg['num_classes']==manifest['num_classes'],'class count binding')
    require(len(set(cfg['architectures']))==len(cfg['architectures']) and len(cfg['architectures'])==2,'two distinct architectures required')
    require(len(set(cfg['strategies']))==len(cfg['strategies']) and bool(cfg['strategies']),'strategy list')
    manifest_check(manifest)
    source_digest=hashlib.sha256(json.dumps(manifest['source_train']+manifest['source_val'],sort_keys=True).encode()).hexdigest()
    sm=read(source_manifest_path) if source_manifest_path else None
    if sm:require(sm['source_train']==manifest['source_train'] and sm['source_val']==manifest['source_val'],'source manifest changed')
    regions={r:t for r,t in manifest['targets'].items() if t.get('role')!='development'}
    expected=len(regions)*len(cfg['strategies'])*len(cfg['seed_pairs'])*len(cfg['budget_fractions'])
    require(expected>0,'no expected experiments')
    barriers={a:read(runs_root/a/'adaptation_complete.json') for a in cfg['architectures']}
    latest=max(b['timestamp'] for b in barriers.values())
    reports=[]
    for arch in cfg['architectures']:
        out=runs_root/arch;binding=read(out/'source_binding.json');receipt=read(out/'receipt.json');sr=read(out/'source_receipt.json')
        evaluation_start=read(out/'evaluation_started.json')
        require(evaluation_start['expected_runs']==expected and evaluation_start['timestamp']>=latest,'evaluation start barrier/count')
        peer_hashes=evaluation_start['peer_barriers_sha256']
        require(set(peer_hashes)>={str((runs_root/a).resolve()) for a in cfg['architectures']},'evaluation peer set incomplete')
        for peer,h in peer_hashes.items():
            marker=Path(peer)/'adaptation_complete.json'
            require(digest(marker)==h and read(marker)['timestamp']<=evaluation_start['timestamp'],'evaluation peer marker SHA/time')
            require(read(marker)['runs']==expected and read(Path(peer)/'receipt.json')['manifest_sha256']==digest(manifest_path),'evaluation peer count/manifest')
        require(binding['config_sha256']==digest(config_path) and binding['arch']==arch and binding['classes']==cfg['num_classes'],'config/model binding')
        require(binding['source_data_digest']==source_digest and binding['data_root']==str(root),'source/data-root binding')
        required_code={str(p.relative_to(root.parent)) for p in (root/'runner.py',root/'models.py',root/'acquisition_v2.py',root.parent/'baseline/engine.py',root.parent/'baseline/model.py',root.parent/'baseline/acquisition.py')}
        require(set(binding['code_sha256'])>=required_code,'code binding required file missing')
        for name,h in binding['code_sha256'].items():
            bound=(root.parent/name).resolve()
            require(bound.is_relative_to(root.parent),'code binding escapes experiment tree')
            require(digest(bound)==h,'code SHA changed '+name)
        for key,p in [('segformer_weights_receipt_sha256',root/'sources/segformer_checkpoint_receipt.json'),('segformer_runtime_receipt_sha256',root/'sources/segformer_runtime_receipt.json'),('unet_weights_sha256',root.parent/'baseline/weights/resnet18-f37072fd.pth')]:require(binding[key]==digest(p),'weights/runtime SHA '+key)
        require(receipt==dict(binding,manifest_sha256=digest(manifest_path),phase_scope='full'),'full receipt binding')
        require(all(sr.get(k)==v for k,v in binding.items()) and sr['phase_scope']=='source','source receipt binding')
        if source_manifest_path:require(sr['manifest_sha256']==digest(source_manifest_path),'source manifest SHA')
        else:require(sr['manifest_sha256']==digest(manifest_path),'source receipt requires --source-manifest if distinct')
        selection=read(out/'recipe_selection.json');pilot=cfg['seed_pairs'][0][0]
        require(selection['candidate_seed']==pilot,'pilot seed')
        candidates=[]
        for recipe in cfg['source_recipes']:
            f=out/'candidates'/recipe['name']/str(pilot);d=read(f/'complete.json');history_check(read(f/'history.json'),d,cfg)
            require(d['recipe']==recipe and d['seed']==pilot and d['arch']==arch,'pilot metadata')
            require(digest(f/'best.pt')==d['checkpoint_sha256'],'pilot model SHA');candidates.append(d)
        fallback=cfg['source_fallback_recipe'];fallback_dir=out/'candidates'/fallback['name']/str(pilot)
        if not any(d['eligible_classes'] for d in candidates):
            d=read(fallback_dir/'complete.json');history_check(read(fallback_dir/'history.json'),d,cfg)
            require(d['recipe']==fallback and d['seed']==pilot and d['arch']==arch,'fallback metadata')
            require(digest(fallback_dir/'best.pt')==d['checkpoint_sha256'],'fallback SHA');candidates.append(d)
        else:require(not fallback_dir.exists(),'untriggered fallback was run')
        require(len(list((out/'candidates').rglob('complete.json')))==len(candidates),'extra/missing recipe pilots')
        require(selection['candidates']==candidates,'pilot receipt mismatch')
        winner=max(candidates,key=lambda d:(d['eligible_classes'],d['best_source_val_miou']))
        require(selection['selected_recipe']==winner['recipe'],'recipe lexicographic winner mismatch')
        gate=read(out/'source_gate.json');sources=[]
        require(selection['selection_time']<=gate['timestamp']<=barriers[arch]['timestamp']<=latest,'source/adaptation timestamp order')
        for seed,_ in cfg['seed_pairs']:
            f=out/'source'/str(seed);d=read(f/'complete.json');history_check(read(f/'history.json'),d,cfg)
            require(d['seed']==seed and d['arch']==arch and d['recipe']==winner['recipe'],'final source metadata')
            require(d['eligible_classes'] and d['convergence_diagnosed'],'source gates failed')
            require(digest(f/'best.pt')==d['checkpoint_sha256'],'source model SHA')
            require(d['source_images']==len({r['group'] for r in manifest['source_train']}) and d['source_patches']==len(manifest['source_train']) and d['validation_patches']==len(manifest['source_val']),'source row counts')
            sources.append(d)
        require(sources[0]==winner,'pilot copied source mismatch')
        transfer=read(out/'source'/str(pilot)/'transfer.json')
        require(transfer['checkpoint_sha256']==winner['checkpoint_sha256'] and Path(transfer['origin']).resolve()==(out/'candidates'/winner['recipe']['name']/str(pilot)).resolve(),'pilot transfer binding')
        require(gate['source_models']==sources and gate['all_classes_learned'] is True and gate['all_plateau_diagnosed'] is True,'source gate receipt mismatch')
        barrier=barriers[arch];require(barrier['runs']==expected and barrier['test_labels_read_by_training'] is False,'adaptation barrier count/declaration')
        require(len(list((out/'adapt').rglob('complete.json')))==expected,'actual complete count')
        expected_records=[]
        for region,t in regions.items():
            rows=t['pool'];budgets=[math.ceil(len(rows)*f) for f in cfg['budget_fractions']]
            require(len(set(budgets))==len(budgets),'rounded budgets collide')
            for si,(seed,qseed) in enumerate(cfg['seed_pairs']):
                for strategy in cfg['strategies']:
                    previous=[]
                    for fraction,budget in zip(cfg['budget_fractions'],budgets):
                        f=out/'adapt'/region/f'm{seed}_q{qseed}'/strategy/f'b{budget}';d=read(f/'complete.json')
                        require(all(d[k]==v for k,v in {'region':region,'train_seed':seed,'selection_seed':qseed,'strategy':strategy,'budget':budget,'arch':arch}.items()),'adapt context')
                        previous=selection_check(read(f/'selection.json'),d,rows,previous,budget)
                        h=read(f/'history.json');require(d['steps']==cfg['adapt_steps'] and len(h)==cfg['adapt_steps'] and all(close(v,v) for v in h),'adapt fixed finite steps')
                        require(d['source_checkpoint_sha256']==sources[si]['checkpoint_sha256'] and d['source_recipe']==winner['recipe']['name'],'paired source/recipe SHA')
                        require(d['model_sha256']==digest(f/'model.pt'),'adapt model SHA')
                        require(close(d['planned_fraction'],fraction) and close(d['actual_fraction'],budget/len(rows)) and d['area_pixels']==65536*budget,'budget area/fraction')
                        area=sum(rows[i]['area_m2'] for i in previous) if all(rows[i].get('area_m2') is not None for i in previous) else None
                        require((d['area_m2'] is None if area is None else close(d['area_m2'],area)),'physical annotation area')
                        require(0<=d['valid_labeled_pixels']<=d['area_pixels'],'valid label area range')
                        require((f/'complete.json').stat().st_mtime<=barrier['timestamp']+.01,'adapt completion after barrier')
                        ev=read(f/'evaluation.json');metrics(ev,cfg['num_classes'])
                        require((f/'evaluation.json').stat().st_mtime>=evaluation_start['timestamp'],'evaluation before recorded start')
                        expected_records.append(dict(d,**ev))
        results=read(out/'all_results.json')
        require(results['config']==cfg and results['classes']==cfg['num_classes'] and results['architecture']==arch,'results config binding')
        require(results['completed_at']>=evaluation_start['timestamp'] and results['completed_at']<= (out/'all_results.json').stat().st_mtime+.01,'evaluation final timestamps')
        key=lambda d:(d['region'],d['train_seed'],d['selection_seed'],d['strategy'],d['budget'])
        require(sorted(results['runs'],key=key)==sorted(expected_records,key=key),'aggregate missing/changed/duplicate runs')
        require(set(results['source_only'])==set(manifest['targets']),'source-only evaluation region set')
        for r,base in results['source_only'].items():
            require(set(base)=={str(s) for s,_ in cfg['seed_pairs']},'source-only seed set')
            for ev in base.values():metrics(ev,cfg['num_classes'])
            supports=[list(map(sum,ev['confusion_matrix'])) for ev in base.values()]
            supports += [list(map(sum,d['confusion_matrix'])) for d in expected_records if d['region']==r]
            require(all(v==supports[0] for v in supports),'holdout ground-truth support differs between paired evaluations')
        reports.append({'architecture':arch,'expected_runs':expected,'verified_runs':len(expected_records)})
    return {'passed':True,'synthetic_fixture':synthetic_fixture,'evidence_scope':'software fixture only; not scientific experiment' if synthetic_fixture else 'formal completed experiment artifacts','architectures':reports,'no_label_pixels_read':True,
            'limitations':['Timestamp check uses filesystem mtimes plus barrier receipts; not cryptographic access logs.',
                           'Cannot prove absence of unlogged holdout reads; checks pool-only IDs, split separation and recorded barrier.',
                           'Does not decode or rehash target labels; manifest row hashes are bound by receipt, not independently revalidated data bytes.',
                           'Checkpoint file SHA binds bytes, not an independent rerun of training.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--source-manifest',type=Path);p.add_argument('--runs-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--synthetic-fixture',action='store_true',help='Only explicitly flagged tests/fixture_runtime* artifacts; never scientific evidence');a=p.parse_args()
    try: result=audit(a.config,a.manifest,a.runs_root,a.source_manifest,a.synthetic_fixture)
    except (AuditError,KeyError,TypeError,ValueError,OSError) as e:result={'passed':False,'error':str(e),'error_type':type(e).__name__}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
