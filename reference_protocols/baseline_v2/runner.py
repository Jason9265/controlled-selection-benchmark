"""Source-only recipe selection, frozen paired adaptation, delayed confirmation.

Legacy numerical operators are reused, but old experiments are never modified.
All model/recipe selection occurs on source validation before target evaluation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.join(os.path.dirname(__file__), 'cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(os.path.dirname(__file__), 'cache/triton'))
import argparse
import hashlib
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'baseline'
sys.path.insert(0, str(LEGACY))
import engine as numeric
from acquisition_v2 import acquire
from models import build_model


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    numeric.write(path, data)


def weighted_ce(logits, labels, weights=None):
    valid = labels != 255
    if not bool(valid.any()):
        raise ValueError('All-ignore training batch')
    safe = labels.masked_fill(~valid, 0)
    loss = -F.log_softmax(logits.float(), 1).gather(1, safe[:, None])[:, 0]
    if weights is None:
        return loss[valid].mean()
    pixel_weights = weights[safe][valid]
    return (loss[valid] * pixel_weights).sum() / pixel_weights.sum()


def source_statistics(pack, classes):
    labels = pack.y.cpu().numpy()
    histograms = np.stack([np.bincount(y.ravel(), minlength=256)[:classes] for y in labels])
    counts = histograms.sum(0)
    if (counts == 0).any():
        raise ValueError('Source training has missing classes; expand source data first')
    frequencies = counts / counts.sum()
    weights = np.clip(np.sqrt(np.median(frequencies) / frequencies), .5, 5.)
    weights /= weights.mean()
    candidates = []
    for c in range(classes):
        indices = np.flatnonzero(histograms[:, c] >= 256)
        if not len(indices):
            indices = np.flatnonzero(histograms[:, c] > 0)
        candidates.append(indices)
    return {'pixels': counts.tolist(), 'frequencies': frequencies.tolist(),
            'class_weights': weights.tolist(), 'sampling_pool_sizes': [len(x) for x in candidates]}, candidates


def sample_source(rng, n, batch, class_pools, ratio):
    balanced = int(round(batch * ratio))
    indices = rng.integers(n, size=batch)
    for j, c in enumerate(rng.integers(len(class_pools), size=balanced)):
        indices[j] = rng.choice(class_pools[c])
    rng.shuffle(indices)
    return indices


def optimizer(model, cfg, arch):
    identifiers = {id(x) for x in model.encoder.parameters()}
    rates = cfg['learning_rates'][arch]
    groups = []
    for encoder, rate in zip((True, False), rates):
        for decay in (True, False):
            params = [p for p in model.parameters() if (id(p) in identifiers) == encoder and (p.ndim > 1) == decay]
            if params:
                groups.append({'params': params, 'lr': rate, 'base_lr': rate,
                               'weight_decay': cfg['weight_decay'] if decay else 0.})
    return torch.optim.AdamW(groups)


def configure_backend(model, arch, cfg):
    backend = cfg['execution_backend'][arch]
    if backend == 'compile_reduce_overhead':
        compiled = torch.compile(model.forward, mode='reduce-overhead')
        def forward(x):
            # Also applies to the shared evaluation/scoring helpers. One complete
            # forward/backward or inference batch ends before the next call.
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(x.contiguous())
        model.forward = forward
    elif backend != 'eager':
        raise ValueError(f'Unknown frozen backend: {backend}')
    return model


def load_model(path, arch, classes, cfg):
    model = build_model(arch, classes, pretrained=False).cuda()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return configure_backend(model, arch, cfg)


def eligible(metrics, threshold):
    return all(iou >= threshold for iou in metrics['iou'])


def source_fit(cfg, arch, recipe, training_seed, source, val, stats, class_pools, folder, progress):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        assert digest(folder / 'best.pt') == done['checkpoint_sha256']
        return done
    if (folder / 'history.json').exists():
        raise RuntimeError(f'Incomplete source attempt preserved at {folder}; use a versioned recovery run')
    numeric.seed(training_seed)
    model = configure_backend(build_model(arch, cfg['num_classes'], pretrained=True).cuda(), arch, cfg)
    opt = optimizer(model, cfg, arch)
    rng = np.random.default_rng(training_seed)
    weights = torch.tensor(stats['class_weights'], device='cuda', dtype=torch.float32) if recipe['weighted'] else None
    history, recent = [], []
    best_key, monitor_key = (-1, -1.), (-1, -1.)
    best_step = 0
    stale = 0
    start = time.perf_counter()
    for step in range(1, cfg['source_max_steps'] + 1):
        for group in opt.param_groups:
            if step <= cfg['source_warmup_steps']:
                group['lr'] = group['base_lr'] * (.1 + .9 * step / cfg['source_warmup_steps'])
            else:
                position = min(step - cfg['source_warmup_steps'], cfg['source_cosine_steps'] - cfg['source_warmup_steps']) / (cfg['source_cosine_steps'] - cfg['source_warmup_steps'])
                group['lr'] = cfg['minimum_lr'] + (group['base_lr'] - cfg['minimum_lr']) * (1 + math.cos(math.pi * position)) / 2
        model.train()
        indices = sample_source(rng, len(source), cfg['source_batch'], class_pools, recipe['balanced_ratio'])
        x, y = source.batch(indices)
        x, y = numeric.augment(x, y, rng)
        opt.zero_grad(set_to_none=True)
        with numeric.amp():
            loss = weighted_ce(model(x)[0], y, weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite source loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        recent.append(float(loss.detach()))
        if step % cfg['source_eval_every']:
            continue
        evaluation = numeric.evaluate(model, val, cfg['num_classes'])
        key = (int(eligible(evaluation, cfg['minimum_class_iou'])), evaluation['miou_fixed'])
        history.append({'step': step, 'train_loss': float(np.mean(recent)), 'validation': evaluation,
                        'eligible': bool(key[0]), 'encoder_lr': opt.param_groups[0]['lr']})
        recent = []
        if key > best_key:
            best_key, best_step = key, step
            numeric.save_model(folder / 'best.pt', model)
        if key[0] > monitor_key[0] or (key[0] == monitor_key[0] and key[1] > monitor_key[1] + cfg['source_min_delta']):
            monitor_key, stale = key, 0
        else:
            stale += 1
        write(folder / 'history.json', history)
        progress('source_training', arch=arch, recipe=recipe['name'], seed=training_seed, step=step,
                 max_steps=cfg['source_max_steps'], val_miou=key[1], minimum_iou=min(evaluation['iou']),
                 eligible=bool(key[0]), stale_checks=stale, loss=history[-1]['train_loss'])
        if step >= cfg['source_min_steps'] and stale >= cfg['source_patience']:
            break
    torch.cuda.synchronize()
    plateau = stale >= cfg['source_patience']
    best = next(x for x in history if x['step'] == best_step)
    done = {'seed': training_seed, 'arch': arch, 'recipe': recipe, 'steps': step, 'best_step': best_step,
            'best_source_val_miou': best_key[1], 'best_source_val_iou': best['validation']['iou'],
            'eligible_classes': bool(best_key[0]), 'convergence_diagnosed': plateau,
            'stop_reason': 'source_validation_plateau' if plateau else 'max_steps_reached_convergence_not_established',
            'selection_policy': 'prefer every source-validation class IoU>=threshold, then fixed-class mIoU',
            'minimum_class_iou_threshold': cfg['minimum_class_iou'], 'seconds': time.perf_counter() - start,
            'source_images': len({x['group'] for x in source.rows}), 'source_patches': len(source),
            'validation_patches': len(val), 'weights': model.pretrained_metadata,
            'checkpoint_sha256': digest(folder / 'best.pt')}
    write(folder / 'complete.json', done)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return done


def source_phase(cfg, manifest, out, arch, progress):
    source = numeric.Pack(manifest['source_train'], ROOT, True, cfg['num_classes'])
    val = numeric.Pack(manifest['source_val'], ROOT, True, cfg['num_classes'])
    stats, class_pools = source_statistics(source, cfg['num_classes'])
    write(out / 'source_statistics.json', stats)
    pilot_seed = cfg['seed_pairs'][0][0]
    candidates = []
    for recipe in cfg['source_recipes']:
        folder = out / 'candidates' / recipe['name'] / str(pilot_seed)
        candidates.append(source_fit(cfg, arch, recipe, pilot_seed, source, val, stats, class_pools, folder, progress))
    if not any(x['eligible_classes'] for x in candidates):
        recipe = cfg['source_fallback_recipe']
        candidates.append(source_fit(cfg, arch, recipe, pilot_seed, source, val, stats, class_pools,
                                     out / 'candidates' / recipe['name'] / str(pilot_seed), progress))
    selected = max(candidates, key=lambda x: (x['eligible_classes'], x['best_source_val_miou']))
    selection = {'candidate_seed': pilot_seed, 'selected_recipe': selected['recipe'], 'candidates': candidates,
                 'selection_basis': 'source validation only; no target metrics', 'selection_time': time.time()}
    selection_file = out / 'recipe_selection.json'
    if selection_file.exists():
        old = read(selection_file)
        assert old['selected_recipe'] == selection['selected_recipe'] and old['candidates'] == candidates
    else:
        write(selection_file, selection)
    if not selected['eligible_classes']:
        raise RuntimeError('Source class-learning gate failed after fixed fallback; no target adaptation authorized by this protocol')
    origin = out / 'candidates' / selected['recipe']['name'] / str(pilot_seed)
    destination = out / 'source' / str(pilot_seed)
    if not destination.exists():
        shutil.copytree(origin, destination)
        write(destination / 'transfer.json', {'origin': str(origin.resolve()), 'checkpoint_sha256': digest(origin / 'best.pt')})
    all_sources = [selected]
    for training_seed, _ in cfg['seed_pairs'][1:]:
        all_sources.append(source_fit(cfg, arch, selected['recipe'], training_seed, source, val, stats, class_pools,
                                      out / 'source' / str(training_seed), progress))
    gate = {'all_classes_learned': all(x['eligible_classes'] for x in all_sources),
            'all_plateau_diagnosed': all(x['convergence_diagnosed'] for x in all_sources),
            'source_models': all_sources, 'timestamp': time.time()}
    write(out / 'source_gate.json', gate)
    del source, val
    gc.collect()
    torch.cuda.empty_cache()
    if not gate['all_classes_learned'] or not gate['all_plateau_diagnosed']:
        raise RuntimeError('Source diagnostic gate not met; retained results, no target adaptation')


def fit_adaptation(model, source, target, stats, class_pools, recipe, cfg, arch, training_seed, progress, context):
    numeric.seed(training_seed)
    model.train()
    rng = np.random.default_rng(training_seed)
    opt = optimizer(model, cfg, arch)
    weights = torch.tensor(stats['class_weights'], device='cuda', dtype=torch.float32) if recipe['weighted'] else None
    half = cfg['adapt_batch'] // 2
    history = []
    start = time.perf_counter()
    for step in range(1, cfg['adapt_steps'] + 1):
        indices = sample_source(rng, len(source), half, class_pools, recipe['balanced_ratio'])
        sx, sy = source.batch(indices)
        tx, ty = target.batch(rng.integers(len(target), size=half))
        x, y = numeric.augment(torch.cat([sx, tx]), torch.cat([sy, ty]), rng)
        opt.zero_grad(set_to_none=True)
        with numeric.amp():
            loss = weighted_ce(model(x)[0], y, weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite adaptation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        history.append(float(loss.detach()))
        if step % 200 == 0:
            progress('adaptation', **context, step=step, total=cfg['adapt_steps'], loss=float(np.mean(history[-200:])))
    torch.cuda.synchronize()
    return history, time.perf_counter() - start


def adapt_phase(cfg, manifest, out, arch, progress):
    gate = read(out / 'source_gate.json')
    if not gate['all_classes_learned'] or not gate['all_plateau_diagnosed']:
        raise RuntimeError('Source gate not passed')
    recipe = read(out / 'recipe_selection.json')['selected_recipe']
    source = numeric.Pack(manifest['source_train'], ROOT, True, cfg['num_classes'])
    stats, class_pools = source_statistics(source, cfg['num_classes'])
    assert stats == read(out / 'source_statistics.json')
    for region, target in manifest['targets'].items():
        if target.get('role') == 'development':
            continue
        rows = target['pool']
        pool = numeric.Pack(rows, ROOT, False, cfg['num_classes'])
        budgets = [math.ceil(len(pool) * f) for f in cfg['budget_fractions']]
        for training_seed, selection_seed in cfg['seed_pairs']:
            source_path = out / 'source' / str(training_seed) / 'best.pt'
            model = load_model(source_path, arch, cfg['num_classes'], cfg)
            start = time.perf_counter()
            initial = numeric.score(model, pool, cfg['num_classes'])
            torch.cuda.synchronize()
            initial_seconds = time.perf_counter() - start
            del model
            gc.collect()
            np.savez_compressed(out / 'source' / str(training_seed) / f'{region}_initial_scores.npz', **initial)
            for strategy in cfg['strategies']:
                selected, prior = [], source_path
                for fraction, budget in zip(cfg['budget_fractions'], budgets):
                    folder = out / 'adapt' / region / f'm{training_seed}_q{selection_seed}' / strategy / f'b{budget}'
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder / 'complete.json').exists():
                        done = read(folder / 'complete.json')
                        assert digest(folder / 'model.pt') == done['model_sha256']
                        selected, prior = done['indices'], folder / 'model.pt'
                        continue
                    start = time.perf_counter()
                    if strategy in ('random', 'uniform_random', 'original_uniform_random'):
                        scores, inference_seconds = None, 0.
                    elif not selected:
                        scores, inference_seconds = initial, initial_seconds
                    else:
                        query = load_model(prior, arch, cfg['num_classes'], cfg)
                        ti = time.perf_counter()
                        scores = numeric.score(query, pool, cfg['num_classes'])
                        torch.cuda.synchronize()
                        inference_seconds = time.perf_counter() - ti
                        del query
                        gc.collect()
                    previous = list(selected)
                    ti = time.perf_counter()
                    selected = acquire(strategy, scores, rows, selected, budget, selection_seed)
                    selector_seconds = time.perf_counter() - ti
                    assert selected[:len(previous)] == previous and len(set(selected)) == budget
                    selection = {'indices': selected, 'ids': [rows[i]['id'] for i in selected],
                                 'groups': [rows[i]['group'] for i in selected], 'previous': previous,
                                 'inference_seconds': inference_seconds, 'selector_seconds': selector_seconds}
                    write(folder / 'selection.json', selection)
                    annotated = numeric.Pack([rows[i] for i in selected], ROOT, True, cfg['num_classes'])
                    model = load_model(source_path, arch, cfg['num_classes'], cfg)
                    context = {'region': region, 'train_seed': training_seed, 'selection_seed': selection_seed,
                               'strategy': strategy, 'budget': budget, 'arch': arch}
                    history, train_seconds = fit_adaptation(model, source, annotated, stats, class_pools, recipe,
                                                            cfg, arch, training_seed, progress, context)
                    numeric.save_model(folder / 'model.pt', model)
                    write(folder / 'history.json', history)
                    done = {**context, **selection, 'steps': len(history), 'planned_fraction': fraction,
                            'actual_fraction': budget / len(rows), 'area_pixels': 65536 * budget,
                            'area_m2': sum(rows[i]['area_m2'] for i in selected) if all(rows[i].get('area_m2') is not None for i in selected) else None,
                            'valid_labeled_pixels': int((annotated.y != 255).sum()), 'train_seconds': train_seconds,
                            'round_wall_seconds': time.perf_counter() - start,
                            'source_checkpoint_sha256': digest(source_path), 'model_sha256': digest(folder / 'model.pt'),
                            'source_recipe': recipe['name']}
                    write(folder / 'complete.json', done)
                    selected, prior = list(selected), folder / 'model.pt'
                    del annotated, model
                    gc.collect()
                    torch.cuda.empty_cache()
        del pool
    del source
    gc.collect()
    torch.cuda.empty_cache()
    write(out / 'adaptation_complete.json', {'timestamp': time.time(), 'runs': len(list((out / 'adapt').rglob('complete.json'))),
                                            'test_labels_read_by_training': False})


def evaluation_phase(cfg, manifest, out, arch, progress, peer_runs):
    expected = sum(t.get('role') != 'development' for t in manifest['targets'].values()) * len(cfg['strategies']) * len(cfg['seed_pairs']) * len(cfg['budget_fractions'])
    peers = set(peer_runs) | {out.parent / a for a in cfg['architectures']}
    for peer in peers:
        if not (peer / 'adaptation_complete.json').exists():
            raise RuntimeError(f'All-architecture evaluation barrier missing for {peer}')
        assert read(peer / 'adaptation_complete.json')['runs'] == expected
        assert len(list((peer / 'adapt').rglob('complete.json'))) == expected
        assert read(peer / 'receipt.json')['manifest_sha256'] == read(out / 'receipt.json')['manifest_sha256']
    if not (out / 'adaptation_complete.json').exists():
        raise RuntimeError('Local adaptation barrier missing')
    files = sorted((out / 'adapt').rglob('complete.json'))
    assert len(files) == expected
    evaluation_start = out / 'evaluation_started.json'
    if not evaluation_start.exists():
        write(evaluation_start, {'timestamp': time.time(), 'expected_runs': expected,
                                 'peer_barriers_sha256': {str(p.resolve()): digest(p / 'adaptation_complete.json') for p in sorted(peers)}})
    results, baselines = [], {}
    for region, target in manifest['targets'].items():
        test = numeric.Pack(target['test'], ROOT, True, cfg['num_classes'])
        baselines[region] = {}
        for training_seed, _ in cfg['seed_pairs']:
            model = load_model(out / 'source' / str(training_seed) / 'best.pt', arch, cfg['num_classes'], cfg)
            baselines[region][str(training_seed)] = numeric.evaluate(model, test, cfg['num_classes'])
            del model
            gc.collect()
        for path in files:
            row = read(path)
            if row['region'] != region:
                continue
            ep = path.parent / 'evaluation.json'
            if ep.exists():
                metrics = read(ep)
            else:
                model = load_model(path.parent / 'model.pt', arch, cfg['num_classes'], cfg)
                metrics = numeric.evaluate(model, test, cfg['num_classes'])
                write(ep, metrics)
                del model
                gc.collect()
            results.append({**row, **metrics})
            progress('confirmation_evaluation', region=region, completed=len(results), expected=expected)
        del test
        gc.collect()
        torch.cuda.empty_cache()
    write(out / 'all_results.json', {'config': cfg, 'source_only': baselines, 'runs': results,
                                    'classes': cfg['num_classes'], 'architecture': arch, 'completed_at': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch', choices=['resnet18_unet', 'segformer_b0'], required=True)
    parser.add_argument('--phase', choices=['source', 'adapt', 'evaluate'], required=True)
    parser.add_argument('--peer-run', type=Path, action='append', default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    cfg, manifest = read(args.config), read(args.manifest)
    # Evaluation/query phases must share the same strict deterministic policy
    # as training, including deterministic CUDA interpolation implementation.
    numeric.seed(cfg['seed_pairs'][0][0])
    numeric.validate(manifest, cfg)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binding = {'config_sha256': digest(args.config), 'arch': args.arch, 'classes': cfg['num_classes'],
               'source_data_digest': hashlib.sha256(json.dumps(manifest['source_train'] + manifest['source_val'], sort_keys=True).encode()).hexdigest(),
               'data_root': str(ROOT), 'code_sha256': {str(p.relative_to(ROOT.parent)): digest(p) for p in
                   [ROOT / 'runner.py', ROOT / 'models.py', ROOT / 'acquisition_v2.py', LEGACY / 'engine.py', LEGACY / 'model.py', LEGACY / 'acquisition.py']},
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
               'segformer_weights_receipt_sha256': digest(ROOT / 'sources/segformer_checkpoint_receipt.json'),
               'segformer_runtime_receipt_sha256': digest(ROOT / 'sources/segformer_runtime_receipt.json'),
               'unet_weights_sha256': digest(LEGACY / 'weights/resnet18-f37072fd.pth')}
    common = out / 'source_binding.json'
    if common.exists():
        assert read(common) == binding, 'Immutable source/code/config binding mismatch'
    else:
        if args.phase != 'source':
            raise RuntimeError('Source binding missing')
        write(common, binding)
    receipt = {**binding, 'manifest_sha256': digest(args.manifest), 'phase_scope': 'source' if args.phase == 'source' else 'full'}
    receipt_path = out / ('source_receipt.json' if args.phase == 'source' else 'receipt.json')
    if receipt_path.exists():
        assert read(receipt_path) == receipt, 'Manifest receipt mismatch'
    else:
        write(receipt_path, receipt)
    def progress(stage, **data):
        record = {'timestamp': time.time(), 'stage': stage, **data}
        write(out / 'progress.json', record)
        print(json.dumps(record), flush=True)
    start = time.perf_counter()
    progress('starting', phase=args.phase, arch=args.arch)
    try:
        if args.phase == 'source':
            source_phase(cfg, manifest, out, args.arch, progress)
        elif args.phase == 'adapt':
            adapt_phase(cfg, manifest, out, args.arch, progress)
        else:
            evaluation_phase(cfg, manifest, out, args.arch, progress, args.peer_run)
    except Exception as exc:
        progress('failed', error=repr(exc))
        raise
    progress('complete', phase=args.phase, wall_seconds=time.perf_counter() - start,
             peak_cuda_allocated=torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    main()
