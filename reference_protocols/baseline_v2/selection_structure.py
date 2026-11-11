"""Read-only, label/effect-free selection audit. Recorded checkpoint hashes only.

Reads configs, manifest metadata, architecture receipt.json and exact adaptation
selection.json/complete.json paths. Never opens masks, models, evaluation or
all_results. Run --dataset oem/loveda; incomplete snapshots get _partial names.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exposure(indices, rows):
    capacity = Counter(r['group'] for r in rows)
    counts = dict.fromkeys(sorted(capacity), 0)
    for i in indices:
        counts[rows[i]['group']] += 1
    # Capacity-aware balanced allocation envelope, including exhausted originals.
    level = 0
    while sum(min(n, level + 1) for n in capacity.values()) <= len(indices):
        level += 1
        if level >= max(capacity.values()):
            break
    low = {g: min(n, level) for g, n in capacity.items()}
    high = {g: min(n, level + 1) for g, n in capacity.items()}
    return {'group_counts': counts, 'pool_originals': len(counts),
            'exposed_originals': sum(v > 0 for v in counts.values()),
            'repeat_original_draws': len(indices) - sum(v > 0 for v in counts.values()),
            'duplicate_patch_indices': len(indices) - len(set(indices)),
            'count_min_including_zero': min(counts.values()),
            'count_max': max(counts.values()),
            'balanced_quota_level': level,
            'within_capacity_aware_balanced_envelope': all(low[g] <= n <= high[g] for g, n in counts.items())}


def compare(a, b):
    same = a['indices'] == b['indices']
    sa, sb = set(a['indices']), set(b['indices'])
    ca, cb = a['exposure']['group_counts'], b['exposure']['group_counts']
    result = {'left': a['strategy'], 'right': b['strategy'],
              'ordered_indices_equal': same, 'index_sets_equal': sa == sb,
              'index_jaccard': len(sa & sb) / len(sa | sb) if sa | sb else 1.,
              'group_count_l1': sum(abs(ca[g] - cb[g]) for g in ca),
              'model_equivalence': 'NOT_IDENTICAL_ORDER'}
    if same:
        sig_a, sig_b = a.get('training_signature'), b.get('training_signature')
        if a['status'] != 'complete' or b['status'] != 'complete' or not sig_a or not sig_b:
            result['model_equivalence'] = 'UNVERIFIED_METADATA_OR_INCOMPLETE'
        elif sig_a != sig_b:
            result['model_equivalence'] = 'DIFFERENT_TRAINING_SETTINGS'
        elif not a.get('model_sha256') or not b.get('model_sha256'):
            result['model_equivalence'] = 'UNVERIFIED_MODEL_HASH'
        else:
            result['model_equivalence'] = 'MATCH' if a['model_sha256'] == b['model_sha256'] else 'RED_MISMATCH'
    return result


def audit(root, dataset):
    config_path = root / 'configs' / (dataset + '.json')
    manifest_path = root / ('prepared' if dataset == 'oem' else 'prepared_loveda') / 'manifest.json'
    cfg, manifest = read(config_path), read(manifest_path)
    config_sha, manifest_sha = sha(config_path), sha(manifest_path)
    result = {'dataset': dataset, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'config_sha256': config_sha, 'manifest_sha256': manifest_sha,
              'scope': 'Selection structure only; no labels, target statistics, evaluation, predictions or model files read.',
              'limitations': ['Checkpoint SHA values are recorded claims, not independently rehashed model files.',
                              'Partial snapshots do not establish full-matrix behavior.',
                              'Original-image exposure is not geographic distance or model quality.',
                              'Files are read during training; a later rerun resolves in-flight or missing records.'],
              'sampler_semantics': {'random': 'uniform patch permutation',
                 'original_uniform_random': 'uniform original among nonexhausted originals, then uniform remaining patch',
                 'uniform_random': 'historical least-exposed-original quota-balanced random',
                 'quota_entropy': 'least-exposed-original quota-balanced entropy'},
              'runs': [], 'missing': [], 'flags': [], 'focus_comparisons': [], 'identical_order_checks': []}
    expected = 0
    for arch in cfg['architectures']:
        archdir = root / 'runs' / dataset / arch
        receipt_path = archdir / 'receipt.json'
        receipt = read(receipt_path) if receipt_path.exists() else {}
        anchored = receipt.get('config_sha256') == config_sha and receipt.get('manifest_sha256') == manifest_sha and receipt.get('arch') == arch
        if receipt and not anchored:
            result['flags'].append({'kind': 'RECEIPT_CONTEXT_MISMATCH', 'arch': arch})
        for region, target in manifest['targets'].items():
            if target.get('role') == 'development':
                continue
            rows = target['pool']
            for train_seed, select_seed in cfg['seed_pairs']:
                for strategy in cfg['strategies']:
                    prior = None
                    for fraction in cfg['budget_fractions']:
                        budget = math.ceil(len(rows) * fraction)
                        expected += 1
                        context = dict(arch=arch, region=region, train_seed=train_seed,
                                       selection_seed=select_seed, strategy=strategy, budget=budget)
                        folder = archdir / 'adapt' / region / f'm{train_seed}_q{select_seed}' / strategy / f'b{budget}'
                        cp, sp = folder / 'complete.json', folder / 'selection.json'
                        if not cp.exists() and not sp.exists():
                            result['missing'].append(context)
                            prior = None
                            continue
                        done = read(cp) if cp.exists() else None
                        sel = read(sp) if sp.exists() else None
                        data = done if done is not None else sel
                        indices = data.get('indices', [])
                        errors = []
                        valid = isinstance(indices, list) and all(type(i) is int and 0 <= i < len(rows) for i in indices)
                        if not valid:
                            errors.append('invalid_indices')
                        elif len(indices) != budget or len(set(indices)) != budget:
                            errors.append('budget_or_duplicate_patch')
                        if valid:
                            for key in ['ids', 'groups']:
                                field = 'id' if key == 'ids' else 'group'
                                if data.get(key) != [rows[i][field] for i in indices]:
                                    errors.append('manifest_' + key + '_mismatch')
                            previous = data.get('previous')
                            if not isinstance(previous, list) or indices[:len(previous)] != previous:
                                errors.append('previous_not_prefix')
                            if prior is not None and previous != prior:
                                errors.append('cross_budget_prefix_mismatch')
                        if done is not None:
                            for key, value in context.items():
                                if done.get(key) != value:
                                    errors.append('context_' + key + '_mismatch')
                        if done is not None and sel is not None:
                            if any(done.get(k) != sel.get(k) for k in ['indices', 'ids', 'groups', 'previous']):
                                errors.append('selection_complete_mismatch')
                        record = {**context, 'status': 'complete' if done is not None else 'selection_only',
                                  'indices': indices, 'errors': errors, 'training_signature': None,
                                  'input_path': str(folder.relative_to(root))}
                        if valid:
                            record['exposure'] = exposure(indices, rows)
                        if done is not None:
                            record['model_sha256'] = done.get('model_sha256')
                            fields = ['source_checkpoint_sha256', 'source_recipe', 'steps']
                            if anchored and not errors and all(done.get(k) is not None for k in fields):
                                record['training_signature'] = {'config_sha256': config_sha, 'arch': arch,
                                    'train_seed': train_seed, **{k: done[k] for k in fields}}
                        if errors:
                            result['flags'].append({**context, 'kind': 'STRUCTURAL_ERROR', 'errors': errors})
                        result['runs'].append(record)
                        prior = indices if valid else None
    aligned = defaultdict(list)
    for run in result['runs']:
        if 'exposure' in run and not run['errors']:
            aligned[tuple(run[k] for k in ['arch', 'region', 'train_seed', 'selection_seed', 'budget'])].append(run)
    randoms = ['random', 'original_uniform_random', 'uniform_random', 'quota_entropy']
    focus = {frozenset(p) for p in itertools.combinations(randoms, 2)}
    if dataset == 'oem':
        focus |= {frozenset(p) for p in [('cai_spatial', 'geo_entropy'), ('geo_coverage', 'coverage'), ('geo_coverage', 'joint_proxy')]}
    for key, runs in aligned.items():
        context = dict(zip(['arch', 'region', 'train_seed', 'selection_seed', 'budget'], key))
        for a, b in itertools.combinations(runs, 2):
            comparison = {**context, **compare(a, b)}
            if frozenset([a['strategy'], b['strategy']]) in focus:
                result['focus_comparisons'].append(comparison)
            if comparison['ordered_indices_equal']:
                result['identical_order_checks'].append(comparison)
            if comparison['model_equivalence'] == 'RED_MISMATCH':
                result['flags'].append({'kind': 'RED_MODEL_HASH_MISMATCH', **comparison})
    completed = sum(r['status'] == 'complete' and not r['errors'] for r in result['runs'])
    result['matrix'] = {'expected': expected, 'complete_valid': completed,
                        'selection_only': sum(r['status'] == 'selection_only' for r in result['runs']),
                        'missing': len(result['missing']), 'status': 'COMPLETE' if completed == expected else 'PARTIAL'}
    result['audit_status'] = 'RED_FLAGS' if result['flags'] else 'NO_FLAGS_IN_OBSERVED_RECORDS'
    return result


def write_report(result, output_dir):
    suffix = '_partial' if result['matrix']['status'] != 'COMPLETE' else ''
    stem = 'selection_structure_' + result['dataset'] + suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = output_dir / (stem + '.json'), output_dir / (stem + '.md')
    jp.write_text(json.dumps(result, indent=2) + '\n')
    lines = [f"# {result['dataset']} selection structure — {result['matrix']['status']}", '',
             json.dumps(result['matrix']), '', result['scope'], '',
             'Audit status: ' + result['audit_status'], '',
             *['- ' + s for s in result['limitations']], '',
             'Historical uniform_random is quota-balanced; original_uniform_random samples originals uniformly per draw.', '',
             '## Flags', '', json.dumps(result['flags'], indent=2) if result['flags'] else 'None among observed records.', '',
             '## Exposure by observed run', '',
             '|Architecture|Region|Train/query seed|Budget|Strategy|State|Exposed originals|Repeated original draws|Count range incl. zero|Balanced envelope|',
             '|---|---|---|---:|---|---|---:|---:|---|---|']
    for r in result['runs']:
        e = r.get('exposure')
        if e:
            lines.append(f"|{r['arch']}|{r['region']}|{r['train_seed']}/{r['selection_seed']}|{r['budget']}|{r['strategy']}|{r['status']}|{e['exposed_originals']}/{e['pool_originals']}|{e['repeat_original_draws']}|{e['count_min_including_zero']}–{e['count_max']}|{e['within_capacity_aware_balanced_envelope']}|")
    lines += ['', '## Aligned method comparisons (observed pairs only)', '',
              '|Architecture|Region|Train/query seed|Budget|Methods|Ordered equal|Jaccard|Group-count L1|Model check|',
              '|---|---|---|---:|---|---|---:|---:|---|']
    for c in result['focus_comparisons']:
        verdict = ('🔴 ' if c['model_equivalence'] == 'RED_MISMATCH' else '') + c['model_equivalence']
        lines.append(f"|{c['arch']}|{c['region']}|{c['train_seed']}/{c['selection_seed']}|{c['budget']}|{c['left']} / {c['right']}|{c['ordered_indices_equal']}|{c['index_jaccard']:.3f}|{c['group_count_l1']}|{verdict}|")
    if any(f['kind'] == 'RED_MODEL_HASH_MISMATCH' for f in result['flags']):
        lines.insert(2, '🔴 IDENTICAL ORDER / SAME RECORDED TRAINING SETTINGS BUT DIFFERENT MODEL SHA: see flags. No experiment was modified.')
    mp.write_text('\n'.join(lines) + '\n')
    return jp, mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['oem', 'loveda'], required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.dataset)
    paths = write_report(result, args.output_dir or args.root / 'reports')
    print(json.dumps({'matrix': result['matrix'], 'audit_status': result['audit_status'], 'reports': [str(p) for p in paths]}))


if __name__ == '__main__':
    main()
