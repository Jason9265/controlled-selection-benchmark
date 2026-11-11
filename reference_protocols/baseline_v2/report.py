"""Report completed engine results without rerunning or selecting methods by outcome.

Reads all_results.json and source training records only. Writes derived figures,
paired-seed statistics, and Markdown; never edits experiments or checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PREDECLARED_PLOTS = ('random', 'original_uniform_random', 'quota_entropy', 'geo_entropy', 'geo_coverage', 'ripu_patch')
FALLBACKS = {'geo_entropy': 'cai_spatial', 'geo_coverage': 'joint_proxy'}
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9')


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Every summary requires exactly three finite paired-seed values')
    return {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1)), 'values': values.tolist(), 'n': 3}


def formatted(stat, scale=100):
    return f"{stat['mean'] * scale:.3f} ± {stat['sample_sd'] * scale:.3f}"


def link(path, output):
    return Path(os.path.relpath(path, Path(output).parent)).as_posix()


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(str(cell).replace('|', '\\|') for cell in row) + ' |' for row in rows]


def region_roles(data):
    """Separate confirmation adaptation from reserved development baselines."""
    cfg = data['config']
    baseline_regions = list(data['source_only'])
    observed = {row['region'] for row in data['runs']}
    # Configuration may use display casing (e.g. Urban), while manifests use
    # canonical ids (urban). Resolve only a unique case-insensitive match; never
    # silently drop a region or merge distinct actual ids.
    canonical = {}
    for region in baseline_regions:
        if not isinstance(region, str) or not region:
            raise ValueError('Actual region ids must be nonempty strings')
        key = region.casefold()
        if key in canonical:
            raise ValueError(f'Ambiguous case-insensitive actual region ids: {canonical[key]!r}, {region!r}')
        canonical[key] = region

    def resolve(names, field):
        resolved = []
        for name in names:
            if not isinstance(name, str) or name.casefold() not in canonical:
                raise ValueError(f'Unmatched configured {field} region: {name!r}; actual ids: {baseline_regions}')
            resolved.append(canonical[name.casefold()])
        return resolved

    declared = cfg.get('confirmation_regions')
    regions = resolve(declared, 'confirmation') if declared is not None else [r for r in baseline_regions if r in observed]
    development = resolve(cfg.get('development_regions', []), 'development')
    if len(set(regions)) != len(regions) or len(set(development)) != len(development):
        raise ValueError('Duplicate declared region')
    if set(regions) & set(development):
        raise ValueError('Confirmation/development region overlap')
    if observed != set(regions):
        raise ValueError(f'Observed adaptation regions differ from confirmation configuration: {observed} vs {set(regions)}')
    if not set(regions + development) <= set(baseline_regions):
        raise ValueError('Missing declared confirmation/development source-only baseline')
    if declared is not None and 'development_regions' in cfg and set(baseline_regions) != set(regions + development):
        raise ValueError('Unexpected source-only region outside configured confirmation/development regions')
    roles = {r: 'confirmation' if r in regions else 'reserved development (source-only)' if r in development
             else 'source-only (role unspecified)' for r in baseline_regions}
    return regions, roles


def validate_results(data):
    cfg = data['config']
    pairs = [tuple(map(int, item)) for item in cfg['seed_pairs']]
    if len(pairs) != 3 or len(set(pairs)) != 3:
        raise ValueError('This report requires exactly three distinct configured seed pairs')
    strategies = cfg['strategies']
    fractions = list(map(float, cfg['budget_fractions']))
    if len(set(strategies)) != len(strategies) or fractions != sorted(set(fractions)):
        raise ValueError('Duplicate strategies or unordered/duplicate budgets')
    classes = int(data['classes'])
    regions, roles = region_roles(data)
    indexed = {}
    for row in data['runs']:
        pair = (int(row['train_seed']), int(row['selection_seed']))
        key = (row['region'], row['strategy'], float(row['planned_fraction']), pair)
        if key in indexed:
            raise ValueError(f'Duplicate result: {key}')
        if row['region'] not in regions or row['strategy'] not in strategies or pair not in pairs or float(row['planned_fraction']) not in fractions:
            raise ValueError(f'Unexpected result: {key}')
        iou = np.asarray(row['iou'], dtype=float)
        if iou.shape != (classes,) or not np.isfinite(iou).all() or (iou < 0).any() or (iou > 1).any():
            raise ValueError(f'Invalid class IoUs: {key}')
        if not np.isclose(float(row['miou_fixed']), iou.mean(), atol=1e-10, rtol=1e-8):
            raise ValueError(f'mIoU is inconsistent with fixed class denominator: {key}')
        if not 0 < float(row['actual_fraction']) <= 1 or int(row['steps']) != int(cfg['adapt_steps']):
            raise ValueError(f'Invalid fraction or unequal adaptation steps: {key}')
        indexed[key] = row
    expected = len(regions) * len(strategies) * len(fractions) * len(pairs)
    if len(indexed) != expected:
        raise ValueError(f'Incomplete results: {len(indexed)} / {expected}; no partial-effect reporting')
    for region in regions:
        for fraction in fractions:
            rows = [indexed[(region, strategy, fraction, pair)] for strategy in strategies for pair in pairs]
            if len({int(r['budget']) for r in rows}) != 1 or not np.allclose([r['actual_fraction'] for r in rows], rows[0]['actual_fraction'], rtol=0, atol=1e-12):
                raise ValueError(f'Unequal annotation budgets: {region}/{fraction}')
    for region in data['source_only']:
        for pair in pairs:
            baseline = data['source_only'][region][str(pair[0])]
            if len(baseline['iou']) != classes or not np.isclose(baseline['miou_fixed'], np.mean(baseline['iou'])):
                raise ValueError(f'Invalid source-only fixed-class metric: {region}/{pair}')
    if not regions:
        raise ValueError('No evaluated regions')
    return cfg, pairs, strategies, fractions, classes, regions, indexed


def aggregate(data):
    cfg, pairs, strategies, fractions, classes, regions, indexed = validate_results(data)
    cells, deltas, aucs, source, interactions = [], [], [], [], []
    _, roles = region_roles(data)
    for region in data['source_only']:
        base_values = [data['source_only'][region][str(pair[0])]['miou_fixed'] for pair in pairs]
        source.append({'region': region, 'role': roles[region], 'miou_fixed': summary(base_values)})
    for region in regions:
        for strategy in strategies:
            for fraction in fractions:
                rows = [indexed[(region, strategy, fraction, pair)] for pair in pairs]
                cell = {'region': region, 'strategy': strategy, 'planned_fraction': fraction,
                        'actual_fraction': float(rows[0]['actual_fraction']), 'budget': int(rows[0]['budget']),
                        'miou_fixed': summary([r['miou_fixed'] for r in rows]),
                        'accuracy': summary([r['accuracy'] for r in rows]),
                        'train_seconds': summary([r['train_seconds'] for r in rows]),
                        'iou_per_seed': [r['iou'] for r in rows]}
                cells.append(cell)
                references = ('entropy', 'quota_entropy', 'coverage', 'geo_entropy') if strategy == 'geo_coverage' else ('entropy', 'quota_entropy')
                for reference in references:
                    if reference not in strategies or strategy == reference:
                        continue
                    values = [indexed[(region, strategy, fraction, pair)]['miou_fixed'] -
                              indexed[(region, reference, fraction, pair)]['miou_fixed'] for pair in pairs]
                    deltas.append({'region': region, 'strategy': strategy, 'reference': reference,
                                   'planned_fraction': fraction, 'actual_fraction': cell['actual_fraction'],
                                   'delta': summary(values)})
                if strategy == 'geo_coverage' and {'geo_coverage', 'geo_entropy', 'coverage', 'entropy'} <= set(strategies):
                    values = [indexed[(region, 'geo_coverage', fraction, pair)]['miou_fixed'] -
                              indexed[(region, 'geo_entropy', fraction, pair)]['miou_fixed'] -
                              indexed[(region, 'coverage', fraction, pair)]['miou_fixed'] +
                              indexed[(region, 'entropy', fraction, pair)]['miou_fixed'] for pair in pairs]
                    interactions.append({'region': region, 'planned_fraction': fraction,
                                         'actual_fraction': cell['actual_fraction'], 'interaction': summary(values)})
            # Integrate only measured points from the declared nominal 1%–5% interval.
            # Rounded actual endpoints are explicitly reported, never replaced by 0 or nominal x.
            chosen = [f for f in fractions if .01 - 1e-12 <= f <= .05 + 1e-12]
            if len(chosen) < 2:
                continue
            areas = []
            x = None
            for pair in pairs:
                rows = [indexed[(region, strategy, fraction, pair)] for fraction in chosen]
                current_x = np.asarray([r['actual_fraction'] for r in rows], dtype=float)
                if (np.diff(current_x) <= 0).any():
                    raise ValueError('AUC requires strictly increasing actual fractions')
                if x is not None and not np.allclose(x, current_x, rtol=0, atol=1e-12):
                    raise ValueError('AUC seeds have unequal actual fractions')
                x = current_x
                y = np.asarray([r['miou_fixed'] for r in rows], dtype=float)
                areas.append(float(np.trapezoid(y, x=x) / (x[-1] - x[0])))
            aucs.append({'region': region, 'strategy': strategy, 'actual_x': x.tolist(),
                         'nominal_x': chosen, 'normalized_auc': summary(areas)})
    auc_deltas = []
    auc_index = {(a['region'], a['strategy']): a for a in aucs}
    for a in aucs:
        for reference in ('entropy', 'quota_entropy'):
            other = auc_index.get((a['region'], reference))
            if other is None or a['strategy'] == reference:
                continue
            delta = np.array(a['normalized_auc']['values']) - np.array(other['normalized_auc']['values'])
            auc_deltas.append({'region': a['region'], 'strategy': a['strategy'], 'reference': reference,
                               'delta': summary(delta)})
    return {'classes': classes, 'seed_pairs': [list(p) for p in pairs], 'source_only': source,
            'confirmation_regions': regions, 'region_roles': roles,
            'cells': cells, 'paired_deltas': deltas, 'interactions': interactions, 'normalized_auc': aucs, 'paired_auc_deltas': auc_deltas}


def save_figure(fig, directory, name):
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'{name}.{extension}', dpi=240, facecolor='white')
    plt.close(fig)


def component_policy(data):
    available = set(data['config']['strategies'])
    geo = ['entropy', 'geo_entropy', 'coverage', 'geo_coverage']
    loveda = ['entropy', 'coverage', 'joint_proxy', 'ripu_patch']
    if set(geo) <= available:
        return {'strategies': geo, 'status': 'original protocol component visualization',
                'description': '原凍結協議四組元件展示：entropy、geo_entropy、coverage、geo_coverage。方法清單不依結果選取。'}
    if 'loveda' in str(data['config'].get('dataset', '')).casefold() and set(loveda) <= available:
        return {'strategies': loveda, 'status': 'v2 predeclared component visualization',
                'description': '本輪LoveDA在正式訓練前固定展示entropy、coverage、joint_proxy、ripu_patch；v1同組圖曾為事後展示，本輪明確重新預定。完整策略仍全部報告。'}
    return None


def draw_components(data, stats, figures):
    policy = component_policy(data)
    if policy is None:
        return
    regions = stats['confirmation_regions']
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4.7), squeeze=False, sharey=True)
    fig.subplots_adjust(left=.13 / len(regions), right=.98, bottom=.22, top=.77, wspace=.16)
    for ax, region in zip(axes[0], regions):
        for index, method in enumerate(policy['strategies']):
            points = sorted((s for s in stats['cells'] if s['region'] == region and s['strategy'] == method), key=lambda s: s['actual_fraction'])
            x = np.array([p['actual_fraction'] for p in points]) * 100
            y = np.array([p['miou_fixed']['mean'] for p in points]) * 100
            sd = np.array([p['miou_fixed']['sample_sd'] for p in points]) * 100
            ax.errorbar(x, y, yerr=sd, color=COLORS[index], label=method, marker=('o', 's', '^', 'D')[index],
                        markersize=4, linewidth=1.3, capsize=2)
            for i in range(3):
                ax.scatter(x, [p['miou_fixed']['values'][i] * 100 for p in points], color=COLORS[index], s=10, alpha=.4)
        ax.set(title=region, xlabel='Queried pool blocks (%)')
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel(f'Fixed-{stats["classes"]}-class mIoU (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995), ncol=2, frameon=False)
    fig.text(.5, .035, policy['status'], fontsize=9, ha='center')
    save_figure(fig, figures, 'component_diagnostics')


def draw_figures(data, stats, histories, figures):
    cfg = data['config']
    regions = stats['confirmation_regions']
    available = set(cfg['strategies'])
    selected = []
    for method in PREDECLARED_PLOTS:
        candidate = method if method in available else FALLBACKS.get(method)
        if candidate in available and candidate not in selected:
            selected.append(candidate)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.dpi': 120})
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4.2), squeeze=False, constrained_layout=True)
    for ax, region in zip(axes[0], regions):
        for index, method in enumerate(selected):
            points = sorted((s for s in stats['cells'] if s['region'] == region and s['strategy'] == method), key=lambda s: s['actual_fraction'])
            x = np.array([p['actual_fraction'] for p in points]) * 100
            y = np.array([p['miou_fixed']['mean'] for p in points]) * 100
            sd = np.array([p['miou_fixed']['sample_sd'] for p in points]) * 100
            ax.errorbar(x, y, yerr=sd, color=COLORS[index], label=method, marker=('o', 's', '^', 'D', 'v', 'P')[index],
                        markersize=4, linewidth=1.3, capsize=2)
            for i in range(3):
                ax.scatter(x, [p['miou_fixed']['values'][i] * 100 for p in points], color=COLORS[index], s=10, alpha=.4)
        ax.set(title=region, xlabel='Queried pool blocks (%)', ylabel=f'Fixed-{stats["classes"]}-class mIoU (%)')
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='outside upper center', ncol=min(3, len(labels)), frameon=False)
    save_figure(fig, figures, 'budget_curves')
    draw_source(histories, stats['classes'], figures)
    draw_components(data, stats, figures)
    return selected


def draw_source(histories, classes, figures):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for index, (train_seed, history) in enumerate(histories.items()):
        x = [row['step'] for row in history]
        axes[0].plot(x, [row['validation']['miou_fixed'] * 100 for row in history], color=COLORS[index % len(COLORS)], label=str(train_seed))
        axes[1].plot(x, [row['train_loss'] for row in history], color=COLORS[index % len(COLORS)], label=str(train_seed))
    axes[0].set(title='Source validation', xlabel='Source optimizer steps', ylabel=f'Fixed-{classes}-class mIoU (%)')
    axes[1].set(title='Source training loss', xlabel='Source optimizer steps', ylabel='Interval mean training cross-entropy')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(title='Training seed', frameon=False)
    save_figure(fig, figures, 'source_diagnostics')


def generate_source_only(run, output, title):
    run, output = Path(run).resolve(), Path(output).resolve()
    histories, completed, classes_seen = {}, {}, set()
    for path in sorted((run / 'source').glob('*/complete.json')):
        record = load(path)
        seed = int(record['seed'])
        history = load(path.parent / 'history.json')
        if seed in completed or not history:
            raise ValueError('Duplicate source seed or empty history')
        histories[seed], completed[seed] = history, record
        for row in history:
            classes_seen.add(len(row['validation']['iou']))
    if len(classes_seen) != 1 or len(completed) != 3:
        raise ValueError('Source-only report requires three completed seeds and a consistent class count')
    classes = classes_seen.pop()
    figures = run / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_source(histories, classes, figures)
    plateau = [seed for seed, row in completed.items() if row['stop_reason'] == 'source_validation_plateau']
    capped = [seed for seed, row in completed.items() if row['stop_reason'] == 'max_steps_reached_convergence_not_established']
    lines = [f'# {title}', '', '本報告僅讀取已完成的源域訓練記錄，未讀取目標域效果，也未獨立重跑訓練。', '',
             f'固定 {classes} 類 mIoU，zero-union 類 IoU=0。以下按訓練種子呈現，不選取表現最佳的種子代替完整結果。', '',
             f'觸發源域驗證平台停止條件的種子：`{plateau}`。達到步數上限、未確立收斂的種子：`{capped}`。平台條件只是預設停止規則，不是泛化收斂證明；不能宣稱全部訓練均已收斂。', '',
             f"![Source diagnostics]({link(figures / 'source_diagnostics.png', output)})", '',
             f"[向量 SVG]({link(figures / 'source_diagnostics.svg', output)})", '']
    lines += table(['訓練種子', '執行步數', 'Best step', 'Best validation mIoU (%)', '最後 validation mIoU (%)', '最後 interval train loss', '停止原因'],
                   [[seed, row['steps'], row['best_step'], f"{row['best_source_val_miou']*100:.3f}",
                     f"{histories[seed][-1]['validation']['miou_fixed']*100:.3f}", f"{histories[seed][-1]['train_loss']:.5f}", row['stop_reason']] for seed, row in completed.items()])
    lines += ['', '## Material Passport', '', '- 已執行：依 source completion 與逐次 history 記錄；本報告未獨立重跑。',
              '- 無目標域準確率、新穎性、SOTA 或 TGRS 投稿就緒主張。',
              f"- 源域訓練同步牆鐘時間合計：{sum(row['seconds'] for row in completed.values()):.2f} 秒；不等同全流程 GPU busy-time。", '']
    for seed in completed:
        for name in ('complete.json', 'history.json'):
            path = run / 'source' / str(seed) / name
            lines.append(f"- [{seed}/{name}]({link(path, output)})：SHA256 `{sha256(path)}`。")
    output.write_text('\n'.join(lines) + '\n')
    return {'classes': classes, 'plateau_seeds': plateau, 'unestablished_seeds': capped}


def generate(run, output, title):
    run, output = Path(run).resolve(), Path(output).resolve()
    data = load(run / 'all_results.json')
    stats = aggregate(data)
    cfg = data['config']
    histories, completed = {}, {}
    for pair in stats['seed_pairs']:
        train_seed = pair[0]
        if train_seed in histories:
            continue
        histories[train_seed] = load(run / 'source' / str(train_seed) / 'history.json')
        completed[train_seed] = load(run / 'source' / str(train_seed) / 'complete.json')
        if not histories[train_seed]:
            raise ValueError(f'No source history: {train_seed}')
    figures = run / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotted = draw_figures(data, stats, histories, figures)
    stats['provenance'] = {'input': str(run / 'all_results.json'), 'input_sha256': sha256(run / 'all_results.json'),
                           'report_code_sha256': sha256(__file__), 'plotted_strategies': plotted,
                           'component_visualization': component_policy(data),
                           'execution': 'engine results summarized; no independent training rerun',
                           'sd_policy': 'sample SD across three paired model/selection seeds; not CI',
                           'auc_policy': 'numpy.trapezoid on actual fractions for measured nominal 1%-5% points; normalized by actual endpoint width; no zero-budget interpolation'}
    stats_path = figures / 'paired_statistics.json'
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    lines = [f'# {title}', '',
             '本報告彙整已執行的 engine 結果，未獨立重跑訓練。保留全部預設策略與正、負差值；不據此宣稱 SOTA 或已具備 TGRS 投稿條件。', '',
             f"主指標為固定 {stats['classes']} 類 mIoU，zero-union 類 IoU 設為 0，分母固定。數值以百分比呈現；差值為百分點。", '',
             '預算百分比指已查詢的256×256圖塊占池內全部圖塊的比例，等價於完整圖塊像素面積比例，包含ignore區域的標註成本；不是已核實地面平方米比例或人工標註時間。LoveDA没有可靠地理座標，不能由本表推導真實地理面積。', '',
             f"三組模型／查詢種子按設定順序為 `{stats['seed_pairs']}`。表中為三次配對運行的平均 ± 樣本標準差（ddof=1）；散點不是独立資料集重複，並非信賴區間。不做像素級顯著性檢定，也不把種子視為地區泛化的獨立抽樣。", '',
             f"類別設定：`{cfg.get('classes', list(range(stats['classes'])))}`。完整逐種子 mIoU、每類 IoU、配對差值與積分資料：[paired_statistics.json]({link(stats_path, output)})。", '',
             '## 預設策略曲線', '',
             f"主圖策略：`{plotted}`。按預設清單選取，缺少地理策略時才使用預定替代策略，未按效果挑選。誤差棒為樣本 SD，半透明點為各種子原始值。", '',
             f"![Budget curves]({link(figures / 'budget_curves.png', output)})", '',
             f"[向量 SVG]({link(figures / 'budget_curves.svg', output)})", '']
    policy = component_policy(data)
    if policy is not None:
        lines += ['## 補充元件展示', '', policy['description'], '',
                  '各確認地區使用相同纵軸刻度；誤差棒為三組配對種子的樣本 SD，散點保留原始值。', '',
                  f"![Component diagnostics]({link(figures / 'component_diagnostics.png', output)})", '',
                  f"[元件圖 SVG]({link(figures / 'component_diagnostics.svg', output)})", '']
    lines += ['## Source-only', '', '開發域只保留獨立 source-only 行，不納入確認域的策略表、配對差值或積分。未宣稱開發域執行過目標適應。', '']
    lines += table(['Region', 'Role', 'Fixed-class mIoU mean ± SD (%)', '三個種子值 (%)'],
                   [[s['region'], s['role'], formatted(s['miou_fixed']), ', '.join(f'{v*100:.3f}' for v in s['miou_fixed']['values'])] for s in stats['source_only']])
    for region in stats['confirmation_regions']:
        lines += ['', f'## {region}：全部策略與預算', '']
        rows = []
        for s in stats['cells']:
            if s['region'] == region:
                rows.append([s['strategy'], f"{s['planned_fraction']*100:.1f}", s['budget'], f"{s['actual_fraction']*100:.5f}",
                             formatted(s['miou_fixed']), *[f'{v*100:.3f}' for v in s['miou_fixed']['values']], formatted(s['accuracy'])])
        lines += table(['策略', '名義圖塊預算 %', 'Blocks', '實際圖塊預算 %', 'mIoU mean ± SD', 'Pair 1', 'Pair 2', 'Pair 3', 'Accuracy mean ± SD'], rows)
        lines += ['', f'### {region}：逐種子配對差值', '', '正值表示該策略高於同一模型／查詢種子的參照策略；負值完整保留。', '']
        lines += table(['策略', '參照', '名義圖塊預算 %', 'Δ mIoU mean ± SD (pp)', 'Pair 1', 'Pair 2', 'Pair 3'],
                       [[s['strategy'], s['reference'], f"{s['planned_fraction']*100:.1f}", formatted(s['delta']),
                         *[f'{v*100:+.3f}' for v in s['delta']['values']]] for s in stats['paired_deltas'] if s['region'] == region])
    if stats['interactions']:
        lines += ['', '## 預定四組互補診斷', '',
                  '逐地區、預算及配對種子計算 interaction = geo_coverage − geo_entropy − coverage + entropy。上方配對表同時保留 geo_coverage 相對 coverage 與 geo_entropy 的增益。這是小樣本交互診斷：正或負值都不構成顯著性、機制互補或因果證明；未根據交互結果挑選圖中策略。', '']
        lines += table(['Region', '名義圖塊預算 %', '實際圖塊預算 %', 'Interaction mean ± SD (pp)', 'Pair 1', 'Pair 2', 'Pair 3'],
                       [[s['region'], f"{s['planned_fraction']*100:.1f}", f"{s['actual_fraction']*100:.5f}", formatted(s['interaction']),
                         *[f'{v*100:+.3f}' for v in s['interaction']['values']]] for s in stats['interactions']])
    lines += ['', '## 有限預算區間積分', '',
              '以名義 1%–5% 範圍內實測點對應的實際圖塊比例為 x，逐種子使用 `numpy.trapezoid` 積分後，除以實際末端減首端的寬度。下表明列因 ceil 取整產生的真實端點；不混用名義橫軸、不加入 0 預算 source-only 點、不外推。不同地區端點可能略異，因此不再合成跨地區積分平均。', '']
    lines += table(['Region', '策略', '實際積分端點 %', 'Normalized AUC mean ± SD (%)', 'Pair 1', 'Pair 2', 'Pair 3'],
                   [[s['region'], s['strategy'], f"{s['actual_x'][0]*100:.5f}–{s['actual_x'][-1]*100:.5f}", formatted(s['normalized_auc']),
                     *[f'{v*100:.3f}' for v in s['normalized_auc']['values']]] for s in stats['normalized_auc']])
    lines += ['', '### 配對積分差值', '']
    lines += table(['Region', '策略', '參照', 'Δ normalized AUC mean ± SD (pp)', 'Pair 1', 'Pair 2', 'Pair 3'],
                   [[s['region'], s['strategy'], s['reference'], formatted(s['delta']), *[f'{v*100:+.3f}' for v in s['delta']['values']]] for s in stats['paired_auc_deltas']])
    lines += ['', '## 源域訓練診斷', '',
              'checkpoint 與停止點只由源域驗證決定。以下呈現原始來源記錄；停止或最高驗證分數本身不證明泛化收斂。', '',
              f"觸發平台停止的種子：`{[seed for seed, row in completed.items() if row['stop_reason'] == 'source_validation_plateau']}`；達步數上限而未確立收斂的種子：`{[seed for seed, row in completed.items() if row['stop_reason'] == 'max_steps_reached_convergence_not_established']}`。平台停止是源驗證診斷，不等於泛化收斂證明。", '',
              f"![Source diagnostics]({link(figures / 'source_diagnostics.png', output)})", '',
              f"[向量 SVG]({link(figures / 'source_diagnostics.svg', output)})", '']
    lines += table(['訓練種子', '執行步數', 'Best step', 'Best source-val mIoU (%)', '停止原因', 'Source seconds'],
                   [[seed, row['steps'], row['best_step'], f"{row['best_source_val_miou']*100:.3f}", row['stop_reason'], f"{row['seconds']:.2f}"] for seed, row in completed.items()])
    selection = load(run / 'recipe_selection.json')
    lines += ['', '### 源配方與逐類門檻', '',
              f"架構：`{data['architecture']}`；凍結後端：`{cfg['execution_backend'][data['architecture']]}`。配方只由2026源驗證選擇：`{selection['selected_recipe']}`。",
              'Best checkpoint先要求所有源驗證類IoU≥0.01，再最大化固定類mIoU；可能不是無條件最高macro分數。此門檻只排除未學會類別。', '']
    lines += table(['候選配方', '種子', '更新步數', 'Best mIoU %', '最低類IoU %', '逐類通過', '平台診斷'],
                   [[r['recipe']['name'], r['seed'], r['steps'], f"{r['best_source_val_miou']*100:.3f}",
                     f"{min(r['best_source_val_iou'])*100:.3f}", r['eligible_classes'], r['convergence_diagnosed']] for r in selection['candidates']])
    lines += ['', '最終三個源模型逐類IoU（%），類序同固定類集合：', '']
    lines += table(['種子', '訓練原圖', '訓練塊', '驗證塊', *cfg['classes']],
                   [[seed, r['source_images'], r['source_patches'], r['validation_patches'],
                     *[f'{v*100:.3f}' for v in r['best_source_val_iou']]] for seed, r in completed.items()])
    source_seconds = sum(float(s['seconds']) for s in completed.values())
    unselected_seconds = sum(float(r['seconds']) for r in selection['candidates'] if r['recipe'] != selection['selected_recipe'])
    lines += ['', f'另有未選用源配方計時 {unselected_seconds:.2f} 秒，未隱藏或併作最終模型訓練時間。', '']
    adaptation_seconds = sum(float(s['train_seconds']) for s in data['runs'])
    lines += ['', '## 計算成本與公平性範圍', '',
              f"源域訓練按模型種子共享，來源記錄合計 {source_seconds:.2f} 秒；已評估 adaptation 運行訓練計時合計 {adaptation_seconds:.2f} 秒。每個策略／預算固定 {cfg['adapt_steps']} 次 optimizer 更新、batch {cfg['adapt_batch']}，從相同種子的源域 checkpoint 重置。來源計時是 CUDA 同步的牆鐘時間，並非 GPU busy-time。", '',
              '第一輪 pool scores 在同一源模型及地區內共用 cache；各策略記錄的 inference_seconds 可能重複歸因相同計算，不能相加充當實際總 wall time。上述訓練秒數亦不等於完整流程耗時：不含所有查詢、評估、I/O 與可能的中斷重跑。未將記錄相加冒充總 GPU 使用時間。', '',
              '## Material Passport', '',
              f"- 執行狀態：依 engine 的完整結果與 source completion 記錄確認已執行；本報告器未獨立重跑。", 
              f"- 結果來源：[all_results.json]({link(run / 'all_results.json', output)})；SHA256 `{stats['provenance']['input_sha256']}`。",
              f"- 報告程式 SHA256：`{stats['provenance']['report_code_sha256']}`。",
              '- 證據邊界：三組配對種子的描述統計；原始散點、所有策略及負差值保留；不作像素偽重複推論。',
              '- 本報告不標記任何文獻為使用者已讀，不授予獨立重現、SOTA、投稿就緒或新穎性認證。', '']
    output.write_text('\n'.join(lines))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='配對主動學習 baseline 實驗報告')
    parser.add_argument('--source-only', action='store_true', help='Read only source records; no all_results.json needed')
    args = parser.parse_args()
    if args.source_only:
        generate_source_only(args.run, args.output, args.title)
    else:
        generate(args.run, args.output, args.title)
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()
