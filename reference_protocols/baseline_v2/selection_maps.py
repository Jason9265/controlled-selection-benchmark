"""Fixed OEM selection geometry: no labels, predictions or evaluation reads.

Example: python selection_maps.py --arch resnet18_unet
Requires both selection.json and complete.json for all three confirmation regions.
Fixed a priori: train seed 2026, query seed 17, nominal annotation fraction 5%.
Original UTM centres are translated by each pool's minimum easting/northing and
scaled to kilometres, without reprojection, jitter, interpolation or fake points.
"""
import argparse
import json
import math
from pathlib import Path

REGIONS = ('daressalaam', 'zanzibar', 'coxsbazar')
TITLES = {'daressalaam': 'Dar es Salaam', 'zanzibar': 'Zanzibar', 'coxsbazar': "Cox's Bazar"}
STRATEGIES = ('cai_spatial', 'geo_entropy')
TRAIN_SEED, QUERY_SEED, FRACTION = 2026, 17, 0.05
SCOPE = 'Fixed seed and 5% annotation; selection geometry only'


def read(path):
    return json.loads(path.read_text())


def split_sets(cai, geo):
    a, b = set(cai), set(geo)
    if len(a) != len(cai) or len(b) != len(geo):
        raise ValueError('Duplicate patch indices')
    return {'cai_only': sorted(a - b), 'geo_only': sorted(b - a), 'both': sorted(a & b)}


def local_coordinates(rows):
    if not rows:
        raise ValueError('Empty pool')
    epsgs = {r.get('epsg') for r in rows}
    if len(epsgs) != 1:
        raise ValueError('Pool centres must share one UTM EPSG')
    epsg = next(iter(epsgs))
    if type(epsg) is not int or not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError('Verified WGS84 UTM EPSG in metres is required')
    xy = []
    for row in rows:
        point, area = row.get('center_xy'), row.get('area_m2')
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) for v in point):
            raise ValueError('Invalid metric centre')
        if type(area) not in (float, int) or not math.isfinite(area) or area <= 0:
            raise ValueError('Positive metric area required')
        xy.append(point)
    origin = [min(p[d] for p in xy) for d in (0, 1)]
    return [[(p[d] - origin[d]) / 1000 for d in (0, 1)] for p in xy], epsg, origin


def load_panels(root, arch):
    cfg = read(root / 'configs/oem.json')
    if arch not in cfg['architectures']:
        raise ValueError('Architecture is not in the frozen OEM configuration')
    if [TRAIN_SEED, QUERY_SEED] not in cfg['seed_pairs'] or FRACTION not in cfg['budget_fractions']:
        raise ValueError('Fixed seed/fraction missing from configuration')
    if set(cfg['confirmation_regions']) != set(REGIONS) or not set(STRATEGIES) <= set(cfg['strategies']):
        raise ValueError('Fixed confirmation regions or methods missing')
    manifest = read(root / 'prepared/manifest.json')
    # Check all required files before accepting any panel or writing outputs.
    required = []
    for region in REGIONS:
        target = manifest['targets'][region]
        if target.get('role') != 'confirmation':
            raise ValueError('Requested region is not a confirmation region')
        budget = math.ceil(len(target['pool']) * FRACTION)
        for strategy in STRATEGIES:
            folder = root / 'runs/oem' / arch / 'adapt' / region / f'm{TRAIN_SEED}_q{QUERY_SEED}' / strategy / f'b{budget}'
            required.extend(folder / name for name in ('selection.json', 'complete.json'))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('Incomplete three-region selection matrix; refusing formal figure. Missing: ' + ', '.join(missing))
    panels = []
    for region in REGIONS:
        rows = manifest['targets'][region]['pool']
        coords, epsg, origin = local_coordinates(rows)
        budget = math.ceil(len(rows) * FRACTION)
        chosen = {}
        for strategy in STRATEGIES:
            folder = root / 'runs/oem' / arch / 'adapt' / region / f'm{TRAIN_SEED}_q{QUERY_SEED}' / strategy / f'b{budget}'
            selection, complete = read(folder / 'selection.json'), read(folder / 'complete.json')
            expected = dict(arch=arch, region=region, train_seed=TRAIN_SEED, selection_seed=QUERY_SEED, strategy=strategy, budget=budget)
            if any(complete.get(k) != v for k, v in expected.items()):
                raise ValueError('Completion context mismatch: ' + str(folder))
            if complete.get('planned_fraction') != FRACTION:
                raise ValueError('Completion nominal fraction mismatch')
            indices = selection.get('indices')
            if not isinstance(indices, list) or len(indices) != budget or any(type(i) is not int or not 0 <= i < len(rows) for i in indices) or len(set(indices)) != budget:
                raise ValueError('Invalid budget or indices: ' + str(folder))
            for field, rowfield in [('ids', 'id'), ('groups', 'group')]:
                if selection.get(field) != [rows[i][rowfield] for i in indices]:
                    raise ValueError('Selection does not match manifest: ' + field)
            if any(selection.get(k) != complete.get(k) for k in ('indices', 'ids', 'groups', 'previous')):
                raise ValueError('Selection/completion disagreement')
            chosen[strategy] = indices
        panels.append(dict(region=region, coordinates=coords, epsg=epsg, origin=origin,
                           budget=budget, pool_size=len(rows), **split_sets(chosen['cai_spatial'], chosen['geo_entropy'])))
    return panels


def make_figure(panels, arch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if tuple(p['region'] for p in panels) != REGIONS:
        raise ValueError('Exactly three panels in the fixed region order are required')
    styles = [('cai_only', '#0072B2', 's', 'cai_spatial only'),
              ('geo_only', '#E69F00', '^', 'geo_entropy only'),
              ('both', '#8E5EA2', 'o', 'Both')]
    with plt.rc_context({'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False}):
        fig = plt.figure(figsize=(15, 6.3), layout='constrained')
        grid = fig.add_gridspec(2, 3, height_ratios=[1, .14])
        for col, panel in enumerate(panels):
            ax = fig.add_subplot(grid[0, col])
            xy = panel['coordinates']
            ax.scatter([p[0] for p in xy], [p[1] for p in xy], s=12, c='#C9CDD2', linewidths=0, zorder=1)
            for key, color, marker, _ in styles:
                points = [xy[i] for i in panel[key]]
                if points:
                    ax.scatter([p[0] for p in points], [p[1] for p in points], s=48, c=color, marker=marker,
                               edgecolors='white', linewidths=.45, zorder=3)
            ax.set_title(f"{TITLES[panel['region']]}\nBudget {panel['budget']} / {panel['pool_size']}; overlap {len(panel['both'])}", pad=10)
            ax.set_xlabel('Local easting (km)')
            ax.set_ylabel('Local northing (km)')
            ax.set_aspect('equal', adjustable='box')
            ax.margins(.08)
            ax.grid(alpha=.16, zorder=0)
            ax.ticklabel_format(style='plain', useOffset=False)
        legend_ax = fig.add_subplot(grid[1, :]); legend_ax.axis('off')
        handles = [Line2D([], [], color='#C9CDD2', marker='o', linestyle='none', markersize=5, label='All pool patches')]
        handles += [Line2D([], [], color=c, marker=m, linestyle='none', markersize=7, label=label) for _, c, m, label in styles]
        legend_ax.legend(handles=handles, loc='center', ncol=4, frameon=False)
        fig.suptitle(f"{arch} | train seed {TRAIN_SEED}, query seed {QUERY_SEED}\n{SCOPE}", fontsize=13)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--arch', choices=['resnet18_unet', 'segformer_b0'], required=True)
    args = parser.parse_args()
    try:
        panels = load_panels(args.root, args.arch)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    fig = make_figure(panels, args.arch)
    out = args.root / 'runs/oem' / args.arch / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        path = out / f'selection_spatial_diagnostics.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.15, facecolor='white')
        print(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == '__main__':
    main()
