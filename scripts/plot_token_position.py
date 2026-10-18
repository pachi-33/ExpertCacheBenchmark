"""Plot token-position summaries; install the optional project dependency .[plots]."""
import argparse
import json
from pathlib import Path
import numpy as np

from expertcache.contracts import TokenPositionSummary
from expertcache.data import digest_file, write_json


def rolling_mean(points, window):
    """Trailing window, weighted by actual expert accesses; never smooth quantiles."""
    hits = np.cumsum([0] + [p.expert_hits for p in points], dtype=np.float64)
    accesses = np.cumsum([0] + [p.expert_accesses for p in points], dtype=np.float64)
    end = np.arange(1, len(points) + 1)
    start = np.maximum(0, end - window)
    return (hits[end] - hits[start]) / (accesses[end] - accesses[start])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary')
    parser.add_argument('--output', required=True, help='new figure directory')
    parser.add_argument('--window', type=int, default=1, help='trailing mean window; default 1 keeps raw token points')
    parser.add_argument('--title', default='Expert cache hits within each request')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('window must be positive')
    summary_path = Path(args.summary).resolve()
    source = json.loads(summary_path.read_text())
    result = TokenPositionSummary.model_validate(source['hit_rate_by_token_position'])
    if not result.overall:
        parser.error('summary has no evaluated token positions')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator, StrMethodFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.edgecolor': '#cbd5e1', 'axes.labelcolor': '#334155', 'text.color': '#0f172a',
        'xtick.color': '#475569', 'ytick.color': '#475569', 'savefig.facecolor': 'white'})
    output.mkdir(parents=True)
    mean_label = 'Mean' if args.window == 1 else f'Mean (trailing {args.window} tokens)'
    points = result.overall
    x = np.array([p.token_index + 1 for p in points])
    xmin, xmax = (.5, 1.5) if len(x) == 1 else (1, int(x[-1]))

    def layout(subtitle):
        fig, (ax, support) = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True,
            gridspec_kw={'height_ratios': [4, 1], 'hspace': .12})
        fig.subplots_adjust(left=.09, right=.97, top=.80, bottom=.18)
        fig.suptitle(args.title, x=.09, y=.97, ha='left', fontsize=20, fontweight='bold')
        fig.text(.09, .90, subtitle, fontsize=11, color='#475569')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('Expert hit rate')
        ax.grid(axis='y', color='#e2e8f0', linewidth=.7)
        support.set_ylabel('Requests', fontsize=10)
        support.set_xlabel('Output token position within request (1-based)')
        support.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        support.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        support.grid(axis='y', color='#f1f5f9')
        support.set_xlim(xmin, xmax)
        ticks = MaxNLocator(8, integer=True).tick_values(1, int(x[-1]))
        support.set_xticks(sorted({1, int(x[-1]), *(int(t) for t in ticks if 1 < t < x[-1])}))
        fig.text(.09, .06, f'Each request/token combines {len(result.metadata.valid_layers)} layers × '
                 f'top-{result.metadata.top_k}. Cache state continues across requests.', fontsize=9, color='#64748b')
        return fig, ax, support

    fig, ax, support = layout(f'{result.metadata.request_count:,} evaluation requests · '
                             f'{result.metadata.request_token_count:,} output tokens · pooled overall curve')
    ax.fill_between(x, [p.p90 for p in points], [p.p10 for p in points], color='#dbeafe',
                    label='P90–P10 (middle 80% of requests)', zorder=1)
    ax.plot(x, [p.p99 for p in points], color='#94a3b8', linewidth=1.2, linestyle=':', label='P99 (lower 1% boundary)')
    ax.plot(x, [p.p50 for p in points], color='#0d9488', linewidth=1.4, linestyle='--', label='P50 / median')
    ax.plot(x, rolling_mean(points, args.window), color='#1d4ed8', linewidth=1.9, label=mean_label)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    support.fill_between(x, [p.request_count for p in points], color='#dbeafe')
    support.plot(x, [p.request_count for p in points], color='#1d4ed8', linewidth=1)
    support.set_ylim(0, points[0].request_count * 1.12)
    support.set_yticks([0, points[0].request_count])
    fig.text(.09, .025, 'Descending percentiles: P10 = Q90, P90 = Q10, P99 = Q1. Percentile curves are unsmoothed.',
             fontsize=9, color='#64748b')
    for ext in ('png', 'svg'):
        fig.savefig(output / f'overall.{ext}', dpi=170)
    plt.close(fig)

    fig, ax, support = layout(f'{len(result.by_task)} task groups · {mean_label.lower()} at each local token position')
    colors = ['#2563eb', '#0d9488', '#d97706', '#9333ea', '#e11d48', '#0891b2']
    for i, (task, curve) in enumerate(sorted(result.by_task.items())):
        xx = [p.token_index + 1 for p in curve]
        color = colors[i % len(colors)]
        ax.plot(xx, rolling_mean(curve, args.window), label=f'{task} (n={curve[0].request_count:,})', color=color, linewidth=1.7)
        support.plot(xx, [p.request_count for p in curve], color=color, linewidth=1)
    max_support = max(curve[0].request_count for curve in result.by_task.values())
    support.set_ylim(0, max_support * 1.12)
    support.set_yticks([0, max_support])
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    fig.text(.09, .025, 'Later positions include only requests that reached that token; no padding or zero-filled suffixes.',
             fontsize=9, color='#64748b')
    for ext in ('png', 'svg'):
        fig.savefig(output / f'by-task.{ext}', dpi=170)
    plt.close(fig)
    write_json(output / 'provenance.json', dict(summary_path=str(summary_path),
        summary_sha256=digest_file(summary_path), window=args.window, matplotlib_version=matplotlib.__version__,
        script_sha256=digest_file(__file__),
        figures={p.name: digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps({'output': str(output.resolve()), 'window': args.window}))


if __name__ == '__main__':
    main()
