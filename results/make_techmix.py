import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pypsa
import yaml
from pathlib import Path

BASE = '/Users/minosandri/Bachelorarbeit/pypsa-eur_dev/results'
SCENARIOS = {
    'Baseline\n(no alpine)': f'{BASE}/Baseline_imp5_simp0_g5/baseline_all_nosimp_no_tes/networks/base_s_adm___2050.nc',
    'B+\n(cold 1985/86)':    f'{BASE}/8586_B+_imp5_simp0_g5/alpine_final/networks/base_s_adm___2050.nc',
    'C+\n(moderate 2017/18)':f'{BASE}/1718_C+_imp5_simp0_g5/alpine_final/networks/base_s_adm___2050.nc',
    'A+\n(warm 2006/07)':    f'{BASE}/0607_A+_imp5_simp0_g5/alpine_final/networks/base_s_adm___2050.nc',
}
def stacked_barh_signed_multi(
    data_dict,
    ax,
    *,
    tech_colors=None,
    nice_names=None,
    y_labels=None,
    annotate_thresh=0.03,
    annotate_decimals=0,
    title=None,
    bar_height=0.6,
    show_legend=True,
):
    if tech_colors is None:
        tech_colors = {}
    if nice_names is None:
        nice_names = {}

    keys = list(data_dict.keys())
    if y_labels is None:
        y_labels = keys

    legend_handles = {}
    max_pos = 0.0
    max_neg = 0.0

    y_positions = np.arange(len(keys))
    for i, key in enumerate(keys):
        s = data_dict[key].dropna()
        pos_sum = s[s > 0].sum() if (s > 0).any() else 0.0
        neg_sum = abs(s[s < 0].sum()) if (s < 0).any() else 0.0
        max_pos = max(max_pos, pos_sum)
        max_neg = max(max_neg, neg_sum)
        _plot_single_bar(
            s,
            ax,
            y_positions[i],
            tech_colors,
            nice_names,
            annotate_thresh,
            bar_height,
            legend_handles,
            annotate_decimals,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.axvline(0, color='k', lw=0.8, alpha=0.4)

    pad = 0.05 * max(max_pos, max_neg, 1e-12)
    ax.set_xlim(-max_neg - pad, max_pos + pad)

    if title:
        ax.set_title(title, pad=12)

    if show_legend and legend_handles:
        ax.legend(handles=list(legend_handles.values()), bbox_to_anchor=(1.04, 1), loc='upper left')


def _plot_single_bar(
    s,
    ax,
    y_pos,
    tech_colors,
    nice_names,
    annotate_thresh,
    bar_height,
    legend_handles,
    annotate_decimals=0,
):
    s = s.dropna()
    pos = s[s > 0].sort_values(ascending=False)
    neg = s[s < 0].sort_values()

    total_pos = pos.sum()
    total_neg_abs = (-neg).sum()

    left_pos = 0.0
    left_neg = 0.0

    for key, v in pos.items():
        color = tech_colors.get(key, '#888888')
        label = nice_names.get(key, key)
        ax.barh(y_pos, v, left=left_pos, height=bar_height, color=color, edgecolor='white')
        if total_pos > 0 and (v / total_pos) >= annotate_thresh:
            ax.text(left_pos + v / 2, y_pos, f'{v:.{annotate_decimals}f}', ha='center', va='center', fontsize=8)
        left_pos += v
        if label not in legend_handles:
            legend_handles[label] = mpatches.Patch(color=color, label=label)

    for key, v in neg.items():
        color = tech_colors.get(key, '#888888')
        label = nice_names.get(key, key)
        ax.barh(y_pos, v, left=left_neg, height=bar_height, color=color, edgecolor='white')
        if total_neg_abs > 0 and (abs(v) / total_neg_abs) >= annotate_thresh:
            ax.text(left_neg + v / 2, y_pos, f'{v:.{annotate_decimals}f}', ha='center', va='center', fontsize=8)
        left_neg += v
        if label not in legend_handles:
            legend_handles[label] = mpatches.Patch(color=color, label=label)


def energy_balance_ch(n):
    eb_bus = n.statistics.energy_balance(drop_zero=False, nice_names=False, groupby=['bus', 'carrier'])
    ch_mask = eb_bus.index.get_level_values('bus').astype(str).str.startswith('CH')
    return eb_bus[ch_mask].groupby('carrier').sum()


def ch_only_carrier(n, c, **kwargs):
    df = n.df(c)
    bus_col = 'bus' if 'bus' in df.columns else 'bus0'
    return df['carrier'].where(df[bus_col].astype(str).str.startswith('CH'))


config_path = Path(BASE).parent / 'config' / 'plotting.default.yaml'
with open(config_path) as f:
    pypsa_carrier_colors = yaml.safe_load(f)['plotting']['tech_colors']

nets = {name: pypsa.Network(path) for name, path in SCENARIOS.items()}

production_by_scenario = {}
capacity_by_scenario = {}
carrier_union = set()

for name in SCENARIOS.keys():
    n = nets[name]
    energy_balance = energy_balance_ch(n)
    production_ch = energy_balance[energy_balance > 0].dropna()
    production_by_scenario[name] = production_ch / 1e6  # TWh
    carrier_union.update(production_ch.index.tolist())

carrier_list = list(carrier_union)

# GW carrier list = production carriers + PHS (storage, not a net generator)
cap_carrier_list = list(set(carrier_list) | {'PHS'})

for name in SCENARIOS.keys():
    n = nets[name]
    stats = n.statistics
    cap = stats.optimal_capacity(drop_zero=False, nice_names=False, groupby=ch_only_carrier)
    cap = cap.drop('Store', level='component', errors='ignore')
    cap = cap.drop('Line', level='component', errors='ignore')
    cap = cap.dropna()
    if isinstance(cap.index, pd.MultiIndex):
        cap = cap.groupby(level='carrier').sum()
    cap = cap.reindex(cap_carrier_list).fillna(0.0)
    capacity_by_scenario[name] = cap / 1e3  # GW

scenario_labels = [s.replace('\n', ' ') for s in SCENARIOS.keys()]

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 9),
    sharex=False,
    gridspec_kw={'height_ratios': [2, 1]},
)

stacked_barh_signed_multi(
    production_by_scenario,
    axes[0],
    tech_colors=pypsa_carrier_colors,
    y_labels=scenario_labels,
    annotate_decimals=1,
    title='Electricity Generation in Switzerland',
    show_legend=True,
)
axes[0].invert_yaxis()
axes[0].set_xlabel('Annual production [TWh]')

stacked_barh_signed_multi(
    capacity_by_scenario,
    axes[1],
    tech_colors=pypsa_carrier_colors,
    y_labels=scenario_labels,
    annotate_decimals=1,
    title='Installed capacity in Switzerland',
    show_legend=False,
)
axes[1].invert_yaxis()
axes[1].set_xlabel('Installed capacity [GW]')

plt.tight_layout()

out = f'{BASE}/production_ch_all'
plt.savefig(out + '.pdf', bbox_inches='tight', dpi=150)
plt.savefig(out + '.png', bbox_inches='tight', dpi=150)
print(f'Saved: {out}.pdf / .png')
