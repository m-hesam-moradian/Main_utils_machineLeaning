"""
Figure 4: SHAP-style Waterfall Plot from Sensitivity
Two side-by-side waterfalls (RFC + KNNC).
"""
import sys, os
sys.path.insert(0, r'd:\ML\task\codes\novelty figures')
from fig_common import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch

copula = load_copula()

fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(1, 2, figure=fig,
                       wspace=0.30, left=0.06, right=0.97, top=0.88, bottom=0.10)

# fig.suptitle('SHAP-style Waterfall — Feature Sensitivity by Morris',
#              x=0.06, y=0.965, ha='left', fontsize=17, fontweight='bold', color=SLATE)
# fig.text(0.06, 0.925,
#          'Each bar shows how much a feature shifts the model output from the baseline.  '
#          'Cumulative line tracks the running prediction value.',
#          ha='left', fontsize=10, style='italic', color=SLATE_2)

def draw_waterfall(ax, model_name, model_color):
    items = sorted(copula[model_name], key=lambda x: x[1], reverse=True)
    features = [f for f, _ in items]
    sens = [s for _, s in items]
    n = len(features)
    total = sum(sens)
    norm_sens = [s / total for s in sens]
    base_value = 0.0
    final_value = sum(sens)
    cum = [base_value]
    for s in sens:
        cum.append(cum[-1] + s)

    labels = ['Base'] + features + ['Final']
    x_pos = np.arange(len(labels))
    bar_width = 0.65

    ax.bar(x_pos[0], 0, bar_width, color='#94A3B8', edgecolor='#475569',
           linewidth=1.0, zorder=3)
    ax.text(x_pos[0], 0 + final_value*0.04, f'Base = 0.000',
            ha='center', va='bottom', fontsize=9, color=SLATE, fontweight='bold')

    for i in range(1, n + 1):
        bottom = cum[i - 1]
        height = sens[i - 1]
        intensity = 0.35 + 0.65 * (sens[i - 1] / max(sens))
        bar_color = to_rgba(model_color, intensity)
        ax.bar(x_pos[i], height, bar_width, bottom=bottom,
                color=bar_color, edgecolor=model_color, linewidth=0.8, zorder=3)
        y_text = bottom + height + final_value * 0.012
        ax.text(x_pos[i], y_text, f'+{sens[i-1]:.4f}',
                ha='center', va='bottom', fontsize=9, color=INK, fontweight='bold')
        if i < n + 1:
            ax.plot([x_pos[i-1] + bar_width/2, x_pos[i] - bar_width/2],
                     [bottom, bottom], color=SLATE_2, linestyle=':', linewidth=0.8, zorder=2)

    ax.bar(x_pos[-1], final_value, bar_width, color=model_color,
            edgecolor=SLATE, linewidth=0.8, hatch='//', alpha=0.85, zorder=3)
    ax.text(x_pos[-1], final_value + final_value * 0.012,
             f'Final = {final_value:.3f}',
             ha='center', va='bottom', fontsize=10, color=SLATE, fontweight='bold')
    ax.plot([x_pos[-2] + bar_width/2, x_pos[-1] - bar_width/2],
             [cum[-1], cum[-1]], color=SLATE_2, linestyle=':', linewidth=0.8, zorder=2)

    cum_x = [x_pos[0]] + list(x_pos[1:n+1]) + [x_pos[-1]]
    cum_y = [0] + list(cum[1:]) + [final_value]
    ax.plot(cum_x, cum_y, color='#B91C1C', linewidth=2.0,
             marker='o', markersize=5, markerfacecolor='white',
             markeredgecolor='#B91C1C', markeredgewidth=1.5, zorder=4,
             label='Cumulative sensitivity')

    ax.set_xticks(x_pos)
    short = {
        'energy_consumption': 'energy cons.', 'inference_time':     'inference t.',
        'memory_usage':       'memory use',    'payload_entropy':    'payload ent.',
        'cpu_usage':           'cpu usage',     'device_type':        'device type',
        'attack_type':        'attack type',   'energy_class':       'energy class',
        'flow_rate':          'flow rate',     'pkt_size':           'pkt size',
        'duration':           'duration',      'protocol':           'protocol',
    }
    feature_labels = ['Base'] + [short.get(f, f.replace('_', ' ')) for f in features] + ['Final']
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=9, color=INK)
    ax.set_ylabel('Morris Sensitivity ($mu^*$)', fontsize=10, color=SLATE_2)
    ax.set_title(f'{model_name}  ·  Total = {final_value:.3f}',
                 loc='left', fontsize=13, fontweight='bold', color=model_color, pad=6)
    ax.set_ylim(0, final_value * 1.25)
    ax.grid(True, axis='y', alpha=0.18, linestyle='--', color=BG_3, linewidth=0.5)
    ax.set_axisbelow(True)

    top_idx = int(np.argmax(sens))
    ax.annotate(f'Top contributor:  {features[top_idx].replace("_", " ")}\n'
                 f'  sensitivity = {sens[top_idx]:.4f}  ({sens[top_idx]/final_value*100:.1f}%)',
                 xy=(x_pos[top_idx + 1], cum[top_idx + 1]),
                 xytext=(x_pos[top_idx + 1] + 2.5, cum[top_idx + 1] + final_value * 0.20),
                 fontsize=9, color=INK, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=model_color, linewidth=0.8, alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=model_color,
                                  linewidth=1.0, connectionstyle='arc3,rad=-0.25'))

    legend_elems = [
        Patch(facecolor=to_rgba(model_color, 0.6), edgecolor=model_color,
              label='Per-feature contribution'),
        plt.Line2D([0], [0], color='#B91C1C', marker='o', markersize=5,
                    markerfacecolor='white', markeredgecolor='#B91C1C',
                    markeredgewidth=1.5, label='Cumulative sensitivity'),
        Patch(facecolor=SLATE, edgecolor=SLATE, hatch='//', alpha=0.6,
              label='Final model output'),
    ]
    ax.legend(handles=legend_elems, loc='upper left',
              bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False, ncol=1)

ax_rfc = fig.add_subplot(gs[0, 0])
draw_waterfall(ax_rfc, 'RFC', CLASSIFIER_PALETTE['RFC'])
ax_knnc = fig.add_subplot(gs[0, 1])
draw_waterfall(ax_knnc, 'KNNC', CLASSIFIER_PALETTE['KNNC'])

# fig.text(0.50, 0.02,
#          'Source:  Morris_Sensitivity(Chi2) sheet  ·  Sensitivities are normalised so the final value equals the sum of all contributions.',
#          ha='center', fontsize=9, color=SLATE_2, style='italic')

out = os.path.join(DOWNLOAD_DIR, 'fig4_shap_waterfall.png')
fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'[OK] Saved: {out}')
