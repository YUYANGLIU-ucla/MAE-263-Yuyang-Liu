"""Shared matplotlib style for all project figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#222222', '#777777'
BLUES = ['#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
plt.rcParams.update({
    'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.titlesize': 8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'legend.frameon': False, 'legend.fontsize': 7, 'lines.linewidth': 1.4, 'lines.markersize': 3.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
})


def save(fig, name):
    fig.savefig(f'figures/{name}.pdf')
    fig.savefig(f'figures/{name}.png', dpi=200)
    plt.close(fig)
