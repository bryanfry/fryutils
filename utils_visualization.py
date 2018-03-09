import matplotlib.pyplot as plt


MATPLOTLIB_DFLT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']


def apply_plot_cosmetics(ax, x_label='Frequency [Hz]', y_label=None, title=None,
                         show_legend=True):
    if x_label:
        ax.set_xlabel(x_label, fontsize=18)
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)
    if title:
        ax.set_title(title, fontsize=22)
    if show_legend:
        ax.legend(fontsize=18, frameon=True, framealpha=0.9, edgecolor='gray')
    ax.tick_params(axis='both', labelsize=14, pad=10)
    return ax
