import numpy as np
import seaborn as sns

from scipy.stats import sem
from analysis import get_baseline
from matplotlib.ticker import FormatStrFormatter


# def compute_summary_stats(X, axis=0, n_se=2):
#     """require X to be 2d"""
#     mu = np.mean(X, axis=axis)
#     er = sem(X, axis=axis)*n_se
#     return mu, er


def get_ylim_bonds(axes):
    ylim_l, ylim_r = axes[0].get_ylim()
    for i, ax in enumerate(axes):
        ylim_l_, ylim_r_ = axes[i].get_ylim()
        ylim_l = ylim_l_ if ylim_l_ < ylim_l else ylim_l
        ylim_r = ylim_r_ if ylim_r_ > ylim_r else ylim_r
    return ylim_l, ylim_r


def get_bw_pal(contrast=100):
    """return black and white color map

    Parameters
    ----------
    contrast : int
        contrast - black vs. white

    Returns
    -------
    list
        list of two rgb values

    """

    bw_pal = sns.color_palette(palette='Greys', n_colors=contrast)
    bw_pal = [bw_pal[-1], bw_pal[0]]
    return bw_pal


def plot_tz_pred_acc(
    pa_mu, pa_er, pa_or_dk_mu, event_bounds, p,
    f, ax,
    alpha=.3,
    title='Performance on the TZ task',

):
    """plot the preformance on the tz task

    Parameters
    ----------
    pa_mu : array
        mu, prediction accuracy
    pa_er : array
        error bar, prediction accuracy
    pa_or_dk_mu : array
        upper bound of accurate predictions + don't knows; i.e. non-errors
    event_bounds : array
        event boundaries
    p : the param class
        parameters
    f : type
        Description of parameter `f`.
    ax : type
        Description of parameter `ax`.
    alpha : float
        Description of parameter `alpha`.
    title : str
        Description of parameter `title`.

    """
    # precompute some stuff
    c_pal = sns.color_palette('colorblind', n_colors=4)
    legend_lab = ['event boundary', 'baseline', 'don\'t know',
                  'mistakes', 'correct prediction']
    total_event_len = np.shape(pa_mu)[0]
    x_ = range(total_event_len)
    ones = np.ones_like(x_)
    baseline = get_baseline(p.env.n_param, 1 / p.env.n_branch)[1:]
    # plot the performance
    ax.errorbar(x=x_, y=pa_mu, yerr=pa_er, color=c_pal[0])
    # plot dk region
    ax.fill_between(x_, pa_mu, pa_or_dk_mu, alpha=alpha, color='grey')
    # plot error region
    ax.fill_between(x_, pa_or_dk_mu, ones, alpha=alpha, color=c_pal[3])
    # plot event boundaries
    for eb in event_bounds:
        ax.axvline(eb-1, ls='--', color=c_pal[3], alpha=1)
    # plot observation baseline
    ax.plot(baseline, color='grey', ls='--')
    # add labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    # xyticks
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks(np.arange(0, total_event_len, p.env.n_param-1))
    # add legend
    f.legend(legend_lab, frameon=False, bbox_to_anchor=(.98, .7))
    sns.despine()
    f.tight_layout()


# precompute some stuff
def plot_rnr_pred_acc(
    pa_mu, pa_er, pa_or_dk_mu, p,
    f, ax,
    alpha=.3,
    title='Performance on the RNR task',
    legend_on=False,

):
    c_pal = sns.color_palette('colorblind', n_colors=4)
    legend_lab = ['baseline', 'don\'t know', 'mistakes', 'correct prediction']
    total_event_len = np.shape(pa_mu)[0]
    x_ = range(total_event_len)
    ones = np.ones_like(x_)
    baseline = get_baseline(p.env.n_param, 1 / p.env.n_branch)[1:]

    alpha = .3
    # plot the performance
    ax.errorbar(x=x_, y=pa_mu, yerr=pa_er, color=c_pal[0])
    # plot dk region
    ax.fill_between(x_, pa_mu, pa_or_dk_mu, alpha=alpha, color='grey')
    # plot error region
    ax.fill_between(x_, pa_or_dk_mu, ones, alpha=alpha, color=c_pal[3])

    # plot observation baseline
    ax.plot(baseline, color='grey', ls='--')
    # add labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    # xyticks
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks(np.arange(0, total_event_len, p.env.n_param-1))
    # add legend
    if legend_on:
        f.legend(legend_lab, frameon=False, bbox_to_anchor=(.98, .78))
    sns.despine()
    f.tight_layout()
