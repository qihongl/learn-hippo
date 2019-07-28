import numpy as np
import seaborn as sns

# from scipy.stats import sem
from analysis import get_baseline, compute_stats
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT
from matplotlib.ticker import FormatStrFormatter


def plot_pred_acc_full(
    pa_mu, pa_er, pa_or_dk_mu, event_bounds, p,
    f, ax,
    alpha=.3,
    title='Performance on the TZ task',
    add_legend=True, legend_loc=(.98, .7),

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
    if add_legend:
        f.legend(legend_lab, frameon=False, bbox_to_anchor=legend_loc)
    sns.despine()
    f.tight_layout()


# precompute some stuff
def plot_pred_acc_rcl(
    pa_mu, pa_er, pa_or_dk_mu, p,
    f, ax,
    alpha=.3,
    title='Performance on the RNR task',
    baseline_on=True,
    legend_on=False,
):
    if baseline_on:
        legend_lab = ['baseline', 'don\'t know',
                      'mistakes', 'correct prediction']
        baseline = get_baseline(p.env.n_param, 1 / p.env.n_branch)[1:]
    else:
        legend_lab = ['don\'t know', 'mistakes', 'correct prediction']
    c_pal = sns.color_palette('colorblind', n_colors=4)
    total_event_len = np.shape(pa_mu)[0]
    x_ = range(total_event_len)
    ones = np.ones_like(x_)

    alpha = .3
    # plot the performance
    ax.errorbar(x=x_, y=pa_mu, yerr=pa_er, color=c_pal[0])
    # plot dk region
    ax.fill_between(x_, pa_mu, pa_or_dk_mu, alpha=alpha, color='grey')
    # plot error region
    ax.fill_between(x_, pa_or_dk_mu, ones, alpha=alpha, color=c_pal[3])

    # plot observation baseline
    if baseline_on:
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
        f.legend(legend_lab, frameon=False, bbox_to_anchor=(.98, .6))
    sns.despine()
    f.tight_layout()


def plot_time_course_for_all_conds(
        matrix, cond_ids, ax,
        n_se=2,
        axis1_start=0, xlabel=None, ylabel=None, title=None,
        frameon=False, add_legend=True,
):
    for i, cond_name in enumerate(TZ_COND_DICT.values()):
        submatrix_ = matrix[cond_ids[cond_name], axis1_start:]
        M_, T_ = np.shape(submatrix_)
        mu_, er_ = compute_stats(submatrix_, axis=0, n_se=n_se)
        ax.errorbar(x=range(T_), y=mu_, yerr=er_, label=cond_name)
    if add_legend:
        ax.legend(frameon=frameon)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def show_weight_stats(agent):
    for name, wts in agent.named_parameters():
        wts_np = to_sqnp(wts)
        wts_norm = np.linalg.norm(wts_np)
        wts_mean = np.mean(wts_np)
        wts_min, wts_max = np.min(wts_np), np.max(wts_np)
        print(name, np.shape(wts_np))
        print('\tNorm: %6.2f, Mean: %6.2f, Min %6.2f, Max: %6.2f' %
              (wts_norm, wts_mean, wts_min, wts_max))
