import seaborn as sns


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


def print_dict(d, indent=0):
    '''
    reference:https://stackoverflow.com/questions/
    3229419/how-to-pretty-print-nested-dictionaries
    '''
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))
