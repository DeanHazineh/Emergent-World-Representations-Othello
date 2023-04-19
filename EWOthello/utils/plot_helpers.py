import matplotlib.pyplot as plt
from EWOthello.data.othello import permit_reverse
from mpl_toolkits.axes_grid1 import make_axes_locatable


def seq_to_boardCoord(sequence):
    return [permit_reverse(move) for move in sequence]


def addAxis(thisfig, n1, n2, maxnumaxis=""):
    counterval = maxnumaxis if maxnumaxis else n1 * n2
    axlist = [thisfig.add_subplot(n1, n2, i + 1) for i in range(counterval)]
    return axlist


def format_ax_boardImage(ax):
    for thisaxis in ax:
        thisaxis.set_yticks(ticks=list(range(0, 8)))
        thisaxis.set_yticklabels("ABCDEFGH")
        thisaxis.set_xticks(ticks=list(range(0, 8)))
        thisaxis.set_xticklabels(list(range(1, 9)))

    return


def addColorbar(thisfig, thisax, thisim, cbartitle=""):
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    cbar = thisfig.colorbar(thisim, cax=cax, orientation="vertical")

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(cbartitle, rotation=90)

    return
