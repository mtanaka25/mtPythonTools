import matplotlib.pyplot as plt
from .mtcolors import ichigo, ruri, rikyu, kincha, sumire, nibi, sumi

line_colors = (ichigo, ruri, rikyu, kincha, sumire, nibi, sumi)
line_width  = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
line_styles = ('-', '-', '-', '-', '-', '-')
labels      = ('Series 1', 'Series 2', 'Series 3', 'Series 4', 'Series 5', 'Series 6')

def multiple_line_plot(x,
                       y,
                       *,
                       line_colors = line_colors,
                       line_width = line_width,
                       line_styles = line_styles,
                       labels = labels,
                       show_legend = True,
                       plot_45_degree_line = False,
                       x_label = None,
                       y_label = None,
                       title = None,
                       xlim = None,
                       ylim = None,
                       savefig = True,
                       fname = 'figure.png',
                       style = 'ggplot'
                       ):
    plt.style.use(style)
    N_y = y.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if plot_45_degree_line:
        ax.plot(x, x, color = sumi, lw = 0.75, ls = ':', label = '45 degree line')
    for n in range(N_y):
        ax.plot(x, y[n,:], color = line_colors[n], lw = line_width[n],
                ls = line_styles[n], label = labels[n])
    if show_legend:
        ax.legend(frameon = False)
    if type(xlim) != type(None):
        ax.set_xlim(xlim)
    if type(ylim) != type(None):
        ax.set_ylim(ylim)
    if type(x_label) != type(None):
        ax.set_xlabel(x_label)
    if type(y_label) != type(None):
        ax.set_ylabel(y_label)
    if type(title) != type(None):
        ax.set_title(title)
    if savefig:
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
