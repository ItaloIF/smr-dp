import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import numpy as np

plt.rcParams['axes.axisbelow'] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 9

def get_ticks_values(lims, num_ticks):
    length = lims[1] - lims[0]
    power = np.floor(np.log10(length))
    base_step = 10 ** power

    factors = [0.1, 0.2, 0.5, 1, 2, 5, 10]

    best_step = base_step

    best_step = min((f * base_step for f in factors),
                    key=lambda s: abs(np.ceil(length / s) - num_ticks))
        
    # return step and also the format of the thiks to show all decimal places
    if best_step < 1:
        fmt = f'%.{abs(int(np.floor(np.log10(best_step))))}f'
    else:
        fmt = '%.0f'
    major_step = best_step
    mantissa = major_step / 10**int(np.floor(np.log10(abs(major_step))))
    for m in (1, 2, 5):
        if np.isclose(mantissa, m, atol=1e-4):
            m_value = m
            break
    
    if m_value == 2:
        minor_step = major_step / 4
    else:
        minor_step = major_step / 5

    return [major_step, minor_step], fmt

def create_ax_format(dim, xlim, ylim, xticks, yticks, fmtx=None, fmty=None, ax_size=False):
    fig, ax = plt.subplots(figsize=dim)

    if ax_size:
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Configure tick parameters
    ax.xaxis.set_tick_params(which='major', size=4, width=0.4, direction='in', top=True)
    ax.xaxis.set_tick_params(which='minor', size=2.5, width=0.2, direction='in', top=True)
    ax.yaxis.set_tick_params(which='major', size=4, width=0.4, direction='in', right=True)
    ax.yaxis.set_tick_params(which='minor', size=2.5, width=0.2, direction='in', right=True)
    
    # Set major and minor tick locators
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xticks[0]))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(xticks[1]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks[0]))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(yticks[1]))
    
    # Set default formatters if None
    if fmtx is None:
        fmtx = '{:.0f}'.format 
    if fmty is None:
        fmty = '{:.0f}'.format
    
    # Format the tick labels
    if callable(fmtx):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: fmtx(x)))
    else:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(fmtx))
    
    if callable(fmty):
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: fmty(y)))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmty))
    
    # Set tick label positions
    ax.tick_params(labeltop=False, labelright=False)
    
    return fig, ax

def plot_acceleration(acc, dt, p_arrival=None, save_path='img/acceleration_plot.png'):
    xticks, fmtx = get_ticks_values((0, len(acc)*dt), 5)
    yticks, fmty = get_ticks_values((min(acc), max(acc)), 5)
    fig, ax = create_ax_format(
        dim=(3.5, 1.5),
        xlim=(0, len(acc)*dt),
        ylim=(min(acc), max(acc)),
        xticks=xticks,
        yticks=yticks,
        fmtx=fmtx,
        fmty=fmty,
    )

    time_array = [i*dt for i in range(len(acc))]
    ax.plot(time_array, acc, color='blue', linewidth=0.5)
    # Plot P-wave arrival if provided
    if p_arrival is not None:
        ax.axvline(x=p_arrival, color='red', linestyle='--', linewidth=0.5, label='P-wave Arrival')
        # ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acc. (cm/s²)')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dis_freq_matrix(dfm, save_path='img/dis_freq_matrix.png'):
    fig, ax = create_ax_format(
        dim=(4.0, 1.0),
        xlim=(0, dfm.shape[1]),
        ylim=(dfm.shape[0], 0),
        xticks=(200, 50),
        yticks=(100, 20),
        fmtx='%.0f',
        fmty='%.0f',
    )

    im = ax.imshow(dfm, aspect='auto', origin='lower', cmap='seismic')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Freq. Index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction(pred, xy, save_path='img/prediction.png'):
    fig, ax = create_ax_format(
        dim=(4.0, 1.0),
        xlim=(0, pred.shape[1]),
        ylim=(pred.shape[0], 0),
        xticks=(200, 50),
        yticks=(100, 20),
        fmtx='%.0f',
        fmty='%.0f',
    )

    im = ax.imshow(pred, aspect='auto', origin='lower', cmap='Reds')
    # xy is a tuple of (freq_index, time_index)
    ax.scatter(xy[1], xy[0], color='white', marker='s', s=0.1, linewidths=0, label='Predicted Corner')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Freq. Index')
    # plt.legend(loc='upper right')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()