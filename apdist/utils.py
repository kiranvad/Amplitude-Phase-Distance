from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np 

## Visualization utils for warping functions
def peak_plot2(x, y, ax, pk, curve, markers):
    peaks, _ = find_peaks(y,**pk)
    ax.plot(x, y, **curve)
    ax.plot(x[peaks], y[peaks], 'x', **markers)
    for i, p in enumerate(peaks):
        ax.text(x[p], y[p], i+1)
        
    return peaks
        
def plot_warping(x, f1, f2, f2_gamma, gamma):
    mosaic = "ABC;DBC"
    fig = plt.figure(figsize=(4*3*1.6, 4*1))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    axs = fig.subplot_mosaic(mosaic)
    
    pk = {'prominence':0.05, 'width':0.01}
    curve = {'color':'tab:blue', 'label':r'$f_{r}$'}
    peaks = {'color':'k'}
    # a : plot two different profiles
    f1_peaks = peak_plot2(x, f1, axs['A'], pk, curve, peaks)
    axs['A'].legend()
    
    curve = {'color':'tab:orange', 'label':r'$f_{q}$'}
    f2_peaks = peak_plot2(x, f2, axs['D'], pk, curve, peaks)
    axs['D'].legend()
    
    # b : plot warped functions
    axs['B'].plot(x, f1, label=r'$f_{r}$')
    axs['B'].plot(x, f2_gamma, label=r'$f_{q}\circ\gamma^{*}$')
    axs['B'].fill_between(x, f1, f2_gamma, color='grey', alpha=0.5)
    axs['B'].legend()
    
    gamma_x = ((gamma)*(max(x)-min(x)))+min(x)
    axs['C'].set_xlabel(r'$x$')
    axs['C'].set_ylabel(r'$x$')
    axs['C'].plot(x, x, label=r'$\Gamma_{I}$')    
    axs['C'].plot(x, gamma_x, label=r'$\gamma^{*}$')
    for i, p in enumerate(f1_peaks):
        if i==0:
            label = r"peaks $f_{r}$" 
        else:
            label = ''
        axs['C'].axvline(x[p], ls='--', lw=1.0, color='grey', label=label)
    
    for i in ['A', 'B', 'D']:
        axs[i].set_xlabel(r'$x$')
        axs[i].set_ylabel(r'$f(x)$')
        
    for i in ['B', 'C']:    
        axs[i].legend(ncol=3,loc='upper center', bbox_to_anchor=[0.5,1.2])
        
    return 