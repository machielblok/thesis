###########################
##### Plot Fig 1 SSRO   ###
###########################
import numpy as np
from analysis.lib.m2.ssro import sequence
from analysis.lib import fitting
from analysis.lib.m2.ssro import ssro, mbi
from analysis.lib.tools import toolbox
from analysis.lib.fitting import fit, ramsey
from analysis.lib.tools import plot
from analysis.lib.math import error
from matplotlib import rc, cm
import h5py
from matplotlib import rc
reload(ramsey)
def plot_single_desr(folder,ax,color,label):

    a=np.load(folder)
    
    sweep_pts=a['xdesr']
    p0=a['datadesr']
    u_p0=a['udatadesr']
    
    ax.errorbar(sweep_pts,p0,yerr=u_p0,mec=color,fmt='o',label=label)
    x_fit=a['xfit']
    y_fit=a['desrfit']
    ax.plot(x_fit,y_fit,'-',color='Grey',linewidth=1)


def plot_desr(do_save=False):


    folders=[folder_uninitialized]
    colors=['RoyalBlue','Crimson']
    labels=['NV 1','NV 2']
    fig = plt.figure(figsize=(2,1.5))
    ax = fig.add_subplot(111)
    
    
    for i in np.arange(len(folders)):
        plot_single_desr(folders[i],ax,colors[i],labels[i])
    ax.set_xlabel('MW frequency (GHz)')#,fontsize=24)
    ax.set_ylabel('P($m_s =0$)')#,fontsize=24)
    ax.tick_params(axis='x')#, labelsize=18)
    ax.tick_params(axis='y')#, labelsize=18)
    ax.set_yticks([0.8,0.9,1])
    ax.set_xticks([2.827,2.829,2.831])
    ax.set_xticklabels(['2.827','2.829','2.831'])
    ax.set_xlim([2.8256,2.832])
    plt.show()
    if do_save:
        #print 'later'
        fig.savefig(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_figs\desr.pdf', bbox_inches='tight')

folder_uninitialized = r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\desr\desr.npz'
plot_desr(do_save=True)
#plot_ssro_hist(do_save=True)
#plot_CR_hist(do_save=True)