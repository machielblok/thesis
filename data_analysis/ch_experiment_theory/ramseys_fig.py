###########################
##### Plot Fig 1 SSRO   ###
###########################
import numpy as np
from analysis.lib.m2.ssro import sequence
from analysis.lib import fitting
from analysis.lib.m2.ssro import ssro, mbi
from analysis.lib.tools import toolbox
from analysis.lib.fitting import fit, ramsey,common
from analysis.lib.tools import plot
from analysis.lib.math import error
from matplotlib import rc, cm
import h5py
from matplotlib import rc
reload(ramsey)
def plot_single_ramsey_curve(folder,ax,color,label,fit_carbon):

    a=np.loadtxt(folder)
    
    guess_A = 0.5
    guess_phi1 = 0*np.pi/2.
    guess_phi2 = 0*np.pi/2.
    guess_phi3 = 0*np.pi/2.
    guess_hf=2.18
    guess_tau = 3
    guess_a = 0.5
    guess_det = 5
    guess_C1 = .2
    sweep_pts=a[:,0]*1e-3
    p0=a[:,2]
    u_p0=a[:,3]
    #ax.set_ylim([0.0,1.05])
    if fit_carbon ==False:
        fit_result = fit.fit1d(sweep_pts, p0, ramsey.fit_ramsey_hyperfinelines_fixed,
            guess_tau, guess_A,guess_a, guess_det,guess_hf,guess_phi1,guess_phi2,guess_phi3,
            #(guess_f3, guess_A3, guess_phi3),
            fixed=[],
            do_print=True, ret=True)
    else:
        guess_A=0.5/6.
        p=np.pi
        guess_phi3=p
        guess_phi2=p
        guess_phi1=p
        fit_result = fit.fit1d(sweep_pts, p0, ramsey.fit_ramsey_14N_fixed_13C_opt,
                guess_tau, guess_A,guess_a, guess_det,guess_hf,guess_phi1,guess_phi2,guess_phi3,guess_C1,
                #(guess_f3, guess_A3, guess_phi3),
                fixed=[],
                do_print=True, ret=True)

    #ax.errorbar(sweep_pts,p0,yerr=u_p0,mec=color,fmt='o',label=label,markeredgewidth=1,mfc='None')
    ax.plot(sweep_pts,p0,'.',mec=color,label=label,markeredgewidth=1,mfc='None')
    x_fit=np.linspace(sweep_pts[0],sweep_pts[-1],501)
    y_fit=fit_result['fitfunc'](x_fit)
    ax.plot(x_fit,y_fit,'-',color='Grey',linewidth=0.75)
    ax.set_xlim([0,4])
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.5,1])
    ax.legend()
def plot_ramsey(do_save=False):


    folders=[folder_nv2,folder_nv1]
    colors=['RoyalBlue','Crimson']
    labels=['NV 1','NV 2']
    fit_carbon=[False,True]
    fig2,(ax1,ax2) = plt.subplots(2,1,figsize=(2.5,2))
    #fig2.clf()
    
    axs=[ax1,ax2]
    for i in np.arange(len(folders)):
        plot_single_ramsey_curve(folders[i],axs[i],colors[i],labels[i],fit_carbon[i])
    ax2.set_xlabel('Free evolution time (us)')#,fontsize=24)
    ax2.set_ylabel('P($m_s =0$)')#,fontsize=24)
    ax2.tick_params(axis='x')#, labelsize=18)
    ax2.tick_params(axis='y')#, labelsize=18)
    ax1.set_xticklabels([])
    ax2.set_xticks([0,1,2,3,4])
    ax1.set_xticks([0,1,2,3,4])
    plt.show()
    if do_save:
        #print 'later'
        fig2.savefig(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_figs\Ramseys.pdf', bbox_inches='tight')


folder_nv1=r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\ramseys\lt1_ramsey.txt'
folder_nv2=r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\ramseys\lt2_ramsey.txt'
plot_ramsey(do_save=False)
#plot_ssro_hist(do_save=True)
#plot_CR_hist(do_save=True)