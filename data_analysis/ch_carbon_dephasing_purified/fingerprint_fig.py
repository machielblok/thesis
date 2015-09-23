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
import analysis.lib.QEC.nuclear_spin_characterisation as SC #used for simulating FP response of spins
import analysis.lib.QEC.hyperfine_params as hyperfine_params; reload(hyperfine_params) 
from matplotlib import rc, cm
import h5py
reload(ramsey)
from analysis.scripts.Quantum_Memory import Simple_Decoupling_Analysis as sda
reload(sda)





def plot_fingerprint(name_contains,nr_ids,xlim,N,figsize,file_name_add='',do_save=False,add_sim=True):
    
    ssro_calib_folder = toolbox.latest_data(contains='AdwinSSRO_SSROCalibration')
    timestamp = None#'20150605163851'

    if nr_ids > 1:
        x, y, yerr, folder, timestamp, all_stitching_points = sda.get_data_multiple_msmts(name_contains, nr_ids, input_timestamp = timestamp, ssro_calib_folder = ssro_calib_folder)
    else:
        x, y, yerr, folder, timestamp = sda.get_data(name_contains, ssro_calib_folder = ssro_calib_folder)
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)

    plt.errorbar(x,y,yerr = yerr, fmt = '.',color='RoyalBlue')
    plt.ylabel('P($m_s =0$)',fontsize=9)
    plt.xlabel(r'$\tau$ ($\mu$s)',fontsize=9)
    # plt.xlabel('Free evolution time (us)',fontsize=25)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    
    Bz=22.75
    Bx=1
    if add_sim:
        y_sim,x_sim=plot_sim_vs_Bx(spin_list=['C1'],Bx_list = [Bx],B_Field = Bz, N =N,xlim=xlim)
        plt.plot(x_sim,y_sim,color='Crimson',linewidth=1.5)
        y_sim,x_sim=plot_sim_vs_Bx(spin_list=['C2'],Bx_list = [Bx],B_Field = Bz, N =N,xlim=xlim)
        plt.plot(x_sim,y_sim,color='DarkGreen',linewidth=1.5)
    
    wl=1e3/(sqrt(Bz**2+Bx**2)*1.0705)
    
    collapses=[wl*1/4.,wl*3/4.,wl*5/4.]
    for i in collapses:
        plt.axvline(i, color='Grey', linestyle='solid')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([0.25,1.05])
    ax.set_yticks([0.25,0.5,0.75,1])
    ax.set_xlim(xlim)
    if do_save:
        filename=r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\fingerpint'+str(file_name_add)+'.pdf'
        fig.savefig(filename, bbox_inches='tight')






def plot_sim_vs_Bx(spin_list=['C1'],Bx_list = [0],B_Field = 12, N =2,xlim=[0,100]):

    for ii in range(len(spin_list)):


        for b in range(len(Bx_list)):
            Bx = Bx_list[b]
            #print spin_list[ii]

            HF_par=[hf[spin_list[ii]]['par']  - hf[spin_list[ii]]['perp']*Bx/B_Field]
            HF_perp=[hf[spin_list[ii]]['perp'] + hf[spin_list[ii]]['par']*Bx/B_Field]

            
            tau_lst = np.linspace(xlim[0]*1e-6,xlim[1]*1e-6,960)
            tau_lst_us = np.linspace(xlim[0],xlim[1],960)
            if ii == 0:
                Mt = SC.dyn_dec_signal(HF_par,HF_perp,B_Field,N,tau_lst)
            else:
                Mt=Mt*SC.dyn_dec_signal(HF_par,HF_perp,B_Field,N,tau_lst)
            #if ii == 0:
            FP_signal = ((Mt+1)/2)
            #else:
            #    FP_signal = FP_signal*((Mt+1)/2)
            #ax.plot(tau_lst*1e6, FP_signal[0,:],'.-',lw=.8,label = 'spin_'+spin_list[ii]+'_Bx_'+str(Bx))
    return FP_signal[0,:],tau_lst_us

# plot_sim_vs_Bx(spin_list=['C1'],Bx_list = [8.5],B_Field = 22, N =8)
# plot_sim_vs_Bx(spin_list=['C1'],Bx_list = [8],B_Field = 22, N =16)

hyperfine_params = {}
hyperfine_params['C1']  = {'par' : 0.227e3, 'perp':.2e3}
hyperfine_params['C2']  = {'par' : -1.02e3, 'perp':0.1925e3}
hf = hyperfine_params

ZFS                 = 2.87180e9
g_factor            = 2.8025e6
f_names=['128pulses_9','200pulses_226']
file_names=['','_zoom']
fig_sizes=[(6,1.5),(3,1.5)]
xlims=[[9,52.55],[30,32]]
add_sim=[False,True]

Ns=[128,200]
savez=[True,True]
ids=[60,20]

for i in np.arange(len(f_names)):
    plot_fingerprint(f_names[i],ids[i],xlim=xlims[i],N =Ns[i],figsize=fig_sizes[i],file_name_add=file_names[i],do_save=savez[i],add_sim=add_sim[i])
    



#folder_uninitialized = r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\desr\desr.npz'
#plot_desr(do_save=True)
#plot_ssro_hist(do_save=True)
#plot_CR_hist(do_save=True)