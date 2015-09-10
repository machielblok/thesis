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

def plot_ssro_hist(do_save=False):
    f=h5py.File(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\ssro\145630_AdwinSSRO_SSROCalibration_Pippin_SIL1\analysis.hdf5')
    #print f.attrs
    cpsh_ms0= f['cpsh/ms0'].value
    cpsh_ms1= f['cpsh/ms1'].value
    fig = plt.figure(figsize=(2.5,1.4))
    ax = fig.add_subplot(111)

    F_ms0= f['fidelity/ms0'].value
    F_ms1= f['fidelity/ms1'].value
    F_mean=f['fidelity/mean_fidelity'].value

    RO_idx = np.where(F_mean==max(F_mean))[0][0]
 
    ax.hist(cpsh_ms0, np.arange(max(cpsh_ms0)+2)-0.5, align='mid',
                        normed=True,facecolor='Grey',edgecolor='None',alpha=0.3,linewidth=0) # , stacked=True)
    #ax.hist(cpsh_ms1, np.arange(max(cpsh_ms1)+2)-0.5, align='mid', 
    #                    normed=True,facecolor='Grey',edgecolor='None',alpha=0.3,linewidth=0) # , stacked=True)  

    #ax.hist(cpsh_ms1, np.arange(max(cpsh_ms1)+2)-0.5, align='mid', label='$m_s = \pm 1$\n F=0.99',
    #                    normed=True,facecolor='None',edgecolor='Crimson',alpha=1,linewidth=2) # , stacked=True)    
    #ax.hist(cpsh_ms0, np.arange(max(cpsh_ms0)+2)-0.5, align='mid', label='$m_s = 0$ \n F=0.88' ,
    #                    normed=True,facecolor='None',edgecolor='RoyalBlue',alpha=1,linewidth=2) # , stacked=True)    
    ax.hist(cpsh_ms0, np.arange(max(cpsh_ms0)+2)-0.5, align='mid', histtype='step',label='$m_s = 0$',
                        normed=True,facecolor='None',edgecolor='RoyalBlue',alpha=1,linewidth=1) # , stacked=True)    
    ax.hist(cpsh_ms1, np.arange(max(cpsh_ms1)+2)-0.5, align='mid', histtype='step',label='$m_s = \pm 1$',
                        normed=True,facecolor='None',edgecolor='Crimson',alpha=1,linewidth=1) # , stacked=True)    
   
    print 'F_1=%.2f'%F_ms1[RO_idx][1]
    print 'CPSH_1=%.2f'%mean(cpsh_ms1)
    P_0_ms1=len(np.where(cpsh_ms1==0)[0])/float(len(cpsh_ms1))
    print 'P(0) = %.2f' % P_0_ms1

    print  'F_0=%.2f'%F_ms0[RO_idx][1] 
    print  'CPSH_0= %.2f'%mean(cpsh_ms0)
    P_larger0_ms0=len(np.where(cpsh_ms0>0)[0])/float(len(cpsh_ms0))
    print 'P(>0) = %.2f' % P_larger0_ms0
    ax.set_xlabel('Photon number',fontsize=8)
    ax.set_ylabel('Probability',fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','0.5','1'])
    ax.set_xticks([0,5,10])
    #ax.set_title(self.default_plot_title + title_suffix)
    ax.set_xlim(-0.75, max(cpsh_ms1)+0.5)
    #ax.legend(prop={'size':8})
    f.close()
    if do_save:
        fig.savefig(r'D:\machielblok\Desktop\PhD\Thesis\thesis\text\ch_theory_and_methods\figures\separate_figs\ssro_hist.pdf', bbox_inches='tight',transparent=True)
        
        #print 'no'

def plot_ssro_mean_fid(do_save=False):
    f=h5py.File(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\ssro\145257_AdwinSSRO_SSROCalibration_Pippin_SIL1\analysis.hdf5')
    #print f.attrs
    cpsh_ms0= f['cpsh/ms0'].value
    cpsh_ms1= f['cpsh/ms1'].value
    
    fig = plt.figure(figsize=(2,1.4))
    ax = fig.add_subplot(111)

    fig2 = plt.figure(figsize=(1,0.75))
    ax2 = fig2.add_subplot(111)

    F_ms0= f['fidelity/ms0'].value
    F_ms1= f['fidelity/ms1'].value
    F_mean=f['fidelity/mean_fidelity'].value

    RO_idx = np.where(F_mean==max(F_mean))[0][0]

    ax.plot(F_ms0[:,0],F_mean,'.',mec='Grey',markeredgewidth=1,mfc='None',label='$F_{ro}$') 
    ax.plot(F_ms0[:,0],F_ms0[:,1],'.',mec='RoyalBlue',markeredgewidth=1,mfc='None',label='$m_s = 0$') 
    ax.plot(F_ms1[:,0],F_ms1[:,1],'.',mec='Crimson',markeredgewidth=1,mfc='None',label='$m_s = \pm 1$') 


    ax2.plot(F_ms1[:,0],F_ms1[:,1],'.',mec='Crimson',markeredgewidth=1,mfc='None') 
    ax2.plot(F_ms0[:,0],F_ms0[:,1],'.',mec='RoyalBlue',markeredgewidth=1,mfc='None') 
    ax2.plot(F_ms0[:,0],F_mean,'.',mec='Grey',markeredgewidth=1,mfc='None') 

 

    
    print 'F_1=%.2f'%F_ms1[RO_idx][1]
    print 'CPSH_1=%.2f'%mean(cpsh_ms1)
    P_0_ms1=len(np.where(cpsh_ms1==0)[0])/float(len(cpsh_ms1))
    print 'P(0) = %.2f' % P_0_ms1

    print  'F_0=%.2f'%F_ms0[RO_idx][1] 
    print  'CPSH_0= %.2f'%mean(cpsh_ms0)
    P_larger0_ms0=len(np.where(cpsh_ms0>0)[0])/float(len(cpsh_ms0))
    print 'P(>0) = %.2f' % P_larger0_ms0
    ax.set_ylabel('Fidelity',fontsize=8)
    ax.set_xlabel(r'Readout duration $\mu s$',fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','0.5','1'])
    ax.set_xticks([0,5,10,15])
    ax.set_ylim(0, 1)
    ax.set_xlim(0,15)
    #ax2.set_ylabel('Fidelity',fontsize=12)
    #ax2.set_xlabel(r'Readout duration $\mu s$',fontsize=12)

    ax2.set_yticks([0.95,1])
    ax2.set_xticks([5,10,15])
    ax2.set_yticklabels(['0.95','1'])
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.set_ylim(0.95, 1)
    ax2.set_xlim(4,15)
    #ax.set_title(self.default_plot_title + title_suffix)

    #ax.legend(loc=8,prop={'size':8})
    f.close()
    if do_save:
        fig.savefig(r'D:\machielblok\Desktop\PhD\Thesis\thesis\text\ch_theory_and_methods\figures\separate_figs\ssro_mean_fid.pdf', bbox_inches='tight',transparent=True)
        fig2.savefig(r'D:\machielblok\Desktop\PhD\Thesis\thesis\text\ch_theory_and_methods\figures\separate_figs\ssro_mean_fid_zoom.pdf', bbox_inches='tight',transparent=True)
           
def plot_CR_hist(do_save=False):
    f=h5py.File(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\ssro\145630_AdwinSSRO_SSROCalibration_Pippin_SIL1\145630_AdwinSSRO_SSROCalibration_Pippin_SIL1.hdf5')
    CR=f['SSROCalibration_Pippin_SIL1']['ms0']['CR_after'].value
    
    fig = plt.figure(figsize=(2.5,1.4))
    ax = fig.add_subplot(111)
 
    ax.hist(CR, np.arange(max(CR)+2)-0.5, align='mid',
                        normed=True,facecolor='Grey',edgecolor='None',alpha=0.3,linewidth=0) # , stacked=True)
    ax.hist(CR, np.arange(max(CR)+2)-0.5,align='mid', histtype='step',
                        normed=True,facecolor='None',edgecolor='Grey',alpha=1,linewidth=1)
 
   
    ax.vlines(30,0,0.1,linewidth=1,color='Grey',linestyle='dashed')

    ax.set_xlabel('Photon number',fontsize=8)
    ax.set_ylabel('Probability',fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_yticks([0,0.04,0.08])
    ax.set_yticklabels(['0','0.04','0.08'])
    #ax.set_xticks([0,5,10])

    ax.set_xlim(0, 50)
    ax.set_ylim(0,0.08)
    
    f.close()
    if do_save:
        fig.savefig(r'D:\machielblok\Desktop\PhD\Thesis\thesis\text\ch_theory_and_methods\figures\separate_figs\CR_hist.pdf', bbox_inches='tight',transparent=True)
        

plot_ssro_mean_fid(do_save=False)
#plot_ssro_hist(do_save=True)
#plot_CR_hist(do_save=True)