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
from analysis.scripts.QEC import carbon_ramsey_analysis as cr 
reload(cr)
from analysis.lib.Qmemory import CarbonDephasing_LT1 as CD
reload(CD)
def Carbon_Ramsey_mult_msmts(timestamp=None, measurement_name = ['adwindata'], ssro_calib_timestamp =None,
            frequency = 1, 
            offset = 0.5, 
            x0 = 0,  
            amplitude = 0.5,  
            decay_constant = 200, 
            phase =0, 
            exponent = 2, 
            plot_fit = False, do_print = False, fixed = [2], show_guess = True,
            return_phase = False,
            return_freq = False,
            return_amp = False,
            return_results = True,
            close_plot = False,
            partstr = 'part',
            contains=[],
            title = 'Carbon'):
    ''' 
    Function to analyze simple decoupling measurements. Loads the results and fits them to a simple exponential.
    Inputs:
    timestamp: in format yyyymmdd_hhmmss or hhmmss or None.
    measurement_name: list of measurement names
    List of parameters (order important for 'fixed') 
    offset, amplitude, decay_constant,exponent,frequency ,phase 
    '''

    if timestamp != None:
        folder = toolbox.data_from_time(timestamp)
    else:
        folder = toolbox.latest_data(title)
        if partstr in folder:
            numberstart = folder.find(partstr)+len(partstr)
            numberofparts = int(folder[numberstart:len(folder)])
            basis_str = folder[folder.rfind('\\')+7:numberstart]
        else:
            numberofparts = 1

    if ssro_calib_timestamp == None: 
        ssro_calib_folder = toolbox.latest_data('SSRO')

    else:
        ssro_dstmp, ssro_tstmp = toolbox.verify_timestamp(ssro_calib_timestamp)
        ssro_calib_folder = toolbox.datadir + '/'+ssro_dstmp+'/'+ssro_tstmp+'_AdwinSSRO_SSROCalibration_111_1_sil18'
        print ssro_calib_folder

    
    #for kk in range(numberofparts):
    for kk,cnts in enumerate(contains):    
        '''
        if partstr in folder:
            folder = toolbox.latest_data(basis_str+str(kk+1))
        else:
            folder = toolbox.latest_data(basis_str)
        '''
        folder = toolbox.latest_data(cnts)    
        a = mbi.MBIAnalysis(folder)
        a.get_sweep_pts()
        a.get_readout_results(name='adwindata')
        a.get_electron_ROC(ssro_calib_folder)
        #ax = a.plot_results_vs_sweepparam(ret='ax')

        if kk == 0:
            cum_sweep_pts = a.sweep_pts
            cum_p0 = a.p0
            cum_u_p0 = a.u_p0
            cum_pts = a.pts
        else:
            cum_sweep_pts = np.concatenate((cum_sweep_pts, a.sweep_pts))
            cum_p0 = np.concatenate((cum_p0, a.p0))
            cum_u_p0 = np.concatenate((cum_u_p0, a.u_p0))
            cum_pts += a.pts

    a.pts   = cum_pts
    a.sweep_pts = cum_sweep_pts
    a.p0    = cum_p0
    a.u_p0  = cum_u_p0

    sorting_order=a.sweep_pts.argsort()
    a.sweep_pts.sort()
    a.p0=a.p0[sorting_order]
    a.u_p0=a.u_p0[sorting_order]

    #ax=a.plot_results_vs_sweepparam(ret='ax')

    x = a.sweep_pts.reshape(-1)[:]
    y = a.p0.reshape(-1)[:]

    ax.plot(x,y)

    p0, fitfunc, fitfunc_str = common.fit_general_exponential_dec_cos(offset, amplitude, 
            x0, decay_constant,exponent,frequency ,phase )

    #plot the initial guess
    if show_guess:
        ax.plot(np.linspace(x[0],x[-1],201), fitfunc(np.linspace(x[0],x[-1],201)), ':', lw=2)

    fit_result = fit.fit1d(x,y, None, p0=p0, fitfunc=fitfunc, do_print=False, ret=True,fixed=fixed)

    ## plot data and fit as function of total time
    if plot_fit == True:
        plot.plot_fit1d(fit_result, np.linspace(x[0],x[-1],1001), ax=ax, plot_data=False)

    
    return a, fit_result


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


a,fit_result = Carbon_Ramsey_mult_msmts(timestamp=None, 
              offset = 0.5, amplitude = 0.4, x0=0, decay_constant = .3, exponent = 2, 
              frequency = 0.003e3, phase =-90, 
              plot_fit = True, show_guess = False,fixed = [2],            
              return_freq = True,
              return_results = False,
              title = '_msm0_freq_C1',
              contains=['120744_CarbonRamseyInitialised_Gretel_msm0_freq_C1','125055_CarbonRamseyInitialised_Gretel_msm0_freq_C1'])
post_fix_names=['pi_200_10nW']
labels=['10nW']
tomo_list=['X','Y']
color_list=['RoyalBlue','DarkOrange']
fit_qsum=False
do_plot=False
fit_xy=True
T=[]
uT=[]
older_list=[['20150811_230126','20150812_012124','20150812_040224','20150812_072855']]
#fig = plt.figure(figsize=(3,1.5))
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(2.5,1.5))

for j in msmnt_list:
    pf=post_fix_names[j]
    ol=older_list[j]
    comb_dict=CD.stitch_tomo_data([pf]*len(ol),older_list=ol,tomo_list=tomo_list,do_plot_and_save=False)
    
rep_time=212e-9+400e-9+12e-6
x_time=rep_time*comb_dict['X']['swp_pts']
#ax1.errorbar(a.sweep_pts*1e3,a.p0[:,0],yerr=a.u_p0[:,0],color=color_list[0],fmt='o',label='Reset electron')
ax1.errorbar(x_time*1e3,comb_dict['X']['Signal'],yerr=comb_dict['X']['Signal_u'],color=color_list[1],fmt='o',label='Reset')
ax2.errorbar(a.sweep_pts*1e3,2*(np.array(a.p0[:,0]))-1,yerr=a.u_p0[:,0]*2,color=color_list[0],fmt='o',label='No Reset')
x=x_time*1e3
y=comb_dict['X']['Signal']
offset = 0
A = 0.8
T = 7
n = 2
f = 2/7.

fit_result=CD.fit_cos(x,y,a=offset,A=A,T=T,n=n,f=f,fixed=[2])
x_fit=np.linspace(x[0],x[-1]*2,500)
y_fit=fit_result['fitfunc'](x_fit)
ax1.plot(x_fit,y_fit,color='Grey')
print 'Reset: tau_decay = ', fit_result['params_dict']['T'],'+-',fit_result['error_dict']['T']

x=np.array(a.sweep_pts)*1e3
y=2*(np.array(a.p0[:,0]))-1

offset = 00
A = 1
T = 300
n = 2
f = 2/300.

fit_result=CD.fit_cos(x,y,a=offset,A=A,T=T,n=n,f=f,fixed=[2,4])
x_fit=np.linspace(x[0],x[-1]*2,500)
y_fit=fit_result['fitfunc'](x_fit)
print 'No Reset: tau_decay = ', fit_result['params_dict']['T'],'+-',fit_result['error_dict']['T']
ax2.plot(x_fit,y_fit,color='Grey')
fontsize=9
ax1.set_xlim([0,15])
ax1.set_xticks([0,7.5,15])
ax1.set_xticklabels(['0','7.5','15'])
ax2.set_xlim([0,750])
ax2.set_yticks([-1,0,1])
ax2.set_yticklabels([])
ax1.set_yticks([-1,0,1])
ax2.set_xticks([0,375,750])
ax1.set_xlabel('Free evolution time (us)',fontsize=fontsize)
ax1.set_ylabel(r'$<X>_C$',fontsize=fontsize)
ax1.legend()
ax2.legend()


ax1.tick_params(axis='x', labelsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize)
ax2.tick_params(axis='x', labelsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize)
do_save=True
if do_save:
    filename=r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\Carbon_ramseys.pdf'
    fig.savefig(filename, bbox_inches='tight')
#plot_ssro_hist(do_save=True)
#plot_CR_hist(do_save=True)