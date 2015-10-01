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
fig = plt.figure(figsize=(3,1.5))
ax = fig.add_subplot(1,1,1)
older=None #'20150809_125600'
name_list=['1400nW','500nW','250nW','100nW','90nW','75nW','50nW','40nW','25nW','20nW','15nW','10nW','5nW']
color_list=['Crimson','RoyalBlue','DarkGreen','DarkOrange','k','Grey','red']*5
label_list=name_list#['1400nW','RoyalBlue','DarkGreen','k','DarkOrange','Grey']
plot_measurements=[1,3,7,10]#[11]#[3,4,5,6,8,9,10]
Powers=[]
tau1=[]
tau2=[]
utau1=[]
utau2=[]
fixed=[0]
for j,i in enumerate(plot_measurements):
    f_res=CD.plot_repump_curve(name_list[i],color_list[j],label_list[i],ax,plot_single=False,older_than=older,do_plot=True,fixed=fixed)

for j,i in enumerate([1,2,3,4,5,6,7,8,9,10]):
    Powers.append(int(name_list[i][:-2]))
    print label_list[i]
    f_res=CD.plot_repump_curve(name_list[i],color_list[j],label_list[i],ax,plot_single=False,older_than=older,do_plot=False,fixed=fixed)
    tau1.append(f_res['params_dict']['tau'])
    utau1.append(f_res['error_dict']['tau'])
    #tau2.append(f_res['params_dict']['tau2'])
    #utau2.append(f_res['error_dict']['tau2'])
ax.hlines(1,0,8,linestyles='dotted',color='Grey')
ax.hlines(0,0,8,linestyles='dotted',color='Grey')
#ax.set_yscale('log')
ax.set_ylim([-0.05,1.05])
ax.set_xlim([0,6])
plt.legend(loc=4)
ax.set_ylabel(r'P($m_s =0$)',fontsize=9)
ax.set_xlabel(r'Repump time ($\mu s$)',fontsize=9)
do_save=False
print'test'
if do_save:
    filename=r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\reset.pdf'
    fig.savefig(filename, bbox_inches='tight')


fig2 = plt.figure(figsize=(1.5,1.75))
ax2 = fig2.add_subplot(1,1,1)
ax2.errorbar(Powers,tau1,yerr=utau1,fmt='o',color='k')


guess_a=0.0
guess_b=0.3
guess_n=2.
x = np.array(Powers)
y = np.array(tau1)
uy = np.array(utau1)
a = fit.Parameter(guess_a, 'a')
b = fit.Parameter(guess_b, 'b')
n = fit.Parameter(guess_n, 'n')

p0 = [a,b,n]
fitfunc_str = ''
fitfunc_str = 'a + b/x^2'

def fitfunc(x):
    return a() + b()/(x**n())

fit_result = fit.fit1d(x,y, None, p0=p0, fitfunc=fitfunc, fixed=[],
        do_print=False, ret=True)
print fit_result
x_fit=np.linspace(x[0],x[-1],500)
y_fit=fit_result['fitfunc'](x_fit)
#ax2.plot(x_fit,y_fit,color='Grey')
ax2.set_ylim([0,5])
ax2.set_xlim([0,500])
ax2.set_xticks([0,250,500])
ax2.set_xlabel(r'$Reset \, power \,(nW)$',fontsize=9)
ax2.set_ylabel(r'$\tau_{reset} \, (\mu s)$',fontsize=9)

if do_save:
    filename=r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\reset_fit_params_vs_power.pdf'
    fig2.savefig(filename, bbox_inches='tight')

#plot_ssro_hist(do_save=True)
#plot_CR_hist(do_save=True)