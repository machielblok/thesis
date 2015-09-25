from analysis.lib.tools import plot
from analysis.lib.fitting import fit, common
reload(common)
from analysis.lib.Qmemory import CarbonDephasing_LT1 as CD
reload(CD)
            ## older than   , prefix, postfix##

d1_XY = ['20150811_142600','','']
dataset_2 = ['20150811_142600','','']



#d1=cd.analyze_and_plot_tomo(older_than='20150811_142600',post_fix_name='')
#d2_xy=cd.analyze_and_plot_tomo(older_than='20150811_133317',post_fix_name='',tomo_list=['X','Y'])
#d2_Z=cd.analyze_and_plot_tomo(older_than='20150811_143941',post_fix_name='',tomo_list=['Z'])


##############################################
## dataset with high resolution, X Y Z tomo 
## untill ~ 800 reps ; only for 100 nW we also have 1000 reps
## 2015 - 08 - 11   overnight measurements

#100nW
older_list_ds1=[['20150810_022116'],['20150810_010819'],['20150809_232519']]
post_fix_names_ds1=['pi_200_100nW','pi_200_500nW','pi_200_10nW']

labels_ds1=['100nW','500nW','10nW']
tomo_list=['X','Y']

##############################################
## dataset with high resolution, X Y Z tomo 
## untill ~ 800 reps ; only for 100 nW we also have 1000 reps
## 2015 - 08 - 11   overnight measurements
#older_list=['20150811_215500','20150811_235451']

post_fix_names_ds2=['pi_200_100nW','pi_200_50nW','pi_200_10nW']
older_list_ds2=['20150811_230126','20150812_012124','20150812_040224','20150812_072855']
labels_ds2=['100nW','50nW','10nW']
tomo_list=['X','Y','Z']

##############################################
## dataset with more reps, X Y  tomo 
## untill  3000 reps
## 2015 - 08 - 12   day measurements
## the 500 nW measurement is from earlier
post_fix_names_ds3=['pi_200_10nW','pi_200_25nW','pi_200_50nW','pi_200_75nW','pi_200_100nW']
older_list_ds3=[['20150811_230126','20150812_012124','20150812_040224','20150812_072855'],
['20150812_171247','20150812_200112'],
['20150812_133641','20150812_142740','20150812_154106'],
['20150812_174859','20150812_204930'],
['20150811_230126','20150812_012124','20150812_040224','20150812_072855']]
labels_ds3=['10nW','25nW','50nW','75nW','100nW']
tomo_list=['X','Y']



##############################################
## dataset with more reps, X Y  tomo 
## untill  3000 reps
## 2015 - 08 - 12   overnight measurements
## with calib of repump time before
post_fix_names_ds4=['pi_200_15nW','pi_200_40nW','pi_200_90']
older_list_ds4=[['20150813_004633','20150813_012557'],
            ['20150813_015744','20150813_024903'],
            ['20150813_032854','20150813_043419']]
labels_ds4=['15nW','40nW','90nW']
tomo_list=['X','Y']


colors=['RoyalBlue','Crimson','DarkGreen','DarkOrange']
color_list=colors
def plot_scaling_curves(post_fix_names,msmnt_list,older_list,labels,ax,fig,tomo_list=['X','Y'],do_plot=True,do_save=True):
    
    fit_xy=False
    fit_qsum=True
    T=[]
    uT=[]
    Powers=[]
    print post_fix_names
    print older_list

    for i,j in enumerate(msmnt_list):
        Powers.append(int(labels[j][:-2]))
        print labels[j][:-2], 'nW'
        

        pf=post_fix_names[j]
        ol=older_list[j]

        ax,fit_res=CD.plot_and_fit_dephasing_curves(pf,ol,tomo_list,ax=ax,label=labels[j],color=color_list[i],fit_xy=fit_xy,fit_qsum=fit_qsum)
        T.append(fit_res['params_dict']['T'])
        uT.append(fit_res['error_dict']['T'])

    #ax.set_ylabel('Sum of positive and negative readout')
    #ax.set_ylim([0,1.05])
    ax.set_xlabel('Repetitions')
    ax.set_xscale('log')
    ax.set_xlim([2e1,4e3])
    ax.set_ylim([0,0.8])
    ax.set_yticks([0,0.2,0.4,0.6,0.8])
    ax.set_ylabel(r'$\sqrt{<X>_C^2+<Y>_C^2}$')
    ax.legend(loc=1)
    if do_save:
            #print 'later'
            fig.savefig(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\Scaling_curves.pdf', bbox_inches='tight')
    return T, uT, Powers
color_list=['Crimson','RoyalBlue','DarkGreen','DarkOrange','k','Grey','red']*5
older_list_scaling_curves=older_list_ds3+[older_list_ds4[0]]
post_fix_names_scaling_curves=post_fix_names_ds3+[post_fix_names_ds4[0]]
labels_scaling_curves = labels_ds3+[labels_ds4[0]]
msmnt_list_scaling_curves=[0,5,1,2]

fig = plt.figure(figsize=(3,2))
ax = fig.add_subplot(1,1,1)
plot_scaling_curves(post_fix_names_scaling_curves,msmnt_list_scaling_curves,older_list_scaling_curves,labels_scaling_curves,ax=ax,fig=fig,do_save=True)


###########
## Ndec vs treset
#############
ol=older_list_ds4+older_list_ds3[0:4]
pf=post_fix_names_ds4+post_fix_names_ds3[0:4]
l = labels_ds4+labels_ds3[0:4]
fig2 = plt.figure(figsize=(3,2))
ax2 = fig2.add_subplot(1,1,1)
pfn=post_fix_names_ds4
ml=[0,1,2,3,4,5,6]
T,uT,P=plot_scaling_curves(pf,ml,ol,l,ax=ax2,fig=fig,do_save=False)
name_list=['1400nW','500nW','250nW','100nW','90nW','75nW','50nW','40nW','25nW','20nW','15nW','10nW','5nW']

label_list=name_list#['1400nW','RoyalBlue','DarkGreen','k','DarkOrange','Grey']
Powers=[]
tau1=[]
tau2=[]
utau1=[]
utau2=[]

mlist=[10,7,4,11,8,6,5]
for j,i in enumerate(mlist):
    Powers.append(int(name_list[i][:-2]))
    print label_list[i]
    f_res=CD.plot_repump_curve(name_list[i],color_list[j],label_list[i],ax,plot_single=False,older_than=None,do_plot=False)
    tau1.append(f_res['params_dict']['tau'])
    utau1.append(f_res['error_dict']['tau'])
    tau2.append(f_res['params_dict']['tau2'])
    utau2.append(f_res['error_dict']['tau2'])
#pfn.append(post_fix_names_ds3[])
print Powers
print P
print 'tau1 = ', tau1
print 'tau2 = ', tau2
fig3 = plt.figure(figsize=(1,1))
ax3 = fig3.add_subplot(1,1,1)
ax3.errorbar(tau1,np.array(T)*1e-3,yerr=np.array(uT)*1e-3,fmt='o',color='Grey')

g_a=.5
g_tau=.2*1e6
g_o=.5
print 'start fit'
decay_constants=np.array(T)*1e-3
fit_result = fit.fit1d(np.array(tau1),decay_constants , common.fit_dephasing_constant_offset,g_a,g_tau,g_o,fixed=[0,1,2],do_print=False, ret=False)
x_fit=np.linspace(tau1[0],tau1[-1],501)
y_fit=fit_result['fitfunc'](x_fit)
ax3.plot(x_fit,y_fit,'-',color='Grey',linewidth=0.75)

ax3.set_xlabel(r'$\tau_{reset} \, (\mu s)$',fontsize=7)
ax3.set_ylabel(r'$N_{decay} 10^3$',fontsize=7)
ax3.tick_params(axis='x', labelsize=7)
ax3.tick_params(axis='y', labelsize=7)
ax3.set_yticks([0,1,2])

do_save=True
if do_save:
    fig3.savefig(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\Ndecay_vs_taureset.pdf', bbox_inches='tight')
'''
###########
## Tomo
#############
fig4 = plt.figure(figsize=(2,1.75))
fig4.clf()

ax4 = fig4.add_subplot(1,1,1)
post_fix_names=['pi_200_100nW']
labels=['100nW']
tomo_list=['X','Y','Z']
color_list=['RoyalBlue','Crimson','Grey']
fit_qsum=False
do_plot=False
fit_xy=True
T=[]
uT=[]
older_list=[['20150811_133502','20150811_144000']]
#fig = plt.figure(figsize=(3,1.5))
msmnt_list=[0]
for j in msmnt_list:
    pf=post_fix_names[j]
    ol=older_list[j]
    comb_dict=CD.stitch_tomo_data([pf]*len(ol),older_list=ol,tomo_list=tomo_list,do_plot_and_save=False)
    
#rep_time=212e-9+400e-9+12e-6
x_time=comb_dict['X']['swp_pts']
#ax1.errorbar(a.sweep_pts*1e3,a.p0[:,0],yerr=a.u_p0[:,0],color=color_list[0],fmt='o',label='Reset electron')
ax4.errorbar(x_time,comb_dict['X']['Signal'],yerr=comb_dict['X']['Signal_u'],color=color_list[1],fmt='o',label=r'$<X>_C$')
ax4.errorbar(x_time,comb_dict['Y']['Signal'],yerr=comb_dict['Y']['Signal_u'],color=color_list[0],fmt='o',label=r'$<Y>_C$')
ax4.errorbar(x_time,comb_dict['Z']['Signal'],yerr=comb_dict['Z']['Signal_u'],color=color_list[2],fmt='.',label=r'$<Z>_C$')
signals=[comb_dict['X']['Signal'],comb_dict['Y']['Signal']]
reps=[comb_dict['X']['swp_pts'],comb_dict['Y']['swp_pts']]
for j,y in enumerate(signals):
    x=reps[j]
    print y
    offset = 0
    A = 0.7
    T = 2000
    n = 2
    f = 1/170.

    fit_result=CD.fit_cos(x,y,a=offset,A=A,T=T,n=n,f=f,fixed=[3],do_print=False)
    #print fit_result
    x_fit=np.linspace(x[0],x[-1],500)
    y_fit=fit_result['fitfunc'](x_fit)
    ax4.plot(x_fit,y_fit,color='Grey')

ax4.set_xlim([0,250])
ax4.set_xticks([0,50,100,150,200,250])
ax4.set_ylim([-0.8,0.8])
ax4.set_yticks([-0.8,0,0.8])
ax4.set_xlabel('Repetitions',fontsize=9)
ax4.set_ylabel(r'Expectation Value',fontsize=9)

#box = ax4.get_position()
#ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax4.legend( loc=2,bbox_to_anchor=(0.5, 1))
#ax4.set_ylim([0.8,1])

do_save=True
if do_save:
    fig4.savefig(r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_carbon_dephasing_purified_figs\tomo.pdf', bbox_inches='tight')
'''