
from numpy import *
import pylab as plt
import numpy as np
from analysis.lib.fitting import fit, common
from analysis.lib.tools import plot
from analysis.lib.nv import nvlevels
reload(nvlevels)
import numpy
from mpl_toolkits.mplot3d import proj3d
 
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return numpy.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,-0.0001,zback]])
 


colors=['Crimson','RoyalBlue','DarkGreen','Crimson','RoyalBlue','DarkGreen']

#datafolders=['1154','1252','1258','1303','1310','1347','1351','1316','1326', '1453']
#RO_time=[0,1,2,3,4,5,6,7,9, 11]
SIL1=[39.4,49.3,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\132500_LaserFrequencyScan_red_scan_coarse_Hans_SIL1_1nW_mw_gv_0.0\132500_LaserFrequencyScan_red_scan_coarse_Hans_SIL1_1nW_mw_gv_0.0.npz',1]
SIL4=[61,63.17,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\142112_LaserFrequencyScan_red_scan_coarse_Hans_SIL4_1nW_mw_gv_0.0\142112_LaserFrequencyScan_red_scan_coarse_Hans_SIL4_1nW_mw_gv_0.0.npz',4]

SIL1111=[74,78.1,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\155115_LaserFrequencyScan_The111no1_enlarged_Sil1_gate22_0V\155115_LaserFrequencyScan_The111no1_enlarged_Sil1_gate22_0V.txt',1111]

SIL5=[74.19,75.9,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\114944_LaserFrequencyScan_Sam_SIL5_MW_0V\114944_LaserFrequencyScan_Sam_SIL5_MW_0V.txt',5]

SIL18=[77.76,82.28,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\135044_LaserFrequencyScan_The111no1_enlarged_Sil18\135044_LaserFrequencyScan_The111no1_enlarged_Sil18.txt',18]

SILHans5=[35.2,50.5,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\154026_LaserFrequencyScan_red_scan_coarse_Hans_SIL5_1nW_mw_gv_0.0\154026_LaserFrequencyScan_red_scan_coarse_Hans_SIL5_1nW_mw_gv_0.0.txt',5]

SIL9=[65.26,77.6,r'M:\tnw\ns\qt\Diamond\Reports and Theses\PhD\Machiel Blok\ch_Experiment_Theory_data\laser_scans\flow_cryo_2012_SIL9.txt',8]
'''
SIL7=[52.7,59.5,r'D:\measuring\data\20130702\122103_LaserFrequencyScan_SIL7_Green_gv_0.0\122103_LaserFrequencyScan_SIL7_Green_gv_0.0.npz',7]
SIL5=[40.8,50.3,r'D:\measuring\data\20130702\173219_LaserFrequencyScan_SIL5_green_gv_0.0\173219_LaserFrequencyScan_SIL5_green_gv_0.0.npz',5]
SIL1 = [71,78.4,r'D:\measuring\data\20130702\171520_LaserFrequencyScan_SIL1_red_gv_0.0\171520_LaserFrequencyScan_SIL1_red_gv_0.0.npz',1]
'''
datafolders=[SIL9,SIL1,SIL18]
fig = plt.figure(figsize=(5,2))
ax = fig.gca(projection='3d')
j=0
n=38.5
xmin=-10
xmax=10
ax.set_xlim([xmin,xmax])
ax.set_ylim([0,8])
ax.set_zlim([1e-3,0.5e1])
lvls=nvlevels.get_levels()

for i in np.arange(6):
    if (i==2)or(i==3):
        trans=lvls[1][:,i]-lvls[1][0,2]
    else:
        trans=lvls[1][:,i]
    ax.plot(trans,lvls[0],np.zeros(len(trans)),color='LightGrey',linewidth=1)
for k,i in enumerate(datafolders):
    if 'txt' in i[2]:
        r=np.loadtxt(i[2])
        print r[1:4,1]
    else:
        result = np.load(i[2])
        r=result['data']
    Z_strain=(i[1]-i[0])/2.
    print 'Z strain', Z_strain
    frq=r[:,1]-Z_strain-i[0]
    
    cnts=r[:,2]
    max_cnts=float(max(cnts))
    for j in np.arange(len(cnts)):
        if cnts[j]>max_cnts*0.3:
            cnts[j]=max_cnts*0.3
    
    # Filter everything outside x-limits
    min_array=np.where(frq<xmin)
    max_array=np.where(frq>xmax)
    if len(min_array[0])==0:
        f_min_idx=0
    else: f_min_idx=max(min_array[0])
    if len(max_array[0])==0:
        f_max_idx=len(frq)-1
    else: f_max_idx=min(max_array[0])
    #print 'min frq', frq[f_min_idx],'at index ', f_min_idx
    #print 'max frq', frq[f_max_idx],'at index ', f_max_idx

    ax.plot(frq[f_min_idx:f_max_idx],Z_strain*np.ones(len(frq[f_min_idx:f_max_idx])),cnts[f_min_idx:f_max_idx]/float(max(cnts)),linewidth=1.5,color=colors[k])
    #ax.plot(frq,Z_strain*np.ones(len(frq)),r[:,2],linewidth=1)
    ax.set_zscale('log')
    
    #if (i[0]!=0):
        #plt.text(i[0]-1,j+0.5,'Ey',fontsize=8)
    #if (i[1]!=0):
        #plt.text(i[1]+0.3,j+0.5,'Ex',fontsize=8)
    name='SIL'+str(i[3])
    #plt.text(n,j+0.1,name,fontsize=10)
    j=j+1
proj3d.persp_transformation = orthogonal_proj
print type(r[:,2])
#lvls[0]    
ax.set_xlim([xmin,xmax])
ax.set_ylim([0,8])
ax.set_yticks([0,3,7])
ax.set_zlim([1e-3,0.075e1])
ax.set_zticks([])
#ax.view_init(elev=70., azim=-90)
ax.view_init(elev=60., azim=-75)
#ax.set_zlim([0,5000])
#plt.ylim([-0.1,j])
plt.xlabel ('Relative Frequency [GHz]')   
plt.ylabel ('Lateral Strain [GHz]')   

ax.grid(False)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

do_save=True
if do_save:
        fig.savefig(r'D:\machielblok\Desktop\PhD\Thesis\thesis\text\ch_theory_and_methods\figures\separate_figs\laserscans.pdf', bbox_inches='tight',transparent=True)
      