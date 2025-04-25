#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:16:22 2022

@author: mercier

Script to generate result figures for NeSVoR reconstructions.

To run the script : 
python -m rosi.simulation.scriptSimulData
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from rosi.registration.tools import separate_slices_in_stacks
from rosi.registration.load import convert2Slices
from rosi.simulation.validation import tre_for_each_slices
import nibabel as nib
from rosi.registration.outliers_detection.multi_start import removeBadSlice
from rosi.registration.outliers_detection.feature import update_features
from rosi.registration.outliers_detection.outliers import sliceFeature
from rosi.reconstruction.link_to_reconstruction import computeRegErrorNesVor
from rosi.registration.intersection import compute_cost_matrix

from steven.merging_folders import merging_folders


#function to compute your error
def error_with_nesvor(dir_stacks,image,dir_motion,dir_nomvt):
    #dir_stacks : the path to directory to the original stacks
    #image : the name of the image, e.g petit1 / petit2 ...
    #dir motion : the path to a directory containing each slice in a nifti file, with corrected motion -> use your algorithm to obtained it
    #dir_nomvt : the path to a directory containing each slices in a nifti file, with the motion not corrected -> you can use svort without motion correction to obtained it
    #(use nesvor register with registration at none and save the slices)
     
    axial = dir_stacks + '/LrAxNifti_%s.nii.gz' %(image)
    sagittal = dir_stacks + '/LrSagNifti_%s.nii.gz' %(image)
    coronal = dir_stacks + '/LrCorNifti_%s.nii.gz' %(image)
    set_of_affines = np.array([nib.load(axial).affine,nib.load(coronal).affine,nib.load(sagittal).affine]) #this will give you the original matrix of the volumes
    tax = dir_stacks + '/transfoAx_%s.npy' %(image) #this is the theorical motion
    tsag = dir_stacks + '/transfoSag_%s.npy' %(image)
    tcor = dir_stacks + '/transfoCor_%s.npy' %(image)
    transfo=np.array([tax,tcor,tsag])
    slice_thickness=3 #you can change it to your slice tickness if needed -> 3 is the one I used for the simulation
    res = computeRegErrorNesVor(dir_motion,slice_thickness,dir_nomvt,set_of_affines,transfo)
    return res

#loop on all your data
#set your variable : path to your directory
dir_stacks_var =  '/home/INT/jia.s/Téléchargements/simu' #'/envau/work/meca/users/jia.s/chloe_arrival/simu/' #'../../Simulation/simu/'
# dir_motion_tp = '/home/INT/jia.s/Documents/PhD/v4/simu_debug' #'/mnt/Data/Chloe/Resultats/manuscript/nesvor/tres_petit'
# dir_nomvt_tp = '/home/INT/jia.s/Documents/PhD/v4/simu_debug_nomvt'#'/mnt/Data/Chloe/Resultats/manuscript/svort_nomvt/tres_petit'
dir_motion_tp = '/home/INT/jia.s/Téléchargements/steven/svort'
dir_nomvt_tp = '/home/INT/jia.s/Téléchargements/steven/svort_nomvt'

dir_motion_other = '/home/INT/jia.s/Documents/PhD/v4/simu_siren_old' #'/mnt/Data/Chloe/Resultats/manuscript/nesvor/'
dir_nomvt_other = '/home/INT/jia.s/Documents/PhD/v4/simu_siren_old_nomvt' #'/mnt/Data/Chloe/Resultats/manuscript/svort_nomvt/'

saving_folder = '/home/INT/jia.s/Documents/PhD/v4/simu_siren_old'


input_folders = [
    "scannerRef_LrAxNifti_moyen4",
    "scannerRef_LrCorNifti_moyen4",
    "scannerRef_LrSagNifti_moyen4"
]

# merging_folders(dir_motion_tp, input_folders, "motion_corrected")
# merging_folders(dir_nomvt_tp, input_folders, "motion_corrected")

#movement = [trespetit,Petit,Moyen,Grand]
movment = ['Moyen'] #["tres_petit","Petit","Moyen","Grand"]
suffix_image = ['moyen'] #["trespetit","petit","moyen","grand"]
error_before_correction_nesvor = []
error_after_correction_nesvor = []
i=0
for m in movment :
    set_error_before = []
    set_error_after = []
    index_image = 4 # for index_image in range(1,5) :
#
    stack = m + str(index_image)
    suffix_stack = suffix_image[i] + str(index_image)
    print(index_image)
    dir_stacks = os.path.join(dir_stacks_var, stack)
    
    # if m == "tres_petit" :
    #     dir_motion = dir_motion_tp + str(index_image) #+ stack
    #     dir_nomvt =  dir_nomvt_tp + str(index_image) #+ stack
    # else:
    #     dir_motion = dir_motion_other + stack
    #     dir_nomvt = dir_nomvt_other + stack
    dir_motion = dir_motion_tp # os.path.join(dir_motion_tp, "motion_corrected")
    dir_nomvt = dir_nomvt_tp # os.path.join(dir_nomvt_tp, "motion_corrected") #+ str(index_image) 

    if os.path.exists(dir_motion) and os.path.exists(dir_nomvt):
        error_before = error_with_nesvor(dir_stacks,suffix_stack,dir_nomvt,dir_nomvt)
        error_after = error_with_nesvor(dir_stacks,suffix_stack,dir_motion,dir_nomvt)
        set_error_before.extend(error_before)
        set_error_after.extend(error_after)
    else :
        print('this file does not exists !')
#
    error_before_correction_nesvor.append(set_error_before)
    error_after_correction_nesvor.append(set_error_after)
    i+=1


#display results : 
fig_name = 'results_figure.pdf'

#fig,axs = plt.subplots(1,4, figsize=(40, 40/4)) 
fig = plt.figure()
#axs = fig.add_subplot(111)
color = ['blue','orange']
#couleur=['green','red','blue','yellow']  
# motionRange=['small','medium','large','extra-large']     
motionRange=['medium']     

cm = plt.cm.get_cmap('tab10')
cm_skip = [cm.colors[i] for i in range(0,len(cm.colors))]

# for index in range(1) :     
# for mvt in range(0,4):
mvt = 0
print("motion",mvt)
if mvt==0:
    datalist=np.concatenate(error_after_correction_nesvor[0])
    before=np.concatenate(error_before_correction_nesvor[0])
elif mvt==1:
    datalist=np.concatenate(error_after_correction_nesvor[1])
    before=np.concatenate(error_before_correction_nesvor[1])
elif mvt==2:
    datalist=np.concatenate(error_after_correction_nesvor[2])
    before=np.concatenate(error_before_correction_nesvor[2])
else :
    datalist=np.concatenate(error_after_correction_nesvor[3])
    before=np.concatenate(error_before_correction_nesvor[3]) 

nous_before = np.array(before)
data = np.array(datalist)
print(len(nous_before),len(data))
    
    #if len(data)==len(nous_before):
        
plt.scatter(nous_before[0:min(len(data),len(nous_before))],data[0:min(len(data),len(nous_before))],marker='.',s=170,alpha=0.1,c=cm_skip[mvt])
plt.ylabel('after registration',fontsize=30)
plt.xlabel('before registration',fontsize=30)
#axs[index].set_title(motionRange[mvt],fontsize=15) 
plt.ylim(0,16)
plt.xlim(0,16)

    #for tick in plt.xaxis.get_majorticklabels():  # example for xaxis
    #        tick.set_fontsize(15) 

    #for tick in plt.yaxis.get_majorticklabels():  # example for xaxis
    #        tick.set_fontsize(15) 


plt.tight_layout()
plt.legend(["small","medium","large","extra-large"], fontsize=15, loc ='upper left')
plt.savefig(os.path.join(saving_folder, fig_name))


