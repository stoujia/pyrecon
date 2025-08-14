#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:16:22 2022

@author: mercier
"""
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from rosi.registration.tools import separate_slices_in_stacks
from rosi.registration.load import convert2Slices
from rosi.simulation.validation import tre_for_each_slices, cumulative_tre
import nibabel as nib
from psnrandssim import PSNR,SSIM
from rosi.registration.outliers_detection.outliers import sliceFeature
import ants

#read data :
error_before_correction_reallysmall=[]  
error_after_correction_reallysmall=[]
error_before_correction_small=[]
error_after_correction_small=[]
error_before_correction_medium=[]
error_after_correction_medium=[]
error_before_correction_grand=[]
error_after_correction_grand=[]

grand_psnr = []
grand_ssim = []
moyen_psnr = []
moyen_ssim = []
petit_psnr = []
petit_ssim = []
tres_petit_psnr = []
tres_petit_ssim = []




optimisation=["LM","BFGS","Powell","TNC","cg","Nelder-Mead"]
#optimisation=["CG"]

for value_optimisation in optimisation:
    errorlist=[]
    psnrlist=[]
    ssimlist=[] 
    for index_image in range(1,6):
        #error_before=[]
        file = '/mnt/Data/Chloe/res/%s/value/simul_data/tres_petit%d/%s/%s' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        petit = '/mnt/Data/Chloe/res/%s/value/simul_data/tres_petit%d/%s/res_test_%s.joblib.gz' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        directory = '../../simu/tres_petit%d/' %(index_image)
        print(petit)
        print(os.path.exists(petit))
        if os.path.exists(petit) : 
            print(petit)
            res = joblib.load(petit)
            #print(res)
            key=[p[0] for p in res]
            element=[p[1] for p in res]
            listSlice=element[key.index('listSlice')]
            parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
            rejected_slices = element[key.index('RejectedSlices')]
            nb_rejected_slices = np.shape(rejected_slices)[0]
            nb_slices = len(listSlice)
            #listFeature=element[key.index('ListError')]
            #proportion_rejected_slices_reallysmall[i-1] = nb_rejected_slices/nb_slices
            transfo_axial =  directory + 'transfoAx_trespetit%d.npy' %(index_image)
            transfo_sagittal  = directory + 'transfoSag_trespetit%d.npy' %(index_image)
            transfo_coronal = directory + 'transfoCor_trespetit%d.npy' %(index_image)
            transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal]) 
            axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
            img_axial_without_movment = nib.load(axial_without_movment)
            axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_trespetit%d.nii.gz' %(index_image)
            img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
            coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
            img_coronal_without_movment = nib.load(coronal_without_movment)
            coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_trespetit%d.nii.gz' %(index_image)
            img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
            sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
            img_sagittal_without_movment = nib.load(sagittal_without_movment)
            sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_trespetit%d.nii.gz' %(index_image)
            img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
            listnomvt = []
            output=convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
            listnomvt.extend(output)
            output=convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
            listnomvt.extend(output)
            output=convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
            listnomvt.extend(output)
            #print(len(listSlice),len(listnomvt))
            img,msk=separate_slices_in_stacks(listSlice)
            img2,msk2=separate_slices_in_stacks(listnomvt)
            print(len(img[0]),len(img2[0]),len(img[1]),len(img2[1]),len(img[2]),len(img2[2]))
            listFeature = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
            tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
            error_after=[feature.get_error() for feature in listFeature]     
            errorlist.extend(error_after) 
            if optimisation=="Nelder-Mead" : 
                for i_slice in listSlice:
                        i_slice.set_parameters([0,0,0,0,0,0])
                tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
                error_before=[feature.get_error() for feature in listFeature]
                error_before_correction_reallysmall.append(error_before)
    error_after_correction_reallysmall.append(errorlist)

for value_optimisation in optimisation:     
    errorlist=[]
    psnrlist=[]
    ssimlist=[] 
    error_before=[]
    for index_image in range(1,6):  
        file='/mnt/Data/Chloe/res/%s/value/simul_data/Petit%d/%s/%s' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        petit = '/mnt/Data/Chloe/res/%s/value/simul_data/Petit%d/%s/res_test_%s.joblib.gz' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
        directory = '../../simu/Petit%d/' %(index_image)
        #print(petit) 
        if os.path.exists(petit) and index_image != 2 and index_image != 3  :    
            print(petit)
            res = joblib.load(petit)
            key=[p[0] for p in res]
            element=[p[1] for p in res]
            listSlice=element[key.index('listSlice')]
            parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
            listFeature=element[key.index('ListError')]
            rejected_slices = element[key.index('RejectedSlices')]
            nb_rejected_slices = np.shape(rejected_slices)[0]
            nb_slices = len(listSlice)
            #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices
            transfo_axial =  directory + 'transfoAx_petit%d.npy' %(index_image)
            transfo_sagittal  = directory + 'transfoSag_petit%d.npy' %(index_image)
            transfo_coronal = directory + 'transfoCor_petit%d.npy' %(index_image)
            transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
            axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
            img_axial_without_movment = nib.load(axial_without_movment)
            axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_petit%d.nii.gz' %(index_image)
            img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
            coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
            img_coronal_without_movment = nib.load(coronal_without_movment)
            coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_petit%d.nii.gz' %(index_image)
            img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
            sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
            img_sagittal_without_movment = nib.load(sagittal_without_movment)
            sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_petit%d.nii.gz' %(index_image)
            img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
            listnomvt = []
            output=convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
            listnomvt.extend(output)
            output=convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
            listnomvt.extend(output)
            output=convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
            listnomvt.extend(output)
            listFeature = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
            tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
            error_after=[feature.get_error() for feature in listFeature]     
            errorlist.extend(error_after) 
            if value_optimisation=="Nelder-Mead" :
                for i_slice in listSlice:
                        i_slice.set_parameters([0,0,0,0,0,0])
                tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
                error_before=[feature.get_error() for feature in listFeature]
                error_before_correction_small.append(error_before)   
    error_after_correction_small.append(errorlist)   
        
for value_optimisation in optimisation:
    errorlist=[]
    psnrlist=[]
    ssimlist=[] 
    for index_image in range(1,6):    
        file = '/mnt/Data/Chloe/res/%s/value/simul_data/Moyen%d/%s/%s' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        petit = '/mnt/Data/Chloe/res/%s/value/simul_data/Moyen%d/%s/res_test_%s.joblib.gz' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
        directory = '../../simu/Moyen%d/' %(index_image)
        print(os.path.exists(petit)) 
        if os.path.exists(petit) :   
            print(petit)
            res = joblib.load(open(petit,'rb'))
            key=[p[0] for p in res]
            element=[p[1] for p in res]
            listSlice=element[key.index('listSlice')]
            listFeature=element[key.index('ListError')]
            parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
            rejected_slices = element[key.index('RejectedSlices')]
            nb_rejected_slices = np.shape(rejected_slices)[0]
            nb_slices = len(listSlice)
            #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices
            transfo_axial =  directory + 'transfoAx_moyen%d.npy' %(index_image)
            transfo_sagittal  = directory + 'transfoSag_moyen%d.npy' %(index_image)
            transfo_coronal = directory + 'transfoCor_moyen%d.npy' %(index_image)
            transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
            axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
            img_axial_without_movment = nib.load(axial_without_movment)
            axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_moyen%d.nii.gz' %(index_image)
            img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
            coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
            img_coronal_without_movment = nib.load(coronal_without_movment)
            coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_moyen%d.nii.gz' %(index_image)
            img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
            sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
            img_sagittal_without_movment = nib.load(sagittal_without_movment)
            sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_moyen%d.nii.gz' %(index_image)
            img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
            listnomvt = []
            output=convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
            listnomvt.extend(output)
            output=convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
            listnomvt.extend(output)
            output=convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
            listnomvt.extend(output)
            listFeature = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
            tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
            error_after=[feature.get_error() for feature in listFeature] 
            errorlist.extend(error_after) 
            if value_optimisation=="Nelder-Mead" :
                for i_slice in listSlice:
                        i_slice.set_parameters([0,0,0,0,0,0])
                tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
                error_before=[feature.get_error() for feature in listFeature]
                error_before_correction_medium.append(error_before)
    error_after_correction_medium.append(errorlist)   

for value_optimisation in optimisation:
    errorlist=[]
    psnrlist=[]
    ssimlist=[] 
    for index_image in range(1,6):
        file = '/mnt/Data/Chloe/res/%s/value/simul_data/Grand%d/%s/%s' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        petit = '/mnt/Data/Chloe/res/%s/value/simul_data/Grand%d/%s/res_test_%s.joblib.gz' %(value_optimisation,index_image,value_optimisation,value_optimisation)
        #directory1 = '/home/mercier/Documents/donnee/test/Grand%d/' %(i)
        directory = '../../simu/Grand%d/' %(index_image)
        print(petit,os.path.exists(petit))
        listerrorimg1img2_after=[]
        if os.path.exists(petit) : 
            res = joblib.load(open(petit,'rb'))
            key=[p[0] for p in res]
            element=[p[1] for p in res]
            listSlice=element[key.index('listSlice')]
            listFeature=element[key.index('ListError')]
            parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
            rejected_slices = element[key.index('RejectedSlices')]
            nb_rejected_slices = np.shape(rejected_slices)[0]
            nb_slices = len(listSlice)
            #proportion_rejected_slices_small[i-1] = nb_rejected_slices/nb_slices
            transfo_axial =  directory + 'transfoAx_grand%d.npy' %(index_image)
            transfo_sagittal  = directory + 'transfoSag_grand%d.npy' %(index_image)
            transfo_coronal = directory + 'transfoCor_grand%d.npy' %(index_image)
            transfo = np.array([transfo_axial,transfo_coronal,transfo_sagittal])
            axial_without_movment = directory + 'LrAxNifti_nomvt.nii.gz'
            img_axial_without_movment = nib.load(axial_without_movment)
            axial_mask_without_movment = directory + 'brain_mask/LrAxNifti_grand%d.nii.gz' %(index_image)
            img_axial_mask_without_movment = nib.load(axial_mask_without_movment)
            coronal_without_movment = directory + 'LrCorNifti_nomvt.nii.gz'
            img_coronal_without_movment = nib.load(coronal_without_movment)
            coronal_mask_without_movment = directory + 'brain_mask/LrCorNifti_grand%d.nii.gz' %(index_image)
            img_coronal_mask_without_movment = nib.load(coronal_mask_without_movment)
            sagittal_without_movment = directory + 'LrSagNifti_nomvt.nii.gz'
            img_sagittal_without_movment = nib.load(sagittal_without_movment)
            sagittal_mask_without_movment = directory + 'brain_mask/LrSagNifti_grand%d.nii.gz' %(index_image)
            img_sagittal_mask_without_movment = nib.load(sagittal_mask_without_movment)
            listnomvt = []
            output = convert2Slices(img_axial_without_movment,img_axial_mask_without_movment,[],0,0)
            listnomvt.extend(output)
            output =convert2Slices(img_coronal_without_movment,img_coronal_mask_without_movment,[],1,1)
            listnomvt.extend(output)
            output = convert2Slices(img_sagittal_without_movment,img_sagittal_mask_without_movment,[],2,2)
            listnomvt.extend(output)
            listFeature = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listSlice]
            tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
            error_after=[feature.get_error() for feature in listFeature] 
            errorlist.extend(error_after) 
            if value_optimisation=="Nelder-Mead":
                for i_slice in listSlice:
                        i_slice.set_parameters([0,0,0,0,0,0])
                tre_for_each_slices(listnomvt,listSlice,listFeature,transfo,[])
                error_before=[feature.get_error() for feature in listFeature]
                error_before_correction_grand.append(error_before)
    error_after_correction_grand.append(errorlist)
        

#fig,axs = plt.subplots(1,1, figsize=(40, 15)) 

fig,axs = plt.subplots(1,len(optimisation),figsize=(10*len(optimisation),15))
#axs = fig.subplots(1,5)
color = ['blue','orange','green','red']
#couleur=['green','red','blue','yellow']  
motionRange=['small','medium','large','extra-large']     


print('test :',value_optimisation)

for index_optimisation in range(0,len(optimisation)) :   
    for mvt in range(0,len(motionRange)):
        print("motion",mvt)
        if mvt==0:
            datalist=error_after_correction_reallysmall[index_optimisation]
            before=np.concatenate(error_before_correction_reallysmall)
        elif mvt==1:
            datalist=error_after_correction_small[index_optimisation]
            before=np.concatenate(error_before_correction_small)
        elif mvt==2:
            datalist=error_after_correction_medium[index_optimisation]
            before=np.concatenate(error_before_correction_medium)
        else :
            datalist=error_after_correction_grand[index_optimisation]
            before=np.concatenate(error_before_correction_grand) 
                
                #data=datalist[value_optimisation]
            
                    #print(len(before))
                    #print(len(data))
                    
                    #add error from each cg test
        nous_before = np.array(before)
        data = np.array(datalist)
        print(len(nous_before),len(data))
        if len(data)==len(nous_before):
                        
            #axs[value_optimisation].
            axs[index_optimisation].scatter(data,nous_before,marker='.',s=170,alpha=0.1,c=color[mvt])
                    #axs[value_optimisation].scatter(ebner_petit, ebner_before, c='blue', marker='.', s=170)
                    #axs[mvt].set_yticks(np.arange(min, max),fontsize=100) 
                    #axs[value_optimisation].
            axs[index_optimisation].set_ylabel('avant recalage',fontsize=30)
                    #axs[value_optimisation].
            axs[index_optimisation].set_xlabel('après recalage',fontsize=30)
                    #axs[value_optimisation].
            axs[index_optimisation].set_title('%s' %(optimisation[index_optimisation]),fontsize=30) 
                    #axs[value_optimisation]
            axs[index_optimisation].set_ylim(0,16)
                    #axs[value_optimisation]
            axs[index_optimisation].set_xlim(0,5)

            #for tick in axs[value_optimisation].xaxis.get_majorticklabels():  # example for xaxis
                    #plt.set_fontsize(15) 

            #for tick in axs[value_optimisation].yaxis.get_majorticklabels():  # example for xaxis
                    #plt.set_fontsize(15) 

    plt.tight_layout()

    #motion = ["small","medium","large"]
    plt.legend(motionRange, fontsize=30, loc ="lower right")
    #["small","medium","large","extra-large"]

    plt.savefig('optimisation_multistart_v3.png')


fig,axs = plt.subplots(len(optimisation),figsize=(40,40))
for index_optimisation in range(0,len(optimisation)): 
     for mvt in range(0,len(motionRange)):
        if mvt==0:
            datalist=error_after_correction_reallysmall[index_optimisation]   
        elif mvt==1:
            datalist=error_after_correction_small[index_optimisation]   
        elif mvt==2:
            datalist=error_after_correction_medium[index_optimisation]    
        else :
            datalist=error_after_correction_grand[index_optimisation]  
        X,Y = cumulative_tre(datalist)
        X = np.array(X)
        Y = np.array(Y)/np.max(Y)
        if index_optimisation==0:
            axs[index_optimisation].plot(X,Y,label=motionRange[mvt])
        else:
            axs[index_optimisation].plot(X,Y)
        axs[index_optimisation].set_xlabel('cumulative tre (in mm)',fontsize=30)
        axs[index_optimisation].set_ylabel('proportion of slices',fontsize=30)
        axs[index_optimisation].set_title('%s' %(optimisation[index_optimisation]),fontsize=30)
        for tick in axs[index_optimisation].xaxis.get_majorticklabels():  # example for xaxis
                    tick.set_fontsize(30) 
        for tick in axs[index_optimisation].yaxis.get_majorticklabels():  # example for xaxis
                    tick.set_fontsize(30) 
axs[0].legend(loc='upper right',fontsize=30)
fig.tight_layout()
plt.savefig('cumulative_tre_test.pdf')


