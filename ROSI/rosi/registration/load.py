#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:20:15 2022

@author: mercier
"""

from nibabel import Nifti1Image,load
import numpy as np
from rosi.reconstruction.rec_ebner import sorted_alphanumeric
from .sliceObject import SliceObject
from .tools import distance_from_mask
from .sliceObject import SliceObject
import os

def convert2Slices(stack : Nifti1Image,
              mask : Nifti1Image,
              listOfSlices : 'list[SliceObject]',
              index_stack : int,
              index_volume : int) -> list:
    
    """
    Take an LR volume as input and convert each slice into a sliceObject. Then, add each slice to the input list
    (listOfSlices).
    """
    
    OutputList = listOfSlices.copy()

    if mask == None: #If the mask wasn't provided, one is created covering the entire volume.
        mask = Nifti1Image(np.ones(stack.get_fdata().shape), stack.affine)
    
    X,Y,Z = stack.shape
    slice_value = np.zeros((X,Y,1))
    slice_mask = np.zeros((X,Y,1))
    
    res=min(mask.header.get_zooms())
    
    for zi in range(Z): #for each slices in the stack
        
        slice_value[:,:,0] = stack.get_fdata()[:,:,zi]
        slice_mask[:,:,0] = mask.get_fdata()[:,:,zi].astype(int)
        slice_mask[np.isnan(slice_mask)]=0
        
        #The slice is linearly cropped according to the distance to the mask
        dist = distance_from_mask(slice_mask)*res
        decrease=np.linspace(1,0,6,dtype=float)
        index=0
        
        for index_dist in range(4,10):
             slice_value[np.where(dist>index_dist)] = decrease[index]*slice_value[np.where(dist>index_dist)]
             index+=1
 
        if ~(np.all(slice_mask==0)): #Check that the slice mask is not null. If it is, the slice will be deleted because it has no interest.
            
           
            mz = np.eye(4)

            #A translation in z is applied to the stack transformation to associate a transformation matrix with each slice : 
            #R_f(k) R_k,2t3d
            mz[2,3]= zi
            slice_transformation = stack.affine @ mz 
            
            new_slice = Nifti1Image(slice_value.copy(),slice_transformation)
            new_object = SliceObject(new_slice,slice_mask.copy(),index_stack,zi,index_volume)
            OutputList.append(new_object)
        else :
            print('mask_nul')
   
    return OutputList


def loadStack(fileImage : str,
              fileMask : str) -> (Nifti1Image,Nifti1Image):
    
    """
    Load stack and mask from files using nibabel library
    """
    
    stack = load(fileImage)
    #stack = stack_original.copy()
    if fileMask == None: ##If the mask wasn't provided, one is created covering the entire image.
          fileMask = np.ones(stack.get_fdata().shape)
          stmask = Nifti1Image(fileMask,stack.affine)
    else :
          stmask = load(fileMask).get_fdata()
          print(stmask[np.where(stmask>0)])
          #check that the mask is a binary image
          #
          stmask = np.round(stmask)
          print(stmask[np.where(stmask>0)])
          stmask = np.array(stmask,dtype=np.int64)
          print(stmask[np.where(stmask>0)])
          data = stmask.reshape(-1)
          stmask = Nifti1Image(stmask,stack.affine,dtype=np.int64)
          #data = stmask.get_fdata()
          #print(data[np.where(data>0)])
          print(data[data<1])
          #)

          if not (np.all((data==0)|(data==1))):
               raise Exception('The mask is not a binary image')
    return stack,stmask


def loadFromdir(dir_input):
     
     """
     Create a listofSlice from the direcory containing all the slices
     """

     list_file = sorted_alphanumeric(file for file in os.listdir(dir_input) if os.path.isdir(dir_input))
     OutputList = []
     list_stack_ortho = []
     list_stack_zi = []
     for file in list_file:
          if not "mask" in file:
            slice_path = os.path.join(dir_input,slice_path)
            nib_slice = load(slice_path)
            mask_path=os.path.join(dir_input,'/mask_' + file)
            nib_mask = load(mask_path)
            
            new_stack = False
            for i in range(0,len(list_stack_ortho)):
                vec_orthogonal = np.norm(nib_slice.affine[0:3,0] @ nib_slice.affine[0:3,1])
                if np.abs(np.cross(vec_orthogonal,list_stack_ortho[i])) < 0.75 :
                    index_stack = i
                    break
                else : 
                    if i == len(list_stack_ortho):
                        new_stack = True
                        index_stack = i +1
            if new_stack : 
                list_stack_ortho.append(vec_orthogonal)
                list_stack_zi.append(1)

            zi = list_stack_zi[index_stack]+1
            list_stack_zi[index_stack]=list_stack_zi[index_stack]+1
            index_volume = index_stack
            new_object = SliceObject(nib_slice,nib_mask.get_fdata(),index_stack,zi,index_volume)
            OutputList.append(new_object)

     return OutputList



def convert2ListSlice(dir_nomvt,dir_slice,slice_thickness,set_of_affines):
    #print(dir_nomvt)
    ##this function take each slices, saved in a nifti, and convert it to sliceObject. It then create a listOfSliceObject, which are used to compute the tre.
    list_file = sorted_alphanumeric(os.listdir(dir_slice))
    #print(list_file)

    listSlice=[];listnomvt=[]
    index=np.zeros(3,np.int32)
    for file in list_file:
        #check that the file is a slice
        
        if not 'mask' in file and not 'image' in file and not 'volume' in file : ##for svrtk you would need to have "slice"
            ##else do nothing
            print('file_name :',file)
            slice_data=nib.load(dir_slice + '/' + file)
            mask_data=nib.load(dir_slice + '/mask_' + file)
            slice_nomvt = nib.load(dir_nomvt + '/' + file)
            mask_slice=nib.load(dir_slice + '/mask_' + file)
            print("affine equal ? ",np.allclose(slice_data.affine,slice_nomvt.affine,1e-1))
                #print(slice_data.affine - slice_nomvt.affine)
            print(slice_data.affine)
            print(slice_nomvt.affine)
                
            T0 = slice_data.affine


            num_stack=which_stack(slice_nomvt.affine,slice_thickness) 
                #print(slice_data.affine)
            stack_affine=set_of_affines[num_stack]
            
            i_slice=where_in_the_stack(stack_affine,slice_nomvt.affine,num_stack)
            slicei = SliceObject(slice_data,mask_data.get_fdata(),num_stack,i_slice, num_stack)
            slicen = SliceObject(slice_nomvt,mask_slice.get_fdata(),num_stack,i_slice, num_stack)
                
            listSlice.append(slicei)
            listnomvt.append(slicen)
    return listnomvt,listSlice