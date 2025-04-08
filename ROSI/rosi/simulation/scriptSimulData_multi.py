#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:25:56 2022

@author: mercier
"""

from rosi.simulation.simul3Ddata import extract_mask,simulateMvt
import nibabel as nib
import numpy as np
import os
import argparse
import sys
from os import getcwd, path, mkdir
import six

#script to simulate LR image with motion from an HR image

#The function 'simulate_mvt' simulates a LR image with inter-slice motion from an HR image.
#SimulateMVt take as parameters : the original HRImage, range motion for rotation, range motion for translation, upsampling parameters (to choose interslice resolution of LR image), image orientation, binary image corresponding to the mask and a boolean (Set to false if you don't want motion)
#And return : the LR image, mask of the LR image, parameters of transformation for each slices, rigid transformation for each slices.


class InputArgparser(object):

    def __init__(self,
                 description=None,
                 prog=None,
                 config_arg="--config"
                ):


        kwargs = {}

        self._parser = argparse.ArgumentParser(**kwargs)
        self._parser.add_argument(
            config_arg,
            help="Configuration file in JSON format.")
        self._parser.add_argument(
            "--version",
            action="version",           
        )
        self._config_arg = config_arg

    def get_parser(self):
        return self._parser

    def parse_args(self):

        # read config file if available
        if self._config_arg in sys.argv:
            self._parse_config_file()

        return self._parser.parse_args()

    def print_arguments(self, args, title="Configuration:"):
        
        for arg in sorted(vars(args)):
            
            vals = getattr(args, arg)

            if type(vals) is list:
                # print list element in new lines, unless only one entry in list
                # if len(vals) == 1:
                #     print(vals[0])
                # else:
                print("")
                for val in vals:
                    print("\t%s" % val)
            else:
                print(vals)
        


    def add_HR1(
        self,
        option_string="--hr",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_HR2(
        self,
        option_string="--hr",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_types(
        self,
        option_string="--types",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Mask(  #if dHCP, mask is the same for the HR reconstructed
        self,
        option_string="--mask",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Output(
        self,
        option_string="--output",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Motion(
        self,
        option_string="--motion",
        nargs="+",
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_stack_Motion( # If not added, no motion between the stacks
        self,
        option_string="--stack-motion",
        nargs="+",
        default=0,
        required=False,
    ):
        self._add_argument(dict(locals()))
        
    def add_Name(
        self,
        option_string="--name",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def _add_argument(self, allvars):

        # Skip variable 'self'
        allvars.pop('self')

        # Get name of argument to add
        option_string = allvars.pop('option_string')

        # Build dictionary for additional, optional parameters
        kwargs = {}
        for key, value in six.iteritems(allvars):
            kwargs[key] = value

        # Add information on default value in case provided
        if 'default' in kwargs.keys():

            if type(kwargs['default']) == list:
                txt = " ".join([str(i) for i in kwargs['default']])
            else:
                txt = str(kwargs['default'])
            txt_default = " [default: %s]" % txt

            # Case where 'required' key is given:
            if 'required' in kwargs.keys():

                # Only add information in case argument is not mandatory to
                # parse
                if kwargs['default'] is not None and not kwargs['required']:
                    kwargs['help'] += txt_default

            # Case where no such field was provided
            else:
                if kwargs['default'] is not None:
                    kwargs['help'] += txt_default

        # Add argument with its options
        self._parser.add_argument(option_string, **kwargs)



if __name__ == '__main__':
        
    root=getcwd()
    
    input_parser = InputArgparser()
    
    input_parser.add_HR1(required=True) #load images
    input_parser.add_HR2(required=True) #load images
    input_parser.add_types(required=True) #load images
    input_parser.add_Mask(required=True) #load masks
    input_parser.add_Output(required=True) #load simulated transformation
    input_parser.add_Name(required=True)
    input_parser.add_Motion(required=True)
    input_parser.add_stack_Motion(required=False)
    args = input_parser.parse_args()


    HRnifti1 = nib.load(args.hr1) #3D isotropic image
    HRnifti2 = nib.load(args.hr2) #3D isotropic image
    types = args.types #list of types of images
    Mask = nib.load(args.mask) #mask associated to the image
    binaryMask = extract_mask(Mask) #convert mask to a biniary mask
    #os.mkdir('/home/mercier/Documents/donnee/test/Grand5/')
    output = args.output
    name1 = args.name + "_"+types[0]
    name2 = args.name + "_"+types[1]

    parameters_motion = args.motion
    motion=np.asarray(parameters_motion,dtype=np.float64)
    
    if not os.path.isdir(output):
        mkdir(output)
    
    
    LrAxNifti1,AxMask1,paramAx1,transfoAx1 = simulateMvt(HRnifti1,motion,motion,6,'axial',binaryMask.get_fdata(),True)#create an axial volume
    LrCorNifti1,CorMask1,paramCor1,transfoCor1 = simulateMvt(HRnifti1,motion,motion,6,'coronal',binaryMask.get_fdata(),True) #create a coronal volume
    LrSagNifti1,SagMask1,paramSag1,transfoSag1 = simulateMvt(HRnifti1,motion,motion,6,'sagittal',binaryMask.get_fdata(),True)#create a sagittal volume

    LrAxNifti2,AxMask2,paramAx2,transfoAx2 = simulateMvt(HRnifti2,motion,motion,6,'axial',binaryMask.get_fdata(),True)#create an axial volume
    LrCorNifti2,CorMask2,paramCor2,transfoCor2 = simulateMvt(HRnifti2,motion,motion,6,'coronal',binaryMask.get_fdata(),True) #create a coronal volume
    LrSagNifti2,SagMask2,paramSag2,transfoSag2 = simulateMvt(HRnifti2,motion,motion,6,'sagittal',binaryMask.get_fdata(),True)#create a sagittal volume

    # We want to add the motion from stacks T1w to T2w
    # 1/ With Motion
    
    
    ##add noise to data 1
    sigma1=np.random.uniform()*0.1
    print(sigma1)
    
    mu=np.mean(LrAxNifti1.get_fdata()[AxMask1.get_fdata()>0])
    var=np.var(LrAxNifti1.get_fdata()[AxMask1.get_fdata()>0])
    print(mu*sigma1)
    print(mu,var)
    data1=LrAxNifti1.get_fdata()+mu*np.random.normal(0,sigma1,LrAxNifti1.get_fdata().shape)
    LrAxNifti1=nib.Nifti1Image(data1, LrAxNifti1.affine)
    
    nib.save(LrAxNifti1,output + '/LrAxNifti_' +name1 +'.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask1,output + '/LrAxNifti_' +name1+ '_mask.nii.gz')
    np.save(output + '/paramAx_' +name1+ '.npy',paramAx1)
    np.save(output + '/transfoAx_' +name1+ '.npy',transfoAx1)
    
    mu=np.mean(LrCorNifti1.get_fdata()[CorMask1.get_fdata()>0])
    var=np.var(LrCorNifti1.get_fdata()[CorMask1.get_fdata()>0])
    print(mu*sigma1)
    print(mu,var)
    data=LrCorNifti1.get_fdata()+mu*np.random.normal(0,sigma1,LrCorNifti1.get_fdata().shape)
    LrCorNifti1=nib.Nifti1Image(data, LrCorNifti1.affine)
    
    nib.save(LrCorNifti1, output +  '/LrCorNifti_' +name1+ '.nii.gz')
    nib.save(CorMask1,output +  '/LrCorNifti_' +name1+ '_mask.nii.gz')
    np.save(output +  '/paramCor_' +name1+ '.npy',paramCor1)
    np.save(output +  '/transfoCor_' +name1+ '.npy',transfoCor1)
    
    mu=np.mean(LrSagNifti1.get_fdata()[SagMask1.get_fdata()>0])
    var=np.var(LrSagNifti1.get_fdata()[SagMask1.get_fdata()>0])
    print(mu*sigma1)
    print(mu,var)
    data=LrSagNifti1.get_fdata()+mu*np.random.normal(0,sigma1,LrSagNifti1.get_fdata().shape)
    LrSagNifti1=nib.Nifti1Image(data, LrSagNifti1.affine)
    
    nib.save(LrSagNifti1,output +  '/LrSagNifti_' +name1+ '.nii.gz')
    nib.save(SagMask1,output +  '/LrSagNifti_' +name1+ '_mask.nii.gz')
    np.save(output +  '/paramSag_' +name1+ '.npy',paramSag1)
    np.save(output +  '/transfoSag_' +name1+ '.npy',transfoSag1)

    ##add noise to data 2
    sigma2=np.random.uniform()*0.1
    print(sigma2)
    
    mu=np.mean(LrAxNifti2.get_fdata()[AxMask2.get_fdata()>0])
    var=np.var(LrAxNifti2.get_fdata()[AxMask2.get_fdata()>0])
    print(mu*sigma2)
    print(mu,var)
    data=LrAxNifti2.get_fdata()+mu*np.random.normal(0,sigma2,LrAxNifti2.get_fdata().shape)
    LrAxNifti2=nib.Nifti1Image(data, LrAxNifti2.affine)
    
    nib.save(LrAxNifti2,output + '/LrAxNifti_' +name2 +'.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask2,output + '/LrAxNifti_' +name2+ '_mask.nii.gz')
    np.save(output + '/paramAx_' +name2+ '.npy',paramAx2)
    np.save(output + '/transfoAx_' +name2+ '.npy',transfoAx2)
    
    mu=np.mean(LrCorNifti2.get_fdata()[CorMask2.get_fdata()>0])
    var=np.var(LrCorNifti2.get_fdata()[CorMask2.get_fdata()>0])
    print(mu*sigma2)
    print(mu,var)
    data=LrCorNifti2.get_fdata()+mu*np.random.normal(0,sigma2,LrCorNifti2.get_fdata().shape)
    LrCorNifti2=nib.Nifti1Image(data, LrCorNifti2.affine)
    
    nib.save(LrCorNifti2, output +  '/LrCorNifti_' +name2+ '.nii.gz')
    nib.save(CorMask2,output +  '/LrCorNifti_' +name2+ '_mask.nii.gz')
    np.save(output +  '/paramCor_' +name2+ '.npy',paramCor2)
    np.save(output +  '/transfoCor_' +name2+ '.npy',transfoCor2)
    
    mu=np.mean(LrSagNifti2.get_fdata()[SagMask2.get_fdata()>0])
    var=np.var(LrSagNifti2.get_fdata()[SagMask2.get_fdata()>0])
    print(mu*sigma2)
    print(mu,var)
    data=LrSagNifti2.get_fdata()+mu*np.random.normal(0,sigma2,LrSagNifti2.get_fdata().shape)
    LrSagNifti2=nib.Nifti1Image(data, LrSagNifti2.affine)
    
    nib.save(LrSagNifti2,output +  '/LrSagNifti_' +name2+ '.nii.gz')
    nib.save(SagMask2,output +  '/LrSagNifti_' +name2+ '_mask.nii.gz')
    np.save(output +  '/paramSag_' +name2+ '.npy',paramSag2)
    np.save(output +  '/transfoSag_' +name2+ '.npy',transfoSag2)

# 2/ Without Motion, (False at the end of the function simulateMvt)

    LrAxNifti1,AxMask1,paramAx1,transfoAx1 = simulateMvt(HRnifti1,motion,motion,6,'axial',binaryMask.get_fdata(),False)#create an axial volume
    LrCorNifti1,CorMask1,paramCor1,transfoCor1 = simulateMvt(HRnifti1,motion,motion,6,'coronal',binaryMask.get_fdata(),False) #create a coronal volume
    LrSagNifti1,SagMask1,paramSag1,transfoSag1 = simulateMvt(HRnifti1,motion,motion,6,'sagittal',binaryMask.get_fdata(),False)#create a sagittal volume

    LrAxNifti2,AxMask2,paramAx2,transfoAx2 = simulateMvt(HRnifti2,motion,motion,6,'axial',binaryMask.get_fdata(),False)#create an axial volume
    LrCorNifti2,CorMask2,paramCor2,transfoCor2 = simulateMvt(HRnifti2,motion,motion,6,'coronal',binaryMask.get_fdata(),False) #create a coronal volume
    LrSagNifti2,SagMask2,paramSag2,transfoSag2 = simulateMvt(HRnifti2,motion,motion,6,'sagittal',binaryMask.get_fdata(),False)#create a sagittal volume

# Contrast 1
    data=LrAxNifti1.get_fdata()#+np.random.normal(0,sigma,LrAxNifti.get_fdata().shape)
    LrAxNifti1=nib.Nifti1Image(data, LrAxNifti1.affine)

    nib.save(LrAxNifti1,output + '/LrAxNifti_nomvt_' + types[0] + '.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask1,output + '/AxMask_nomvt_' + types[0] + '.nii.gz')
    np.save(output + '/paramAx_nomvt_' + types[0] + '.npy',paramAx1)
    np.save(output + '/transfoAx_nomvt_' + types[0] + '.npy',transfoAx1)
    
    data=LrCorNifti1.get_fdata()#+np.random.normal(0,sigma,LrCorNifti.get_fdata().shape)
    LrCorNifti1=nib.Nifti1Image(data, LrCorNifti1.affine)
    
    nib.save(LrCorNifti1, output +  '/LrCorNifti_nomvt_' + types[0] + '.nii.gz')
    nib.save(CorMask1,output +  '/CorMask_nomvt_' + types[0] + '.nii.gz')
    np.save(output +  '/paramCor_nomvt_' + types[0] + '.npy',paramCor1)
    np.save(output +  '/transfoCor_nomvt_' + types[0] + '.npy',transfoCor1)
    
    data=LrSagNifti1.get_fdata()#+np.random.normal(0,sigma,LrSagNifti.get_fdata().shape)
    LrSagNifti1=nib.Nifti1Image(data, LrSagNifti1.affine)
    
    nib.save(LrSagNifti1,output +  '/LrSagNifti_nomvt_' + types[0] + '.nii.gz')
    nib.save(SagMask1,output +  '/SagMask_nomvt_' + types[0] + '.nii.gz')
    np.save(output +  '/paramSag_nomvt_' + types[0] + '.npy',paramSag1)
    np.save(output +  '/transfoSag_nomvt_' + types[0] + '.npy',transfoSag1)

# Contrast 2
    data=LrAxNifti2.get_fdata()#+np.random.normal(0,sigma,LrAxNifti.get_fdata().shape)
    LrAxNifti2=nib.Nifti1Image(data, LrAxNifti2.affine)

    nib.save(LrAxNifti2,output + '/LrAxNifti_nomvt_' + types[1] + '.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask2,output + '/AxMask_nomvt_' + types[1] + '.nii.gz')
    np.save(output + '/paramAx_nomvt_' + types[1] + '.npy',paramAx2)
    np.save(output + '/transfoAx_nomvt_' + types[1] + '.npy',transfoAx2)
    
    data=LrCorNifti2.get_fdata()#+np.random.normal(0,sigma,LrCorNifti.get_fdata().shape)
    LrCorNifti2=nib.Nifti1Image(data, LrCorNifti2.affine)
    
    nib.save(LrCorNifti2, output +  '/LrCorNifti_nomvt_' + types[1] + '.nii.gz')
    nib.save(CorMask2,output +  '/CorMask_nomvt_' + types[1] + '.nii.gz')
    np.save(output +  '/paramCor_nomvt_' + types[1] + '.npy',paramCor2)
    np.save(output +  '/transfoCor_nomvt_' + types[1] + '.npy',transfoCor2)
    
    data=LrSagNifti2.get_fdata()#+np.random.normal(0,sigma,LrSagNifti.get_fdata().shape)
    LrSagNifti2=nib.Nifti1Image(data, LrSagNifti2.affine)
    
    nib.save(LrSagNifti2,output +  '/LrSagNifti_nomvt_' + types[1] + '.nii.gz')
    nib.save(SagMask2,output +  '/SagMask_nomvt_' + types[1] + '.nii.gz')
    np.save(output +  '/paramSag_nomvt_' + types[1] + '.npy',paramSag2)
    np.save(output +  '/transfoSag_nomvt_' + types[1] + '.npy',transfoSag2)



