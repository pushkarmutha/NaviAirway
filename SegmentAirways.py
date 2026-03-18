import numpy as np
import torch
import SimpleITK as sitk
import os
from lungmask import mask
from os.path import join

from func.model_arch import SegAirwayModel
from func.model_run import semantic_segment_crop_and_cat
from func.post_process import post_process
from func.ulti import load_one_CT_img

def read_and_reorient(image_path):
    image = sitk.ReadImage(image_path)

    reoriented_image = sitk.DICOMOrient(image, 'LPS')
    # Save or return the reoriented image
    sitk.WriteImage(reoriented_image, image_path)
    return reoriented_image

def getdirs(path_to_folder):
    return [f.name for f in os.scandir(path_to_folder) if f.is_dir()]

def bbox2_3D(mask):
    # Computes a bounding box for the mask
    r = np.any(mask, axis=(1, 2))
    c = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax

def segmentAirway(raw_img_path, lung_path, savepath):
    sitkim = read_and_reorient(raw_img_path)
    if os.path.isfile(Lung_path)==False:
        lm = mask.apply(sitkim)
        lmg = sitk.GetImageFromArray(np.uint8(lm>0))
        lmg.CopyInformation(sitkim)
        sitk.WriteImage(lmg, Lung_path)

    in_img = load_one_CT_img(raw_img_path)
    lmg = load_one_CT_img(lung_path)
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(lmg)
    raw_img = in_img[rmin:rmax, cmin:cmax, zmin:zmax]

    seg_result_semi_supervise_learning = semantic_segment_crop_and_cat(raw_img, model_semi_supervise_learning, device,
                                                                   crop_cube_size=[32, 128, 128], stride=[16, 64, 64],
                                                                   windowMin=-1000, windowMax=600)
    seg_onehot_semi_supervise_learning = np.array(seg_result_semi_supervise_learning>threshold, dtype=np.uint8)
#
    seg_result = semantic_segment_crop_and_cat(raw_img, model, device,
                                           crop_cube_size=[32, 128, 128], stride=[16, 64, 64],
                                           windowMin=-1000, windowMax=600)
    seg_onehot = np.array(seg_result>threshold, dtype=np.uint8)
#
    seg_onehot_comb = np.array((seg_onehot+seg_onehot_semi_supervise_learning)>0, dtype=np.uint8)
    seg_processed,_ = post_process(seg_onehot_comb, threshold=threshold)
#
    op = np.zeros_like(lmg)
    op[rmin:rmax, cmin:cmax, zmin:zmax] = seg_processed
    outimage = sitk.GetImageFromArray(np.uint8(op>0))
    outimage.CopyInformation(sitkim)
    sitk.WriteImage(outimage, savepath)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.5

model=SegAirwayModel(in_channels=1, out_channels=2)
model.to(device)
load_path = "checkpoint/checkpoint.pkl"
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])

model_semi_supervise_learning=SegAirwayModel(in_channels=1, out_channels=2)
model_semi_supervise_learning.to(device)
load_path = "checkpoint/checkpoint_semi_supervise_learning.pkl"
checkpoint = torch.load(load_path)
model_semi_supervise_learning.load_state_dict(checkpoint['model_state_dict'])

path_to_folder = '<path_to_patient_folder_containing CT, LungMask, etc.>'
CT_filename = 'CT.nii.gz'
LungMask_filename = 'LungMask.nii.gz'
Output_filename = 'Airway.nii.gz'

CT_path = os.path.join(path_to_folder, CT_filename)
Lung_path = os.path.join(path_to_folder, LungMask_filename)
Output_path = os.path.join(path_to_folder, Output_filename)

print("Segmenting Airways...")
segmentAirway(CT_path, Lung_path, Output_path)
