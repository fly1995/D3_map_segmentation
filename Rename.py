import glob
import nibabel as nb
import os

file_dir = r'E:\D4_map_segmentation\data\test_results\D_UNet_pp+MDFA_Net\es\\'


imgname = glob.glob(file_dir+'*.nii.gz')
for file_name in imgname:
    name = file_name[file_name.rindex("\\") + 1:]
    newname = name.replace('D', 'S')
    os.rename(os.path.join(file_dir, name), os.path.join(file_dir,newname))


