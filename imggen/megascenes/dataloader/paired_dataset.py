import json
import cv2
from PIL import Image
import numpy as np
import random
import os 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sys import getsizeof
import glob
from .data_helpers import *
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math
import hashlib

class PairedDataset(Dataset):
    def __init__(self, target_res=256, split='train', pose_cond='warp_plus_pose'):
        
        # warp_plus_pose: scale translation based on 20th quantile of depth
        # inpainting: only use warped depth as pose condition, no additional pose condition for crossattn
        # zeronvs_baseline: no warped depth
        self.pose_cond = pose_cond
        self.split = split
        self.scaleclip = 1e-7
        self.target_res = target_res
        
        if split == 'train':
            with open('data/splits/trainsplit.pkl', 'rb') as f: 
                paired_images = pickle.load(f) # path1, path2, dict1, dict2, scale for path1 (dict keys: extrinsics, intrinsics)

        else:
            with open('data/splits/testsplit.pkl', 'rb') as f: 
                paired_images = pickle.load(f)
                # first 10000 are used for validation
                # 10000: are used for testing 
                paired_images = paired_images[10000:] 
                
        paired_images = paired_images[:1]
        self.paired_images = paired_images


    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):

        if len(self.paired_images[idx])==4:
            path1, path2, dict1, dict2 = self.paired_images[idx]
        else:
            path1, path2, dict1, dict2, scales = self.paired_images[idx]
        try:
            path1 = "/mydata/data/seunghoonjeong/DreamGrasp/assets/doll/0.png"
            path2 = "/mydata/data/seunghoonjeong/DreamGrasp/assets/doll/1.png"
            # img_ref = np.array(Image.open(path1)) /127.5-1.0
            # img_target = np.array(Image.open(path2)) /127.5-1.0 # HxWx3
            img_ref = resize_with_padding(path1, 512, black=False) /127.5-1.0
            img_target = resize_with_padding(path2, 512, black=False) /127.5-1.0
        except Exception as error:
            print("exception when loading image: ", error, path1, path2)
            img_ref = np.zeros((256,256,3)) -1.0
            img_target = np.zeros((256,256,3)) -1.0


        if self.pose_cond not in ['zeronvs_baseline'] or self.split!='train':
            warpname = "/mydata/data/seunghoonjeong/DreamGrasp/results/doll/warp_stage0/stacked_19.png"
            try:
                high_warped_depth = Image.open(warpname) # original warped image with high resolution
                warped_depth = resize_with_padding(high_warped_depth, 64, black=False) /127.5-1.0 
                high_warped_depth = resize_with_padding(high_warped_depth, 512, black=False) /127.5-1.0 
            except Exception as error:
                print("exception when loading warped depth:", error, path1, path2, warpname)
                warped_depth = np.zeros((32,32,3)) -1.0
                high_warped_depth = np.zeros((256,256,3)) -1.0

        if self.pose_cond == 'inpainting':
            retdict = dict(image_target=img_target, image_ref=img_ref, warped_depth=warped_depth, highwarp=high_warped_depth)
            if self.split!='train':
                return retdict, idx
            return retdict


        if self.pose_cond == 'sdinpaint':
            mask = np.zeros_like(high_warped_depth)
            mask[np.all(high_warped_depth == [-1,-1,-1], axis=-1)] = [255,255,255]
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).convert("L")).astype(np.float32)/255.0
            mask = torch.tensor(mask).unsqueeze(0)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            #masked_image = torch.tensor(img_target).permute(2,0,1) * (mask < 0.5)
            masked_image = np.array(Image.open(warpname))
            masked_image[np.all(masked_image == [0,0,0], axis=-1)] = [127,127,127]
            masked_image = resize_with_padding(Image.fromarray(masked_image), 256, black=False) /127.5-1.0 
            #ipdb.set_trace()
            retdict = dict(image_target=img_target,  image_ref=img_ref, highwarp=high_warped_depth, mask=mask, masked_image=masked_image, txt="photograph of a beautiful scene, highest quality settings")
            if self.split!='train':
                return retdict, idx
            return retdict

        dict1_extrinsics = np.array([
            [0.9999919533729553, -0.0039535281248390675, 0.0007246877648867667, 0.0], 
            [0.00395191228017211, 0.9999897480010986, 0.0022180795203894377, 0.0], 
            [-0.0007334495312534273, -0.00221519754268229, 0.9999973177909851, 0.0], 
            [0.0, 0.0, 0.0, 1.0]])
        dict2_extrinsics = np.array([
            [-0.8861775994300842, -0.40905532240867615, 0.21763068437576294, -0.06119517982006073], 
            [0.424800306558609, -0.529701292514801, 0.7341398596763611, -0.4925293028354645], 
            [-0.18502455949783325, 0.7430278658866882, 0.6431761980056763, 0.27335068583488464], 
            [0.0, 0.0, 0.0, 1.0]
        ])
        height = 512
        width = 512
        focal_length = 588.2550048828125
        cx = 256.0
        cy = 256.0
        
        sensor_diagonal = math.sqrt(width**2 + height**2)
        diagonal_fov = 2 * math.atan(sensor_diagonal / (2 * focal_length)) # assuming fx = fy

        ext_ref = np.linalg.inv(dict1_extrinsics)
        ext_target = np.linalg.inv(dict2_extrinsics) # using c2w, following zeronvs


        if self.pose_cond == 'zero123':
            tref = ext_ref[:3, -1] / np.clip(scales, a_min=self.scaleclip, a_max=None)
            ttarget = ext_target[:3, -1] / np.clip(scales, a_min=self.scaleclip, a_max=None)
            theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(tref[None, :])
            theta_target, azimuth_target, z_target = cartesian_to_spherical(ttarget[None, :])
            
            d_theta = theta_target - theta_cond
            d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
            d_z = z_target - z_cond
            
            d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()]).float()

            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=d_T, highwarp=high_warped_depth)
            if self.split!='train':
                return retdict, idx
            return retdict

        fov = torch.tensor(diagonal_fov) # target fov, invariant to resizing
        rel_pose = np.linalg.inv(ext_ref) @ ext_target # 4x4

        # if self.pose_cond in ['warp_plus_pose', 'zeronvs_baseline']:
        #     depth_ref = np.load( self.imgname_to_depthname(path1) ) # HxW
        #     scales = np.quantile( depth_ref[::8, ::8].reshape(-1), q=0.2 ) 
        rel_pose[:3, -1] /= np.clip(scales, a_min=self.scaleclip, a_max=None) # scales preprocessed for faster data loading

        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        
        
        if self.pose_cond == 'warp_plus_pose':
            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=rel_pose, warped_depth=warped_depth, highwarp=high_warped_depth)
        elif self.pose_cond == 'zeronvs_baseline':
            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=rel_pose) # , highwarp=high_warped_depth
            if self.split!='train':
                retdict['highwarp'] = high_warped_depth
    
        if self.split!='train':
            return retdict, idx
        return retdict


