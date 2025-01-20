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
    def __init__(self, target_res=512, split='train', pose_cond='warp_plus_pose', data_dir='data', category='apple'):
        
        # warp_plus_pose: scale translation based on 20th quantile of depth
        # inpainting: only use warped depth as pose condition, no additional pose condition for crossattn
        # zeronvs_baseline: no warped depth
        self.pose_cond = pose_cond
        self.split = split
        self.scaleclip = 1e-7
        self.target_res = target_res
        
        # if split == 'train':
        #     with open('data/splits/trainsplit.pkl', 'rb') as f: 
        #         paired_images = pickle.load(f) # path1, path2, dict1, dict2, scale for path1 (dict keys: extrinsics, intrinsics)

        # else:
        #     with open('data/splits/testsplit.pkl', 'rb') as f: 
        #         paired_images = pickle.load(f)
        #         # first 10000 are used for validation
        #         # 10000: are used for testing 
        #         paired_images = paired_images[10000:] 
        
        
        random.seed(42)
        self.data_dir = data_dir
        if category is not None:
            with open(os.path.join(data_dir, f'paired_data_{split}.json'), 'r') as f:
                paired_images = json.load(f)[category]
        else:
            with open(os.path.join(data_dir, f'paired_data_{split}.json'), 'r') as f:
                paired_images_json = json.load(f)
            categories = paired_images_json.keys()
            paired_images = []
            for category in categories:
                paired_images += paired_images_json[category]

        print(paired_images[0])
        self.paired_images = paired_images

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):

        # if len(self.paired_images[idx])==4:
        #     path1, path2, dict1, dict2 = self.paired_images[idx]
        # else:
        #     path1, path2, dict1, dict2, scales = self.paired_images[idx]

        category = self.paired_images[idx]['category']
        seq_name = self.paired_images[idx]['seq_name']
        query_idx = self.paired_images[idx]['query_idx']
        query_image_num = self.paired_images[idx]['query_image_num']
        ref_idx_list = self.paired_images[idx]['ref_idx_candidates']
        ref_image_num_list = self.paired_images[idx]['ref_image_nums']
        warped_image_path = self.paired_images[idx]['warped_image_path']
        warped_mask_path = self.paired_images[idx]['warped_mask_path']

        # get ref image from ref_idx_list (randomly)
        ref_idx_idx = random.randint(0, len(ref_idx_list)-1)
        ref_idx = ref_idx_list[ref_idx_idx]
        ref_image_num = ref_image_num_list[ref_idx_idx]

        pose_info = np.load(os.path.join(self.data_dir, category, seq_name, f"poses_{self.split}.npy"), allow_pickle=True)
        focal_info = np.load(os.path.join(self.data_dir, category, seq_name, f"focals_{self.split}.npy"), allow_pickle=True)
        # pps_length = np.load(os.path.join(self.data_dir, category, seq_name, f"pps_{self.split}.npy"), allow_pickle=True)

        try:
            path1 = os.path.join(self.data_dir, category, seq_name, "images", "frame%06d.jpg" % query_image_num)
            path2 = os.path.join(self.data_dir, category, seq_name, "images", "frame%06d.jpg" % ref_image_num)
            # img_ref = np.array(Image.open(path1)) /127.5-1.0
            # img_target = np.array(Image.open(path2)) /127.5-1.0 # HxWx3
            img_target = resize_with_padding(path1, int(self.target_res), black=False) /127.5-1.0
            img_ref = resize_with_padding(path2, int(self.target_res), black=False) /127.5-1.0
        except Exception as error:
            print("exception when loading image: ", error, path1, path2)
            img_ref = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0
            img_target = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0


        if self.pose_cond not in ['zeronvs_baseline'] or self.split!='train':
            try:
                high_warped_depth = Image.open(warped_image_path) # original warped image with high resolution
                warped_depth = resize_with_padding(high_warped_depth, int(self.target_res) // 8, black=False) /127.5-1.0 
                high_warped_depth = resize_with_padding(high_warped_depth, int(self.target_res), black=False) /127.5-1.0 
            except Exception as error:
                print("exception when loading warped depth:", error, path1, path2, warped_image_path)
                warped_depth = np.zeros((int(self.target_res) // 8, int(self.target_res) // 8, 3)) -1.0
                high_warped_depth = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0

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
            masked_image = np.array(Image.open(warped_image_path))
            masked_image[np.all(masked_image == [0,0,0], axis=-1)] = [127,127,127]
            masked_image = resize_with_padding(Image.fromarray(masked_image), self.target_res, black=False) /127.5-1.0 
            #ipdb.set_trace()
            retdict = dict(image_target=img_target, image_ref=img_ref, highwarp=high_warped_depth, mask=mask, masked_image=masked_image, txt="photograph of a beautiful scene, highest quality settings")
            if self.split!='train':
                return retdict, idx
            return retdict

        dict1_extrinsics = pose_info[query_idx]
        dict2_extrinsics = pose_info[ref_idx]
        height = self.target_res
        width = self.target_res
        focal_length = focal_info[query_idx][0]
        cx = self.target_res / 2.0
        cy = self.target_res / 2.0
        
        sensor_diagonal = math.sqrt(width**2 + height**2)
        diagonal_fov = 2 * math.atan(sensor_diagonal / (2 * focal_length)) # assuming fx = fy

        ext_ref = np.linalg.inv(dict2_extrinsics)
        ext_target = np.linalg.inv(dict1_extrinsics) # using c2w, following zeronvs

        scales = 1.0 # default scale
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

class TempPairedDataset(PairedDataset):
    def __init__(self, seq_info, generate_idx):
        breakpoint()
        self.pose_cond = 'warp_plus_pose'
        self.split = 'test'
        self.scaleclip = 1e-7

        assert seq_info.default_h == seq_info.default_w
        self.target_res = seq_info.default_h

        random.seed(42)
        self.data_dir = data_dir
        with open(os.path.join(data_dir, f'paired_data_{split}.json'), 'r') as f:
            paired_images = json.load(f)[category]

        print(paired_images[0])
        self.paired_images = paired_images
