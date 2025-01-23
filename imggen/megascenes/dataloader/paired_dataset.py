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
        self.pose_cond = 'warp_plus_pose'
        self.split = 'test'
        self.scaleclip = 1e-7

        assert seq_info.default_h == seq_info.default_w
        self.target_res = seq_info.default_h

        random.seed(42)

        paired_images = []
        for idx in generate_idx:
            overlaps = seq_info.views[idx].overlaps
            # get biggest overlap
            reference_idx = overlaps.index(max(overlaps))
            print(f"Generate image {idx} from reference {reference_idx} ({overlaps[reference_idx]})")
            ref_image = (seq_info.views[reference_idx].image.cpu().detach().numpy() + 1) * 127.5
            ref_image = ref_image.astype(np.uint8)

            warped_image = (seq_info.views[idx].warped.cpu().detach().numpy() + 1) * 127.5
            warped_image = np.transpose(warped_image[0], (1,2,0)).astype(np.uint8)
            
            paired_data = {
                "reference_idx" : reference_idx,
                "reference_pose" : seq_info.views[reference_idx].pose.cpu().detach().numpy(),
                "reference_focal" : seq_info.views[reference_idx].focal,
                "reference_image" : ref_image,
                "target_idx" : idx,
                "target_pose" : seq_info.views[idx].pose.cpu().detach().numpy(),
                "target_focal" : seq_info.views[idx].focal,
                "warped_image" : warped_image,
                "warped_mask" : seq_info.views[idx].mask,
            }
            paired_images.append(paired_data)

        self.paired_images = paired_images

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        ref_idx = self.paired_images[idx]["reference_idx"]
        target_idx = self.paired_images[idx]["target_idx"]

        try:
            img_target = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0
            img_ref = Image.fromarray(self.paired_images[idx]["reference_image"])
            img_ref = resize_with_padding(img_ref, int(self.target_res), black=False) /127.5-1.0 
        except Exception as error:
            print("exception when loading image:", error)
            img_target = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0
            img_ref = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0


        if self.pose_cond not in ['zeronvs_baseline'] or self.split!='train':
            try:
                high_warped_depth = Image.fromarray(self.paired_images[idx]["warped_image"])
                warped_depth = resize_with_padding(high_warped_depth, int(self.target_res) // 8, black=False) /127.5-1.0 
                high_warped_depth = resize_with_padding(high_warped_depth, int(self.target_res), black=False) /127.5-1.0 
            except Exception as error:
                print("exception when loading warped depth:", error)
                high_warped_depth = np.zeros((int(self.target_res),int(self.target_res),3)) -1.0
                warped_depth = np.zeros((int(self.target_res) // 8, int(self.target_res) // 8, 3)) -1.0

        dict1_extrinsics = self.paired_images[idx]['target_pose']
        dict2_extrinsics = self.paired_images[idx]['reference_pose']
        height = self.target_res
        width = self.target_res

        assert self.paired_images[idx]['target_focal'] == self.paired_images[idx]['reference_focal']
        focal_length = self.paired_images[idx]['target_focal'] 
        cx = self.target_res / 2.0
        cy = self.target_res / 2.0
        
        sensor_diagonal = math.sqrt(width**2 + height**2)
        diagonal_fov = 2 * math.atan(sensor_diagonal / (2 * focal_length)) # assuming fx = fy

        ext_ref = np.linalg.inv(dict2_extrinsics)
        ext_target = np.linalg.inv(dict1_extrinsics) # using c2w, following zeronvs

        scales = 1.0 # default scale

        fov = torch.tensor(diagonal_fov) # target fov, invariant to resizing
        rel_pose = np.linalg.inv(ext_ref) @ ext_target # 4x4

        # if self.pose_cond in ['warp_plus_pose', 'zeronvs_baseline']:
        #     depth_ref = np.load( self.imgname_to_depthname(path1) ) # HxW
        #     scales = np.quantile( depth_ref[::8, ::8].reshape(-1), q=0.2 ) 
        rel_pose[:3, -1] /= np.clip(scales, a_min=self.scaleclip, a_max=None) # scales preprocessed for faster data loading

        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        
        # for debug
        # img_ref = Image.open("imggen/megascenes/quant_eval/apple_test/refimgs/0.png")
        # img_target = Image.open("imggen/megascenes/quant_eval/apple_test/tarimgs/0.png")
        # img_ref = resize_with_padding(img_ref, int(self.target_res), black=False) /127.5-1.0 
        # img_target = resize_with_padding(img_target, int(self.target_res), black=False) /127.5-1.0
        # high_warped_depth = Image.open("imggen/megascenes/quant_eval/apple_test/masks/0.png")
        # warped_depth = resize_with_padding(high_warped_depth, int(self.target_res) // 8, black=False) /127.5-1.0 
        # high_warped_depth = resize_with_padding(high_warped_depth, int(self.target_res), black=False) /127.5-1.0 
        # rel_pose = torch.Tensor([
        #     -0.9408, -0.3204,  0.1108,  5.3313,  0.2051, -0.2776,  0.9386,  2.1735,
        #     -0.2699,  0.9057,  0.3268,  2.4848,  0.0000,  0.0000,  0.0000,  1.0000,
        #     0.8943,  0.7798,  0.6260])
                
        if self.pose_cond == 'warp_plus_pose':
            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=rel_pose, warped_depth=warped_depth, highwarp=high_warped_depth)
        elif self.pose_cond == 'zeronvs_baseline':
            retdict = dict(image_target=img_target,  image_ref=img_ref, rel_pose=rel_pose) # , highwarp=high_warped_depth
            if self.split!='train':
                retdict['highwarp'] = high_warped_depth
    
        if self.split!='train':
            return retdict, idx, ref_idx, target_idx
        return retdict
