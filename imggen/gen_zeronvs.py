import os
import yaml
import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

# add sys path for dust3r
import sys
sys.path.append("imggen/megascenes")
from ldm.util import instantiate_from_config
from ldm.logger import ImageLogger
from dataloader import TempPairedDataset
from accelerate import Accelerator

def load_gen_model(output_dir, model_dir, model_iteration, device):
    config_file = yaml.safe_load(open("imggen/megascenes/configs/warp_plus_pose/config.yaml"))
    train_configs = config_file.get('training', {})
    model = instantiate_from_config(config_file['model']).eval()
    img_logger = ImageLogger(log_directory=output_dir, log_images_kwargs=train_configs['log_images_kwargs'])
    accelerator = Accelerator()
    model = accelerator.prepare(model)

    if model_iteration is None:
        saved_checkpoints = glob.glob(os.path.join(model_dir, 'iter*'))
        assert len(saved_checkpoints) > 0, "No saved checkpoints found"
        saved_checkpoints = sorted(saved_checkpoints, key=lambda x: int(x.split('iter_')[-1]))
        saved_checkpoints.reverse()
        model_path = saved_checkpoints[0]
        model_iteration = int(model_path.split('iter_')[-1])
    else:
        model_path = os.path.join(model_dir, f'iter_{model_iteration}')
    
    print("Loading model from", model_path)
    accelerator.load_state(model_path)

    return {
        "accelerator": accelerator,
        "model": model,
        "model_path": model_path,
        "img_logger": img_logger,
        "model_iteration": model_iteration
    }

def choose_idx_to_generate(seq_info, known_area_ratio):
    generate_cnt = min(2 ** seq_info.cur_stage, seq_info.target_poses_cnt - seq_info.generated_poses_cnt)
    min_known_area_ratio = 0.5
    max_azimuth_diff = np.pi / 4

    # get lowest known area ratio that is greater than min_known_area_ratio
    candidates = [(i, known_area_ratio[i]) for i in range(seq_info.length) if known_area_ratio[i] >= min_known_area_ratio and seq_info.views[i].stage == -1]
    candidates.sort(key=lambda x: x[1])
    
    if len(candidates) < generate_cnt:
        additional_candidates = [(i, known_area_ratio[i]) for i in range(seq_info.length) if known_area_ratio[i] < min_known_area_ratio and seq_info.views[i].stage == -1]
        additional_candidates.sort(key=lambda x: -x[1])
        additional_generate_cnt = generate_cnt - len(candidates)
        candidates += additional_candidates[:additional_generate_cnt]

    if len(candidates) == generate_cnt:
        generate_idx = [candidates[i][0] for i in range(len(candidates))]
    else: # len(candidates) > generate_cnt
        generate_idx = []
        reference_idx = [i for i in range(seq_info.length) if seq_info.views[i].stage != -1]
        # print("Reference idx:" + str(reference_idx))

        not_so_far_from_reference_idx = []
        not_so_far_from_reference_info = []
        for candidate_idx, candidate_info in enumerate(candidates):
            min_azimuth = np.pi
            for ref in reference_idx:
                if seq_info.pair_angles[candidate_info[0], ref] < min_azimuth:
                    min_azimuth = seq_info.pair_angles[candidate_info[0], ref]
            if min_azimuth < max_azimuth_diff:
                not_so_far_from_reference_idx.append(candidate_idx)
                not_so_far_from_reference_info.append((min_azimuth, candidate_idx))
        not_so_far_from_reference_info.sort(key=lambda x: -x[0])
        # print("Not so far from reference info:")
        # for info in not_so_far_from_reference_info:
        #     print(f"  {info[0]} -> {candidates[info[1]][0]}")

        while len(generate_idx) < generate_cnt:
            if len(generate_idx) == 0:
                # get max min_azimuth candidate
                generate_idx.append(candidates[not_so_far_from_reference_info[0][1]][0])

            else:
                azimuth_from_known = []
                for candidate_idx in not_so_far_from_reference_idx:
                    candidate_info = candidates[candidate_idx]
                    if candidate_idx in generate_idx:
                        continue

                    min_azimuth = np.pi
                    for j in generate_idx + reference_idx:
                        if seq_info.pair_angles[candidate_info[0], j] < min_azimuth:
                            min_azimuth = seq_info.pair_angles[candidate_info[0], j]

                    azimuth_from_known.append((candidate_idx, min_azimuth))

                if len(azimuth_from_known) == 0:
                    for i, candidate_info in enumerate(candidates):
                        if i in generate_idx:
                            continue

                        min_azimuth = np.pi
                        for j in generate_idx + reference_idx:
                            if seq_info.pair_angles[candidate_info[0], j] < min_azimuth:
                                min_azimuth = seq_info.pair_angles[candidate_info[0], j]
                        azimuth_from_known.append((i, min_azimuth))

                # get candidate with max min_azimuth
                azimuth_from_known.sort(key=lambda x: -x[1])
                max_min_azimuth_candidate_idx = azimuth_from_known[0][0]
                generate_idx.append(candidates[max_min_azimuth_candidate_idx][0])

    print(f"Generate {generate_cnt} images:")
    for idx in generate_idx:
        print(f"  {idx} -> {known_area_ratio[idx]}")
    return generate_idx

def generate_images(gen_model, seq_info, generate_idx):
    accelerator = gen_model["accelerator"]
    model = gen_model["model"]
    img_logger = gen_model["img_logger"]
    model_iteration = gen_model["model_iteration"]

    savepath = os.path.join(seq_info.project_dir, f"gen_stage{seq_info.cur_stage}")
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(os.path.join(savepath, 'generations'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'refimgs'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'masks'), exist_ok=True)

    dataset = TempPairedDataset(seq_info, generate_idx)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4,
        drop_last=False, shuffle=False, persistent_workers=False, pin_memory=False
    )
    dataloader = accelerator.prepare(dataloader)

    for ii, batch in enumerate(dataloader):
        batch, dataidx, ref_idx, target_idx = batch

        refimg = batch["image_ref"].cuda().float()
        mask = (batch['highwarp'].float().cuda()+1)/2 # [-1,1]->[0,1], zeros are pixels without information
        pred = img_logger.log_img(model, batch, model_iteration, split='test', returngrid='train', warpeddepth=None, onlyretimg=True).permute(0,2,3,1) # from chw to hwc, in range -1,1

        for i in range(pred.shape[0]):
            seq_info.set_generated_image(target_idx[i], pred[i])            
            predimg = ((pred[i].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
            Image.fromarray(predimg).save(os.path.join(savepath, 'generations', f'{target_idx[i]}.png'))
            Image.fromarray(predimg).save(os.path.join(seq_info.project_dir, 'sequence', f'{target_idx[i]}.png'))
        for i in range(pred.shape[0]):
            ref = ((refimg[i].cpu().numpy()+1)/2*255).astype(np.uint8)
            Image.fromarray(ref).save(os.path.join(savepath, 'refimgs', f'{target_idx[i]}_{ref_idx[i]}.png'))
            m = ((mask[i].cpu().numpy())*255).astype(np.uint8)
            Image.fromarray(m).save(os.path.join(savepath, 'masks', f'{target_idx[i]}.png'))
