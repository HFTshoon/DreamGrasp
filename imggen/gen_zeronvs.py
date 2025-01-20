from torch.utils.data import DataLoader

# add sys path for dust3r
import sys
sys.path.append("imggen/megascenes")
from ldm.util import instantiate_from_config
from ldm.logger import ImageLogger
from dataloader import *

def load_gen_model(output_dir, model_dir, device):
    config_file = yaml.safe_load(open("imggen/megascenes/configs/warp_plus_pose/config.yaml"))
    model = instantiate_from_config(config_file['model']).eval().to(device)
    model.train()
    model.load_state_dict(torch.load(model_dir, map_location=device))

    img_logger = ImageLogger(log_directory=output_dir, log_images_kwargs=train_configs['log_images_kwargs'])
    return {
        "model": model,
        "img_logger": img_logger
    }

def choose_idx_to_generate(seq_info, known_area_ratio):
    generate_cnt = 2 ** seq_info.cur_stage
    min_known_area_ratio = 0.5

    # get lowest known area ratio that is greater than min_known_area_ratio
    candidates = [(i, known_area_ratio[i]) for i in range(seq_info.length) if known_area_ratio[i] >= min_known_area_ratio and seq_info.views[i].stage == -1]
    candidates.sort(key=lambda x: x[1])
    
    generate_idx = [candidates[i][0] for i in range(generate_cnt)]

    if len(generate_idx) < generate_cnt:
        additional_candidates = [(i, known_area_ratio[i]) for i in range(seq_info.length) if known_area_ratio[i] < min_known_area_ratio and seq_info.views[i].stage == -1]
        additional_candidates.sort(key=lambda x: -x[1])
        additional_generate_cnt = generate_cnt - len(generate_idx)
        generate_idx += [additional_candidates[i][0] for i in range(additional_generate_cnt)]

    print(f"Generate {generate_cnt} images:")
    for idx in generate_idx:
        print(f"  {idx} -> {known_area_ratio[idx]}")
    return generate_idx

def generate_images(gen_model, seq_info, generate_idx):
    for idx in generate_idx:
        overlaps = seq_info.views[idx].overlaps
        # get biggest overlap
        reference_idx = overlaps.index(max(overlaps))
        print(f"Generate image {idx} from reference {reference_idx} ({overlaps[reference_idx]})")

    dataset = TempPariedDataset(seq_info, generate_idx)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4,
        drop_last=False, shuffle=False, persistent_workers=False, pin_memory=False
    )

    for ii, batch in enumerate(dataloader):
        batch, dataidx = batch

        refimg = batch["image_ref"].cuda().float()
        tarimg = batch["image_target"].cuda().float()
        mask = (batch['highwarp'].float().cuda()+1)/2 # [-1,1]->[0,1], zeros are pixels without information
        pred = img_logger.log_img(model, batch, resume, split='test', returngrid='train', warpeddepth=None, onlyretimg=True).permute(0,2,3,1) # from chw to hwc, in range -1,1

        if args.save_generations:
            for i in range(pred.shape[0]):
                if dataidx[i]%args.savefreq==0:
                    predimg = ((pred[i].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                    Image.fromarray(predimg).save(join(savepath, 'generations', f'{dataidx[i]}.png'))
        if args.save_data:
            for i in range(pred.shape[0]):
                if dataidx[i]%args.savefreq==0:
                    ref = ((refimg[i].cpu().numpy()+1)/2*255).astype(np.uint8)
                    Image.fromarray(ref).save(join(savepath, 'refimgs', f'{dataidx[i]}.png'))
                    tar = ((tarimg[i].cpu().numpy()+1)/2*255).astype(np.uint8)
                    Image.fromarray(tar).save(join(savepath, 'tarimgs', f'{dataidx[i]}.png'))
                    m = ((mask[i].cpu().numpy())*255).astype(np.uint8)
                    Image.fromarray(m).save(join(savepath, 'masks', f'{dataidx[i]}.png'))
