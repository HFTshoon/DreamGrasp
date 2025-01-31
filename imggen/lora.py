import os
import yaml
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import sys
sys.path.append("imggen/megascenes")
from ldm.util import instantiate_from_config
from dataloader import LoraPairedDataset
from ldm.logger import ImageLogger

from minlora import get_lora_state_dict

def train_lora(gen_model, seq_info):
    config_dir = "imggen/megascenes/configs/warp_plus_pose"
    batch_size = 1
    workers = 4
    dataset_dir = seq_info.project_dir
    exp_dir = os.path.join(seq_info.project_dir, "lora")

    config_file = yaml.safe_load(open(os.path.join(config_dir, 'config.yaml')))
    train_configs = config_file.get('training', {})

    os.makedirs(exp_dir, exist_ok=True)

    dataset = LoraPairedDataset(seq_info)

    print("size of dataset: ", len(dataset))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers,
        drop_last=True, shuffle=True, persistent_workers=(workers!=0)
    )

    img_logger = ImageLogger(log_directory=exp_dir, log_images_kwargs=train_configs['log_images_kwargs'])
  
    # load model
    model = gen_model["model"]
    model.train()
    model.learning_rate = float(train_configs.get('learning_rate', 1e-4))
    
    total_iterations = 100
    optimizer, scheduler = model.configure_optimizers(use_lora=True)

    # accelerator = gen_model["accelerator"]
    # model, dataloader, optimizer, scheduler = accelerator.prepare( model, dataloader, optimizer, scheduler )
    module = model.module if isinstance(model, DistributedDataParallel) else model
    starting_iter = 0
    num_processes = 1
    print("number of acc processes: ", num_processes)

    # start training loop
    progress_bar = tqdm(initial=starting_iter, total=total_iterations, disable=False)
    global_step = starting_iter
    local_step = 0

    while True:
        progress_bar.set_description(f"Training step {global_step}")
        for _, batch in enumerate(dataloader):
            if local_step == 0: # log image in the beginning for sanity check and comparisons
                grid = img_logger.log_img(module, batch, global_step, split='test', returngrid='train', has_target=True)
                # accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                #     grid = img_logger.log_img(module, batch, global_step, split='test', returngrid='train', has_target=True)
                #     accelerator.log( {"train_table":log_image_table(grid)} )

            loss, loss_dict = model(batch)
            progress_bar.set_postfix(loss_dict)

            loss.backward()
            optimizer.step()
            scheduler.step()    
            optimizer.zero_grad()
            progress_bar.update(num_processes)
            global_step += num_processes
            local_step += 1

            lr = optimizer.param_groups[0]['lr']
            loss_dict.update({'lr': lr}) 

            # with accelerator.accumulate(model):
            #     loss, loss_dict = model(batch)
            #     progress_bar.set_postfix(loss_dict)

            #     accelerator.backward(loss)
            #     optimizer.step()
            #     scheduler.step()    
            #     optimizer.zero_grad()
            #     progress_bar.update(num_processes)
            #     global_step += num_processes
            #     local_step += 1

            #     lr = optimizer.param_groups[0]['lr']
            #     loss_dict.update({'lr': lr}) 

            if global_step >= total_iterations:                        
                grid = img_logger.log_img(module, batch, global_step, split='test', returngrid='train', has_target=True)
                lora_state_dict = get_lora_state_dict(model)
                # save lora state dict
                torch.save(lora_state_dict, os.path.join(exp_dir, "lora_state_dict.pth"))
                breakpoint()
                print("Training complete!")
                model.eval()
                gen_model["model"] = model
                return gen_model

            # if global_step >= total_iterations:
            #     accelerator.wait_for_everyone()
            #     if accelerator.is_main_process:                        
            #         grid = img_logger.log_img(module, batch, global_step, split='test', returngrid='train', has_target=True)
            #         accelerator.log( {"train_table":log_image_table(grid)} )
            #     lora_state_dict = get_lora_state_dict(model)
            #     # save lora state dict
            #     torch.save(lora_state_dict, os.path.join(exp_dir, "lora_state_dict.pth"))
            #     breakpoint()
            #     accelerator.end_training()
            #     print("Training complete!")
            #     model.eval()
            #     gen_model["model"] = model
            #     return gen_model


def log_image_table(grid, test=False):
    column = "cond/target/sample" if not test else "cond/samples"
    table = wandb.Table(columns=[column])
    for g in grid:
        table.add_data(wandb.Image(g))
    return table