import numpy as np
import torch

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_views(panorama_height, panorama_width, window_size=64, stride=8, loop_closure=False):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    
    if loop_closure:
        # NOTE: Only when height is (8 * window_size)
        assert panorama_height == window_size

        for i in range(window_size // stride - 1):
            h_start = 0
            h_end = window_size
            w_start = int(panorama_width - window_size + (i + 1) * stride)
            w_end = (i + 1) * stride
            views.append((h_start, h_end, w_start, w_end))

    return views


def set_latent_view(latent, h_start, h_end, w_start, w_end):
    '''Set the latent for each window'''
    if w_end > w_start:
        latent_view = latent[:, :, h_start:h_end, w_start:w_end].detach()
    else:
        latent_view = latent[:, :, h_start:h_end, w_start:]
        latent_view = torch.cat([
            latent_view, 
            latent[:, :, h_start:h_end, :w_end]
        ], dim=-1)

    return latent_view

def set_image_view(image, h_start, h_end, w_start, w_end):
    '''Set the latent for each window'''
    if w_end > w_start:
        image_view = image[:, :, h_start:h_end, w_start:w_end].detach()
    else:
        image_view = image[:, :, h_start:h_end, w_start:]
        image_view = torch.cat([
            image_view, 
            image[:, :, h_start:h_end, :w_end]
        ], dim=-1)

    return image_view


def exponential_decay_list(init_weight, decay_rate, num_steps):
    weights = [init_weight * (decay_rate ** i) for i in range(num_steps)]
    return torch.tensor(weights)