#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from gaussian_renderer import gsplat_render as render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import sys
from  autoencoder.model import Autoencoder
import torch.nn as nn
import sklearn
import sklearn.decomposition
from tqdm import tqdm


def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature


def test(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,model_path):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,load_iteration=30000)
    gaussians.training_setup(opt)
    gaussians.active_sh_degree=gaussians.max_sh_degree
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    first_iter += 1

    sence_name=model_path.split('/')[1]

    print('-------Doing sence: ',sence_name)
    if '512' not in model_path:
        no_AE=False
    else:
        no_AE=True
    print('no_AE: ',no_AE)
    
    checkpoint=torch.load(f'autoencoder/Sences/{sence_name}/latest_AE_ckpt.pth')

    encoder_hidden_dims = [256, 128, 64, 32, 9]
    decoder_hidden_dims = [32, 64, 128, 256, 1024, 512]
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    model.load_state_dict(checkpoint)
    model.eval()

    os.makedirs(f'{model_path}/render/feature_map', exist_ok=True)
    os.makedirs(f'{model_path}/render/feature_map_vis', exist_ok=True)

    
    with torch.no_grad():
        viewpoint_stack = scene.getTrainCameras().copy()
        num=len(viewpoint_stack)

        for view_id in tqdm(range(num)) :
            
            viewpoint_cam = viewpoint_stack.pop()
            image_name=viewpoint_cam.image_name

            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            feature_map,  viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(360, 480), mode='bilinear', align_corners=True).squeeze(0)
            ch,h,w=feature_map.shape

            if no_AE:
                feature_map=feature_map
            else:
                feature_map=model.decode(feature_map.permute(1,2,0)).permute(2,0,1)
            
            

            torch.save(feature_map,f'{model_path}/render/feature_map/'+image_name+'.pt')
            feature_map_vis2 = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis2.cpu().numpy() * 255).astype(np.uint8)).save(f'{model_path}/render/feature_map_vis/'+image_name+'.png')



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    test(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.model_path)
  
    # All done
    print("\nTesting complete.")
