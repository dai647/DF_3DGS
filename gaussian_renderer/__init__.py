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

import torch
import math
from scene.gaussian_model import GaussianModel
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
def gsplat_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    img_height = int(viewpoint_camera.image_height)
    img_width = int(viewpoint_camera.image_width)

    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
        means3d=pc.get_xyz,
        scales=pc.get_scaling,
        glob_scale=scaling_modifier,
        quats=pc.get_rotation,
        viewmat=viewpoint_camera.world_view_transform.T,
        fx=focal_length_x,
        fy=focal_length_y,
        cx=img_width / 2.,
        cy=img_height / 2.,
        img_height=img_height,
        img_width=img_width,
        block_width=16,
    )

    try:
        xys.retain_grad()
    except:
        pass


    def rasterize_features(input_features, bg):
        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            input_features,
            pc.get_opacity,
            img_height=img_height,
            img_width=img_width,
            block_width=16,
            background=bg,
            return_alpha=False,
        ).permute(2, 0, 1)


    semantic_features = pc.get_semantic_feature.squeeze(1)
    output_semantic_feature_map_list = []
    chunk_size = 64 if semantic_features.shape[-1]%4==0 else 3
    bg_color = torch.zeros((chunk_size,), dtype=torch.float, device=bg_color.device)
    for i in range(semantic_features.shape[-1] // chunk_size):
        start = i * chunk_size
        output_semantic_feature_map_list.append(rasterize_features(
            semantic_features[..., start:start + chunk_size],
            bg_color,
        ))
    feature_map = torch.concat(output_semantic_feature_map_list, dim=0)

    return {

        'feature_map': feature_map,
        "viewspace_points": xys,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
