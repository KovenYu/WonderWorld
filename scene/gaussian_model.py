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

import numpy as np
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from io import BytesIO
from simple_knn._C import distCUDA2
from utils.general import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system import mkdir_p
from utils.sh import RGB2SH
from utils.graphics import BasicPointCloud
from utils.general import strip_symmetric, build_scaling_rotation, normal2rotation, rotation2normal
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import copy
from tqdm import tqdm

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = lambda x: (torch.tanh(x) * 0.51).clamp(-0.5, 0.5) + 0.5
        self.inverse_opacity_activation = lambda y: torch.atanh((y - 0.5) / 0.51)

        self.rotation_activation = torch.nn.functional.normalize
        self.color_activation = lambda x: (torch.tanh(x) * 0.51).clamp(-0.5, 0.5) + 0.5
        self.inverse_color_activation = lambda y: torch.atanh((y - 0.5) / 0.51)


    def __init__(self, sh_degree : int, previous_gaussian=None, floater_dist2_threshold=0.0002):
        """
        args:
            previous_gaussian : GaussianModel; We take all of its 3DGS particles, freeze them and use them for rendering only.
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.floater_dist2_threshold = floater_dist2_threshold
        self.setup_functions()

        if previous_gaussian is not None:
            self._xyz_prev = torch.cat([previous_gaussian._xyz.detach(), previous_gaussian._xyz_prev], dim=0)
            self._features_dc_prev = torch.cat([previous_gaussian._features_dc.detach(), previous_gaussian._features_dc_prev], dim=0)
            self._scaling_prev = torch.cat([previous_gaussian._scaling.detach(), previous_gaussian._scaling_prev], dim=0)
            self._rotation_prev = torch.cat([previous_gaussian._rotation.detach(), previous_gaussian._rotation_prev], dim=0)
            self._opacity_prev = torch.cat([previous_gaussian._opacity.detach(), previous_gaussian._opacity_prev], dim=0)
            self.filter_3D_prev = torch.cat((previous_gaussian.filter_3D.detach(), previous_gaussian.filter_3D_prev), dim=0)
            self.visibility_filter_all = previous_gaussian.visibility_filter_all
            self.is_sky_filter = previous_gaussian.is_sky_filter
            self.delete_mask_all = previous_gaussian.delete_mask_all
        else:
            self._xyz_prev = torch.empty(0).cuda()
            self._features_dc_prev = torch.empty(0).cuda()
            self._scaling_prev = torch.empty(0).cuda()
            self._rotation_prev = torch.empty(0).cuda()
            self._opacity_prev = torch.empty(0).cuda()
            self.filter_3D_prev = torch.empty(0).cuda()
            self.visibility_filter_all = torch.empty(0, dtype=torch.bool).cuda()
            self.is_sky_filter = torch.empty(0, dtype=torch.bool).cuda()
            self.delete_mask_all = torch.empty(0, dtype=torch.bool).cuda()
            
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        return features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


    @property
    def get_scaling_all(self):
        # return self.scaling_activation(self._scaling)
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_prev], dim=0))
    
    @property
    def get_scaling_with_3D_filter_all(self):
        # scales = self.get_scaling
        scales = self.get_scaling_all
        
        # scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.square(scales) + torch.square(torch.cat([self.filter_3D, self.filter_3D_prev], dim=0))
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation_all(self):
        # return self.rotation_activation(self._rotation)
        return self.rotation_activation(torch.cat([self._rotation, self._rotation_prev], dim=0))
    
    @property
    def get_xyz_all(self):
        # return self._xyz
        return torch.cat([self._xyz, self._xyz_prev], dim=0)
    
    @property
    def get_features_all(self):
        # features_dc = self._features_dc
        features_dc = torch.cat([self._features_dc, self._features_dc_prev], dim=0)
        return features_dc
    
    @property
    def get_opacity_all(self):
        # return self.opacity_activation(self._opacity)
        return self.opacity_activation(torch.cat([self._opacity, self._opacity_prev], dim=0))
    
    @property
    def get_opacity_with_3D_filter_all(self):
        # opacity = self.opacity_activation(self._opacity)
        opacity = self.get_opacity_all
        # apply 3D filter
        # scales = self.get_scaling
        scales = self.get_scaling_all
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        # scales_after_square = scales_square + torch.square(self.filter_3D) 
        scales_after_square = scales_square + torch.square(torch.cat([self.filter_3D, self.filter_3D_prev], dim=0))
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance_all(self, scaling_modifier = 1):
        # return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        return self.covariance_activation(self.get_scaling_all, scaling_modifier, torch.cat([self._rotation, self._rotation_prev], dim=0))


    @torch.no_grad()
    def compute_3D_filter(self, cameras, initialize_scaling=False):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x

            screen_normal = torch.tensor([[0, 0, -1]], device=xyz.device, dtype=torch.float32)
            point_normals_in_screen = rotation2normal(self.get_rotation) @ R
            point_normals_in_screen_xoz = F.normalize(point_normals_in_screen[:, [0, 2]], dim=1)
            screen_normal_xoz = F.normalize(screen_normal[:, [0, 2]], dim=1)
            cos_xz = torch.sum(point_normals_in_screen_xoz * screen_normal_xoz, dim=1)
            # assert torch.all(cos_xz >= 0), "All normals should be in the same direction of the screen normal. Current min value: {}".format(cos_xz.min())
            point_normals_in_screen_yoz = F.normalize(point_normals_in_screen[:, [1, 2]], dim=1)
            screen_normal_yoz = F.normalize(screen_normal[:, [1, 2]], dim=1)
            cos_yz = torch.sum(point_normals_in_screen_yoz * screen_normal_yoz, dim=1)
            # assert torch.all(cos_yz >= 0), "All normals should be in the same direction of the screen normal. Current min value: {}".format(cos_yz.min())
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length
        self.filter_3D = filter_3D[..., None]

        x_scale = distance / focal_length / cos_xz.clamp(min=1e-1)
        y_scale = distance / focal_length / cos_yz.clamp(min=1e-1)

        if initialize_scaling:
            print('Initializing scaling...')
            dist_scales = torch.exp(self._scaling)
            nyquist_scales = self.filter_3D.clone().repeat(1, 3)
            nyquist_scales[:, 0:1] = x_scale[..., None]
            nyquist_scales[:, 1:2] = y_scale[..., None]
            nyquist_scales *= 0.7
            scaling = torch.log(nyquist_scales)
            # scaling[:, 2] = torch.log(torch.tensor(0))
            # mixed_scales = (dist_scales * nyquist_scales).sqrt()
            # scaling = torch.log(mixed_scales)
            optimizable_tensors = self.replace_tensor_to_optimizer(scaling, 'scaling')
            self._scaling = optimizable_tensors['scaling']
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, is_sky: bool = False):
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        floater_mask = dist2 > self.floater_dist2_threshold
        print("Floater ratio: {} %".format(floater_mask.float().mean().item()*100))
        dist2 = dist2[~floater_mask]
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()[~floater_mask]
        fused_color = self.inverse_color_activation((torch.tensor(np.asarray(pcd.colors)).float().cuda() * 1.01).clamp(0, 1))[~floater_mask]
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales[:, 2] = torch.log(torch.tensor(0))
        # pcd.normals[:] = pcd.normals.mean()
        normals = pcd.normals
        
        rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")[~floater_mask]
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.15 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if self._xyz.numel() == 0:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
        else:
            print("Adding these points to the existing model that has ", self.get_xyz.shape[0], " points")
            self._xyz = nn.Parameter(torch.cat((self._xyz, fused_point_cloud), dim=0).requires_grad_(True))
            self._features_dc = nn.Parameter(torch.cat((self._features_dc, features[:,:,0:1].transpose(1, 2).contiguous()), dim=0).requires_grad_(True))
            self._scaling = nn.Parameter(torch.cat((self._scaling, scales), dim=0).requires_grad_(True))
            self._rotation = nn.Parameter(torch.cat((self._rotation, rots), dim=0).requires_grad_(True))
            self._opacity = nn.Parameter(torch.cat((self._opacity, opacities), dim=0).requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        visibility_filter_current = torch.ones((fused_point_cloud.shape[0]), device="cuda").bool()
        visibility_filter_prev = self.visibility_filter_all
        self.visibility_filter_all = torch.cat((visibility_filter_current, visibility_filter_prev), dim=0)

        delete_mask_current = torch.zeros((fused_point_cloud.shape[0]), device="cuda").bool()
        delete_mask_prev = self.delete_mask_all
        self.delete_mask_all = torch.cat((delete_mask_current, delete_mask_prev), dim=0)
        
        is_sky_filter_prev = self.is_sky_filter
        if is_sky:
            is_sky_filter_current = torch.ones((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        else:
            is_sky_filter_current = torch.zeros((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        self.is_sky_filter = torch.cat((is_sky_filter_current, is_sky_filter_prev), dim=0)
    
    @torch.no_grad()
    def delete_points(self, tdgs_cam):
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here

        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen_x = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        in_screen_y = torch.logical_and(y >= 0, y < tdgs_cam.image_height)
        in_screen = torch.logical_and(in_screen_x, in_screen_y)
        
        delete_mask = torch.logical_and(in_screen, ~self.is_sky_filter)
        self.delete_mask_all = self.delete_mask_all | delete_mask
        
    @torch.no_grad()
    def set_inscreen_points_to_visible(self, tdgs_cam):
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here

        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        self.visibility_filter_all = self.visibility_filter_all | in_screen

    def my_load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(0., 0.99))
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False, use_higher_freq=True):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def yield_splat_data_raw(self, path):
        print('yielding splat data raw...')
        def apply_activation(x):
            return np.clip(np.tanh(x) * 0.51, -0.5, 0.5) + 0.5
        
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(), self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()], dim=0).cpu().numpy()
        current_opacity_with_filter = self.get_opacity_with_3D_filter_all
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()        
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        vert = el
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) * apply_activation(vert["opacity"])
        )
        buffer = BytesIO()
        
        for idx in tqdm(sorted_indices):
            v = el[idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            color = np.array(
                [
                    apply_activation(v["f_dc_0"]),
                    apply_activation(v["f_dc_1"]),
                    apply_activation(v["f_dc_2"]),
                    apply_activation(v["opacity"]),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        splat_data = buffer.getvalue()
        with open(path, 'wb') as f:
            f.write(splat_data)
        print('splat data raw yielded')
        return splat_data
    
    def yield_splat_data(self, path):
        print('yielding splat data...')
        def apply_activation(x):
            return np.clip(np.tanh(x) * 0.51, -0.5, 0.5) + 0.5
        
        # filter_all = ~self.delete_mask_all & (~self.is_sky_filter)
        filter_all = ~self.delete_mask_all
        filter_all = filter_all.cpu()
        
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        xyz = xyz[filter_all]
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(), self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()], dim=0).cpu().numpy()
        f_dc = f_dc[filter_all]
        current_opacity_with_filter = self.get_opacity_with_3D_filter_all
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()        
        opacities = opacities[filter_all]
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()
        scale = scale[filter_all]
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()
        rotation = rotation[filter_all]
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()
        filters_3D = filters_3D[filter_all]
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        vert = el
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) * apply_activation(vert["opacity"])
        )
        buffer = BytesIO()
        
        for idx in tqdm(sorted_indices):
            v = el[idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            color = np.array(
                [
                    apply_activation(v["f_dc_0"]),
                    apply_activation(v["f_dc_1"]),
                    apply_activation(v["f_dc_2"]),
                    apply_activation(v["opacity"]),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        splat_data = buffer.getvalue()
        with open(path, 'wb') as f:
            f.write(splat_data)
        print('splat data yielded')
        return splat_data

    def save_ply(self, path, use_higher_freq=True, use_splat=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(), self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()], dim=0).cpu().numpy()
        current_opacity_with_filter = torch.cat([self.get_opacity_with_3D_filter, self.get_opacity_with_3D_filter_all], dim=0)
        opacities = self.inverse_opacity_activation(current_opacity_with_filter).detach().cpu().numpy()
        scale = torch.cat([self.scaling_inverse_activation(self.get_scaling_with_3D_filter), self.scaling_inverse_activation(self.get_scaling_with_3D_filter_all)], dim=0).cpu().numpy()
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True, use_higher_freq=use_higher_freq)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        if use_splat:
            vert = el
            sorted_indices = np.argsort(
                -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
                / (1 + np.exp(-vert["opacity"]))
            )
            buffer = BytesIO()
            for idx in sorted_indices:
                v = el[idx]
                position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
                scales = np.exp(
                    np.array(
                        [v["scale_0"], v["scale_1"], v["scale_2"]],
                        dtype=np.float32,
                    )
                )
                rot = np.array(
                    [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                    dtype=np.float32,
                )
                SH_C0 = 0.28209479177387814
                color = np.array(
                    [
                        0.5 + SH_C0 * v["f_dc_0"],
                        0.5 + SH_C0 * v["f_dc_1"],
                        0.5 + SH_C0 * v["f_dc_2"],
                        1 / (1 + np.exp(-v["opacity"])),
                    ]
                )
                buffer.write(position.tobytes())
                buffer.write(scales.tobytes())
                buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
                buffer.write(
                    ((rot / np.linalg.norm(rot)) * 128 + 128)
                    .clip(0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )

            splat_data = buffer.getvalue()
            with open(path, "wb") as f:
                f.write(splat_data)
        else:
            PlyData([el]).write(path)
    
    def save_ply_with_filter(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filters_3D = self.filter_3D.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)

    def save_ply_all_with_filter(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(), self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()], dim=0).cpu().numpy()
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()        
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
        
    def load_ply_with_filter(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(np.asarray(plydata.elements[0]["filter_3D"]), dtype=torch.float, device="cuda")[:, None]

        self.active_sh_degree = self.max_sh_degree
        
    def save_ply_combined(self, gaussian, path, use_higher_freq=True, use_splat=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz_1 = self._xyz.detach().cpu().numpy()
        xyz_2 = gaussian._xyz.detach().cpu().numpy()
        xyz = np.concatenate((xyz_1, xyz_2), axis=0)
        normals = np.zeros_like(xyz)
        f_dc_1 = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_2 = gaussian._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = np.concatenate((f_dc_1, f_dc_2), axis=0)
    
        current_opacity_with_filter_1 = self.get_opacity_with_3D_filter
        opacities_1 = self.inverse_opacity_activation(current_opacity_with_filter_1).detach().cpu().numpy()
        current_opacity_with_filter_2 = gaussian.get_opacity_with_3D_filter
        opacities_2 = self.inverse_opacity_activation(current_opacity_with_filter_2).detach().cpu().numpy()
        opacities = np.concatenate((opacities_1, opacities_2), axis=0)
    
        scale_1 = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        scale_2 = gaussian.scaling_inverse_activation(gaussian.get_scaling_with_3D_filter).detach().cpu().numpy()
        scale = np.concatenate((scale_1, scale_2), axis=0)
    
        rotation_1 = self._rotation.detach().cpu().numpy()
        rotation_2 = gaussian._rotation.detach().cpu().numpy()
        rotation = np.concatenate((rotation_1, rotation_2), axis=0)
    
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True, use_higher_freq=use_higher_freq)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        if use_splat:
            vert = el
            sorted_indices = np.argsort(
                -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
                / (1 + np.exp(-vert["opacity"]))
            )
            buffer = BytesIO()
            for idx in sorted_indices:
                v = el[idx]
                position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
                scales = np.exp(
                    np.array(
                        [v["scale_0"], v["scale_1"], v["scale_2"]],
                        dtype=np.float32,
                    )
                )
                rot = np.array(
                    [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                    dtype=np.float32,
                )
                SH_C0 = 0.28209479177387814
                color = np.array(
                    [
                        0.5 + SH_C0 * v["f_dc_0"],
                        0.5 + SH_C0 * v["f_dc_1"],
                        0.5 + SH_C0 * v["f_dc_2"],
                        1 / (1 + np.exp(-v["opacity"])),
                    ]
                )
                buffer.write(position.tobytes())
                buffer.write(scales.tobytes())
                buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
                buffer.write(
                    ((rot / np.linalg.norm(rot)) * 128 + 128)
                    .clip(0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )

            splat_data = buffer.getvalue()
            with open(path, "wb") as f:
                f.write(splat_data)
        else:
            PlyData([el]).write(path)
            
    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = inverse_sigmoid(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    self.optimizer.state[group['params'][0]] = stored_state
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if len(valid_points_mask) < len(self.visibility_filter_all):  # Assuming that visibility filter is arranged such that current points have smaller index
            current = self.visibility_filter_all[:len(valid_points_mask)]
            prev = self.visibility_filter_all[len(valid_points_mask):]
            self.visibility_filter_all = torch.cat((current[valid_points_mask], prev), dim=0)
            current_sky = self.is_sky_filter[:len(valid_points_mask)]
            prev_sky = self.is_sky_filter[len(valid_points_mask):]
            self.is_sky_filter = torch.cat((current_sky[valid_points_mask], prev_sky), dim=0)
            current_delete_mask = self.delete_mask_all[:len(valid_points_mask)]
            prev_delete_mask = self.delete_mask_all[len(valid_points_mask):]
            self.delete_mask_all = torch.cat((current_delete_mask[valid_points_mask], prev_delete_mask), dim=0)
        else:
            self.visibility_filter_all = self.visibility_filter_all[valid_points_mask]
            self.is_sky_filter = self.is_sky_filter[valid_points_mask]
            self.delete_mask_all = self.delete_mask_all[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        n_added_points = new_xyz.shape[0] - self.get_xyz.shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if n_added_points > 0:
            assert len(self.visibility_filter_all) == 0, 'We have not yet implemented visibility filter densification.'

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1