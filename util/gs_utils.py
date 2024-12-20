from plyfile import PlyData, PlyElement
import numpy as np
import torch
import torch.nn as nn
import os, sys
from io import BytesIO

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

class Gaussians(nn.Module):
    def __init__(self, max_sh_degree=3):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


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

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cpu"))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cpu").transpose(1, 2).contiguous())
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cpu"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cpu"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cpu"))

        self.active_sh_degree = self.max_sh_degree

    def load_my_pc(self, path, depth_cutoff=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        red = np.asarray(plydata.elements[0]["red"], dtype=np.float32) / 255.0
        green = np.asarray(plydata.elements[0]["green"], dtype=np.float32) / 255.0
        blue = np.asarray(plydata.elements[0]["blue"], dtype=np.float32) / 255.0

        rgb_dc = np.zeros((xyz.shape[0], 3, 1))
        rgb_dc[:, 0, 0] = red
        rgb_dc[:, 1, 0] = green
        rgb_dc[:, 2, 0] = blue
        f_dc = RGB2SH(rgb_dc)

        if depth_cutoff is not None:
            valid_mask = xyz[:, 2] < depth_cutoff
            xyz = xyz[valid_mask]
            f_dc = f_dc[valid_mask]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cpu"))
        self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous())

    def load_from_my_point_cloud(self, xyz, rgb):
        """
        xyz (torch.Tensor): The point cloud tensor of shape [N, 3].
        rgb (torch.Tensor): The color tensor of shape [N, 3] with values in [0, 1].
        """
        rgb_dc = rgb[..., None]  # [N, 3, 1]
        f_dc = RGB2SH(rgb_dc)

        self._xyz = nn.Parameter(xyz.cpu())
        self._features_dc = nn.Parameter(f_dc.cpu().transpose(1, 2).contiguous())

    def set_gs(self, opacity_values=0.99, scale_values=2.7):
        xyz = self._xyz.detach().cpu().numpy()

        # Set opacity to 1, will be activated by sigmoid
        opacities_values = np.ones((xyz.shape[0], 1)) * opacity_values
        opacities = self.inverse_opacity_activation(torch.tensor(opacities_values, dtype=torch.float, device="cpu"))

        # Set scaling to 1, will be activated by exp
        scales_values = np.ones((xyz.shape[0], 3)) * scale_values
        scales = self.scaling_inverse_activation(torch.tensor(scales_values, dtype=torch.float, device="cpu"))

        # Set rotation to identity quaternion
        rots = np.zeros((xyz.shape[0], 4))
        rots[:, 0] = 1.0

        self._opacity = nn.Parameter(opacities)
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cpu"))

    def save_ply(self, path, use_splat=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
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

    def yield_splat_data(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
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
        return splat_data

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


@torch.no_grad()
def save_pc_as_3dgs(save_path, xyz, rgb, opacity_values=0.99, scale_values=5e-4):
    """
    xyz (torch.Tensor): The point cloud tensor of shape [N, 3].
    rgb (torch.Tensor): The color tensor of shape [N, 3] with values in [0, 1].
    opacity_values (float): The actual opacity value (after activation) to use for the splatting.
    scale_values (float): The actual scale value (after activation) to use for the splatting.
    """
    use_splat = str(save_path).endswith('.splat')
    gaussians = Gaussians()
    gaussians.load_from_my_point_cloud(xyz, rgb)
    gaussians.set_gs(opacity_values=opacity_values, scale_values=scale_values)
    gaussians.save_ply(save_path, use_splat=use_splat)

@torch.no_grad()
def convert_pc_to_splat(xyz, rgb, opacity_values=0.99, scale_values=5e-4):
    """
    xyz (torch.Tensor): The point cloud tensor of shape [N, 3].
    rgb (torch.Tensor): The color tensor of shape [N, 3] with values in [0, 1].
    opacity_values (float): The actual opacity value (after activation) to use for the splatting.
    scale_values (float): The actual scale value (after activation) to use for the splatting.
    """
    gaussians = Gaussians()
    gaussians.load_from_my_point_cloud(xyz, rgb)
    gaussians.set_gs(opacity_values=opacity_values, scale_values=scale_values)
    splat_data = gaussians.yield_splat_data()
    return splat_data

def process_ply_to_splat(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
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

    return buffer.getvalue()

def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)


if __name__ == "__main__":
    gaussians = Gaussians()
    my_pc_path = '/viscam/projects/wonderland/gaussian-splatting/my_point_clouds/kf1_point_cloud_marigold.ply'
    gaussians.load_my_pc(my_pc_path, depth_cutoff=None)
    xyz_max_value = 20
    xyz_scale = xyz_max_value / gaussians._xyz[:, 2].max()
    scale_values = xyz_max_value / 1000
    gaussians.set_gs(xyz_scale=xyz_scale, opacity_values=0.99, scale_values=scale_values)
    my_pc_save_path = '/viscam/projects/wonderland/gaussian-splatting/my_point_clouds/kf1_point_cloud_marigold_3dgs.ply'
    gaussians.save_ply(my_pc_save_path)