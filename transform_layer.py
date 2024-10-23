# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

def homogenize_mat(mat):
    ''' Adds a row (bottom) and column (right) to the matrix `mat` with all zeros and a 1 in the lower right corner.
    Is used to make the matrix work as transformation for homogeneous coordinates. '''
    assert torch.is_tensor(mat)
    ret = torch.eye(mat.size(-1)+1, dtype=mat.dtype, device=mat.device)
    if mat.ndim > 2:
        flat_mat = mat.view(-1, mat.size(-2), mat.size(-1))
        num_mats = flat_mat.size(0)
        ret = ret[None].expand(num_mats, -1, -1)
    else: flat_mat = mat
    ret[..., :mat.size(-2), :mat.size(-1)] = flat_mat
    return ret.reshape(*mat.shape[:-2], mat.size(-2)+1, mat.size(-1)+1)

def homogenize_vec(vec, dim=None):
    ''' Adds an additional component to `vec` with value 1 to make it a homogeneous coordinate. '''
    assert torch.is_tensor(vec) and 3 in vec.shape
    if dim is None: dim = vec.ndim - list(reversed(vec.shape)).index(3) - 1
    ad_shape = list(vec.shape); ad_shape[dim] = 1
    nu = torch.ones(ad_shape, dtype=vec.dtype, device=vec.device)
    return torch.cat([vec, nu], dim=dim)

class Transform3D(nn.Module):
    def __init__(self, im_feat_dim):
        super().__init__()
        self.pos = nn.Linear(im_feat_dim, 2)

    def get_rot_mat(self, mu, rho):
        ''' Computes a batch of rotation matrices from a batch of angles `mu` and `rho`

        Args:
            mu (Tensor): azimuth angle. Shape (BS)
            rho (Tensor): elevation angle. Shape (BS)

        Returns:
            Tensor: Rotation matrices of shape (BS, 4, 4)
        '''
        zero, one = torch.zeros_like(mu), torch.ones_like(mu)
        rot_x = torch.stack([
            torch.stack([one,   zero,           zero,           zero], dim=-1),
            torch.stack([zero,  torch.cos(rho), torch.sin(rho), zero], dim=-1),
            torch.stack([zero, -torch.sin(rho), torch.cos(rho), zero], dim=-1),
            torch.stack([zero,  zero,           zero,            one], dim=-1)
        ], dim=-1)
        rot_y = torch.stack([
            torch.stack([torch.cos(mu), zero, -torch.sin(mu), zero], dim=-1),
            torch.stack([zero,          one,   zero,          zero], dim=-1),
            torch.stack([torch.sin(mu), zero,  torch.cos(mu), zero], dim=-1),
            torch.stack([zero,          zero,  zero,           one], dim=-1)
        ], dim=-1)

    def forward(self, x, im_feat):
        mu, rho = self.pos(im_feat).permute(1,0)
        X = torch.linspace(-1, 1, x.size(-1))
        Y = torch.linspace(-1, 1, x.size(-2))
        Z = torch.linspace(-1, 1, x.size(-3))

        x, y, z = torch.meshgrid(X, Y, Z)
        coords = torch.stack([x, y, z], dim=-1)

        ## TODO: create rotation matrices, apply to coords


        return F.grid_sample(x, coords)

# %%
