import torch
import random
import scipy.ndimage as ndimage
from einops import repeat
# exemplar torch C H W
def grid_map_generate(exemplar, grid_size=4, min_rate=0.1, max_rate = 0.5):
    C, H, W = exemplar.shape
    grid_H = ((H // 4) + 1) * grid_size
    grid_W = ((W // 4) + 1) * grid_size

    grid_num_H = 384 // grid_H
    grid_num_W = 384 // grid_W
    
    xs = torch.arange(0, grid_num_W * grid_W, grid_W)
    ys = torch.arange(0, grid_num_H * grid_H, grid_H)

    rate = (random.random() * max_rate) + min_rate
    
    # grid_x, grid_y = torch.meshgrid(x, y)
    grid_map = torch.zeros(3,384,384)
    density_map = torch.zeros(384,384)
    
    block = torch.zeros(3,grid_H,grid_W)
    block[:,:H,:W] = exemplar
    grid_map_exp = repeat(block,'c h w->c (nh h) (nw w)',nh=grid_num_H, nw= grid_num_W)
    grid_map[:,:int(grid_num_H*grid_H),:int(grid_num_W*grid_W)] = grid_map_exp

    density_map_block = torch.zeros(grid_H,grid_W)
    density_map_block[int(grid_H/2),int(grid_W/2)] = 1
    density_map_exp = repeat(density_map_block,'h w->(nh h) (nw w)',nh=grid_num_H, nw= grid_num_W)
    density_map[:int(grid_num_H*grid_H),:int(grid_num_W*grid_W)] = density_map_exp

    reresized_density = ndimage.gaussian_filter(density_map.numpy(), sigma=(1, 1), order=0)
    reresized_density = reresized_density * 60
    reresized_density = torch.from_numpy(reresized_density)
    return grid_map, reresized_density

if __name__ == "__main__":
    exemplar = torch.rand(3,14,10)
    grid_map_generate(exemplar=exemplar)
