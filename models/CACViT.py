from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange,repeat
from timm.models.vision_transformer import PatchEmbed
from models.Block.Blocks import Block
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np

class SupervisedMAE(nn.Module):
    """ CntVit with VisionTransformer backbone
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate = 0):
        super().__init__()
        ## Setting the model
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        ## Global Setting
        self.patch_size = patch_size
        self.img_size = img_size
        ex_size = 64
        self.norm_pix_loss = norm_pix_loss
        ## Global Setting

        ## Encoder specifics
        self.scale_embeds = nn.Linear(2,embed_dim,bias = True)
        self.patch_embed_exemplar = PatchEmbed(ex_size, patch_size, in_chans+1, embed_dim)
        num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, embed_dim), requires_grad=False)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
        Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        for i in range(depth)])

        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        ### decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        ### decoder blocks
        ## Decoder specifics
        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(513, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  
        ## Regressor
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w1 = self.patch_embed_exemplar.proj.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scale_embedding(self, exemplars, scale_infos):
        method = 1
        if method == 0:
            bs, n, c, h, w = exemplars.shape
            scales_batch = []
            for i in range(bs): 
                scales = []
                for j in range(n):
                    w_scale = torch.linspace(0,scale_infos[i,j,0],w)
                    w_scale = repeat(w_scale,'w->h w',h=h).unsqueeze(0)
                    h_scale = torch.linspace(0,scale_infos[i,j,1],h)
                    h_scale = repeat(h_scale,'h->h w',w=w).unsqueeze(0)
                    scale = torch.cat((w_scale,h_scale),dim=0)
                    scales.append(scale)
                scales = torch.stack(scales)
                scales_batch.append(scales)
            scales_batch = torch.stack(scales_batch)

        if method == 1:
            bs, n, c, h, w = exemplars.shape
            scales_batch = []
            for i in range(bs): 
                scales = []
                for j in range(n):
                    w_scale = torch.linspace(0,scale_infos[i,j,0],w)
                    w_scale = repeat(w_scale,'w->h w',h=h).unsqueeze(0)
                    h_scale = torch.linspace(0,scale_infos[i,j,1],h)
                    h_scale = repeat(h_scale,'h->h w',w=w).unsqueeze(0)
                    scale = w_scale + h_scale
                    scales.append(scale)
                scales = torch.stack(scales)
                scales_batch.append(scales)
            scales_batch = torch.stack(scales_batch)

        scales_batch = scales_batch.to(exemplars.device)
        exemplars = torch.cat((exemplars,scales_batch),dim=2)

        return exemplars
    
    def forward_encoder(self, x, y, scales=None):
        y_embed = []
        y = rearrange(y,'b n c w h->n b c w h')
        for box in y:
            box = self.patch_embed_exemplar(box)
            box = box + self.pos_embed_exemplar
            y_embed.append(box)
        y_embed = torch.stack(y_embed, dim=0)
        box_num,_,n,d = y_embed.shape
        y = rearrange(y_embed, 'box_num batch n d->batch (box_num  n) d')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        _, l, d = x.shape
        attns = []
        x_y = torch.cat((x,y),axis=1)
        for i, blk in enumerate(self.blocks):
            x_y, attn = blk(x_y)
            attns.append(attn)
        x_y = self.norm(x_y)    
        x = x_y[:,:l,:]
        for i in range(box_num):
            y[:,i*n:(i+1)*n,:] = x_y[:,l+i*n:l+(i+1)*n,:]
        y = rearrange(y,'batch  (box_num  n) d->box_num batch n d',box_num = box_num,n=n)
        return x, y
    
    def forward_decoder(self,x,y,scales=None):
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        b,l_x,d = x.shape
        y_embeds = []
        num, batch, l, dim = y.shape
        for i in range(num):
            y_embed = self.decoder_embed(y[i])
            y_embed = y_embed + self.decoder_pos_embed_exemplar
            y_embeds.append(y_embed)
        y_embeds = torch.stack(y_embeds)
        num, batch, l, dim = y_embeds.shape
        y_embeds = rearrange(y_embeds,'n b l d -> b (n l) d')
        x = torch.cat((x,y_embeds),axis=1)
        attns = []
        xs = []
        ys = []
        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == 2:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:,:l_x,:])
            ys.append(x[:,l_x:,:])
        return xs,ys,attns

    def AttentionEnhance(self, attns,l=24,n=1):
        l_x = int(l*l)
        l_y = int(4*4)
        r = self.img_size//self.patch_size
        attns = torch.mean(attns,dim=1)

        attns_x2y = attns[:, l_x:, :l_x]
        attns_x2y = rearrange(attns_x2y,'b (n ly) l->b n ly l',ly = l_y)
        attns_x2y = attns_x2y * n.unsqueeze(-1).unsqueeze(-1)
        attns_x2y = attns_x2y.sum(2)

        attns_x2y = torch.mean(attns_x2y, dim=1).unsqueeze(-1)
        attns_x2y = rearrange(attns_x2y,'b (w h) c->b c w h',w = r, h = r)
        return attns_x2y

    def MacherMode(self,xs,ys,attn,scales=None,name='0.jpg'):
        x = xs[-1]
        B,L,D = x.shape
        y = ys[-1]
        B,Ly,D = y.shape
        n = int(Ly/16)
        r2 = (scales[:,:,0] + scales[:,:,1]) ** 2
        n = 16 / (r2 * 384)
        density_feature = rearrange(x,'b (w h) d->b d w h',w = 24)
        density_enhance = self.AttentionEnhance(attn[-1],l=int(np.sqrt(L)),n=n)
        density_feature2 = torch.cat((density_feature.contiguous(), density_enhance.contiguous()), axis=1)

        return density_feature2
    
    def Regressor(self, feature):
        feature = F.interpolate(
                                self.decode_head0(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head1(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head2(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head3(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = feature.squeeze(-3)
        return feature
    
    def forward(self, samples,name = None): 
        imgs = samples[0]
        boxes = samples[1]
        scales = samples[2]
        boxes = self.scale_embedding(boxes,scales)
        latent, y_latent= self.forward_encoder(imgs, boxes, scales=scales)
        xs,ys,attns = self.forward_decoder(latent,y_latent)
        density_feature = self.MacherMode(xs,ys,attns,scales,name=None)
        density_map = self.Regressor(density_feature)

        return density_map

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
    
