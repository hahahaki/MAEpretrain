# code is partially adopted from https://github.com/facebookresearch/mae

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils import get_2d_sincos_pos_embed, get_plugin, register_plugin
import matplotlib.pyplot as plt


@register_plugin('model', "MAESTER")
def model_config(cfg):
    img_size = cfg["img_size"]
    patch_size = cfg["patch_size"]
    print("img_size and patch_size configed")
    model = MAESTER_MODEL(
        img_size=img_size, 
        in_chans=1,
        patch_size=patch_size, 
        embed_dim=cfg["embed_dim"], 
        depth=cfg["depth"], 
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"], 
        decoder_depth=cfg["decoder_depth"],
        decoder_num_heads=cfg["decoder_num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        pos_encode_w = cfg["pos_encode_w"],
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

class MAESTER_MODEL(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size= 8, #16,
        in_chans=1,
        embed_dim= 1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        pos_encode_w=1.0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # need to reassign the embed_dim, otherwise wrong, but why
        #embed_dim= 29
        self.embed_dim = embed_dim
        self.pos_encode_w = pos_encode_w
        #print("img_size:", img_size)
        #print("patch_size:", patch_size)
    
        #print("dim", embed_dim)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        #print("num_patches:", num_patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        #print(self.pos_embed.shape)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *3)
        L = H * W
        """
        # here imgs is (N, C = 1, H, W), C is channel
       # print("patchify dim:", imgs.shape)
        p = self.patch_embed.patch_size[0]
        #imgs = imgs.unsqueeze(0)  # This adds the batch dimension, making the shape [1, 1, 232, 232]
        # I annotate the assert
        #assert imgs.shape[2] == imgs.shapeten and imgs.shape[2] % p == 0
        # I modifed the dimension here
        h = w = imgs.shape[2] // p
        # Ensure the image dimensions are divisible by the patch size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def infer_latent(self, x):
        return self._infer(x)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        #print(x.shape)
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x) # NCHW -> NLC
        print("pos_embed shape", self.pos_embed.shape)
        print("xshape", x.shape)
        # add pos embed w/o cls token
        # I added start here:
        # dimension should be [batch_size, num_patches, embed_dim]
        #required_elements = 1 * 841 * 29  # Total elements for the target shape
        #current_elements = x.numel()
        # Calculate the number of padding elements needed
        #padding_elements = required_elements - current_elements
        # Create a padding tensor of the appropriate size
        #padding = torch.zeros(padding_elements, dtype=x.dtype, device=x.device)
        # Concatenate x with the padding tensor
        #x = torch.cat((x.flatten(), padding)).view(1, 841, 29)
        # x is 3D
        
        x = x + self.pos_embed[:, 1:, :] * self.pos_encode_w

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x change to [1, 226, 192] because mask?
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        '''
        # check here; what is the x like? after encoder
        # unpachify
        print("xdecoder", x.shape)
        p = 8
        h = w = 30
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        im = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        plt.imshow(im, cmap='gray')
        plt.axis('off')  # Turn off axis numbers and labels
        plt.savefig('/home/codee/scratch/result/test_decoder.png')
        plt.close()
        '''
        
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def _infer(self, x):
        """
        Infers the output of the model given an input tensor.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size) representing the input.

        Returns:
            A tensor of shape (batch_size, sequence_length, hidden_size) representing the output of the model.
        """
        # Forward the encoder without random masking
        x, mask, ids_restore = self.forward_encoder(x, 0.0)

        # Create a mask tensor with the same shape as x
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )

        # Remove the cls token from x
        x_ = x[:, 1:, :]

        # Unshuffle the tensor using the ids_restore tensor
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # Append the cls token to the tensor
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Return the output tensor
        print("output tensor", x)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # this is the main function
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*1]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
