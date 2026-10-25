from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, Union

from safetensors.torch import load_file as load_safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Flux2VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    decoder_block_out_channels: Optional[Tuple[int, ...]] = None
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    patch_size: Tuple[int, int] = (2, 2)
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True


class _ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        sc = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return sc + h


class _Attention2D(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=channels, eps=eps, affine=True
        )
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        normed = self.group_norm(x).view(b, c, h * w).transpose(1, 2)
        q = self.to_q(normed).unsqueeze(1)
        k = self.to_k(normed).unsqueeze(1)
        v = self.to_v(normed).unsqueeze(1)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = self.to_out[0](attn.squeeze(1)).transpose(1, 2).view(b, c, h, w)
        return residual + out


class _Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.conv(x_padded)


class _Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x_up)


class _DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        add_downsample: bool,
        groups: int,
        eps: float,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else out_channels
            resnets.append(
                _ResnetBlock2D(c_in, out_channels, groups=groups, eps=eps)
            )
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = (
            nn.ModuleList([_Downsample2D(out_channels)])
            if add_downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x


class _UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        add_upsample: bool,
        groups: int,
        eps: float,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else out_channels
            resnets.append(
                _ResnetBlock2D(c_in, out_channels, groups=groups, eps=eps)
            )
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([_Upsample2D(out_channels)])
            if add_upsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for us in self.upsamplers:
                x = us(x)
        return x


class _UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int, eps: float):
        super().__init__()
        self.attentions = nn.ModuleList(
            [_Attention2D(channels, groups=groups, eps=eps)]
        )
        self.resnets = nn.ModuleList(
            [
                _ResnetBlock2D(channels, channels, groups=groups, eps=eps),
                _ResnetBlock2D(channels, channels, groups=groups, eps=eps),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class _Flux2Encoder(nn.Module):
    def __init__(self, config: Flux2VAEConfig):
        super().__init__()
        ch = config.block_out_channels
        self.conv_in = nn.Conv2d(
            config.in_channels, ch[0], kernel_size=3, stride=1, padding=1
        )
        down_blocks = []
        c_in = ch[0]
        for i, c_out in enumerate(ch):
            is_final = i == len(ch) - 1
            down_blocks.append(
                _DownEncoderBlock2D(
                    in_channels=c_in,
                    out_channels=c_out,
                    num_layers=config.layers_per_block,
                    add_downsample=not is_final,
                    groups=config.norm_num_groups,
                    eps=config.norm_eps,
                )
            )
            c_in = c_out
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = _UNetMidBlock2D(
            ch[-1], groups=config.norm_num_groups, eps=config.norm_eps
        )
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=ch[-1],
            eps=config.norm_eps,
            affine=True,
        )
        self.conv_out = nn.Conv2d(
            ch[-1], 2 * config.latent_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = F.silu(self.conv_norm_out(h))
        return self.conv_out(h)


class _Flux2Decoder(nn.Module):
    def __init__(self, config: Flux2VAEConfig):
        super().__init__()
        dec_ch = (
            config.decoder_block_out_channels
            if config.decoder_block_out_channels is not None
            else config.block_out_channels
        )
        rev_ch = list(reversed(dec_ch))
        self.conv_in = nn.Conv2d(
            config.latent_channels, rev_ch[0], kernel_size=3, stride=1, padding=1
        )
        self.mid_block = _UNetMidBlock2D(
            rev_ch[0], groups=config.norm_num_groups, eps=config.norm_eps
        )
        up_blocks = []
        c_in = rev_ch[0]
        for i, c_out in enumerate(rev_ch):
            is_final = i == len(rev_ch) - 1
            up_blocks.append(
                _UpDecoderBlock2D(
                    in_channels=c_in,
                    out_channels=c_out,
                    num_layers=config.layers_per_block + 1,
                    add_upsample=not is_final,
                    groups=config.norm_num_groups,
                    eps=config.norm_eps,
                )
            )
            c_in = c_out
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=rev_ch[-1],
            eps=config.norm_eps,
            affine=True,
        )
        self.conv_out = nn.Conv2d(
            rev_ch[-1], config.out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        h = F.silu(self.conv_norm_out(h))
        return self.conv_out(h)


class Flux2AutoencoderKL(nn.Module):
    def __init__(self, config: Optional[Flux2VAEConfig] = None):
        super().__init__()
        self.config = config or Flux2VAEConfig()
        self.encoder = _Flux2Encoder(self.config)
        self.decoder = _Flux2Decoder(self.config)
        self.quant_conv = (
            nn.Conv2d(
                2 * self.config.latent_channels,
                2 * self.config.latent_channels,
                kernel_size=1,
            )
            if self.config.use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(
                self.config.latent_channels,
                self.config.latent_channels,
                kernel_size=1,
            )
            if self.config.use_post_quant_conv
            else None
        )
        p_h, p_w = self.config.patch_size
        packed_ch = self.config.latent_channels * p_h * p_w
        self.bn = nn.BatchNorm2d(
            packed_ch,
            eps=self.config.batch_norm_eps,
            momentum=self.config.batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        )

    @classmethod
    def from_config_dir(cls, vae_dir: Union[str, Path]) -> Flux2AutoencoderKL:
        cfg_file = Path(vae_dir) / "config.json"
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            blk_ch = tuple(raw.get("block_out_channels", [128, 256, 512, 512]))
            dec_blk_raw = raw.get("decoder_block_out_channels", None)
            dec_blk_ch = tuple(dec_blk_raw) if dec_blk_raw is not None else blk_ch
            cfg = Flux2VAEConfig(
                in_channels=int(raw.get("in_channels", 3)),
                out_channels=int(raw.get("out_channels", 3)),
                latent_channels=int(raw.get("latent_channels", 32)),
                block_out_channels=blk_ch,
                decoder_block_out_channels=dec_blk_ch,
                layers_per_block=int(raw.get("layers_per_block", 2)),
                norm_num_groups=int(raw.get("norm_num_groups", 32)),
                norm_eps=float(raw.get("norm_eps", 1e-6)),
                patch_size=tuple(raw.get("patch_size", [2, 2])),
                batch_norm_eps=float(raw.get("batch_norm_eps", 1e-4)),
                batch_norm_momentum=float(raw.get("batch_norm_momentum", 0.1)),
                use_quant_conv=bool(raw.get("use_quant_conv", True)),
                use_post_quant_conv=bool(raw.get("use_post_quant_conv", True)),
            )
            return cls(cfg)
        return cls()

    @classmethod
    def from_pretrained_local(
        cls,
        vae_dir: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> Flux2AutoencoderKL:
        path = Path(vae_dir)
        sf_files = sorted(path.glob("*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")
        state_dict = {}
        for sf in sf_files:
            state_dict.update(load_safetensors(str(sf), device="cpu"))
        if "bn.num_batches_tracked" not in state_dict:
            state_dict["bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

        model = cls.from_config_dir(path)
        if "decoder.conv_out.weight" in state_dict:
            actual_dec_ch0 = int(state_dict["decoder.conv_out.weight"].shape[1])
            expected_dec_ch0 = int(model.decoder.conv_out.weight.shape[1])
            if actual_dec_ch0 != expected_dec_ch0 and actual_dec_ch0 == 96:
                model.config.decoder_block_out_channels = (96, 192, 384, 384)
                model = cls(model.config)

        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device=device, dtype=dtype)
        model.eval().requires_grad_(False)
        return model

    @torch.no_grad()
    def encode_to_packed_latents(self, x: torch.Tensor) -> torch.Tensor:
        param_dev = next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype
        x_in = x.to(device=param_dev, dtype=param_dtype)
        h = self.encoder(x_in)
        moments = self.quant_conv(h) if self.quant_conv is not None else h
        mean, _ = torch.chunk(moments.float(), 2, dim=1)
        b, c, h_lat, w_lat = mean.shape
        ph, pw = self.config.patch_size
        packed = (
            mean.view(b, c, h_lat // ph, ph, w_lat // pw, pw)
            .permute(0, 1, 3, 5, 2, 4)
            .contiguous()
            .view(b, c * ph * pw, h_lat // ph, w_lat // pw)
        )
        r_mean = self.bn.running_mean.float().view(1, -1, 1, 1).to(device=packed.device)
        r_var = self.bn.running_var.float().view(1, -1, 1, 1).to(device=packed.device)
        return (packed - r_mean) * torch.rsqrt(r_var + self.config.batch_norm_eps)

    @torch.no_grad()
    def decode_packed_latents(self, z_packed: torch.Tensor) -> torch.Tensor:
        param_dev = next(self.parameters()).device
        param_dtype = next(self.parameters()).dtype
        z = z_packed.to(device=param_dev, dtype=torch.float32)
        r_mean = self.bn.running_mean.float().view(1, -1, 1, 1).to(device=z.device)
        r_var = self.bn.running_var.float().view(1, -1, 1, 1).to(device=z.device)
        unnorm = z * torch.sqrt(r_var + self.config.batch_norm_eps) + r_mean
        b, c_pack, h_lat, w_lat = unnorm.shape
        ph, pw = self.config.patch_size
        c = c_pack // (ph * pw)
        unpacked_view = (
            unnorm.view(b, c, ph, pw, h_lat, w_lat)
            .permute(0, 1, 4, 2, 5, 3)
            .contiguous()
            .view(b, c, h_lat * ph, w_lat * pw)
            .to(dtype=param_dtype)
        )
        z_dec = (
            self.post_quant_conv(unpacked_view)
            if self.post_quant_conv is not None
            else unpacked_view
        )
        return self.decoder(z_dec).float().clamp(-1.0, 1.0)