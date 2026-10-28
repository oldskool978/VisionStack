# ShadowBoxer/models/modeling_vae.py
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, Union

from safetensors.torch import load_file as load_safetensors
import torch
from torch import nn
from torch.nn import functional as F

_SDPA_QUERY_CHUNK_SIZE = 2048


@dataclass
class Flux2VAEConfig:
    act_fn: str = "silu"
    batch_norm_eps: float = 0.0001
    batch_norm_momentum: float = 0.1
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    decoder_block_out_channels: Tuple[int, ...] = (96, 192, 384, 384)
    in_channels: int = 3
    latent_channels: int = 32
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    out_channels: int = 3
    patch_size: Tuple[int, int] = (2, 2)
    sample_size: int = 1024
    use_post_quant_conv: bool = True
    use_quant_conv: bool = True

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> Flux2VAEConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in raw.items():
            if k in valid and v is not None:
                if k in (
                    "block_out_channels",
                    "decoder_block_out_channels",
                    "patch_size",
                ):
                    filtered[k] = tuple(v)
                else:
                    filtered[k] = v
        return cls(**filtered)


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
        h = self.norm1(x)
        h = F.silu(h, inplace=True)
        h = self.conv1(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        h = self.norm2(h)
        h = F.silu(h, inplace=True)
        h = self.conv2(h)
        return h.add_(x)


class _AttentionBlock2D(nn.Module):
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
        seq_len = h * w
        normed = self.group_norm(x).reshape(b, c, seq_len).transpose(1, 2).contiguous()
        q = self.to_q(normed).unsqueeze(1)
        k = self.to_k(normed).unsqueeze(1)
        v = self.to_v(normed).unsqueeze(1)
        del normed
        if seq_len <= _SDPA_QUERY_CHUNK_SIZE * 2:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            ).squeeze(1)
        else:
            out = torch.empty((b, seq_len, c), dtype=q.dtype, device=q.device)
            for start in range(0, seq_len, _SDPA_QUERY_CHUNK_SIZE):
                end = min(start + _SDPA_QUERY_CHUNK_SIZE, seq_len)
                out[:, start:end, :] = F.scaled_dot_product_attention(
                    q[:, :, start:end, :], k, v, dropout_p=0.0, is_causal=False
                ).squeeze(1)
        del q, k, v
        out = self.to_out[0](out).transpose(1, 2).reshape(b, c, h, w)
        return out.add_(x)


class _Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.conv(x)


class _Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        add_downsample: bool,
        groups: int = 32,
        eps: float = 1e-6,
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
            nn.ModuleList([_Downsample2D(out_channels)]) if add_downsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            h = resnet.norm1(x)
            h = F.silu(h, inplace=True)
            h = resnet.conv1(h)
            if resnet.conv_shortcut is not None:
                x = resnet.conv_shortcut(x)
            h = resnet.norm2(h)
            h = F.silu(h, inplace=True)
            h = resnet.conv2(h)
            x = h.add_(x)
            del h
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
                x = downsampler.conv(x)
        return x


class _UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        add_upsample: bool,
        groups: int = 32,
        eps: float = 1e-6,
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
            nn.ModuleList([_Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            h = resnet.norm1(x)
            h = F.silu(h, inplace=True)
            h = resnet.conv1(h)
            if resnet.conv_shortcut is not None:
                x = resnet.conv_shortcut(x)
            h = resnet.norm2(h)
            h = F.silu(h, inplace=True)
            h = resnet.conv2(h)
            x = h.add_(x)
            del h
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = upsampler.conv(x)
        return x


class _UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.attentions = nn.ModuleList(
            [_AttentionBlock2D(channels, groups=groups, eps=eps)]
        )
        self.resnets = nn.ModuleList(
            [
                _ResnetBlock2D(channels, channels, groups=groups, eps=eps),
                _ResnetBlock2D(channels, channels, groups=groups, eps=eps),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


class _Flux2Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = ch
            is_final = i == len(block_out_channels) - 1
            down_blocks.append(
                _DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    add_downsample=not is_final,
                    groups=groups,
                    eps=eps,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = _UNetMidBlock2D(
            block_out_channels[-1], groups=groups, eps=eps
        )
        self.conv_norm_out = nn.GroupNorm(
            num_groups=groups,
            num_channels=block_out_channels[-1],
            eps=eps,
            affine=True,
        )
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            for resnet in block.resnets:
                h = resnet.norm1(x)
                h = F.silu(h, inplace=True)
                h = resnet.conv1(h)
                if resnet.conv_shortcut is not None:
                    x = resnet.conv_shortcut(x)
                h = resnet.norm2(h)
                h = F.silu(h, inplace=True)
                h = resnet.conv2(h)
                x = h.add_(x)
                del h
            if block.downsamplers is not None:
                for downsampler in block.downsamplers:
                    x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
                    x = downsampler.conv(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = F.silu(x, inplace=True)
        return self.conv_out(x)


class _Flux2Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (96, 192, 384, 384),
        layers_per_block: int = 2,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(
            in_channels, reversed_channels[0], kernel_size=3, stride=1, padding=1
        )
        self.mid_block = _UNetMidBlock2D(
            reversed_channels[0], groups=groups, eps=eps
        )
        up_blocks = []
        output_channel = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            prev_output_channel = output_channel
            output_channel = ch
            is_final = i == len(reversed_channels) - 1
            up_blocks.append(
                _UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final,
                    groups=groups,
                    eps=eps,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(
            num_groups=groups,
            num_channels=block_out_channels[0],
            eps=eps,
            affine=True,
        )
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid_block(z)
        for block in self.up_blocks:
            for resnet in block.resnets:
                h = resnet.norm1(z)
                h = F.silu(h, inplace=True)
                h = resnet.conv1(h)
                if resnet.conv_shortcut is not None:
                    z = resnet.conv_shortcut(z)
                h = resnet.norm2(h)
                h = F.silu(h, inplace=True)
                h = resnet.conv2(h)
                z = h.add_(z)
                del h
            if block.upsamplers is not None:
                for upsampler in block.upsamplers:
                    z = F.interpolate(z, scale_factor=2.0, mode="nearest")
                    z = upsampler.conv(z)
        z = self.conv_norm_out(z)
        z = F.silu(z, inplace=True)
        return self.conv_out(z)


class Flux2AutoencoderKL(nn.Module):
    def __init__(
        self,
        config: Optional[Flux2VAEConfig] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_block_out_channels: Optional[Tuple[int, ...]] = (96, 192, 384, 384),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        batch_norm_eps: float = 0.0001,
        batch_norm_momentum: float = 0.1,
        patch_size: Tuple[int, int] = (2, 2),
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()
        if config is not None:
            in_channels = config.in_channels
            out_channels = config.out_channels
            latent_channels = config.latent_channels
            block_out_channels = config.block_out_channels
            decoder_block_out_channels = (
                config.decoder_block_out_channels
                if config.decoder_block_out_channels is not None
                else config.block_out_channels
            )
            layers_per_block = config.layers_per_block
            norm_num_groups = config.norm_num_groups
            norm_eps = getattr(config, "norm_eps", 1e-6)
            batch_norm_eps = config.batch_norm_eps
            batch_norm_momentum = config.batch_norm_momentum
            patch_size = config.patch_size
            use_quant_conv = config.use_quant_conv
            use_post_quant_conv = config.use_post_quant_conv
        elif decoder_block_out_channels is None:
            decoder_block_out_channels = block_out_channels

        self.config = Flux2VAEConfig(
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            block_out_channels=tuple(block_out_channels),
            decoder_block_out_channels=tuple(decoder_block_out_channels),
            in_channels=in_channels,
            latent_channels=latent_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            out_channels=out_channels,
            patch_size=tuple(patch_size),
            use_post_quant_conv=use_post_quant_conv,
            use_quant_conv=use_quant_conv,
        )

        packed_channels = latent_channels * patch_size[0] * patch_size[1]
        self.bn = nn.BatchNorm2d(
            packed_channels,
            eps=batch_norm_eps,
            momentum=batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.encoder = _Flux2Encoder(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            groups=norm_num_groups,
            eps=norm_eps,
        )
        self.decoder = _Flux2Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=tuple(decoder_block_out_channels),
            layers_per_block=layers_per_block,
            groups=norm_num_groups,
            eps=norm_eps,
        )
        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
            if use_post_quant_conv
            else None
        )

    @classmethod
    def from_config_dir(cls, model_dir: Union[str, Path]) -> Flux2AutoencoderKL:
        cfg_file = Path(model_dir) / "config.json"
        if cfg_file.is_file():
            return cls(config=Flux2VAEConfig.from_json_file(cfg_file))
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
        target_dev = torch.device(device) if isinstance(device, str) else device
        state_dict = {}
        for sf in sf_files:
            raw = load_safetensors(str(sf), device="cpu")
            for k, v in raw.items():
                if v.is_floating_point():
                    t_dtype = torch.float32 if k.startswith("bn.") else dtype
                    state_dict[k] = v.to(device=target_dev, dtype=t_dtype)
                else:
                    state_dict[k] = v.to(device=target_dev)
            del raw

        if "bn.num_batches_tracked" not in state_dict:
            state_dict["bn.num_batches_tracked"] = torch.tensor(
                0, dtype=torch.long, device=target_dev
            )

        model = cls.from_config_dir(path)
        if "decoder.conv_out.weight" in state_dict:
            actual_dec_ch0 = int(state_dict["decoder.conv_out.weight"].shape[1])
            expected_dec_ch0 = int(model.decoder.conv_out.weight.shape[1])
            if actual_dec_ch0 != expected_dec_ch0 and actual_dec_ch0 == 96:
                model.config.decoder_block_out_channels = (96, 192, 384, 384)
                model = cls(config=model.config)

        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device=target_dev, dtype=dtype)
        model.eval().requires_grad_(False)
        return model

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.conv_in.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.decoder.conv_in.weight.device

    def decode(self, z: torch.Tensor, return_dict: bool = False) -> Tuple[torch.Tensor]:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return (dec,)

    @torch.inference_mode()
    def decode_packed_latents(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, enabled=False):
            z = latents.to(device=self.device, dtype=torch.float32)
            del latents
            eps = self.config.batch_norm_eps
            std = (self.bn.running_var.float() + eps).sqrt().to(z).view(1, -1, 1, 1)
            mean = self.bn.running_mean.float().to(z).view(1, -1, 1, 1)
            z = torch.addcmul(mean, z, std)
            del std, mean
            b, c, h, w = z.shape
            z = (
                z.reshape(b, c // 4, 2, 2, h, w)
                .permute(0, 1, 4, 2, 5, 3)
                .reshape(b, c // 4, 2 * h, 2 * w)
            )
            if z.dtype != self.dtype:
                z = z.to(dtype=self.dtype)
            if self.post_quant_conv is not None:
                z = self.post_quant_conv(z)
            decoded = self.decoder(z)
            del z
            return decoded.float().clamp_(-1.0, 1.0)

    @torch.inference_mode()
    def encode_to_packed_latents(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, enabled=False):
            x = images.to(device=self.device, dtype=self.dtype)
            del images
            h = self.encoder(x)
            del x
            if self.quant_conv is not None:
                h = self.quant_conv(h)
            posterior_mode, _ = torch.chunk(h.float(), 2, dim=1)
            del h
            b, c, h2, w2 = posterior_mode.shape
            z_patch = (
                posterior_mode.reshape(b, c, h2 // 2, 2, w2 // 2, 2)
                .permute(0, 1, 3, 5, 2, 4)
                .reshape(b, c * 4, h2 // 2, w2 // 2)
            )
            del posterior_mode
            eps = self.config.batch_norm_eps
            inv_std = (
                torch.rsqrt(self.bn.running_var.float() + eps)
                .to(z_patch)
                .view(1, -1, 1, 1)
            )
            mean = self.bn.running_mean.float().to(z_patch).view(1, -1, 1, 1)
            z_patch = z_patch.sub_(mean).mul_(inv_std)
            del inv_std, mean
            return z_patch.float()