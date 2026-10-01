






from __future__ import annotations
import sys
sys.dont_write_bytecode = True






import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class ModelConfig:








    image_channels: int = 3
    image_size: int = 256
    latent_channels: int = 8
    latent_size: int = 32
    target_widths: tuple[int, int, int, int] = (32, 64, 128, 128)
    condition_widths: tuple[int, int, int, int, int, int] = (
        32,
        64,
        128,
        128,
        192,
        256,
    )
    unet_widths: tuple[int, int, int] = (128, 192, 256)
    time_input_dim: int = 128
    time_hidden_dim: int = 512
    attention_heads: int = 8
    group_norm_groups: int = 8
    phase_widths: tuple[int, int, int, int, int, int] = (128, 64, 32, 16, 8, 8)
    phase_size: int = 2048

    @classmethod
    def paper_defaults(cls) -> "ModelConfig":


        return cls()

    def validate(self) -> None:
        expected = type(self)()
        if self != expected:
            changed = {
                key: (value, getattr(expected, key))
                for key, value in asdict(self).items()
                if value != getattr(expected, key)
            }
            raise ValueError(
                "Only the exact Supplementary Tables 2--6 architecture is "
                f"supported; changed fields (received, required): {changed}"
            )
        if self.image_size // 8 != self.latent_size:
            raise ValueError("image_size/8 must equal latent_size")
        if self.latent_size * 2 ** len(self.phase_widths) != self.phase_size:
            raise ValueError("six PixelShuffle stages must map 32 to 2048")
        if self.unet_widths[-1] % self.attention_heads:
            raise ValueError("bottleneck width must divide evenly into 8 heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionConfig:


    train_steps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    inference_evaluations: int = 8

    def __post_init__(self) -> None:
        if self.train_steps != 1000:
            raise ValueError("Supplementary Note 5 requires T=1000")
        if not math.isclose(self.beta_start, 1.0e-4):
            raise ValueError("Supplementary equation S43 requires beta_start=1e-4")
        if not math.isclose(self.beta_end, 2.0e-2):
            raise ValueError("Supplementary equation S43 requires beta_end=0.02")
        if self.inference_evaluations != 8:
            raise ValueError("This simulation deliverable is fixed to K=8")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)




EXPECTED_PARAMETER_COUNTS: dict[str, int] = {
    "target_encoder": 934_408,
    "aux_decoder": 934_211,
    "condition_encoder": 3_434_816,
    "condition_encoder_early": 924_928,
    "condition_encoder_late": 2_509_888,
    "denoiser": 13_575_304,
    "phase_decoder": 1_487_787,
    "digital_pattern_generator": 18_497_907,
}


def _require_bchw(
    tensor: Tensor,
    *,
    channels: int | None = None,
    spatial: int | None = None,
    name: str = "tensor",
) -> None:


    if tensor.ndim != 4:
        raise ValueError(f"{name} must be BCHW; received {tuple(tensor.shape)}")
    if channels is not None and tensor.shape[1] != channels:
        raise ValueError(
            f"{name} must have {channels} channels; received {tensor.shape[1]}"
        )
    if spatial is not None and tensor.shape[-2:] != (spatial, spatial):
        raise ValueError(
            f"{name} must be {spatial}x{spatial}; received {tuple(tensor.shape[-2:])}"
        )


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _module_dtype(module: nn.Module) -> torch.dtype:


    return next(module.parameters()).dtype


def logits_to_phase(logits: Tensor) -> Tensor:






    return TAU * torch.sigmoid(logits)


class ResidualBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class DepthwiseResidualBlock(nn.Module):


    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.depthwise1 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.pointwise1 = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.depthwise2 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.pointwise2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.pointwise1(self.depthwise1(F.silu(self.norm1(x))))
        x = self.pointwise2(self.depthwise2(F.silu(self.norm2(x))))
        return x + residual


class TargetEncoder(nn.Module):


    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        a, b, c, d = config.target_widths
        self.input_conv = nn.Conv2d(config.image_channels, a, 3, padding=1)
        self.residuals = nn.ModuleList(
            (ResidualBlock(a), ResidualBlock(b), ResidualBlock(c), ResidualBlock(d))
        )
        self.downsample = nn.ModuleList(
            (
                nn.Conv2d(a, b, 3, stride=2, padding=1),
                nn.Conv2d(b, c, 3, stride=2, padding=1),
                nn.Conv2d(c, d, 3, stride=2, padding=1),
            )
        )
        self.output_norm = nn.GroupNorm(8, d)
        self.output_conv = nn.Conv2d(d, config.latent_channels, 3, padding=1)

    def forward(self, target: Tensor) -> Tensor:
        _require_bchw(target, channels=3, spatial=256, name="target")
        x = self.residuals[0](self.input_conv(target))
        for down, residual in zip(self.downsample, self.residuals[1:]):
            x = residual(down(x))
        return self.output_conv(F.silu(self.output_norm(x)))


class AuxiliaryDecoder(nn.Module):


    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        
        a, b, _, d = config.target_widths
        widths = (d, d, b, a)
        self.input_conv = nn.Conv2d(config.latent_channels, d, 3, padding=1)
        self.residuals = nn.ModuleList(ResidualBlock(width) for width in widths)
        self.up_convs = nn.ModuleList(
            nn.Conv2d(in_width, out_width, 3, padding=1)
            for in_width, out_width in zip(widths[:-1], widths[1:])
        )
        self.output_norm = nn.GroupNorm(8, a)
        self.output_conv = nn.Conv2d(a, config.image_channels, 3, padding=1)

    def forward(self, raw_latent: Tensor) -> Tensor:
        _require_bchw(raw_latent, channels=8, spatial=32, name="raw_latent")
        x = self.residuals[0](self.input_conv(raw_latent))
        for conv, residual in zip(self.up_convs, self.residuals[1:]):
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = residual(conv(x))
        return torch.sigmoid(self.output_conv(F.silu(self.output_norm(x))))


class ConditionFeatures(NamedTuple):


    c32: Tensor
    c16: Tensor
    c8: Tensor


class ConditionEncoder(nn.Module):







    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        w = config.condition_widths
        self.input_conv = nn.Conv2d(config.image_channels, w[0], 3, padding=1)
        self.residual0 = ResidualBlock(w[0])

        self.early_downs = nn.ModuleList(
            nn.Conv2d(w[index], w[index + 1], 3, stride=2, padding=1)
            for index in range(3)
        )
        self.early_residuals = nn.ModuleList(
            ResidualBlock(w[index]) for index in range(1, 4)
        )

        self.late_down1 = nn.Conv2d(w[3], w[4], 3, stride=2, padding=1)
        self.late_residual1 = ResidualBlock(w[4])
        self.late_down2 = nn.Conv2d(w[4], w[5], 3, stride=2, padding=1)
        self.late_residual2 = ResidualBlock(w[5])

    def early_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.input_conv.parameters()
        yield from self.residual0.parameters()
        yield from self.early_downs.parameters()
        yield from self.early_residuals.parameters()

    def late_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.late_down1,
            self.late_residual1,
            self.late_down2,
            self.late_residual2,
        ):
            yield from module.parameters()

    def forward(self, condition: Tensor) -> ConditionFeatures:
        _require_bchw(condition, channels=3, spatial=256, name="condition")
        x = self.residual0(self.input_conv(condition))
        for down, residual in zip(self.early_downs, self.early_residuals):
            x = residual(down(x))
        c32 = x
        c16 = self.late_residual1(self.late_down1(c32))
        c8 = self.late_residual2(self.late_down2(c16))
        return ConditionFeatures(c32, c16, c8)


class SinusoidalTimeEmbedding(nn.Module):


    def __init__(self, input_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        if input_dim % 2:
            raise ValueError("sinusoidal time input dimension must be even")
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.input_dim // 2
        
        exponent = -math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * exponent
        )
        angles = timesteps.to(torch.float32).reshape(-1, 1) * frequencies[None, :]
        embedded = torch.cat((angles.sin(), angles.cos()), dim=1)
        
        
        embedded = embedded.to(dtype=self.linear1.weight.dtype)
        return self.linear2(F.silu(self.linear1(embedded)))


class TimeResidualBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        
        self.time_modulation = nn.Linear(time_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        scale, bias = self.time_modulation(F.silu(time_embedding)).chunk(2, dim=1)
        x = self.norm2(x)
        x = x * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        x = self.conv2(F.silu(x))
        return x + residual


class BottleneckAttention(nn.Module):


    def __init__(self, channels: int = 256, heads: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must divide evenly into heads")
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        head_dim = channels // self.heads
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        tokens = height * width
        q = q.reshape(batch, self.heads, head_dim, tokens).transpose(-1, -2)
        k = k.reshape(batch, self.heads, head_dim, tokens).transpose(-1, -2)
        v = v.reshape(batch, self.heads, head_dim, tokens).transpose(-1, -2)
        
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(-1, -2).reshape(
            batch, channels, height, width
        )
        return x + self.output(attended)


class LatentDenoiser(nn.Module):


    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        a, b, c = config.unet_widths
        ca, cb, cc = config.condition_widths[3:]

        self.time_embedding = SinusoidalTimeEmbedding(
            config.time_input_dim, config.time_hidden_dim
        )
        self.input_conv = nn.Conv2d(config.latent_channels, a, 3, padding=1)
        self.condition_projections = nn.ModuleList(
            (nn.Conv2d(ca, a, 1), nn.Conv2d(cb, b, 1), nn.Conv2d(cc, c, 1))
        )

        self.encoder32a = TimeResidualBlock(a, a, config.time_hidden_dim)
        self.encoder32b = TimeResidualBlock(a, a, config.time_hidden_dim)
        self.down16 = nn.Conv2d(a, b, 3, stride=2, padding=1)
        self.encoder16a = TimeResidualBlock(b, b, config.time_hidden_dim)
        self.encoder16b = TimeResidualBlock(b, b, config.time_hidden_dim)
        self.down8 = nn.Conv2d(b, c, 3, stride=2, padding=1)
        self.encoder8a = TimeResidualBlock(c, c, config.time_hidden_dim)
        self.encoder8b = TimeResidualBlock(c, c, config.time_hidden_dim)

        self.bottleneck1 = TimeResidualBlock(c, c, config.time_hidden_dim)
        self.attention = BottleneckAttention(c, config.attention_heads)
        self.bottleneck2 = TimeResidualBlock(c, c, config.time_hidden_dim)

        self.up16 = nn.Conv2d(c, b, 3, padding=1)
        self.decoder16a = TimeResidualBlock(2 * b, b, config.time_hidden_dim)
        self.decoder16b = TimeResidualBlock(b, b, config.time_hidden_dim)
        self.up32 = nn.Conv2d(b, a, 3, padding=1)
        self.decoder32a = TimeResidualBlock(2 * a, a, config.time_hidden_dim)
        self.decoder32b = TimeResidualBlock(a, a, config.time_hidden_dim)

        self.output_norm = nn.GroupNorm(8, a)
        self.output_conv = nn.Conv2d(a, config.latent_channels, 3, padding=1)

    def forward(
        self,
        latent: Tensor,
        timesteps: Tensor,
        features: ConditionFeatures | Sequence[Tensor],
    ) -> Tensor:
        _require_bchw(latent, channels=8, spatial=32, name="latent")
        if len(features) != 3:
            raise ValueError("condition features must contain c32, c16 and c8")
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(latent.shape[0])
        if timesteps.ndim != 1 or timesteps.shape[0] != latent.shape[0]:
            raise ValueError("timesteps must be a scalar or a length-B vector")

        time = self.time_embedding(timesteps)
        c32, c16, c8 = (
            projection(feature)
            for projection, feature in zip(self.condition_projections, features)
        )

        x32 = self.input_conv(latent) + c32
        x32 = self.encoder32b(self.encoder32a(x32, time), time)
        skip32 = x32

        x16 = self.down16(x32) + c16
        x16 = self.encoder16b(self.encoder16a(x16, time), time)
        skip16 = x16

        x8 = self.down8(x16) + c8
        x8 = self.encoder8b(self.encoder8a(x8, time), time)
        x8 = self.bottleneck2(
            self.attention(self.bottleneck1(x8, time)), time
        )

        x = F.interpolate(x8, size=skip16.shape[-2:], mode="nearest")
        x = self.up16(x)
        x = self.decoder16b(
            self.decoder16a(torch.cat((x, skip16), dim=1), time), time
        )

        x = F.interpolate(x, size=skip32.shape[-2:], mode="nearest")
        x = self.up32(x)
        x = self.decoder32b(
            self.decoder32a(torch.cat((x, skip32), dim=1), time), time
        )
        return self.output_conv(F.silu(self.output_norm(x)))


class PixelShuffleStage(nn.Module):


    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, 4 * out_channels, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.refine = DepthwiseResidualBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.refine(self.shuffle(self.expand(x)))


class PhaseDecoder(nn.Module):


    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        first = config.phase_widths[0]
        shallow_channels = config.condition_widths[3]
        self.input_conv = nn.Conv2d(
            config.latent_channels + shallow_channels, first, 3, padding=1
        )
        self.input_residual = ResidualBlock(first)

        stages: list[nn.Module] = []
        current = first
        for out_channels in config.phase_widths:
            stages.append(PixelShuffleStage(current, out_channels))
            current = out_channels
        self.stages = nn.ModuleList(stages)
        self.output_norm = nn.GroupNorm(8, current)
        self.output_conv = nn.Conv2d(current, config.image_channels, 1)

    def _forward_impl(self, final_latent: Tensor, c32: Tensor) -> Tensor:
        x = torch.cat((final_latent, c32), dim=1)
        x = self.input_residual(self.input_conv(x))
        for stage in self.stages:
            x = stage(x)
        
        
        return self.output_conv(F.silu(self.output_norm(x)))

    def forward(
        self,
        final_latent: Tensor,
        c32: Tensor,
        *,
        use_checkpoint: bool = False,
    ) -> Tensor:
        _require_bchw(final_latent, channels=8, spatial=32, name="final_latent")
        _require_bchw(c32, channels=128, spatial=32, name="c32")
        if final_latent.shape[0] != c32.shape[0]:
            raise ValueError("final_latent and c32 must use the same batch size")
        if use_checkpoint and torch.is_grad_enabled():
            
            
            
            return checkpoint(
                self._forward_impl,
                final_latent,
                c32,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(final_latent, c32)


class LinearDiffusionSchedule:


    def __init__(
        self,
        config: DiffusionConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = DiffusionConfig() if config is None else config
        beta = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.train_steps,
            dtype=torch.float32,
            device=device,
        )
        alpha = 1.0 - beta
        self.alpha_bar = torch.cat(
            (
                torch.ones(1, dtype=torch.float32, device=device),
                torch.cumprod(alpha, dim=0),
            )
        )

    def on(self, reference: Tensor) -> Tensor:







        return self.alpha_bar.to(device=reference.device, dtype=torch.float32)


def direct_noising(
    clean_latent: Tensor,
    timesteps: Tensor,
    noise: Tensor,
    schedule: LinearDiffusionSchedule,
) -> Tensor:


    if clean_latent.shape != noise.shape:
        raise ValueError("clean_latent and noise must have identical shapes")
    alpha_bar = schedule.on(clean_latent)
    alpha = alpha_bar[timesteps.to(device=alpha_bar.device)].view(-1, 1, 1, 1)
    
    
    clean32 = clean_latent.float()
    noise32 = noise.float()
    return alpha.sqrt() * clean32 + (1.0 - alpha).sqrt() * noise32


def ddim_nodes(config: DiffusionConfig | None = None) -> tuple[int, ...]:


    config = DiffusionConfig() if config is None else config
    total = config.train_steps
    evaluations = config.inference_evaluations
    nodes = tuple(
        round(total * (1.0 - index / evaluations))
        for index in range(evaluations + 1)
    )
    if nodes != (1000, 875, 750, 625, 500, 375, 250, 125, 0):
        raise AssertionError(f"unexpected K=8 node schedule: {nodes}")
    return nodes


@dataclass(frozen=True)
class DDIMResult:


    final_latent: Tensor
    nodes: tuple[int, ...]
    evaluations: int


def _denoise_with_optional_checkpoint(
    denoiser: LatentDenoiser,
    state: Tensor,
    timestep: Tensor,
    features: ConditionFeatures,
    enabled: bool,
) -> Tensor:
    
    
    
    state_for_network = state.to(dtype=_module_dtype(denoiser))
    if not enabled or not torch.is_grad_enabled():
        return denoiser(state_for_network, timestep, features)

    def call(
        latent: Tensor, t: Tensor, c32: Tensor, c16: Tensor, c8: Tensor
    ) -> Tensor:
        return denoiser(latent, t, ConditionFeatures(c32, c16, c8))

    return checkpoint(
        call,
        state_for_network,
        timestep,
        features.c32,
        features.c16,
        features.c8,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def deterministic_ddim_k8(
    denoiser: LatentDenoiser,
    initial_noise: Tensor,
    features: ConditionFeatures,
    schedule: LinearDiffusionSchedule,
    *,
    use_checkpoint: bool = False,
) -> DDIMResult:


    nodes = ddim_nodes(schedule.config)
    state = initial_noise.float()
    alpha_bar = schedule.on(state)
    batch = state.shape[0]

    for current, following in zip(nodes, nodes[1:]):
        timestep = torch.full(
            (batch,), current, dtype=torch.long, device=state.device
        )
        predicted_noise = _denoise_with_optional_checkpoint(
            denoiser, state, timestep, features, use_checkpoint
        ).float()
        alpha_current = alpha_bar[current]
        clean_estimate = (
            state - torch.sqrt(1.0 - alpha_current) * predicted_noise
        ) / torch.sqrt(alpha_current).clamp_min(1.0e-8)
        alpha_next = alpha_bar[following]
        state = (
            torch.sqrt(alpha_next) * clean_estimate
            + torch.sqrt(1.0 - alpha_next) * predicted_noise
        )

    return DDIMResult(
        final_latent=state,
        nodes=nodes,
        evaluations=schedule.config.inference_evaluations,
    )


@dataclass(frozen=True)
class GenerationResult:


    condition_features: ConditionFeatures
    latent: DDIMResult
    phase_logits: Tensor


class EPIModel(nn.Module):








    def __init__(
        self,
        config: ModelConfig | None = None,
        diffusion: DiffusionConfig | None = None,
        *,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.config = ModelConfig.paper_defaults() if config is None else config
        self.config.validate()
        self.diffusion_config = DiffusionConfig() if diffusion is None else diffusion
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.target_encoder = TargetEncoder(self.config)
        self.aux_decoder = AuxiliaryDecoder(self.config)
        self.condition_encoder = ConditionEncoder(self.config)
        self.denoiser = LatentDenoiser(self.config)
        self.phase_decoder = PhaseDecoder(self.config)

        
        
        self.register_buffer(
            "latent_mean", torch.zeros(1, self.config.latent_channels, 1, 1)
        )
        self.register_buffer(
            "latent_std", torch.ones(1, self.config.latent_channels, 1, 1)
        )
        self.register_buffer(
            "latent_stats_fitted", torch.tensor(False, dtype=torch.bool)
        )

        self.assert_parameter_counts()

    def components(self) -> OrderedDict[str, nn.Module]:
        return OrderedDict(
            target_encoder=self.target_encoder,
            aux_decoder=self.aux_decoder,
            condition_encoder=self.condition_encoder,
            denoiser=self.denoiser,
            phase_decoder=self.phase_decoder,
        )

    def parameter_summary(self) -> dict[str, int]:
        component_counts = {
            name: _parameter_count(module) for name, module in self.components().items()
        }
        early = sum(
            parameter.numel()
            for parameter in self.condition_encoder.early_parameters()
        )
        late = sum(
            parameter.numel()
            for parameter in self.condition_encoder.late_parameters()
        )
        dpg = (
            component_counts["condition_encoder"]
            + component_counts["denoiser"]
            + component_counts["phase_decoder"]
        )
        return {
            **component_counts,
            "condition_encoder_early": early,
            "condition_encoder_late": late,
            "digital_pattern_generator": dpg,
        }

    def audit_parameter_counts(self) -> dict[str, dict[str, int | bool]]:
        actual = self.parameter_summary()
        return {
            name: {
                "actual": actual[name],
                "expected": expected,
                "matches": actual[name] == expected,
            }
            for name, expected in EXPECTED_PARAMETER_COUNTS.items()
        }

    def assert_parameter_counts(self) -> None:
        mismatches = {
            name: row
            for name, row in self.audit_parameter_counts().items()
            if not bool(row["matches"])
        }
        if mismatches:
            raise RuntimeError(
                "Network parameter counts disagree with Supplementary Tables "
                f"1--6: {mismatches}"
            )

    @torch.no_grad()
    def set_latent_stats(self, mean: Tensor, std: Tensor) -> None:


        expected = (1, self.config.latent_channels, 1, 1)
        if mean.ndim == 1:
            mean = mean.reshape(expected)
        if std.ndim == 1:
            std = std.reshape(expected)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError(f"mean and std must broadcast as {expected}")
        if not bool(torch.all(torch.isfinite(mean))):
            raise ValueError("latent mean contains a non-finite value")
        if not bool(torch.all(torch.isfinite(std))) or bool(torch.any(std <= 0)):
            raise ValueError("latent std must be finite and strictly positive")
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.to(self.latent_std))
        self.latent_stats_fitted.fill_(True)

    def normalize_latent(self, raw_latent: Tensor) -> Tensor:
        if not bool(self.latent_stats_fitted):
            raise RuntimeError("latent statistics have not been fitted after A0")
        return (raw_latent - self.latent_mean) / self.latent_std

    def denormalize_latent(self, standardized_latent: Tensor) -> Tensor:
        if not bool(self.latent_stats_fitted):
            raise RuntimeError("latent statistics have not been fitted after A0")
        return standardized_latent * self.latent_std + self.latent_mean

    def encode_target_raw(self, target: Tensor) -> Tensor:


        return self.target_encoder(target.to(dtype=_module_dtype(self.target_encoder)))

    def encode_target_standardized(self, target: Tensor) -> Tensor:


        raw = self.target_encoder(target.to(dtype=_module_dtype(self.target_encoder)))
        return self.normalize_latent(raw)

    def reconstruct_target(self, target: Tensor) -> Tensor:


        target = target.to(dtype=_module_dtype(self.target_encoder))
        return self.aux_decoder(self.target_encoder(target))

    def encode_condition(self, condition: Tensor) -> ConditionFeatures:
        condition = condition.to(dtype=_module_dtype(self.condition_encoder))
        return self.condition_encoder(condition)

    def generate_latent(
        self,
        condition: Tensor,
        *,
        initial_noise: Tensor | None = None,
        schedule: LinearDiffusionSchedule | None = None,
        checkpoint_denoiser: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ConditionFeatures, DDIMResult]:


        _require_bchw(condition, channels=3, spatial=256, name="condition")
        condition_for_network = condition.to(
            dtype=_module_dtype(self.condition_encoder)
        )
        features = self.condition_encoder(condition_for_network)
        if initial_noise is None:
            initial_noise = torch.randn(
                (
                    condition.shape[0],
                    self.config.latent_channels,
                    self.config.latent_size,
                    self.config.latent_size,
                ),
                device=condition.device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            _require_bchw(
                initial_noise,
                channels=self.config.latent_channels,
                spatial=self.config.latent_size,
                name="initial_noise",
            )
            if initial_noise.shape[0] != condition.shape[0]:
                raise ValueError("condition and initial_noise batch sizes differ")

        schedule = (
            LinearDiffusionSchedule(self.diffusion_config, device=condition.device)
            if schedule is None
            else schedule
        )
        use_checkpoint = (
            self.gradient_checkpointing
            if checkpoint_denoiser is None
            else checkpoint_denoiser
        )
        result = deterministic_ddim_k8(
            self.denoiser,
            initial_noise,
            features,
            schedule,
            use_checkpoint=bool(use_checkpoint),
        )
        return features, result

    def decode_phase_logits(
        self,
        final_latent: Tensor,
        c32: Tensor,
        *,
        checkpoint_decoder: bool | None = None,
    ) -> Tensor:
        use_checkpoint = (
            self.gradient_checkpointing
            if checkpoint_decoder is None
            else checkpoint_decoder
        )
        decoder_dtype = _module_dtype(self.phase_decoder)
        return self.phase_decoder(
            final_latent.to(dtype=decoder_dtype),
            c32.to(dtype=decoder_dtype),
            use_checkpoint=bool(use_checkpoint),
        )

    def decode_phase(
        self,
        final_latent: Tensor,
        c32: Tensor,
        *,
        checkpoint_decoder: bool | None = None,
    ) -> Tensor:


        return logits_to_phase(
            self.decode_phase_logits(
                final_latent, c32, checkpoint_decoder=checkpoint_decoder
            )
        )

    def generate_phase_logits(
        self,
        condition: Tensor,
        *,
        initial_noise: Tensor | None = None,
        schedule: LinearDiffusionSchedule | None = None,
        checkpoint_denoiser: bool | None = None,
        checkpoint_decoder: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> GenerationResult:


        features, latent_result = self.generate_latent(
            condition,
            initial_noise=initial_noise,
            schedule=schedule,
            checkpoint_denoiser=checkpoint_denoiser,
            generator=generator,
        )
        phase_logits = self.decode_phase_logits(
            latent_result.final_latent,
            features.c32,
            checkpoint_decoder=checkpoint_decoder,
        )
        return GenerationResult(features, latent_result, phase_logits)

    def forward(
        self,
        condition: Tensor,
        *,
        initial_noise: Tensor | None = None,
    ) -> Tensor:


        return self.generate_phase_logits(
            condition, initial_noise=initial_noise
        ).phase_logits


def noise_prediction_loss(
    model: EPIModel,
    condition: Tensor,
    clean_standardized_latent: Tensor,
    schedule: LinearDiffusionSchedule,
    *,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
) -> Tensor:


    batch = clean_standardized_latent.shape[0]
    if timesteps is None:
        timesteps = torch.randint(
            1,
            schedule.config.train_steps + 1,
            (batch,),
            device=clean_standardized_latent.device,
        )
    if noise is None:
        noise = torch.randn_like(clean_standardized_latent)
    noised = direct_noising(
        clean_standardized_latent, timesteps, noise, schedule
    )
    denoiser_dtype = _module_dtype(model.denoiser)
    predicted = model.denoiser(
        noised.to(dtype=denoiser_dtype),
        timesteps,
        model.encode_condition(condition),
    )
    return F.mse_loss(predicted.float(), noise.float())







import math
from dataclasses import asdict, dataclass
from typing import Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


TAU: Final[float] = 2.0 * math.pi
RGB_WAVELENGTHS_M: Final[tuple[float, float, float]] = (635e-9, 532e-9, 450e-9)


@dataclass(frozen=True)
class OpticsSpec:








    slm_pixels: int = 2048
    wavelengths_m_rgb: tuple[float, float, float] = RGB_WAVELENGTHS_M
    pixel_pitch_m: float = 6.4e-6
    distance_cws_to_odr_m: float = 0.50
    focal_length_m: float = 0.20
    reference_wavelength_m: float = 532e-9
    padding_pixels: int = 1024
    
    
    
    camera_fov_pixels: int = 1024
    camera_pixels: int = 256
    intensity_gain: float = 0.10
    camera_saturation: bool = False
    incident_mode: Literal["plane", "sqrt_condition"] = "plane"
    bandlimit: bool = True
    checkpoint_propagation: bool = True
    odr_phase_epsilon: float = 1.0e-4
    odr_init_seed: int = 20260929
    
    
    
    odr_initialization: Literal["quadratic_lens", "uniform"] = "quadratic_lens"
    camera_sampling: Literal["projected_slm", "reference_bins"] = "projected_slm"

    
    
    
    
    
    manufacturer_fill_factor: float = 0.93
    manufacturer_reflectivity_635: float = 0.45

    def __post_init__(self) -> None:
        
        
        if self.slm_pixels != 2048:
            raise ValueError("the manuscript optical model requires 2048 x 2048 SLMs")
        if len(self.wavelengths_m_rgb) != 3 or min(self.wavelengths_m_rgb) <= 0:
            raise ValueError("three positive wavelengths in RGB order are required")
        if self.pixel_pitch_m <= 0:
            raise ValueError("pixel_pitch_m must be positive")
        if self.distance_cws_to_odr_m <= 0:
            raise ValueError("distance_cws_to_odr_m must be positive")
        if self.focal_length_m <= 0:
            raise ValueError("focal_length_m must be positive")
        if self.reference_wavelength_m not in self.wavelengths_m_rgb:
            raise ValueError("reference_wavelength_m must be one of the RGB wavelengths")
        if self.padding_pixels < 0:
            raise ValueError("padding_pixels cannot be negative")
        if not 0 < self.camera_fov_pixels <= self.slm_pixels:
            raise ValueError("camera_fov_pixels must lie inside the SLM aperture")
        if self.camera_fov_pixels % self.camera_pixels:
            raise ValueError("camera_fov_pixels must be divisible by camera_pixels")
        if self.intensity_gain <= 0 or not math.isfinite(self.intensity_gain):
            raise ValueError("intensity_gain must be finite and positive")
        if self.incident_mode not in {"plane", "sqrt_condition"}:
            raise ValueError("incident_mode must be 'plane' or 'sqrt_condition'")
        if not 0 < self.odr_phase_epsilon < 0.5:
            raise ValueError("odr_phase_epsilon must lie in (0, 0.5)")
        if self.odr_initialization not in {"quadratic_lens", "uniform"}:
            raise ValueError(
                "odr_initialization must be 'quadratic_lens' or 'uniform'"
            )
        if self.camera_sampling not in {"projected_slm", "reference_bins"}:
            raise ValueError(
                "camera_sampling must be 'projected_slm' or 'reference_bins'"
            )
        if not 0 < self.manufacturer_fill_factor <= 1:
            raise ValueError("manufacturer_fill_factor must lie in (0, 1]")
        if not 0 < self.manufacturer_reflectivity_635 <= 1:
            raise ValueError("manufacturer_reflectivity_635 must lie in (0, 1]")

    def to_dict(self) -> dict[str, object]:


        return asdict(self)

    def derived_geometry(self) -> dict[str, float]:


        return {
            "active_aperture_m": self.slm_pixels * self.pixel_pitch_m,
            "reference_focal_pitch_m": self.focal_pitch_m(
                self.reference_wavelength_m
            ),
            "camera_fov_m": self.camera_fov_m,
            "camera_reference_span_bins": self.camera_reference_span_bins,
            "camera_integration_samples": float(self.camera_fov_pixels),
            "camera_pixel_pitch_m": self.camera_pixel_pitch_m,
        }

    def focal_pitch_m(self, wavelength_m: float) -> float:


        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        return (
            wavelength_m
            * self.focal_length_m
            / (self.slm_pixels * self.pixel_pitch_m)
        )

    @property
    def camera_fov_m(self) -> float:
        if self.camera_sampling == "projected_slm":
            
            
            
            return (
                self.slm_pixels
                * self.pixel_pitch_m
                * self.focal_length_m
                / self.distance_cws_to_odr_m
            )
        return self.camera_fov_pixels * self.focal_pitch_m(
            self.reference_wavelength_m
        )

    @property
    def camera_reference_span_bins(self) -> float:


        return self.camera_fov_m / self.focal_pitch_m(self.reference_wavelength_m)

    @property
    def camera_pixel_pitch_m(self) -> float:
        return self.camera_fov_m / self.camera_pixels


def phase_from_logits(logits: Tensor) -> Tensor:


    if not logits.is_floating_point():
        raise TypeError("phase logits must be floating point")
    
    
    return TAU * torch.sigmoid(logits.float())


def circular_phase_regularizer(phase: Tensor) -> Tensor:


    if phase.ndim != 4:
        raise ValueError("phase must have shape B x C x H x W")
    horizontal = 1.0 - torch.cos(phase[..., 1:] - phase[..., :-1])
    vertical = 1.0 - torch.cos(phase[..., 1:, :] - phase[..., :-1, :])
    return (horizontal.sum() + vertical.sum()) / (
        horizontal.numel() + vertical.numel()
    )


def bandlimited_transfer_function(
    height: int,
    width: int,
    *,
    pixel_pitch_m: float,
    wavelength_m: float,
    distance_m: float,
    device: torch.device,
    bandlimit: bool = True,
) -> Tensor:







    if height <= 0 or width <= 0:
        raise ValueError("transfer-function dimensions must be positive")
    if pixel_pitch_m <= 0 or wavelength_m <= 0 or distance_m == 0:
        raise ValueError("pitch/wavelength must be positive and distance non-zero")

    
    
    
    
    
    
    
    
    fy = torch.fft.fftfreq(height, d=pixel_pitch_m, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(width, d=pixel_pitch_m, device=device, dtype=torch.float64)
    transverse_sq = fy[:, None].square() + fx[None, :].square()
    carrier_frequency = 1.0 / wavelength_m
    radial_term = carrier_frequency**2 - transverse_sq
    propagating = radial_term >= 0

    
    
    longitudinal = torch.sqrt(radial_term.clamp_min(0.0))
    residual_frequency = -transverse_sq / (longitudinal + carrier_frequency)
    residual_phase = TAU * distance_m * residual_frequency
    transfer = torch.polar(torch.ones_like(residual_phase), residual_phase).to(
        torch.complex64
    )
    mask = propagating

    if bandlimit:
        extent_x = width * pixel_pitch_m
        extent_y = height * pixel_pitch_m
        fx_limit = 1.0 / (
            wavelength_m
            * math.sqrt(1.0 + (2.0 * abs(distance_m) / extent_x) ** 2)
        )
        fy_limit = 1.0 / (
            wavelength_m
            * math.sqrt(1.0 + (2.0 * abs(distance_m) / extent_y) ** 2)
        )
        mask = mask & (fx[None, :].abs() <= fx_limit)
        mask = mask & (fy[:, None].abs() <= fy_limit)

    return transfer * mask.to(dtype=transfer.dtype)


def _propagate_with_transfer(
    field: Tensor,
    transfer: Tensor,
    padding_pixels: int,
) -> Tensor:


    if field.ndim != 4 or field.shape[1] != 1 or not torch.is_complex(field):
        raise ValueError("field must be complex B x 1 x H x W")
    if padding_pixels:
        field = F.pad(
            field,
            (padding_pixels, padding_pixels, padding_pixels, padding_pixels),
            mode="constant",
            value=0.0,
        )
    if tuple(transfer.shape) != tuple(field.shape[-2:]):
        raise ValueError("transfer function does not match the padded field")
    propagated = torch.fft.ifft2(torch.fft.fft2(field) * transfer[None, None])
    if padding_pixels:
        stop_y = propagated.shape[-2] - padding_pixels
        stop_x = propagated.shape[-1] - padding_pixels
        propagated = propagated[
            ..., padding_pixels:stop_y, padding_pixels:stop_x
        ]
    return propagated


def bandlimited_angular_spectrum(
    field: Tensor,
    *,
    pixel_pitch_m: float,
    wavelength_m: float,
    distance_m: float,
    padding_pixels: int = 0,
    bandlimit: bool = True,
) -> Tensor:


    if padding_pixels < 0:
        raise ValueError("padding_pixels cannot be negative")
    if not torch.is_complex(field):
        field = field.to(torch.complex64)
    else:
        field = field.to(torch.complex64)
    height = field.shape[-2] + 2 * padding_pixels
    width = field.shape[-1] + 2 * padding_pixels
    transfer = bandlimited_transfer_function(
        height,
        width,
        pixel_pitch_m=pixel_pitch_m,
        wavelength_m=wavelength_m,
        distance_m=distance_m,
        device=field.device,
        bandlimit=bandlimit,
    )
    return _propagate_with_transfer(field, transfer, padding_pixels)


def focal_plane_intensity(
    field: Tensor,
    *,
    wavelength_m: float,
    reference_wavelength_m: float = 532e-9,
) -> Tensor:












    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError("field must have shape B x 1 x H x W")
    if wavelength_m <= 0 or reference_wavelength_m <= 0:
        raise ValueError("wavelengths must be positive")
    if not torch.is_complex(field):
        field = field.to(torch.complex64)
    else:
        field = field.to(torch.complex64)
    
    
    
    
    shifted = torch.fft.ifftshift(field, dim=(-2, -1))
    normalization = math.sqrt(field.shape[-2] * field.shape[-1])
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(shifted, norm="backward") / normalization,
        dim=(-2, -1),
    )
    wavelength_scale = (reference_wavelength_m / wavelength_m) ** 2
    return spectrum.abs().square() * wavelength_scale


def common_focal_sampling_grid(
    native_pixels: int,
    output_samples: int,
    *,
    wavelength_m: float,
    reference_wavelength_m: float,
    device: torch.device,
    reference_span_bins: float | None = None,
    invert_axes: bool = False,
) -> Tensor:










    if not 0 < output_samples <= native_pixels:
        raise ValueError("output_samples must lie inside the native Fourier grid")
    if wavelength_m <= 0 or reference_wavelength_m <= 0:
        raise ValueError("wavelengths must be positive")
    ratio = reference_wavelength_m / wavelength_m
    span_bins = (
        float(output_samples)
        if reference_span_bins is None
        else float(reference_span_bins)
    )
    if span_bins <= 0 or not math.isfinite(span_bins):
        raise ValueError("reference_span_bins must be finite and positive")
    half_span = (span_bins / 2.0) * ratio
    if half_span > native_pixels / 2.0:
        raise ValueError("requested physical field of view exceeds this wavelength grid")

    
    output_coordinate = torch.arange(
        output_samples, device=device, dtype=torch.float64
    )
    if reference_span_bins is None:
        
        output_coordinate = output_coordinate - output_samples / 2.0
    else:
        
        
        output_coordinate = (
            output_coordinate + 0.5 - output_samples / 2.0
        ) * (span_bins / output_samples)
    if invert_axes:
        output_coordinate = -output_coordinate
    native_index = native_pixels / 2.0 + output_coordinate * ratio
    normalized = 2.0 * native_index / (native_pixels - 1.0) - 1.0
    grid_y, grid_x = torch.meshgrid(normalized, normalized, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).to(torch.float32).unsqueeze(0)


def resample_focal_intensity(
    intensity: Tensor,
    *,
    wavelength_m: float,
    reference_wavelength_m: float,
    output_samples: int,
    grid: Tensor | None = None,
    reference_span_bins: float | None = None,
    invert_axes: bool = False,
) -> Tensor:


    if intensity.ndim != 4 or intensity.shape[1] != 1:
        raise ValueError("intensity must have shape B x 1 x H x W")
    if intensity.shape[-2] != intensity.shape[-1]:
        raise ValueError("native focal plane must be square")
    if grid is None:
        grid = common_focal_sampling_grid(
            intensity.shape[-1],
            output_samples,
            wavelength_m=wavelength_m,
            reference_wavelength_m=reference_wavelength_m,
            device=intensity.device,
            reference_span_bins=reference_span_bins,
            invert_axes=invert_axes,
        )
    if tuple(grid.shape) != (1, output_samples, output_samples, 2):
        raise ValueError("sampling grid has the wrong shape")
    return F.grid_sample(
        intensity,
        grid.expand(intensity.shape[0], -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class OpticalSystem(nn.Module):














    def __init__(self, spec: OpticsSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or OpticsSpec()

        if self.spec.odr_initialization == "quadratic_lens":
            
            
            
            
            size = self.spec.slm_pixels
            coordinate = (
                torch.arange(size, dtype=torch.float64) - size / 2.0
            ) * self.spec.pixel_pitch_m
            radius_squared = coordinate.square()[:, None] + coordinate.square()[None, :]
            phase_fractions: list[Tensor] = []
            for wavelength_m in self.spec.wavelengths_m_rgb:
                phase = (
                    -math.pi
                    * radius_squared
                    / (wavelength_m * self.spec.distance_cws_to_odr_m)
                )
                phase_fractions.append(
                    (torch.remainder(phase, TAU) / TAU).to(torch.float32)
                )
            phase_fraction = torch.stack(phase_fractions, dim=0).unsqueeze(0)
        else:
            
            
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.spec.odr_init_seed)
            phase_fraction = torch.rand(
                (1, 3, self.spec.slm_pixels, self.spec.slm_pixels),
                generator=generator,
                dtype=torch.float32,
            )

        
        
        epsilon = self.spec.odr_phase_epsilon
        if self.spec.odr_initialization == "uniform":
            phase_fraction = phase_fraction * (1.0 - 2.0 * epsilon) + epsilon
        else:
            phase_fraction = phase_fraction.clamp(epsilon, 1.0 - epsilon)
        initial_odr = torch.logit(phase_fraction)
        self.odr_logits = nn.Parameter(initial_odr)

        
        
        
        self.register_buffer(
            "intensity_gain",
            torch.tensor(float(self.spec.intensity_gain), dtype=torch.float32),
        )

        
        
        self._transfer_cache: dict[tuple[object, ...], Tensor] = {}
        self._focal_grid_cache: dict[tuple[object, ...], Tensor] = {}

    @property
    def odr_parameter_count(self) -> int:
        return self.odr_logits.numel()

    @torch.no_grad()
    def set_intensity_gain(self, value: float) -> None:









        if value <= 0 or not math.isfinite(value):
            raise ValueError("camera gain must be finite and positive")
        self.intensity_gain.fill_(float(value))

    @torch.no_grad()
    def set_camera_gain(self, value: float) -> None:


        self.set_intensity_gain(value)

    def clear_transfer_cache(self) -> None:


        self._transfer_cache.clear()
        self._focal_grid_cache.clear()

    @torch.no_grad()
    def precompute_transfers(self, device: torch.device | str) -> None:


        target = torch.device(device)
        for wavelength in self.spec.wavelengths_m_rgb:
            self._transfer(wavelength, self.spec.distance_cws_to_odr_m, target)
            self._focal_grid(wavelength, target)

    def _transfer(
        self,
        wavelength_m: float,
        distance_m: float,
        device: torch.device,
    ) -> Tensor:
        padded = self.spec.slm_pixels + 2 * self.spec.padding_pixels
        key = (
            device.type,
            device.index,
            padded,
            self.spec.pixel_pitch_m,
            wavelength_m,
            distance_m,
            self.spec.bandlimit,
        )
        transfer = self._transfer_cache.get(key)
        if transfer is None:
            transfer = bandlimited_transfer_function(
                padded,
                padded,
                pixel_pitch_m=self.spec.pixel_pitch_m,
                wavelength_m=wavelength_m,
                distance_m=distance_m,
                device=device,
                bandlimit=self.spec.bandlimit,
            ).detach()
            self._transfer_cache[key] = transfer
        return transfer

    def _propagate(self, field: Tensor, wavelength_m: float, distance_m: float) -> Tensor:
        transfer = self._transfer(wavelength_m, distance_m, field.device)
        use_checkpoint = (
            self.spec.checkpoint_propagation
            and self.training
            and torch.is_grad_enabled()
            and field.requires_grad
        )
        if use_checkpoint:
            return checkpoint(
                _propagate_with_transfer,
                field,
                transfer,
                self.spec.padding_pixels,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return _propagate_with_transfer(field, transfer, self.spec.padding_pixels)

    def _focal_grid(self, wavelength_m: float, device: torch.device) -> Tensor:
        key = (
            device.type,
            device.index,
            self.spec.slm_pixels,
            self.spec.camera_fov_pixels,
            self.spec.camera_reference_span_bins,
            self.spec.camera_sampling,
            wavelength_m,
            self.spec.reference_wavelength_m,
        )
        grid = self._focal_grid_cache.get(key)
        if grid is None:
            grid = common_focal_sampling_grid(
                self.spec.slm_pixels,
                self.spec.camera_fov_pixels,
                wavelength_m=wavelength_m,
                reference_wavelength_m=self.spec.reference_wavelength_m,
                device=device,
                reference_span_bins=(
                    self.spec.camera_reference_span_bins
                    if self.spec.camera_sampling == "projected_slm"
                    else None
                ),
                invert_axes=self.spec.camera_sampling == "projected_slm",
            ).detach()
            self._focal_grid_cache[key] = grid
        return grid

    def _focal_intensity(self, field: Tensor, wavelength_m: float) -> Tensor:
        def operation(value: Tensor) -> Tensor:
            return focal_plane_intensity(
                value,
                wavelength_m=wavelength_m,
                reference_wavelength_m=self.spec.reference_wavelength_m,
            )

        use_checkpoint = (
            self.spec.checkpoint_propagation
            and self.training
            and torch.is_grad_enabled()
            and field.requires_grad
        )
        if use_checkpoint:
            return checkpoint(
                operation,
                field,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return operation(field)

    def _incident_field_channel(self, condition: Tensor, channel: int) -> Tensor:
        batch = condition.shape[0]
        size = self.spec.slm_pixels
        if self.spec.incident_mode == "plane":
            return torch.ones(
                (batch, 1, size, size),
                device=condition.device,
                dtype=torch.complex64,
            )

        
        
        
        intensity = condition[:, channel : channel + 1].float().clamp(0.0, 1.0)
        if tuple(intensity.shape[-2:]) != (size, size):
            intensity = F.interpolate(
                intensity,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
        return torch.sqrt(intensity.clamp_min(0.0)).to(torch.complex64)

    def _camera_map(self, intensity: Tensor, wavelength_m: float) -> Tensor:


        fov = self.spec.camera_fov_pixels
        intensity = resample_focal_intensity(
            intensity,
            wavelength_m=wavelength_m,
            reference_wavelength_m=self.spec.reference_wavelength_m,
            output_samples=fov,
            grid=self._focal_grid(wavelength_m, intensity.device),
        )
        bin_size = fov // self.spec.camera_pixels
        intensity = F.avg_pool2d(intensity, kernel_size=bin_size, stride=bin_size)
        intensity = intensity * self.intensity_gain.to(intensity)
        if self.spec.camera_saturation:
            intensity = intensity.clamp(0.0, 1.0)
        return intensity

    def forward(self, cws_logits: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        if cws_logits.ndim != 4 or tuple(cws_logits.shape[1:]) != (
            3,
            self.spec.slm_pixels,
            self.spec.slm_pixels,
        ):
            raise ValueError("cws_logits must have shape B x 3 x 2048 x 2048")
        if condition.ndim != 4 or condition.shape[1] != 3:
            raise ValueError("condition must have shape B x 3 x H x W")
        if cws_logits.shape[0] != condition.shape[0]:
            raise ValueError("cws_logits and condition must have the same batch size")
        if cws_logits.device != condition.device:
            raise ValueError("cws_logits and condition must be on the same device")
        if self.odr_logits.device != cws_logits.device:
            raise ValueError("move OpticalSystem to the same device as cws_logits")

        camera_channels: list[Tensor] = []
        cws_regularizers: list[Tensor] = []
        odr_regularizers: list[Tensor] = []

        
        
        with torch.autocast(device_type=cws_logits.device.type, enabled=False):
            for channel, wavelength_m in enumerate(self.spec.wavelengths_m_rgb):
                cws_phase = phase_from_logits(cws_logits[:, channel : channel + 1])
                odr_phase = phase_from_logits(self.odr_logits[:, channel : channel + 1])
                incident = self._incident_field_channel(condition, channel)

                after_cws = incident * torch.exp(1j * cws_phase)
                at_odr = self._propagate(
                    after_cws, wavelength_m, self.spec.distance_cws_to_odr_m
                )
                after_odr = at_odr * torch.exp(1j * odr_phase)
                focal_intensity = self._focal_intensity(after_odr, wavelength_m)
                camera_channels.append(
                    self._camera_map(focal_intensity, wavelength_m)
                )

                cws_regularizers.append(circular_phase_regularizer(cws_phase))
                odr_regularizers.append(circular_phase_regularizer(odr_phase))

            reconstruction = torch.cat(camera_channels, dim=1)
            
            phase_regularizer = torch.stack(cws_regularizers).mean()
            phase_regularizer = phase_regularizer + torch.stack(odr_regularizers).mean()
        return reconstruction, phase_regularizer

    def odr_phase(self) -> Tensor:


        return phase_from_logits(self.odr_logits)

    @torch.no_grad()
    def export_odr_phase(self, *, cpu: bool = True) -> Tensor:


        phase = self.odr_phase()[0].detach()
        return phase.cpu() if cpu else phase


def _gaussian_window(channels: int, reference: Tensor) -> Tensor:
    coordinate = torch.arange(11, device=reference.device, dtype=reference.dtype) - 5
    vector = torch.exp(-coordinate.square() / (2.0 * 1.5**2))
    vector = vector / vector.sum()
    return torch.outer(vector, vector).expand(channels, 1, 11, 11).contiguous()


def _ssim_and_contrast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    window = _gaussian_window(x.shape[1], x)
    mu_x = F.conv2d(x, window, groups=x.shape[1])
    mu_y = F.conv2d(y, window, groups=y.shape[1])
    variance_x = (F.conv2d(x.square(), window, groups=x.shape[1]) - mu_x.square()).clamp_min(0)
    variance_y = (F.conv2d(y.square(), window, groups=y.shape[1]) - mu_y.square()).clamp_min(0)
    covariance = F.conv2d(x * y, window, groups=x.shape[1]) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    luminance = (2.0 * mu_x * mu_y + c1) / (mu_x.square() + mu_y.square() + c1)
    contrast_structure = (2.0 * covariance + c2) / (variance_x + variance_y + c2)
    spatial = (2, 3)
    return (
        (luminance * contrast_structure).mean(dim=spatial),
        contrast_structure.mean(dim=spatial),
    )


def ms_ssim_index(output: Tensor, target: Tensor) -> Tensor:








    if output.ndim != 4 or output.shape != target.shape:
        raise ValueError("MS-SSIM tensors must share B x C x H x W shape")
    if min(output.shape[-2:]) < 161:
        raise ValueError("five-scale 11 x 11 MS-SSIM requires dimensions >= 161")
    
    
    weights = torch.tensor(
        (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        device=output.device,
        dtype=torch.float32,
    )
    current_output = output.float()
    current_target = target.float()
    factors: list[Tensor] = []
    with torch.autocast(device_type=output.device.type, enabled=False):
        for level in range(5):
            score, contrast = _ssim_and_contrast(current_output, current_target)
            factors.append((score if level == 4 else contrast).clamp(1.0e-6, 1.0))
            if level != 4:
                padding = (
                    current_output.shape[-2] % 2,
                    current_output.shape[-1] % 2,
                )
                current_output = F.avg_pool2d(current_output, 2, padding=padding)
                current_target = F.avg_pool2d(current_target, 2, padding=padding)
        stacked = torch.stack(factors, dim=0)
        return (stacked ** weights[:, None, None]).prod(dim=0).mean()


def image_objective(output: Tensor, target: Tensor) -> Tensor:


    if output.shape != target.shape:
        raise ValueError("output and target must have identical shapes")
    charbonnier = torch.sqrt((output - target).float().square() + 1.0e-3**2).mean()
    return charbonnier + 0.1 * (1.0 - ms_ssim_index(output, target))


def simulation_objective(
    output: Tensor,
    target: Tensor,
    phase_regularizer: Tensor,
) -> Tensor:


    return image_objective(output, target) + 1.0e-4 * phase_regularizer





"""Read paired images without modifying source data.
The legacy acquisition layout is supported alongside generic input/gt trees.
Images retain their complete field of view when resized to the network grid.
All selected pairs enter training; folder names alone do not create a holdout.
"""



import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor


_TDV11_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_TDV11_PRETRAIN_DOMAINS = {
    "2d": ("droplet", "mist", "tissue"),
    "3d": ("droplet", "mist", "tissue"),
    "sim": ("droplet", "mist", "tissue", "multi-source"),
}
_TDV11_FINE_DOMAINS = ("droplet", "mist", "tissue")
_TDV11_TEST_DOMAINS = ("multi-source",)


@dataclass(frozen=True)
class TransferPair:


    stage: str
    family: str
    domain: str
    key: str
    group: str
    condition: Path
    target: Path


def _tdv11_natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _tdv11_direct_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    nested = [path for path in directory.iterdir() if path.is_dir()]
    if nested:
        raise ValueError(
            f"nested directories are not part of the v11 data contract: {nested[:3]}"
        )
    unexpected = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() not in _TDV11_IMAGE_EXTENSIONS
    ]
    if unexpected:
        raise ValueError(
            f"unsupported files would otherwise be dropped under {directory}: "
            f"{unexpected[:3]}"
        )
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _TDV11_IMAGE_EXTENSIONS
        ),
        key=lambda path: _tdv11_natural_key(path.name),
    )


def _tdv11_check_immediate_directories(
    root: Path, expected: Iterable[str]
) -> None:
    expected_set = set(expected)
    actual = {path.name for path in root.iterdir() if path.is_dir()}
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)
    if missing or unexpected:
        raise ValueError(
            f"directory contract mismatch under {root}; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _tdv11_canonical_id(family: str, domain: str, path: Path) -> str:
    stem = path.stem
    suffix = path.suffix.lower()
    if family == "2d":
        if suffix != ".bmp" or re.fullmatch(r"\d{3}", stem) is None:
            raise ValueError(f"unexpected pre-train/2d filename: {path.name}")
        return f"{int(stem):04d}"

    if family == "3d":
        match = re.fullmatch(r"(.+)-(shui|yu|wu)-5", stem, re.IGNORECASE)
        if suffix != ".png" or match is None:
            raise ValueError(f"unexpected pre-train/3d filename: {path.name}")
        required_token = {
            "gt": "shui",
            "droplet": "shui",
            "mist": "yu",
            "tissue": "wu",
        }[domain]
        if match.group(2).lower() != required_token:
            raise ValueError(
                f"unexpected token for pre-train/3d/{domain}: {path.name}"
            )
        return match.group(1).lower()

    if family == "sim":
        pattern = r"1 \((\d+)\)" if domain == "gt" else r"IMG \((\d+)\)"
        match = re.fullmatch(pattern, stem, re.IGNORECASE)
        if suffix != ".png" or match is None:
            raise ValueError(f"unexpected pre-train/sim/{domain} filename: {path.name}")
        return f"{int(match.group(1)):04d}"

    if family == "real":
        match = re.fullmatch(r"group_(\d+)_(\d+)", stem, re.IGNORECASE)
        if suffix not in {".jpg", ".jpeg"} or match is None:
            raise ValueError(f"unexpected grouped real-data filename: {path.name}")
        return f"group_{int(match.group(1)):02d}_{int(match.group(2)):03d}"

    raise ValueError(f"unknown data family: {family}")


def _tdv11_index_domain(
    directory: Path, family: str, domain: str
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in _tdv11_direct_image_files(directory):
        local_id = _tdv11_canonical_id(family, domain, path)
        if local_id in indexed:
            raise ValueError(
                f"duplicate canonical identity {local_id!r} under {directory}: "
                f"{indexed[local_id].name}, {path.name}"
            )
        indexed[local_id] = path
    if not indexed:
        raise ValueError(f"no images found under {directory}")
    return indexed


def _tdv11_pair_domains(
    *,
    stage: str,
    family: str,
    root: Path,
    condition_domains: Sequence[str],
) -> list[TransferPair]:
    expected_domains = ("gt", *condition_domains)
    _tdv11_check_immediate_directories(root, expected_domains)
    indexed = {
        domain: _tdv11_index_domain(root / domain, family, domain)
        for domain in expected_domains
    }
    target_ids = set(indexed["gt"])
    pairs: list[TransferPair] = []
    for domain in condition_domains:
        condition_ids = set(indexed[domain])
        missing = sorted(target_ids - condition_ids, key=_tdv11_natural_key)
        extra = sorted(condition_ids - target_ids, key=_tdv11_natural_key)
        if missing or extra:
            raise ValueError(
                f"unpaired v11 files in {stage}/{family}/{domain}; "
                f"missing_vs_gt={missing[:10]}, extra_vs_gt={extra[:10]}"
            )
        for local_id in sorted(target_ids, key=_tdv11_natural_key):
            if family == "real":
                group_match = re.match(r"group_(\d+)_", local_id)
                if group_match is None:
                    raise AssertionError(local_id)
                group = f"group_{int(group_match.group(1)):02d}"
            else:
                group = f"{family}:{local_id}"
            pairs.append(
                TransferPair(
                    stage=stage,
                    family=family,
                    domain=domain,
                    key=f"{stage}/{family}/{domain}/{local_id}",
                    group=group,
                    condition=indexed[domain][local_id],
                    target=indexed["gt"][local_id],
                )
            )
    return pairs


def discover_transfer_pairs(
    dataset_root: str | Path,
    stage: str,
) -> list[TransferPair]:







    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if stage not in {"pre-train", "fine-tune", "test"}:
        raise ValueError("stage must be 'pre-train', 'fine-tune' or 'test'")

    stage_root = root / stage
    if not stage_root.is_dir():
        raise FileNotFoundError(stage_root)

    if stage == "pre-train":
        _tdv11_check_immediate_directories(stage_root, _TDV11_PRETRAIN_DOMAINS)
        pairs: list[TransferPair] = []
        for family in ("2d", "3d", "sim"):
            pairs.extend(
                _tdv11_pair_domains(
                    stage=stage,
                    family=family,
                    root=stage_root / family,
                    condition_domains=_TDV11_PRETRAIN_DOMAINS[family],
                )
            )
    elif stage == "fine-tune":
        pairs = _tdv11_pair_domains(
            stage=stage,
            family="real",
            root=stage_root,
            condition_domains=_TDV11_FINE_DOMAINS,
        )
    else:
        pairs = _tdv11_pair_domains(
            stage=stage,
            family="real",
            root=stage_root,
            condition_domains=_TDV11_TEST_DOMAINS,
        )

    keys = [pair.key for pair in pairs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("v11 discovery produced duplicate pair keys")
    return sorted(pairs, key=lambda pair: _tdv11_natural_key(pair.key))










@lru_cache(maxsize=256)
def _tdv11_resized_rgb_bytes(
    path_string: str,
    output_size: int,
    size_bytes: int,
    mtime_ns: int,
) -> bytes:
    del size_bytes, mtime_ns  
    with Image.open(path_string) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Use one image file per frame; multi-page images are not supported")
        if image.mode in {"I", "F", "I;16", "I;16B", "I;16L"}:
            raise ValueError("Input images must be calibrated 8-bit images; convert high-bit-depth data using a fixed camera calibration before loading")
        image.load()
        rgb = image.convert("RGB")
        if rgb.size != (output_size, output_size):
            
            
            rgb = rgb.resize(
                (output_size, output_size),
                resample=Image.Resampling.LANCZOS,
            )
        return rgb.tobytes()


def load_rgb_tensor(path: str | Path, size: int = 256) -> Tensor:


    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in _TDV11_IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {source}")
    if int(size) <= 0:
        raise ValueError("size must be a positive integer")
    output_size = int(size)
    stat = source.stat()
    payload = _tdv11_resized_rgb_bytes(
        str(source), output_size, stat.st_size, stat.st_mtime_ns
    )
    
    
    array = np.frombuffer(payload, dtype=np.uint8).reshape(
        output_size, output_size, 3
    ).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _tdv11_relative_path(
    path: Path,
    pair: TransferPair,
    dataset_root: Path | None,
) -> str:
    resolved = path.resolve()
    if dataset_root is not None:
        return Path(os.path.relpath(resolved, Path(dataset_root).resolve())).as_posix()
    parts = resolved.parts
    try:
        start = parts.index(pair.stage)
    except ValueError:
        
        
        return path.name
    return Path(*parts[start:]).as_posix()


def data_signature(
    pairs: Sequence[TransferPair],
    dataset_root: str | Path | None = None,
) -> str:


    root = None if dataset_root is None else Path(dataset_root).expanduser().resolve()
    rows = []
    for pair in pairs:
        rows.append(
            (
                pair.stage,
                pair.family,
                pair.domain,
                pair.key,
                pair.group,
                _tdv11_relative_path(pair.condition, pair, root),
                _tdv11_relative_path(pair.target, pair, root),
            )
        )
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()







import contextlib
import csv
import ctypes
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np



TRANSFER_SEED=20260930


def hide_internal(path):

    path = Path(path)
    if os.name == 'nt' and path.exists():
        a = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if a != -1:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), a | 2)

def _windows_attributes(path):

    path = Path(path)
    if os.name != 'nt' or not path.exists():
        return None
    value = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    return None if value == -1 else value

def _clear_hidden(path):

    attributes = _windows_attributes(path)
    if attributes is not None and attributes & 2:
        ctypes.windll.kernel32.SetFileAttributesW(str(Path(path)), attributes & ~2)
        return attributes
    return None

def _atomic_replace(temporary, destination, *, hide_after=False):

    temporary, destination = Path(temporary), Path(destination)
    previous_attributes = _clear_hidden(destination)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if previous_attributes is not None and destination.exists():
            ctypes.windll.kernel32.SetFileAttributesW(
                str(destination), previous_attributes
            )
        raise
    if hide_after or (previous_attributes is not None and previous_attributes & 2):
        hide_internal(destination)

def write_json(path, value, internal=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    _atomic_replace(temporary, path, hide_after=internal)

@contextlib.contextmanager
def exclusive_run(path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    hide_internal(path)
    locked = False
    try:
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError(f'Another process holds this run lock: {path}') from error
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

def transfer_device(name):
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Install a compatible PyTorch build or select --device cpu.')
    return torch.device(name)

def seed_transfer(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def mixed_precision(device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == 'cuda')

@torch.no_grad()
def initialize_icnr(model):







    for module in model.phase_decoder.modules():
        if isinstance(module, PixelShuffleStage):
            weight = module.expand.weight
            base = torch.empty((weight.shape[0] // 4, *weight.shape[1:]), device=weight.device)
            nn.init.kaiming_normal_(base)
            weight.copy_(base.repeat_interleave(4, dim=0))
            if module.expand.bias is not None:
                module.expand.bias.zero_()



def transfer_batch(pairs, device, augment=False):
    conditions, targets = [], []
    for pair in pairs:
        c = load_rgb_tensor(pair.condition)
        t = load_rgb_tensor(pair.target)
        if augment:
            turns = random.randrange(4)
            c, t = torch.rot90(c, turns, (-2, -1)), torch.rot90(t, turns, (-2, -1))
            if random.random() < .5:
                c, t = c.flip(-1), t.flip(-1)
        conditions.append(c)
        targets.append(t)
    return torch.stack(conditions).to(device), torch.stack(targets).to(device)

def stable_noise(pair_key, device, seed=TRANSFER_SEED):
    value = int.from_bytes(hashlib.sha256(f'{seed}|{pair_key}'.encode('utf-8')).digest()[:8], 'little')
    generator = torch.Generator(device='cpu').manual_seed(value & ((1 << 63) - 1))
    return torch.randn((1, 8, 32, 32), generator=generator).to(device)

def predict_transfer(model, optics, condition, key, seed=TRANSFER_SEED, amp=True):

    with mixed_precision(condition.device, amp):
        generated = model.generate_phase_logits(condition, initial_noise=stable_noise(key, condition.device, seed),
                                                checkpoint_denoiser=False, checkpoint_decoder=False)
    output, _ = optics(generated.phase_logits, condition)
    return output, generated.phase_logits

def _cpu_state(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}

def _capture_rng(device):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
    }

def _restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _rgb8(tensor):
    return tensor.detach().float().cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()

def visual_sheet(rows, destination, title, reconstruction_label='Optical reconstruction'):

    tile = 256
    canvas = Image.new('RGB', (tile * 3, 54 + len(rows) * (tile * 2 + 24)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8,5), title, fill='black')
    for j,label in enumerate(('Condition', reconstruction_label, 'GT')):
        draw.text((j*tile+8,30),label,fill='black')
    y = 54
    for key, c, p, t in rows:
        draw.text((8,y+4),str(key),fill='black')
        y += 24
        arrays = [_rgb8(v) for v in (c,p,t)]
        for j,a in enumerate(arrays):
            canvas.paste(Image.fromarray(a),(j*tile,y))
        y += tile
        
        
        for j,a in enumerate(arrays):
            detail = Image.fromarray(a[96:160,96:160]).resize((tile,tile),Image.Resampling.NEAREST)
            canvas.paste(detail,(j*tile,y))
        y += tile
    canvas.save(destination)







def _phase_optimizer(model, optics, phase, rates):






    model.requires_grad_(False).eval()
    optics.requires_grad_(False).eval()
    model.zero_grad(set_to_none=True)
    optics.zero_grad(set_to_none=True)
    groups = []

    def enable(module, name, rate):
        if not math.isfinite(float(rate)) or rate <= 0:
            raise ValueError(f"Invalid learning rate for {name}")
        module.requires_grad_(True).train()
        groups.append({"params": list(module.parameters()), "lr": float(rate), "name": name})

    if phase == "autoencoder":
        enable(model.target_encoder, "E", rates["digital"])
        enable(model.aux_decoder, "R", rates["digital"])
    elif phase == "joint_pretrain":
        if not bool(model.latent_stats_fitted):
            raise ValueError("Complete latent construction before joint pretraining")
        enable(model.condition_encoder, "C", rates["digital"])
        enable(model.denoiser, "U", rates["digital"])
        enable(model.phase_decoder, "D", rates["decoder"])
        enable(optics, "ODR", rates["odr"])
    else:
        raise ValueError(f"The two-stage workflow has no stage {phase!r}")
    return torch.optim.Adam(groups)


def _training_objective(model, optics, condition, target, phase, schedule, settings):







    if phase == "autoencoder":
        latent_loss = image_objective(model.reconstruct_target(target), target)
        return latent_loss, {"latent": latent_loss}
    if phase != "joint_pretrain":
        raise ValueError(f"Unsupported training stage: {phase}")
    with torch.no_grad():
        target_latent = model.encode_target_standardized(target)
    diffusion = noise_prediction_loss(model, condition, target_latent, schedule)
    generated = model.generate_phase_logits(condition, schedule=schedule)
    reconstruction, regularizer = optics(generated.phase_logits, condition)
    image_loss = image_objective(reconstruction, target)
    optical = image_loss + 1.0e-4 * regularizer
    return diffusion + optical, {
        "diffusion": diffusion,
        "optical": optical,
        "image": image_loss,
        "phase_regularizer": regularizer,
    }


def training_entry_v13(stage, argv=None):

    import argparse
    parser = argparse.ArgumentParser(
        description=("Stage 1: construct and freeze the target latent space." if stage == "latent"
                     else "Stage 2: train the K=8 digital generator and SLM2 jointly.")
    )
    parser.add_argument("--dataset", type=Path, help="Paired dataset folder; opens a chooser when omitted")
    parser.add_argument("--output", type=Path, default=default_v13_run_home(), help="Parent folder for timestamped runs")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--epochs", type=int, default=24 if stage == "latent" else 4)
    parser.add_argument("--batch-size", type=int, default=32 if stage == "latent" else 4)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--resume", type=Path, help="Matching resume checkpoint inside the original run/support folder")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA bfloat16 mixed precision")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--monitor-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    if stage == "joint":
        parser.add_argument("--latent-checkpoint", type=Path, help="Completed stage-1 latent_space_v13.pt; chooser if omitted")
    args = parser.parse_args(argv)
    dataset = args.dataset or choose_dataset_v13("latent construction" if stage == "latent" else "joint pretraining")
    settings = {
        "dataset_root": str(dataset), "run_home": str(args.output), "device": args.device,
        "seed": args.seed, "amp": not args.no_amp, "augment": False,
        "checkpoint_every": args.checkpoint_every, "monitor_every": args.monitor_every,
        "print_every": args.print_every, "resume_checkpoint": str(args.resume) if args.resume else None,
    }
    if stage == "latent":
        settings.update(ae_epochs=args.epochs, digital_batch=args.batch_size)
        return run_latent_construction(settings)
    latent_checkpoint = args.latent_checkpoint
    if latent_checkpoint is None and args.resume is None:
        latent_checkpoint = choose_checkpoint_v13()
    settings.update(joint_epochs=args.epochs, optical_batch=args.batch_size,
                    latent_checkpoint=str(latent_checkpoint) if latent_checkpoint else None)
    return run_joint_pretraining(settings)



import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


V13_FORMAT = "epi-two-stage-v13"
V13_ROLE_LATENT = "latent_space"
V13_ROLE_PRETRAIN = "pre_train"
V13_WORKFLOW = "strict_two_stage"
V13_PHASES = ("autoencoder", "joint_pretrain")


def default_v13_run_home() -> Path:


    return Path(__file__).resolve().parent.parent / "training_outputs_v13"


def latest_completed_latent_v13(run_home) -> Path:
    candidates = []
    for path in Path(run_home).glob("*/latent_space_v13.pt"):
        marker = path.parent / "completed_latent_v13.json"
        if marker.is_file():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if (
                record.get("status") == "completed"
                and record.get("checkpoint") == path.name
                and record.get("role") == V13_ROLE_LATENT
            ):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed v13 latent checkpoint was found. Run latent space construction.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_completed_pretrain_v13(run_home) -> Path:
    candidates = []
    for path in Path(run_home).glob("*/pretrained_v13.pt"):
        marker = path.parent / "completed_pretrain_v13.json"
        if marker.is_file():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if (
                record.get("status") == "completed"
                and record.get("checkpoint") == path.name
                and record.get("role") == V13_ROLE_PRETRAIN
            ):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed v13 joint checkpoint was found. Run pre-train.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _v13_data_fingerprint(pairs, dataset_root) -> str:


    root = Path(dataset_root).expanduser().resolve()
    digest = hashlib.sha256(data_signature(pairs, root).encode("ascii"))
    paths = {pair.condition.resolve() for pair in pairs}
    paths.update(pair.target.resolve() for pair in pairs)
    for path in sorted(paths, key=lambda value: value.as_posix().lower()):
        stat = path.stat()
        
        
        
        
        
        try:
            relative = Path(os.path.relpath(path, root)).as_posix()
        except ValueError:
            relative = path.as_posix()
        digest.update(
            f"{relative}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _v13_epoch_permutation(length: int, seed: int, phase: str, epoch: int):
    digest = hashlib.sha256(f"{seed}|{phase}|{epoch}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "little"))
    order = list(range(length))
    generator.shuffle(order)
    return order


def _v13_monitor_pairs(pairs):


    buckets = {}
    for pair in pairs:
        buckets.setdefault((pair.stage, pair.family, pair.domain), []).append(pair)
    selected = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda pair: _tdv11_natural_key(pair.key))
        selected.append(rows[len(rows) // 2])

    groups = {}
    for pair in pairs:
        if pair.stage == "test":
            groups.setdefault(pair.group, []).append(pair)
    for group in sorted(groups, key=_tdv11_natural_key):
        rows = sorted(groups[group], key=lambda pair: _tdv11_natural_key(pair.key))
        selected.append(rows[len(rows) // 2])
    return list({pair.key: pair for pair in selected}.values())


def _v13_target_pairs(pairs):


    return list({pair.target.resolve(): pair for pair in pairs}.values())


def _v13_stage_settings(settings, stage_mode: str) -> None:
    if stage_mode not in {"latent", "joint", "both"}:
        raise ValueError("stage_mode must be 'latent', 'joint' or 'both'")
    required = ["checkpoint_every", "monitor_every", "print_every"]
    if stage_mode in {"latent", "both"}:
        required.extend(("ae_epochs", "digital_batch"))
    if stage_mode in {"joint", "both"}:
        required.extend(("joint_epochs", "optical_batch"))
    for name in required:
        if type(settings.get(name)) is not int or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(settings.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if type(settings.get("amp")) is not bool:
        raise ValueError("amp must be bool")
    if settings.get("augment") is not False:
        raise ValueError("strict v13 training requires augment=False")


def _v13_phase_name(progress) -> str:
    index = int(progress.get("phase_index", 0))
    if index >= 2:
        return "joint_pretrain"
    return V13_PHASES[index]


def save_v13_checkpoint(
    path,
    model,
    optics,
    settings,
    progress,
    optimizer=None,
    *,
    role=None,
    checkpoint_kind="training",
):


    progress_copy = dict(progress)
    role = role or progress_copy.get("role")
    if role not in {V13_ROLE_LATENT, V13_ROLE_PRETRAIN}:
        raise ValueError("v13 checkpoint role must be latent_space or pre_train")
    payload = {
        "format": V13_FORMAT,
        "role": role,
        "workflow": V13_WORKFLOW,
        "phase": _v13_phase_name(progress_copy),
        "checkpoint_kind": str(checkpoint_kind),
        "model_config": model.config.to_dict(),
        "diffusion_config": model.diffusion_config.to_dict(),
        "optics_spec": optics.spec.to_dict(),
        "model_state": _cpu_state(model),
        "optics_state": _cpu_state(optics),
        "settings": dict(settings),
        "progress": progress_copy,
        "data_scope": settings.get("data_scope"),
        "stage_counts": dict(settings.get("stage_counts", {})),
        "test_data_used_for_training": bool(
            settings.get("test_data_used_for_training", False)
        ),
        "held_out_pairs": int(settings.get("held_out_pairs", 0)),
        "objective_contract": {
            "latent": "S66: Charbonnier + 0.1*(1-MS-SSIM)",
            "joint": "Ldiff + Lopt^K=8",
            "forbidden": [
                "R(generated_latent)",
                "generated-target latent regression",
                "latent alignment",
                "A1/A2 warm-up",
                "texture/spectrum/periodicity additions",
            ],
        },
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    payload["rng"] = _capture_rng(next(model.parameters()).device)
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    torch.save(payload, temporary)
    _atomic_replace(temporary, path)


def load_v13_checkpoint(path, device, require_final=False):


    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != V13_FORMAT:
        raise ValueError("An epi-two-stage-v13 checkpoint is required")
    if payload.get("workflow") != V13_WORKFLOW:
        raise ValueError("Checkpoint is not from the strict two-stage workflow")
    if (
        payload.get("model_config") != ModelConfig().to_dict()
        or payload.get("diffusion_config") != DiffusionConfig().to_dict()
        or payload.get("optics_spec") != OpticsSpec().to_dict()
    ):
        raise ValueError("Checkpoint architecture, K=8 or optical geometry differs")
    progress = payload.get("progress", {})
    if require_final:
        history = progress.get("history", [])
        if (
            payload.get("role") != V13_ROLE_PRETRAIN
            or payload.get("phase") != "joint_pretrain"
            or progress.get("phase_index") != 2
            or [row.get("phase") for row in history] != list(V13_PHASES)
        ):
            raise ValueError("Inference requires a completed strict two-stage checkpoint")
        if not bool(payload["model_state"]["latent_stats_fitted"]):
            raise ValueError("Final checkpoint has no fitted latent statistics")

    model = EPIModel(
        ModelConfig(**payload["model_config"]),
        DiffusionConfig(**payload["diffusion_config"]),
        gradient_checkpointing=True,
    ).to(device)
    optics = OpticalSystem(OpticsSpec(**payload["optics_spec"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    optics.load_state_dict(payload["optics_state"], strict=True)
    return model, optics, payload


def _v13_validate_latent_payload(payload, settings) -> None:
    progress = payload.get("progress", {})
    history = progress.get("history", [])
    if (
        payload.get("role") != V13_ROLE_LATENT
        or payload.get("phase") != "joint_pretrain"
        or progress.get("phase_index") != 1
        or [row.get("phase") for row in history] != ["autoencoder"]
    ):
        raise ValueError("Joint training requires a completed v13 latent-space checkpoint")
    if not bool(payload["model_state"]["latent_stats_fitted"]):
        raise ValueError("Latent-space checkpoint has no fitted statistics")
    prior = payload.get("settings", {})
    for name in ("data_fingerprint", "pair_signature", "training_pairs", "target_paths"):
        if prior.get(name) != settings.get(name):
            raise ValueError(f"Latent checkpoint and selected dataset differ: {name}")


def _v13_write_pairs(path, pairs) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "use",
                "original_stage",
                "family",
                "domain",
                "key",
                "group",
                "condition",
                "target",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    "training",
                    pair.stage,
                    pair.family,
                    pair.domain,
                    pair.key,
                    pair.group,
                    pair.condition,
                    pair.target,
                ]
            )
    _atomic_replace(temporary, path, hide_after=True)


def _v13_loss_columns():
    return [
        "phase",
        "epoch",
        "samples",
        "updates",
        "total_loss",
        "latent_loss",
        "diffusion_loss",
        "optical_loss",
        "image_loss",
        "phase_regularizer",
        "monitor_loss",
        "selected_for_next_stage",
    ]


def _v13_write_epoch_losses(path, rows) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_v13_loss_columns())
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in writer.fieldnames})
    _atomic_replace(temporary, path)


def _v13_write_loss_curve(path, rows, phase) -> None:






    selected = sorted(
        (row for row in rows if row.get("phase") == phase),
        key=lambda row: int(row["epoch"]),
    )
    if not selected:
        return
    if phase == "autoencoder":
        specifications = (("total_loss", "total", 1.0, (34, 96, 180)),)
    elif phase == "joint_pretrain":
        specifications = (
            ("total_loss", "total", 1.0, (25, 85, 165)),
            ("diffusion_loss", "diffusion", 1.0, (220, 90, 45)),
            ("optical_loss", "optical", 1.0, (35, 150, 85)),
            ("image_loss", "image", 1.0, (155, 75, 175)),
            (
                "phase_regularizer",
                "1e-4 phase regularizer",
                1.0e-4,
                (120, 120, 120),
            ),
        )
    else:
        raise ValueError(phase)

    series = []
    for column, label, scale, color in specifications:
        points = []
        for row in selected:
            value = row.get(column, "")
            if value == "" or value is None:
                continue
            value = float(value) * scale
            if not math.isfinite(value) or value <= 0:
                continue
            points.append((int(row["epoch"]), value))
        if points:
            series.append((label, color, points))
    if not series:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required to export v13 loss curves"
        ) from error

    colors = {
        tuple(int(channel) for channel in color): tuple(channel / 255 for channel in color)
        for _, color, _ in series
    }
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
        }
    ):
        figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        for label, color, points in series:
            axis.plot(
                [epoch for epoch, _ in points],
                [value for _, value in points],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                label=label,
                color=colors[color],
            )
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Epoch-mean loss")
        axis.set_title(
            "Latent construction" if phase == "autoencoder" else "Joint pretraining"
        )
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, which="both", linewidth=0.6, alpha=0.28)
        axis.legend(frameon=False, ncol=2)

        png = Path(path)
        pdf = png.with_suffix(".pdf")
        temporary_png = png.with_name(png.stem + ".writing" + png.suffix)
        temporary_pdf = pdf.with_name(pdf.stem + ".writing" + pdf.suffix)
        figure.savefig(temporary_png, dpi=220, facecolor="white")
        figure.savefig(temporary_pdf, facecolor="white")
        plt.close(figure)
        _atomic_replace(temporary_png, png)
        _atomic_replace(temporary_pdf, pdf)


@torch.no_grad()
def _v13_fit_latent_stats(model, target_pairs, device, amp, batch_size=32):


    total = torch.zeros(8, dtype=torch.float64)
    square = torch.zeros_like(total)
    count = 0
    model.eval()
    for start in range(0, len(target_pairs), batch_size):
        target = torch.stack(
            [load_rgb_tensor(pair.target) for pair in target_pairs[start : start + batch_size]]
        ).to(device)
        with mixed_precision(device, amp):
            latent = model.encode_target_raw(target)
        latent = latent.double().cpu()
        total += latent.sum((0, 2, 3))
        square += latent.square().sum((0, 2, 3))
        count += latent.shape[0] * latent.shape[2] * latent.shape[3]
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(0.0)
    std = (variance + 1.0e-6).sqrt()
    model.set_latent_stats(mean.float().to(device), std.float().to(device))
    return {
        "target_paths": len(target_pairs),
        "latent_elements_per_channel": int(count),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epsilon": 1.0e-6,
    }


def _v13_rgb8(tensor):
    return (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def _v13_visual_sheet(rows, destination, title, labels):
    tile = 256
    canvas = Image.new("RGB", (tile * 3, 54 + len(rows) * (tile * 2 + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    for column, label in enumerate(labels):
        draw.text((column * tile + 8, 30), label, fill="black")
    y = 54
    for key, first, prediction, target in rows:
        draw.text((8, y + 4), str(key), fill="black")
        y += 24
        arrays = [_v13_rgb8(value) for value in (first, prediction, target)]
        for column, array in enumerate(arrays):
            canvas.paste(Image.fromarray(array), (column * tile, y))
        y += tile
        for column, array in enumerate(arrays):
            crop = Image.fromarray(array[96:160, 96:160]).resize(
                (tile, tile), Image.Resampling.NEAREST
            )
            canvas.paste(crop, (column * tile, y))
        y += tile
    canvas.save(destination)


@torch.no_grad()
def _v13_monitor(model, optics, pairs, phase, settings, preview_prefix=None):


    device = next(model.parameters()).device
    rng = _capture_rng(device)
    model.eval()
    optics.eval()
    schedule = LinearDiffusionSchedule(model.diffusion_config, device=device)
    totals = {name: 0.0 for name in ("total", "latent", "diffusion", "optical", "image", "phase_regularizer")}
    metrics = []
    rows = []
    try:
        for pair in pairs:
            condition, target = transfer_batch([pair], device, False)
            with mixed_precision(device, settings["amp"]):
                if phase == "autoencoder":
                    prediction = model.reconstruct_target(target)
                    latent_loss = image_objective(prediction, target)
                    values = {
                        "total": latent_loss,
                        "latent": latent_loss,
                    }
                    first = target
                elif phase == "joint_pretrain":
                    clean = model.encode_target_standardized(target)
                    number = int.from_bytes(
                        hashlib.sha256(pair.key.encode("utf-8")).digest()[:4],
                        "little",
                    )
                    diffusion_loss = noise_prediction_loss(
                        model,
                        condition,
                        clean,
                        schedule,
                        timesteps=torch.tensor([1 + number % 1000], device=device),
                        noise=stable_noise(pair.key, device, settings["seed"]),
                    )
                    generated = model.generate_phase_logits(
                        condition,
                        initial_noise=stable_noise(pair.key, device, settings["seed"]),
                        schedule=schedule,
                        checkpoint_denoiser=False,
                        checkpoint_decoder=False,
                    )
                    prediction, phase_regularizer = optics(
                        generated.phase_logits, condition
                    )
                    image_loss = image_objective(prediction, target)
                    optical_loss = image_loss + 1.0e-4 * phase_regularizer
                    values = {
                        "total": diffusion_loss + optical_loss,
                        "diffusion": diffusion_loss,
                        "optical": optical_loss,
                        "image": image_loss,
                        "phase_regularizer": phase_regularizer,
                    }
                    first = condition
                else:
                    raise ValueError(phase)
            for name, value in values.items():
                totals[name] += float(value)
            mse = (prediction.float() - target).square().mean().clamp_min(1.0e-12)
            metrics.append(
                {
                    "key": pair.key,
                    "psnr": float(-10.0 * torch.log10(mse)),
                    "ms_ssim": float(ms_ssim_index(prediction, target)),
                }
            )
            rows.append(
                (
                    pair.key,
                    first[0].detach().cpu(),
                    prediction[0].detach().cpu(),
                    target[0].detach().cpu(),
                )
            )
        count = len(pairs)
        report = {
            "phase": phase,
            "evaluation_type": "training-monitor",
            "sample_count": count,
            "losses": {
                name: value / count
                for name, value in totals.items()
                if value != 0.0 or name in {"total", "latent" if phase == "autoencoder" else "diffusion"}
            },
            "metrics": metrics,
            "independent_validation": False,
        }
        if preview_prefix and rows:
            prefix = Path(preview_prefix)
            labels = (
                ("Target input", "AE reconstruction", "GT")
                if phase == "autoencoder"
                else ("Condition input", "Optical output", "GT")
            )
            title = (
                "v13 latent construction; training monitor"
                if phase == "autoencoder"
                else "v13 strict joint K=8; training monitor"
            )
            preview_paths = []
            for start in range(0, len(rows), 4):
                destination = prefix.with_name(
                    prefix.stem + f"_{start // 4 + 1:02d}_v13.png"
                )
                _v13_visual_sheet(rows[start : start + 4], destination, title, labels)
                preview_paths.append(str(destination))
            report["preview_paths"] = preview_paths
        return report
    finally:
        _restore_rng(rng, device)


def _v13_expected_parts(phase):
    if phase == "autoencoder":
        return ("latent",)
    if phase == "joint_pretrain":
        return ("diffusion", "optical", "image", "phase_regularizer")
    raise ValueError(phase)


def _v13_check_objective_contract(total, parts, phase):
    if not isinstance(parts, dict):
        raise TypeError("_training_objective must return (total, tensor_dict)")
    missing = [name for name in _v13_expected_parts(phase) if name not in parts]
    if missing:
        raise ValueError(f"{phase} loss parts are missing: {missing}")
    for name, value in parts.items():
        if not torch.is_tensor(value) or value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"Invalid scalar loss part {name!r}")
    if phase == "autoencoder":
        expected = parts["latent"]
    else:
        expected_optical = parts["image"] + 1.0e-4 * parts["phase_regularizer"]
        if not bool(torch.allclose(parts["optical"].float(), expected_optical.float(), rtol=2e-4, atol=2e-5)):
            raise ValueError("optical loss must equal image + 1e-4*phase_regularizer")
        expected = parts["diffusion"] + parts["optical"]
    if not bool(torch.allclose(total.float(), expected.float(), rtol=2e-4, atol=2e-5)):
        raise ValueError(f"{phase} total loss violates the strict objective contract")


def _v13_optimizer_parameters(optimizer):
    seen = set()
    result = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                result.append(parameter)
    return result


def _v13_validate_phase_recovery(payload, phase, phase_index, epochs, batch, pool_size):
    progress = payload.get("progress", {})
    if payload.get("format") != V13_FORMAT or payload.get("workflow") != V13_WORKFLOW:
        raise ValueError("Resume checkpoint is not strict v13")
    if progress.get("phase_index") != phase_index or progress.get("active_phase") != phase:
        raise ValueError("Resume checkpoint belongs to a different phase")
    epoch = progress.get("epoch")
    cursor = progress.get("cursor")
    step = progress.get("phase_step")
    values = (epoch, cursor, step, progress.get("epoch_loss_samples"))
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Resume epoch/cursor/step/loss count is invalid")
    if epoch > epochs or cursor > pool_size or (epoch == epochs and cursor != 0):
        raise ValueError("Resume position exceeds the configured phase")
    if cursor != pool_size and cursor % batch:
        raise ValueError("Resume cursor is not a valid batch boundary")
    per_epoch = math.ceil(pool_size / batch)
    if step != epoch * per_epoch + math.ceil(cursor / batch):
        raise ValueError("Resume update count disagrees with epoch and cursor")
    if progress["epoch_loss_samples"] != cursor:
        raise ValueError("Resume loss accumulator does not match its cursor")
    if step and "optimizer_state" not in payload:
        raise ValueError("Partial phase checkpoint is missing optimizer state")


def _v13_compare_resume_settings(previous, current, names):
    changed = [name for name in names if previous.get(name) != current.get(name)]
    if changed:
        raise ValueError(f"Resume settings or data changed: {changed}")


def _v13_validate_resume_home(previous, current_home):

    if not previous.get("run_home") or not previous.get("run_dir"):
        raise ValueError("Resume checkpoint is missing its original output directory")
    original_home = Path(previous["run_home"]).expanduser().resolve()
    if Path(current_home).expanduser().resolve() != original_home:
        raise ValueError("Resume output differs from the original run; use the original --output directory")
    if Path(previous["run_dir"]).expanduser().resolve().parent != original_home:
        raise ValueError("Resume run directory does not belong to its recorded output parent")


def _v13_prepare_data(settings):
    pairs = discover_training_pairs_v13(Path(settings["dataset_root"]))
    targets = _v13_target_pairs(pairs)
    settings["data_fingerprint"] = _v13_data_fingerprint(
        pairs, settings["dataset_root"]
    )
    settings["pair_signature"] = data_signature(pairs, settings["dataset_root"])
    settings["training_pairs"] = len(pairs)
    settings["target_paths"] = len(targets)
    settings["stage_counts"] = {
        stage: sum(pair.stage == stage for pair in pairs)
        for stage in sorted({pair.stage for pair in pairs})
    }
    stages_used = [
        stage for stage in settings["stage_counts"]
        if settings["stage_counts"][stage] > 0
    ]
    settings["data_scope"] = "+".join(stages_used)
    settings["test_data_used_for_training"] = settings["stage_counts"].get("test", 0) > 0
    settings["held_out_pairs"] = 0
    return pairs, targets, _v13_monitor_pairs(pairs)


def _v13_new_progress(phase, phase_index, history=None, epoch_losses=None, role=None):
    return {
        "role": role,
        "phase_index": phase_index,
        "active_phase": phase,
        "epoch": 0,
        "cursor": 0,
        "phase_step": 0,
        "history": list(history or []),
        "epoch_losses": list(epoch_losses or []),
        "best_complete_score": float("inf"),
        "best_complete_epoch": None,
        "epoch_loss_sums": {},
        "epoch_loss_samples": 0,
    }


def _v13_restore_weights(path, model, optics):
    selected = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state"], strict=True)
    optics.load_state_dict(selected["optics_state"], strict=True)
    return selected


def _v13_train_one_phase(
    model,
    optics,
    pool,
    monitor_pairs,
    phase,
    phase_index,
    epochs,
    batch_size,
    rates,
    role,
    settings,
    run_dir,
    progress,
    resume_payload=None,
):


    device = next(model.parameters()).device
    support = run_dir / "support"
    support.mkdir(exist_ok=True)
    hide_internal(support)
    previews = run_dir / "fit previews"
    previews.mkdir(exist_ok=True)
    candidates = run_dir / "epoch candidates"
    candidates.mkdir(exist_ok=True)
    loss_csv = run_dir / "loss_epoch_v13.csv"

    optimizer = _phase_optimizer(model, optics, phase, rates)
    
    model.zero_grad(set_to_none=True)
    optics.zero_grad(set_to_none=True)
    active_parameters = _v13_optimizer_parameters(optimizer)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        _restore_rng(resume_payload["rng"], device)
    schedule = LinearDiffusionSchedule(model.diffusion_config, device=device)
    per_epoch = math.ceil(len(pool) / batch_size)
    total_steps = per_epoch * epochs
    resume_file = support / f"resume_{phase}_v13.pt"
    best_file = run_dir / f"best_{phase}_v13.pt"
    latest_file = run_dir / f"latest_{phase}_v13.pt"

    if progress["phase_step"] == 0:
        baseline = _v13_monitor(
            model,
            optics,
            monitor_pairs,
            phase,
            settings,
            previews / f"{phase}_baseline",
        )
        write_json(run_dir / f"{phase}_baseline_report_v13.json", baseline, internal=False)

    optimizer_entered = False
    epoch_commit_unsafe = False
    pre_rng = None
    try:
        start_epoch = int(progress["epoch"])
        for epoch in range(start_epoch, epochs):
            order = _v13_epoch_permutation(len(pool), settings["seed"], phase, epoch)
            cursor = int(progress["cursor"]) if epoch == start_epoch else 0
            seen = np.zeros(len(pool), dtype=np.bool_)
            seen[order[:cursor]] = True
            if epoch != start_epoch:
                progress["epoch_loss_sums"] = {}
                progress["epoch_loss_samples"] = 0

            for offset in range(cursor, len(pool), batch_size):
                pre_rng = _capture_rng(device)
                optimizer.zero_grad(set_to_none=True)
                for module in model.components().values():
                    module.train(any(parameter.requires_grad for parameter in module.parameters()))
                optics.train(optics.odr_logits.requires_grad)

                indices = order[offset : offset + batch_size]
                if bool(seen[indices].any()):
                    raise AssertionError("An item appeared twice in one epoch")
                selected = [pool[index] for index in indices]
                if phase == "autoencoder":
                    target = torch.stack(
                        [load_rgb_tensor(pair.target) for pair in selected]
                    ).to(device)
                    condition = target
                else:
                    condition, target = transfer_batch(selected, device, False)

                factor = 0.2 + 0.8 * 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * progress["phase_step"]
                        / max(1, total_steps - 1)
                    )
                )
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * factor

                tick = time.perf_counter()
                with mixed_precision(device, settings["amp"]):
                    total_loss, parts = _training_objective(
                        model,
                        optics,
                        condition,
                        target,
                        phase,
                        schedule,
                        settings,
                    )
                if not torch.is_tensor(total_loss) or total_loss.ndim != 0:
                    raise TypeError("_training_objective total must be a scalar tensor")
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError(f"Non-finite total loss in {phase}")
                _v13_check_objective_contract(total_loss, parts, phase)

                total_loss.backward()
                
                
                grad_norm = torch.nn.utils.clip_grad_norm_(active_parameters, 1.0)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError(f"Non-finite gradient norm in {phase}")
                optimizer_entered = True
                optimizer.step()

                seen[indices] = True
                batch_count = len(selected)
                names = ["total", *_v13_expected_parts(phase)]
                packed = torch.stack(
                    [total_loss.detach(), *(parts[name].detach() for name in names[1:])]
                ).double().cpu().tolist()
                sums = dict(progress.get("epoch_loss_sums", {}))
                for name, value in zip(names, packed):
                    sums[name] = float(sums.get(name, 0.0)) + float(value) * batch_count
                progress["epoch_loss_sums"] = sums
                progress["epoch_loss_samples"] = int(progress["epoch_loss_samples"]) + batch_count
                progress.update(
                    epoch=epoch,
                    cursor=min(offset + batch_size, len(pool)),
                    phase_step=int(progress["phase_step"]) + 1,
                )
                pre_rng = None
                optimizer_entered = False
                del total_loss, parts, condition, target

                step = int(progress["phase_step"])
                if step % settings["checkpoint_every"] == 0:
                    save_v13_checkpoint(
                        resume_file,
                        model,
                        optics,
                        settings,
                        progress,
                        optimizer,
                        role=role,
                        checkpoint_kind="resume",
                    )
                    hide_internal(resume_file)
                if step % settings["print_every"] == 0 or progress["cursor"] == len(pool):
                    elapsed = time.perf_counter() - tick
                    print(
                        f"{phase}: epoch {epoch + 1}/{epochs}; "
                        f"images {progress['cursor']}/{len(pool)}; "
                        f"step {step}/{total_steps}; batch_seconds={elapsed:.3f}",
                        flush=True,
                    )

            if not bool(seen.all()):
                raise AssertionError("Incomplete no-replacement epoch coverage")
            if progress["epoch_loss_samples"] != len(pool):
                raise AssertionError("Epoch loss does not cover every item")

            monitor = _v13_monitor(
                model,
                optics,
                monitor_pairs,
                phase,
                settings,
                previews / f"{phase}_epoch{epoch + 1:03d}",
            )
            write_json(
                run_dir / f"{phase}_epoch{epoch + 1:03d}_report_v13.json",
                monitor,
                internal=False,
            )
            means = {
                name: value / len(pool)
                for name, value in progress["epoch_loss_sums"].items()
            }
            row = {
                "phase": phase,
                "epoch": epoch + 1,
                "samples": len(pool),
                "updates": per_epoch,
                "total_loss": means.get("total", ""),
                "latent_loss": means.get("latent", ""),
                "diffusion_loss": means.get("diffusion", ""),
                "optical_loss": means.get("optical", ""),
                "image_loss": means.get("image", ""),
                "phase_regularizer": means.get("phase_regularizer", ""),
                "monitor_loss": monitor["losses"]["total"],
                "selected_for_next_stage": False,
            }
            
            
            
            
            
            save_v13_checkpoint(
                resume_file,
                model,
                optics,
                settings,
                progress,
                optimizer,
                role=role,
                checkpoint_kind="epoch_commit_boundary",
            )
            hide_internal(resume_file)
            epoch_commit_unsafe = True
            progress["epoch_losses"].append(row)
            score = float(monitor["losses"]["total"])
            if score < float(progress["best_complete_score"]):
                progress["best_complete_score"] = score
                progress["best_complete_epoch"] = epoch + 1

            epoch_file = candidates / f"{phase}_epoch{epoch + 1:03d}_v13.pt"
            save_v13_checkpoint(
                epoch_file,
                model,
                optics,
                settings,
                progress,
                role=role,
                checkpoint_kind="complete_epoch_candidate",
            )
            save_v13_checkpoint(
                latest_file,
                model,
                optics,
                settings,
                progress,
                role=role,
                checkpoint_kind="latest_complete_epoch",
            )
            if progress["best_complete_epoch"] == epoch + 1:
                save_v13_checkpoint(
                    best_file,
                    model,
                    optics,
                    settings,
                    progress,
                    role=role,
                    checkpoint_kind="best_complete_epoch",
                )
            write_json(
                run_dir / f"{phase}_epoch{epoch + 1:03d}_coverage_v13.json",
                {
                    "phase": phase,
                    "epoch": epoch + 1,
                    "pool_size": len(pool),
                    "unique_seen": int(seen.sum()),
                    "every_item_seen_exactly_once": True,
                    "drop_last": False,
                    "monitor_is_training": True,
                },
                internal=False,
            )
            _v13_write_epoch_losses(loss_csv, progress["epoch_losses"])
            _v13_write_loss_curve(
                run_dir
                / (
                    "latent_loss_curve_v13.png"
                    if phase == "autoencoder"
                    else "joint_loss_curve_v13.png"
                ),
                progress["epoch_losses"],
                phase,
            )

            progress.update(
                epoch=epoch + 1,
                cursor=0,
                epoch_loss_sums={},
                epoch_loss_samples=0,
            )
            save_v13_checkpoint(
                resume_file,
                model,
                optics,
                settings,
                progress,
                optimizer,
                role=role,
                checkpoint_kind="resume",
            )
            hide_internal(resume_file)
            epoch_commit_unsafe = False
            print(
                f"{phase}: full epoch {epoch + 1} complete; "
                f"training-monitor={score:.6f}; "
                f"best_epoch={progress['best_complete_epoch']}",
                flush=True,
            )
    except BaseException:
        if not optimizer_entered and not epoch_commit_unsafe:
            if pre_rng is not None:
                _restore_rng(pre_rng, device)
            optimizer.zero_grad(set_to_none=True)
            save_v13_checkpoint(
                resume_file,
                model,
                optics,
                settings,
                progress,
                optimizer,
                role=role,
                checkpoint_kind="resume",
            )
            hide_internal(resume_file)
        elif epoch_commit_unsafe:
            print(
                "Epoch commit was interrupted; preserving the last complete "
                f"recovery boundary at {resume_file}",
                flush=True,
            )
        print(f"Interrupted. Recover from {resume_file}", flush=True)
        raise

    selected = _v13_restore_weights(best_file, model, optics)
    selected_epoch = int(selected["progress"]["best_complete_epoch"])
    for row in progress["epoch_losses"]:
        if row["phase"] == phase:
            row["selected_for_next_stage"] = row["epoch"] == selected_epoch
    _v13_write_epoch_losses(loss_csv, progress["epoch_losses"])
    _v13_write_loss_curve(
        run_dir
        / (
            "latent_loss_curve_v13.png"
            if phase == "autoencoder"
            else "joint_loss_curve_v13.png"
        ),
        progress["epoch_losses"],
        phase,
    )
    result = {
        "phase": phase,
        "completed_epochs": epochs,
        "selected_epoch": selected_epoch,
        "pool_size": len(pool),
        "updates_executed": total_steps,
        "all_images_covered": True,
        "training_monitor_loss": float(progress["best_complete_score"]),
    }
    del optimizer, selected
    return result


def _v13_base_settings(settings, stage_mode):
    cfg = dict(settings)
    cfg["stage_mode"] = stage_mode
    _v13_stage_settings(cfg, stage_mode)
    cfg["dataset_root"] = str(Path(cfg["dataset_root"]).expanduser().resolve())
    cfg["run_home"] = str(Path(cfg["run_home"]).expanduser().resolve())
    return cfg


def _v13_create_run_dir(home, suffix):
    base = home / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}_v13"
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = home / f"{base.name}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise FileExistsError("Could not allocate a unique v13 run directory")


def _v13_run_latent(settings):
    cfg = _v13_base_settings(settings, "latent")
    device = transfer_device(cfg["device"])
    pairs, target_pairs, monitor_pairs = _v13_prepare_data(cfg)
    home = Path(cfg["run_home"])
    home.mkdir(parents=True, exist_ok=True)
    resume_path = cfg.get("resume_checkpoint")
    resume_payload = None
    if resume_path:
        resume_path = Path(resume_path).expanduser().resolve()
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        previous = resume_payload.get("settings", {})
        _v13_validate_resume_home(previous, home)
        _v13_compare_resume_settings(
            previous,
            cfg,
            (
                "data_fingerprint",
                "pair_signature",
                "seed",
                "amp",
                "augment",
                "ae_epochs",
                "digital_batch",
            ),
        )
        _v13_validate_phase_recovery(
            resume_payload,
            "autoencoder",
            0,
            cfg["ae_epochs"],
            cfg["digital_batch"],
            len(target_pairs),
        )
        run_dir = Path(previous["run_dir"]).resolve()
        if resume_path != run_dir / "support" / "resume_autoencoder_v13.pt":
            raise ValueError("Latent resume checkpoint must remain in its original run")
    else:
        run_dir = _v13_create_run_dir(home, "latent_space")
    cfg["run_dir"] = str(run_dir)

    lock = home / "training.lock"
    with exclusive_run(lock):
        if resume_payload is None:
            run_dir.mkdir(exist_ok=False)
            seed_transfer(cfg["seed"])
            model = EPIModel(gradient_checkpointing=True).to(device)
            optics = OpticalSystem(OpticsSpec()).to(device)
            initialize_icnr(model)
            model.latent_stats_fitted.fill_(False)
            progress = _v13_new_progress(
                "autoencoder", 0, role=V13_ROLE_LATENT
            )
        else:
            model, optics, loaded = load_v13_checkpoint(resume_path, device)
            progress = dict(loaded["progress"])
            _restore_rng(loaded["rng"], device)
            resume_payload = loaded
        model.assert_parameter_counts()

        write_json(run_dir / "settings_v13.json", cfg, internal=False)
        _v13_write_pairs(run_dir / "all_training_pairs_v13.csv", pairs)
        result = _v13_train_one_phase(
            model,
            optics,
            target_pairs,
            monitor_pairs,
            "autoencoder",
            0,
            cfg["ae_epochs"],
            cfg["digital_batch"],
            {"digital": 1.0e-4},
            V13_ROLE_LATENT,
            cfg,
            run_dir,
            progress,
            resume_payload,
        )
        stats = _v13_fit_latent_stats(
            model, target_pairs, device, cfg["amp"], cfg["digital_batch"]
        )
        progress["history"].append(result)
        progress.update(
            phase_index=1,
            active_phase="joint_pretrain",
            epoch=0,
            cursor=0,
            phase_step=0,
            best_complete_score=float("inf"),
            best_complete_epoch=None,
            epoch_loss_sums={},
            epoch_loss_samples=0,
        )
        latent_checkpoint = run_dir / "latent_space_v13.pt"
        save_v13_checkpoint(
            latent_checkpoint,
            model,
            optics,
            cfg,
            progress,
            role=V13_ROLE_LATENT,
            checkpoint_kind="completed_latent_space",
        )
        final_report = _v13_monitor(
            model,
            optics,
            monitor_pairs,
            "autoencoder",
            cfg,
            run_dir / "fit previews" / "latent_final",
        )
        write_json(run_dir / "latent_final_report_v13.json", final_report, internal=False)
        write_json(run_dir / "latent_statistics_v13.json", stats, internal=False)
        write_json(
            run_dir / "latent_input_output_v13.json",
            {
                "role": V13_ROLE_LATENT,
                "input": "clear target RGB tensor, 3x256x256 in [0,1]",
                "latent_output": "standardized target latent, 8x32x32",
                "diagnostic_output": "R(E(target)) RGB reconstruction, 3x256x256",
                "loss": "S66 Charbonnier + 0.1*(1-MS-SSIM)",
                "checkpoint": str(latent_checkpoint),
                "target_paths": len(target_pairs),
                "generated_latent_used": False,
            },
            internal=False,
        )
        write_json(
            run_dir / "completed_latent_v13.json",
            {
                "status": "completed",
                "checkpoint": latent_checkpoint.name,
                "role": V13_ROLE_LATENT,
                "phase_index": 1,
                "history": progress["history"],
                "training_pairs": len(pairs),
                "target_paths": len(target_pairs),
                "data_scope": cfg["data_scope"],
                "stage_counts": cfg["stage_counts"],
                "test_data_used_for_training": cfg["test_data_used_for_training"],
                "held_out_pairs": cfg["held_out_pairs"],
            },
            internal=False,
        )
        print(f"LATENT COMPLETE: {latent_checkpoint}", flush=True)
    return run_dir


def _v13_run_joint(settings):
    cfg = _v13_base_settings(settings, "joint")
    device = transfer_device(cfg["device"])
    pairs, target_pairs, monitor_pairs = _v13_prepare_data(cfg)
    home = Path(cfg["run_home"])
    home.mkdir(parents=True, exist_ok=True)
    resume_path = cfg.get("resume_checkpoint")
    resume_payload = None

    if resume_path:
        resume_path = Path(resume_path).expanduser().resolve()
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        previous = resume_payload.get("settings", {})
        _v13_validate_resume_home(previous, home)
        
        
        
        
        
        cfg["latent_checkpoint"] = previous.get("latent_checkpoint")
        cfg["latent_checkpoint_sha256"] = previous.get(
            "latent_checkpoint_sha256"
        )
        if not cfg["latent_checkpoint"] or not cfg["latent_checkpoint_sha256"]:
            raise ValueError("Joint resume checkpoint has no latent-source identity")
        _v13_compare_resume_settings(
            previous,
            cfg,
            (
                "data_fingerprint",
                "pair_signature",
                "latent_checkpoint_sha256",
                "seed",
                "amp",
                "augment",
                "joint_epochs",
                "optical_batch",
            ),
        )
        _v13_validate_phase_recovery(
            resume_payload,
            "joint_pretrain",
            1,
            cfg["joint_epochs"],
            cfg["optical_batch"],
            len(pairs),
        )
        run_dir = Path(previous["run_dir"]).resolve()
        if resume_path != run_dir / "support" / "resume_joint_pretrain_v13.pt":
            raise ValueError("Joint resume checkpoint must remain in its original run")
    else:
        latent_checkpoint = cfg.get("latent_checkpoint")
        if latent_checkpoint:
            latent_checkpoint = Path(latent_checkpoint).expanduser().resolve()
        else:
            latent_checkpoint = latest_completed_latent_v13(home)
        cfg["latent_checkpoint"] = str(latent_checkpoint)
        cfg["latent_checkpoint_sha256"] = hashlib.sha256(
            latent_checkpoint.read_bytes()
        ).hexdigest()
        run_dir = _v13_create_run_dir(home, "joint_pretrain")
    cfg["run_dir"] = str(run_dir)

    lock = home / "training.lock"
    with exclusive_run(lock):
        if resume_payload is None:
            run_dir.mkdir(exist_ok=False)
            seed_transfer(cfg["seed"])
            model, optics, latent_payload = load_v13_checkpoint(
                cfg["latent_checkpoint"], device
            )
            _v13_validate_latent_payload(latent_payload, cfg)
            progress = _v13_new_progress(
                "joint_pretrain",
                1,
                history=latent_payload["progress"]["history"],
                epoch_losses=latent_payload["progress"].get("epoch_losses", []),
                role=V13_ROLE_PRETRAIN,
            )
            resume_for_phase = None
        else:
            model, optics, loaded = load_v13_checkpoint(resume_path, device)
            progress = dict(loaded["progress"])
            _restore_rng(loaded["rng"], device)
            resume_for_phase = loaded
        model.assert_parameter_counts()
        if device.type == "cuda":
            optics.precompute_transfers(device)

        write_json(run_dir / "settings_v13.json", cfg, internal=False)
        _v13_write_pairs(run_dir / "all_training_pairs_v13.csv", pairs)
        _v13_write_epoch_losses(
            run_dir / "loss_epoch_v13.csv", progress.get("epoch_losses", [])
        )
        _v13_write_loss_curve(
            run_dir / "latent_loss_curve_v13.png",
            progress.get("epoch_losses", []),
            "autoencoder",
        )
        result = _v13_train_one_phase(
            model,
            optics,
            pairs,
            monitor_pairs,
            "joint_pretrain",
            1,
            cfg["joint_epochs"],
            cfg["optical_batch"],
            {"digital": 1.0e-4, "decoder": 5.0e-5, "odr": 3.0e-5},
            V13_ROLE_PRETRAIN,
            cfg,
            run_dir,
            progress,
            resume_for_phase,
        )
        progress["history"].append(result)
        progress.update(
            phase_index=2,
            active_phase="complete",
            epoch=0,
            cursor=0,
            phase_step=0,
            best_complete_score=float("inf"),
            best_complete_epoch=None,
            epoch_loss_sums={},
            epoch_loss_samples=0,
        )
        final = run_dir / "pretrained_v13.pt"
        save_v13_checkpoint(
            final,
            model,
            optics,
            cfg,
            progress,
            role=V13_ROLE_PRETRAIN,
            checkpoint_kind="completed_two_stage_pretrain",
        )
        final_report = _v13_monitor(
            model,
            optics,
            monitor_pairs,
            "joint_pretrain",
            cfg,
            run_dir / "fit previews" / "optical_final",
        )
        write_json(run_dir / "optical_final_report_v13.json", final_report, internal=False)
        np.save(
            run_dir / "shared_odr_phase_v13.npy",
            optics.export_odr_phase(cpu=True).float().numpy(),
        )
        write_json(
            run_dir / "optical_input_output_v13.json",
            {
                "role": V13_ROLE_PRETRAIN,
                "input": "condition RGB tensor plus independent Gaussian latent",
                "digital_output": "K=8 final latent followed once by 3x2048x2048 CWS phase",
                "optical_output": "two-SLM focal-plane RGB intensity, 3x256x256",
                "loss": "Ldiff + Limg + 1e-4*(CWS phase regularizer + ODR phase regularizer)",
                "checkpoint": str(final),
                "training_pairs": len(pairs),
                "target_passed_to_reverse_generator": False,
                "auxiliary_decoder_used_in_joint_training": False,
                "latent_regression_used": False,
                "K": 8,
            },
            internal=False,
        )
        write_json(
            run_dir / "completed_pretrain_v13.json",
            {
                "status": "completed",
                "checkpoint": final.name,
                "role": V13_ROLE_PRETRAIN,
                "phase": "joint_pretrain",
                "phase_index": 2,
                "history": progress["history"],
                "training_pairs": len(pairs),
                "target_paths": len(target_pairs),
                "data_scope": cfg["data_scope"],
                "stage_counts": cfg["stage_counts"],
                "test_data_used_for_training": cfg["test_data_used_for_training"],
                "held_out_pairs": cfg["held_out_pairs"],
                "evaluation_type": "training-fit",
                "selection": "minimum deterministic training-monitor loss among complete epochs",
            },
            internal=False,
        )
        print(f"JOINT COMPLETE: {final}", flush=True)
    return run_dir


def run_latent_construction(settings):


    cfg = dict(settings)
    cfg["stage_mode"] = "latent"
    return _v13_run_latent(cfg)


def run_joint_pretraining(settings):


    cfg = dict(settings)
    cfg["stage_mode"] = "joint"
    return _v13_run_joint(cfg)


def run_two_stage_training(settings):


    stage_mode = str(settings.get("stage_mode", "")).strip().lower()
    if stage_mode == "latent":
        return run_latent_construction(settings)
    if stage_mode == "joint":
        return run_joint_pretraining(settings)
    if stage_mode == "both":
        if settings.get("resume_checkpoint"):
            raise ValueError("Internal stage_mode='both' does not accept one shared resume file")
        latent_settings = dict(settings)
        latent_settings["stage_mode"] = "latent"
        latent_settings["resume_checkpoint"] = None
        latent_run = run_latent_construction(latent_settings)
        joint_settings = dict(settings)
        joint_settings["stage_mode"] = "joint"
        joint_settings["resume_checkpoint"] = None
        joint_settings["latent_checkpoint"] = str(latent_run / "latent_space_v13.pt")
        return run_joint_pretraining(joint_settings)
    raise ValueError("stage_mode must be 'latent', 'joint' or 'both'")



import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn


_V13_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_V13_KNOWN_STAGES = ("pre-train", "fine-tune", "test")


@dataclass(frozen=True)
class SelectedPairV13:


    stage: str
    family: str
    domain: str
    key: str
    group: str
    condition: Path
    target: Path | None


@dataclass(frozen=True)
class DatasetSelectionV13:


    selected_path: Path
    layout: str
    pairs: tuple[SelectedPairV13, ...]
    has_targets: bool

    def summary(self) -> dict[str, object]:
        return {
            "selected_path": str(self.selected_path),
            "layout": self.layout,
            "pairs": len(self.pairs),
            "has_targets": self.has_targets,
            "stages": {
                stage: sum(pair.stage == stage for pair in self.pairs)
                for stage in sorted({pair.stage for pair in self.pairs})
            },
            "domains": {
                domain: sum(pair.domain == domain for pair in self.pairs)
                for domain in sorted({pair.domain for pair in self.pairs})
            },
        }


def _v13_natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _v13_images_recursive(directory: Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    rows = sorted(
        (
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _V13_IMAGE_EXTENSIONS
        ),
        key=lambda path: _v13_natural_key(path.relative_to(directory).as_posix()),
    )
    if not rows:
        raise ValueError(f"所选文件夹没有可读取图像：{directory}")
    resolved = [str(path).casefold() for path in rows]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"所选文件夹包含大小写冲突的图像路径：{directory}")
    return rows


def _v13_group_from_path(path: Path) -> str:
    stem = path.stem
    for pattern, prefix in (
        (r"group_(\d+)_\d+", "group"),
        (r"sanheyi\d*_(\d+)_\d+", "sanheyi"),
    ):
        match = re.fullmatch(pattern, stem, re.IGNORECASE)
        if match:
            return f"{prefix}_{int(match.group(1)):03d}"
    match = re.fullmatch(r"(.+?)[_-](\d+)", stem)
    return match.group(1).lower() if match else stem.lower()


def _v13_wrap_known_pairs(
    rows: Sequence[TransferPair], selected_path: Path
) -> tuple[SelectedPairV13, ...]:
    wrapped = tuple(
        SelectedPairV13(
            stage=row.stage,
            family=row.family,
            domain=row.domain,
            key=row.key,
            group=row.group,
            condition=row.condition.resolve(),
            target=row.target.resolve(),
        )
        for row in rows
    )
    if len({row.key for row in wrapped}) != len(wrapped):
        raise RuntimeError(f"已知数据结构产生重复 key：{selected_path}")
    if len({row.condition for row in wrapped}) != len(wrapped):
        raise RuntimeError(f"已知数据结构重复使用 condition：{selected_path}")
    return wrapped


def discover_training_pairs_v13(dataset_root: str | Path) -> list[SelectedPairV13]:







    selection = discover_selected_pairs_v13(dataset_root, purpose="fine_tune")
    pairs = list(selection.pairs)
    if not pairs or any(pair.target is None for pair in pairs):
        raise ValueError("Training requires a nonempty paired condition/GT dataset")
    if len({pair.key for pair in pairs}) != len(pairs):
        raise RuntimeError("v13 预训练发现重复 pair key")
    if len({pair.condition.resolve() for pair in pairs}) != len(pairs):
        raise RuntimeError("v13 预训练发现重复 condition 路径")
    return pairs


def _v13_generic_pairs(input_root: Path, target_root: Path | None) -> tuple[SelectedPairV13, ...]:
    conditions = _v13_images_recursive(input_root)
    input_root = input_root.resolve()
    if target_root is None:
        rows = []
        for condition in conditions:
            relative = condition.relative_to(input_root)
            domain = relative.parts[0] if len(relative.parts) > 1 else "input"
            rows.append(
                SelectedPairV13(
                    stage="selected",
                    family="generic",
                    domain=domain,
                    
                    
                    
                    key=f"selected/generic/{relative.as_posix()}",
                    group=_v13_group_from_path(condition),
                    condition=condition,
                    target=None,
                )
            )
        return tuple(rows)

    target_root = target_root.resolve()
    targets = _v13_images_recursive(target_root)
    relative_targets: dict[str, Path] = {}
    relative_stem_targets: dict[str, list[Path]] = {}
    basename_targets: dict[str, list[Path]] = {}
    stem_targets: dict[str, list[Path]] = {}
    for target in targets:
        relative_path = target.relative_to(target_root)
        relative = relative_path.as_posix().casefold()
        if relative in relative_targets:
            raise ValueError(f"GT 中存在大小写冲突的相对路径：{target}")
        relative_targets[relative] = target
        relative_stem_targets.setdefault(relative_path.with_suffix("").as_posix().casefold(), []).append(target)
        basename_targets.setdefault(target.name.casefold(), []).append(target)
        stem_targets.setdefault(target.stem.casefold(), []).append(target)

    used_targets: set[Path] = set()
    rows: list[SelectedPairV13] = []
    for condition in conditions:
        relative_path = condition.relative_to(input_root)
        exact = relative_targets.get(relative_path.as_posix().casefold())
        if exact is not None:
            target = exact
        else:
            candidate_groups = (
                relative_stem_targets.get(relative_path.with_suffix("").as_posix().casefold(), []),
                basename_targets.get(condition.name.casefold(), []),
                stem_targets.get(condition.stem.casefold(), []),
            )
            candidates = next((group for group in candidate_groups if group), [])
            if len(candidates) != 1:
                reason = "没有同名或同 stem 的 GT" if not candidates else "同名或同 stem 的 GT 不唯一"
                raise ValueError(
                    f"无法无歧义配对 {condition}：{reason}；请整理为 input 与 gt/target 同名结构"
                )
            target = candidates[0]
        used_targets.add(target)
        domain = relative_path.parts[0] if len(relative_path.parts) > 1 else "input"
        rows.append(
            SelectedPairV13(
                stage="selected",
                family="generic",
                domain=domain,
                key=f"selected/generic/{relative_path.as_posix()}",
                group=_v13_group_from_path(target),
                condition=condition,
                target=target,
            )
        )
    unused = sorted(set(targets) - used_targets, key=lambda path: str(path).casefold())
    if unused:
        raise ValueError(
            "GT 中存在没有被任何 input 使用的图像；为避免猜测已停止。"
            f"示例：{unused[:3]}"
        )
    if len({row.key for row in rows}) != len(rows):
        raise RuntimeError("generic input 与 gt/target 产生重复 pair key")
    return tuple(rows)


def discover_selected_pairs_v13(
    selected_path: str | Path,
    purpose: str = "test",
) -> DatasetSelectionV13:







    if purpose not in {"fine_tune", "test"}:
        raise ValueError("purpose 必须是 fine_tune 或 test")
    selected = Path(selected_path).expanduser().resolve()
    if not selected.is_dir():
        raise FileNotFoundError(selected)

    stage_directories = {name for name in _V13_KNOWN_STAGES if (selected / name).is_dir()}
    if stage_directories:
        if stage_directories != set(_V13_KNOWN_STAGES):
            raise ValueError(
                "检测到不完整的已知数据根。请补齐三个阶段，或直接选择其中一个"
                f"阶段文件夹；当前为 {sorted(stage_directories)}"
            )
        rows = [pair for stage in _V13_KNOWN_STAGES for pair in discover_transfer_pairs(selected, stage)]
        pairs = _v13_wrap_known_pairs(rows, selected)
        return DatasetSelectionV13(selected, "known-root-all-stages", pairs, True)

    if selected.name in _V13_KNOWN_STAGES:
        parent_stages = {name for name in _V13_KNOWN_STAGES if (selected.parent / name).is_dir()}
        if parent_stages == set(_V13_KNOWN_STAGES):
            rows = discover_transfer_pairs(selected.parent, selected.name)
            pairs = _v13_wrap_known_pairs(rows, selected)
            return DatasetSelectionV13(selected, f"known-stage:{selected.name}", pairs, True)

    input_child = selected / "input"
    target_children = [path for path in (selected / "gt", selected / "target") if path.is_dir()]
    if len(target_children) > 1:
        raise ValueError("generic 数据根同时包含 gt 和 target；无法判断哪一个是配对目标")
    target_child = target_children[0] if target_children else None
    if input_child.is_dir() or target_child is not None:
        if not input_child.is_dir():
            raise ValueError("generic 数据根包含 gt/target 但缺少 input")
        if target_child is None and purpose == "fine_tune":
            raise ValueError("微调必须选择同时包含 input 和 gt/target 的数据")
        pairs = _v13_generic_pairs(input_child, target_child)
        return DatasetSelectionV13(
            selected,
            f"generic-input-{target_child.name}" if target_child is not None else "generic-input-only",
            pairs,
            target_child is not None,
        )

    if selected.name.casefold() == "input":
        sibling_targets = [path for path in (selected.parent / "gt", selected.parent / "target") if path.is_dir()]
        if len(sibling_targets) > 1:
            raise ValueError("input 旁边同时存在 gt 和 target；无法判断哪一个是配对目标")
        sibling_target = sibling_targets[0] if sibling_targets else None
        if sibling_target is None and purpose == "fine_tune":
            raise ValueError("微调所选 input 文件夹旁边没有 gt/target 文件夹")
        pairs = _v13_generic_pairs(selected, sibling_target)
        return DatasetSelectionV13(
            selected,
            f"generic-input-{sibling_target.name}" if sibling_target is not None else "generic-input-only",
            pairs,
            sibling_target is not None,
        )

    if purpose == "fine_tune":
        raise ValueError(
            "无法识别微调数据结构。请选择完整已知数据根、一个已知阶段，"
            "或包含 input 与 gt/target 的文件夹。"
        )
    pairs = _v13_generic_pairs(selected, None)
    return DatasetSelectionV13(selected, "selected-input-only", pairs, False)


def selected_data_signature_v13(selection: DatasetSelectionV13) -> str:


    digest = hashlib.sha256()
    digest.update(selection.layout.encode("utf-8"))
    for pair in selection.pairs:
        row = [pair.stage, pair.family, pair.domain, pair.key, pair.group]
        for path in (pair.condition, pair.target):
            if path is None:
                row.extend([None, None, None])
            else:
                stat = path.stat()
                row.extend([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def choose_dataset_v13(purpose: str) -> Path:


    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as error:  
        raise RuntimeError("当前 Python 没有可用的 Tk 文件选择器；请用 --dataset 指定路径") from error
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    normalized = str(purpose).strip().lower().replace("-", "_")
    purpose_labels = {
        "latent": "潜空间构建",
        "latent construction": "潜空间构建",
        "pre_train": "预训练",
        "pretraining": "预训练",
        "joint_pretraining": "联合预训练",
        "joint pretraining": "联合预训练",
        "fine_tune": "微调",
        "test": "测试",
        "inference": "测试",
    }
    title = f"选择{purpose_labels.get(normalized, '运行')}数据文件夹"
    try:
        value = filedialog.askdirectory(parent=root, title=title, mustexist=True)
    finally:
        root.destroy()
    if not value:
        raise RuntimeError("未选择数据文件夹")
    return Path(value).resolve()


def choose_checkpoint_v13() -> Path:


    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as error:  
        raise RuntimeError("当前 Python 没有可用的 Tk 文件选择器；请用 --checkpoint 指定路径") from error
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        value = filedialog.askopenfilename(
            parent=root,
            title="选择 v13 潜空间、预训练或微调检查点",
            filetypes=(("PyTorch checkpoint", "*.pt"), ("All files", "*.*")),
        )
    finally:
        root.destroy()
    if not value:
        raise RuntimeError("未选择检查点")
    return Path(value).resolve()


def _write_selected_manifest_v13(path: Path, selection: DatasetSelectionV13) -> None:
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "family", "domain", "key", "group", "condition", "target"])
        for pair in selection.pairs:
            writer.writerow(
                [pair.stage, pair.family, pair.domain, pair.key, pair.group, pair.condition, pair.target or ""]
            )
    _atomic_replace(temporary, path, hide_after=True)


def _tensor_dict_hash_v13(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _frozen_state_v13(model: EPIModel, optics: OpticalSystem, mode: str) -> dict[str, torch.Tensor]:
    if mode not in {"C1", "C2"}:
        raise ValueError("mode 必须是 C1 或 C2")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        frozen = (
            name.startswith("target_encoder.")
            or name.startswith("aux_decoder.")
            or name.startswith("phase_decoder.")
            or name in {"latent_mean", "latent_std", "latent_stats_fitted"}
            or name.startswith("condition_encoder.input_conv.")
            or name.startswith("condition_encoder.residual0.")
            or name.startswith("condition_encoder.early_downs.")
            or name.startswith("condition_encoder.early_residuals.")
        )
        if mode == "C1" and name.startswith("condition_encoder."):
            frozen = True
        if frozen:
            state[f"model.{name}"] = tensor.detach().cpu().clone()
    for name, tensor in optics.state_dict().items():
        state[f"optics.{name}"] = tensor.detach().cpu().clone()
    return state


def _frozen_category_hashes_v13(state: dict[str, torch.Tensor]) -> dict[str, str]:
    categories: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        if name.startswith("model.target_encoder."):
            category = "E_target_encoder"
        elif name.startswith("model.aux_decoder."):
            category = "R_aux_decoder"
        elif name.startswith("model.phase_decoder."):
            category = "D_phase_decoder"
        elif name.startswith("model.condition_encoder."):
            category = "C_frozen"
        elif name.startswith("model.latent_"):
            category = "latent_statistics"
        elif name.startswith("optics."):
            category = "optics_and_ODR"
        else:
            category = "other"
        categories.setdefault(category, {})[name] = tensor
    return {name: _tensor_dict_hash_v13(values) for name, values in sorted(categories.items())}


def _assert_frozen_equal_v13(
    reference: dict[str, torch.Tensor],
    model: EPIModel,
    optics: OpticalSystem,
    mode: str,
) -> dict[str, object]:
    current = _frozen_state_v13(model, optics, mode)
    if set(reference) != set(current):
        missing = sorted(set(reference) - set(current))
        extra = sorted(set(current) - set(reference))
        raise AssertionError(f"冻结状态键发生变化；missing={missing[:3]}, extra={extra[:3]}")
    changed = [name for name in reference if not torch.equal(reference[name], current[name])]
    if changed:
        raise AssertionError(f"冻结参数被修改：{changed[:10]}")
    before = _tensor_dict_hash_v13(reference)
    after = _tensor_dict_hash_v13(current)
    if before != after:
        raise AssertionError("冻结参数整体哈希不一致")
    return {
        "verified_equal": True,
        "tensor_count": len(reference),
        "sha256_before": before,
        "sha256_after": after,
        "category_sha256": _frozen_category_hashes_v13(current),
    }


def _configure_otl_v13(
    model: EPIModel,
    optics: OpticalSystem,
    mode: str,
    denoiser_lr: float,
    condition_late_lr: float,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], dict[str, int]]:
    if mode not in {"C1", "C2"}:
        raise ValueError("mode 必须是 C1 或 C2")
    if (
        not math.isfinite(float(denoiser_lr))
        or not math.isfinite(float(condition_late_lr))
        or denoiser_lr <= 0
        or condition_late_lr <= 0
    ):
        raise ValueError("学习率必须是有限正数")
    model.eval()
    optics.eval()
    model.requires_grad_(False)
    optics.requires_grad_(False)

    model.denoiser.requires_grad_(True)
    model.denoiser.train()
    groups = [{"params": list(model.denoiser.parameters()), "lr": float(denoiser_lr), "name": "U"}]
    if mode == "C2":
        late_modules = (
            model.condition_encoder.late_down1,
            model.condition_encoder.late_residual1,
            model.condition_encoder.late_down2,
            model.condition_encoder.late_residual2,
        )
        late_parameters: list[nn.Parameter] = []
        for module in late_modules:
            module.requires_grad_(True)
            module.train()
            late_parameters.extend(module.parameters())
        groups.append({"params": late_parameters, "lr": float(condition_late_lr), "name": "C_late"})

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in trainable)
    expected_u = model.parameter_summary()["denoiser"]
    expected = expected_u + (model.parameter_summary()["condition_encoder_late"] if mode == "C2" else 0)
    if count != expected:
        raise AssertionError(f"{mode} trainable parameter count {count} != {expected}")
    if mode == "C2" and count != 16_085_192:
        raise AssertionError(f"Supplementary C2 requires 16,085,192 parameters, found {count}")
    if any(parameter.requires_grad for parameter in optics.parameters()):
        raise AssertionError("OTL simulation must keep every optical/ODR parameter frozen")
    
    
    model.zero_grad(set_to_none=True)
    optics.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(groups)
    return optimizer, trainable, {
        "total": count,
        "U_denoiser": expected_u,
        "C_late": model.parameter_summary()["condition_encoder_late"] if mode == "C2" else 0,
    }


def _validate_fine_progress_v13(payload: dict) -> None:

    settings = payload.get("settings", {})
    progress = payload.get("progress", {})
    epochs, batch, pool = (settings.get(name) for name in ("epochs", "batch_size", "selected_pairs"))
    if any(type(value) is not int or value <= 0 for value in (epochs, batch, pool)):
        raise ValueError("Fine-tuning checkpoint has invalid epoch/batch/dataset settings")
    if payload.get("selected_data", {}).get("pairs") != pool:
        raise ValueError("Fine-tuning checkpoint pair counts disagree")
    status = progress.get("status")
    if status not in {"running", "completed"}:
        raise ValueError("Fine-tuning checkpoint status is invalid")
    expected_kind = "fine_tune_final" if status == "completed" else "fine_tune_resume"
    if payload.get("checkpoint_kind") != expected_kind:
        raise ValueError("Fine-tuning checkpoint kind and status disagree")
    epoch, cursor, step, count = (progress.get(name) for name in ("epoch", "cursor", "step", "epoch_loss_count"))
    if any(type(value) is not int or value < 0 for value in (epoch, cursor, step, count)):
        raise ValueError("Fine-tuning epoch/cursor/step/loss count is invalid")
    if epoch > epochs or cursor > pool or (epoch == epochs and cursor != 0):
        raise ValueError("Fine-tuning resume position exceeds the configured training")
    if cursor != pool and cursor % batch:
        raise ValueError("Fine-tuning cursor is not a valid batch boundary")
    if step != epoch * math.ceil(pool / batch) + math.ceil(cursor / batch):
        raise ValueError("Fine-tuning step count disagrees with epoch and cursor")
    loss_sum = progress.get("epoch_loss_sum")
    if (count != cursor or isinstance(loss_sum, bool) or not isinstance(loss_sum, (int, float))
            or not math.isfinite(loss_sum) or loss_sum < 0 or (count == 0 and loss_sum != 0)):
        raise ValueError("Fine-tuning loss accumulator is invalid")
    history = progress.get("history")
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Fine-tuning history length differs from completed epochs")
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict) or row.get("epoch") != index or row.get("pairs") != pool:
            raise ValueError("Fine-tuning epoch history is inconsistent")
        value = row.get("mean_image_objective")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Fine-tuning epoch history has a non-finite or invalid loss")
    if status == "completed" and (epoch != epochs or cursor != 0):
        raise ValueError("Fine-tuning final marker disagrees with epoch progress")
    if status == "running" and ("optimizer_state" not in payload or not isinstance(payload.get("rng"), dict)):
        raise ValueError("Fine-tuning recovery requires optimizer and RNG states")


def _load_fine_checkpoint_v13(path: str | Path, device) -> tuple[EPIModel, OpticalSystem, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != V13_FORMAT or payload.get("role") != "fine_tune":
        raise ValueError("需要 v13 fine_tune 检查点")
    if payload.get("workflow") != "strict_two_stage_then_otl" or payload.get("phase") not in {"C1", "C2"}:
        raise ValueError("fine_tune 检查点的流程或阶段无效")
    verification = payload.get("frozen_verification", {})
    if verification.get("verified_equal") is not True:
        raise ValueError("fine_tune 检查点没有通过冻结参数核对")
    if verification.get("sha256_before") != verification.get("sha256_after"):
        raise ValueError("fine_tune 检查点声明的冻结参数前后哈希不一致")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_checkpoint_sha256", ""))):
        raise ValueError("fine_tune 检查点缺少有效的来源检查点哈希")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("selected_data_signature", ""))):
        raise ValueError("fine_tune 检查点缺少有效的数据签名")
    settings = payload.get("settings", {})
    if settings.get("mode") != payload.get("phase"):
        raise ValueError("fine_tune 检查点的 mode 与 phase 不一致")
    if settings.get("source_checkpoint_sha256") != payload.get("source_checkpoint_sha256"):
        raise ValueError("fine_tune 检查点的来源哈希记录不一致")
    if settings.get("selected_data_signature") != payload.get("selected_data_signature"):
        raise ValueError("fine_tune 检查点的数据签名记录不一致")
    if (
        payload.get("model_config") != ModelConfig().to_dict()
        or payload.get("diffusion_config") != DiffusionConfig().to_dict()
        or payload.get("optics_spec") != OpticsSpec().to_dict()
    ):
        raise ValueError("fine_tune 检查点的网络、K=8 或光学几何不一致")
    if not bool(payload["model_state"]["latent_stats_fitted"]):
        raise ValueError("fine_tune 检查点缺少已拟合 latent statistics")
    _validate_fine_progress_v13(payload)
    model = EPIModel(
        ModelConfig(**payload["model_config"]),
        DiffusionConfig(**payload["diffusion_config"]),
        gradient_checkpointing=True,
    ).to(device)
    optics = OpticalSystem(OpticsSpec(**payload["optics_spec"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    optics.load_state_dict(payload["optics_state"], strict=True)
    current_frozen = _frozen_state_v13(model, optics, payload["phase"])
    current_hash = _tensor_dict_hash_v13(current_frozen)
    current_categories = _frozen_category_hashes_v13(current_frozen)
    if current_hash != verification.get("sha256_after"):
        raise ValueError("fine_tune 检查点内容与冻结参数哈希不一致")
    if current_categories != verification.get("category_sha256"):
        raise ValueError("fine_tune 检查点内容与冻结模块分类哈希不一致")
    counts = payload.get("trainable_parameter_counts", {})
    expected_u = model.parameter_summary()["denoiser"]
    expected_late = model.parameter_summary()["condition_encoder_late"] if payload["phase"] == "C2" else 0
    if counts != {"total": expected_u + expected_late, "U_denoiser": expected_u, "C_late": expected_late}:
        raise ValueError("fine_tune 检查点的可训练参数计数不一致")
    return model, optics, payload


def load_inference_checkpoint_v13(path: str | Path, device):


    preview = torch.load(Path(path), map_location="cpu", weights_only=False)
    if preview.get("format") != V13_FORMAT:
        raise ValueError("需要 v13 检查点")
    if preview.get("role") == "pre_train":
        return load_v13_checkpoint(path, device, require_final=True)
    if preview.get("role") == "fine_tune":
        if preview.get("progress", {}).get("status") != "completed":
            raise ValueError("测试只能读取已完成的 fine_tune 检查点")
        return _load_fine_checkpoint_v13(path, device)
    raise ValueError(f"不支持的 v13 checkpoint role：{preview.get('role')}")


def _save_fine_checkpoint_v13(
    path: Path,
    model: EPIModel,
    optics: OpticalSystem,
    settings: dict,
    progress: dict,
    source_checkpoint: Path,
    selection: DatasetSelectionV13,
    frozen_verification: dict[str, object],
    trainable_counts: dict[str, int],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload = {
        "format": V13_FORMAT,
        "role": "fine_tune",
        "workflow": "strict_two_stage_then_otl",
        "phase": settings["mode"],
        "checkpoint_kind": "fine_tune_final" if progress.get("status") == "completed" else "fine_tune_resume",
        "model_config": model.config.to_dict(),
        "diffusion_config": model.diffusion_config.to_dict(),
        "optics_spec": optics.spec.to_dict(),
        "model_state": _cpu_state(model),
        "optics_state": _cpu_state(optics),
        "settings": dict(settings),
        "progress": dict(progress),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": settings["source_checkpoint_sha256"],
        "selected_data": selection.summary(),
        "selected_data_signature": settings["selected_data_signature"],
        "trainable_parameter_counts": dict(trainable_counts),
        "frozen_verification": dict(frozen_verification),
        "adaptation_loss": "image_objective(O_sim(D(U_K8(C(condition)))), GT)",
        "target_used_by_generator": False,
        "hardware_otl_executed": False,
        "simulation_gradient_only": True,
        "rng": _capture_rng(next(model.parameters()).device),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_name(path.name + ".writing")
    torch.save(payload, temporary)
    _atomic_replace(temporary, path)


def _fine_settings_v13(settings: dict) -> dict:
    cfg = dict(settings)
    required_positive_int = ("epochs", "batch_size", "checkpoint_every", "print_every")
    for name in required_positive_int:
        if type(cfg.get(name)) is not int or cfg[name] <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if type(cfg.get("seed")) is not int:
        raise ValueError("seed 必须是整数")
    for name in ("amp", "augment"):
        if type(cfg.get(name)) is not bool:
            raise ValueError(f"{name} 必须是 bool")
    cfg["mode"] = str(cfg.get("mode", "C2")).upper()
    if cfg["mode"] not in {"C1", "C2"}:
        raise ValueError("mode 必须是 C1 或 C2")
    for name in ("denoiser_lr", "condition_late_lr"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0:
            raise ValueError(f"{name} 必须是正数")
        cfg[name] = float(value)
    return cfg


def run_fine_tune_v13(settings: dict) -> Path:


    cfg = _fine_settings_v13(settings)
    selection = discover_selected_pairs_v13(cfg["dataset"], purpose="fine_tune")
    if not selection.has_targets or any(pair.target is None for pair in selection.pairs):
        raise ValueError("fine_tune requires one GT target for every condition")
    cfg["dataset"] = str(selection.selected_path)
    cfg["selected_data_signature"] = selected_data_signature_v13(selection)
    cfg["selected_pairs"] = len(selection.pairs)
    cfg["simulation_gradient_only"] = True
    cfg["hardware_otl_executed"] = False

    run_home = Path(cfg["run_home"]).expanduser().resolve()
    cfg["run_home"] = str(run_home)
    run_home.mkdir(parents=True, exist_ok=True)
    source_checkpoint = Path(cfg["checkpoint"]).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    source_sha256 = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    cfg["checkpoint"] = str(source_checkpoint)
    cfg["source_checkpoint_sha256"] = source_sha256
    device = transfer_device(cfg["device"])
    lock = run_home / "training.lock"

    with exclusive_run(lock):
        seed_transfer(cfg["seed"])
        resume_path = Path(cfg["resume_checkpoint"]).expanduser().resolve() if cfg.get("resume_checkpoint") else None
        if resume_path is None:
            model, optics, source_payload = load_v13_checkpoint(source_checkpoint, device, require_final=True)
            run_dir = run_home / f"{time.strftime('%Y%m%d_%H%M%S')}_fine_tune_{cfg['mode']}_v13"
            run_dir.mkdir(exist_ok=False)
            progress = {"status": "running", "epoch": 0, "cursor": 0, "step": 0, "history": [], "epoch_loss_sum": 0.0, "epoch_loss_count": 0}
        else:
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
            model, optics, resume_payload = _load_fine_checkpoint_v13(resume_path, device)
            if "optimizer_state" not in resume_payload:
                raise ValueError("resume fine_tune checkpoint 缺少 optimizer_state")
            prior = resume_payload["settings"]
            _v13_validate_resume_home(prior, run_home)
            compare = (
                "mode", "epochs", "batch_size", "denoiser_lr", "condition_late_lr",
                "seed", "amp", "augment", "selected_data_signature", "source_checkpoint_sha256",
            )
            changed = [name for name in compare if prior.get(name) != cfg.get(name)]
            if changed:
                raise ValueError(f"resume settings/data changed: {changed}")
            run_dir = Path(prior["run_dir"]).resolve()
            if resume_path != run_dir / "support" / "resume_fine_tune_v13.pt":
                raise ValueError("resume 文件必须保留在原运行目录的 support 中")
            progress = dict(resume_payload["progress"])
            if progress.get("status") == "completed":
                raise ValueError("该 fine_tune checkpoint 已完成，不应作为 resume")

        cfg["run_dir"] = str(run_dir)
        support = run_dir / "support"
        support.mkdir(exist_ok=True)
        hide_internal(support)
        write_json(run_dir / "settings_v13.json", cfg, internal=False)
        _write_selected_manifest_v13(support / "fine_tune_pairs_v13.csv", selection)

        
        if resume_path is None:
            frozen_reference = _frozen_state_v13(model, optics, cfg["mode"])
        else:
            source_model, source_optics, _ = load_v13_checkpoint(source_checkpoint, torch.device("cpu"), require_final=True)
            frozen_reference = _frozen_state_v13(source_model, source_optics, cfg["mode"])
            del source_model, source_optics
            _assert_frozen_equal_v13(frozen_reference, model, optics, cfg["mode"])

        optimizer, trainable, trainable_counts = _configure_otl_v13(
            model, optics, cfg["mode"], cfg["denoiser_lr"], cfg["condition_late_lr"]
        )
        if resume_path is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            
            
            
            _restore_rng(resume_payload["rng"], device)
        model.zero_grad(set_to_none=True)
        optics.zero_grad(set_to_none=True)
        if device.type == "cuda":
            optics.precompute_transfers(device)

        resume_file = support / "resume_fine_tune_v13.pt"
        history_file = support / "fine_tune_history_v13.csv"
        if not history_file.exists():
            with history_file.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(["epoch", "cursor", "step", "loss", "seconds", "gradient_norm"])
        else:
            _clear_hidden(history_file)

        optimizer_step_entered = False
        epoch_commit_unsafe = False
        rng_before = None
        try:
            for epoch in range(int(progress["epoch"]), cfg["epochs"]):
                order = _v13_epoch_permutation(len(selection.pairs), cfg["seed"], f"fine_{cfg['mode']}", epoch)
                cursor = int(progress["cursor"]) if epoch == int(progress["epoch"]) else 0
                for offset in range(cursor, len(order), cfg["batch_size"]):
                    chosen = [selection.pairs[index] for index in order[offset:offset + cfg["batch_size"]]]
                    rng_before = _capture_rng(device)
                    optimizer.zero_grad(set_to_none=True)
                    condition, target = transfer_batch(chosen, device, augment=cfg["augment"])
                    tick = time.perf_counter()
                    try:
                        with mixed_precision(device, cfg["amp"]):
                            generated = model.generate_phase_logits(condition)
                            reconstruction, _ = optics(generated.phase_logits, condition)
                            loss = image_objective(reconstruction, target)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("fine_tune loss is non-finite")
                        loss.backward()
                        gradient_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
                        if not bool(torch.isfinite(gradient_norm)):
                            raise FloatingPointError("fine_tune gradients are non-finite")
                        if any(parameter.grad is not None for parameter in optics.parameters()):
                            raise AssertionError("frozen optical parameters received gradients")
                        optimizer_step_entered = True
                        optimizer.step()
                        value = float(loss.detach())
                        progress["cursor"] = min(offset + cfg["batch_size"], len(order))
                        progress["step"] = int(progress["step"]) + 1
                        progress["epoch_loss_sum"] = (
                            float(progress["epoch_loss_sum"]) + value * len(chosen)
                        )
                        progress["epoch_loss_count"] = (
                            int(progress["epoch_loss_count"]) + len(chosen)
                        )
                        optimizer_step_entered = False
                        rng_before = None
                    except BaseException:
                        if not optimizer_step_entered and rng_before is not None:
                            _restore_rng(rng_before, device)
                            optimizer.zero_grad(set_to_none=True)
                        raise
                    del loss, condition, target, reconstruction, generated
                    with history_file.open("a", encoding="utf-8", newline="") as stream:
                        csv.writer(stream).writerow([epoch + 1, progress["cursor"], progress["step"], value, time.perf_counter() - tick, float(gradient_norm)])
                    if progress["step"] % cfg["checkpoint_every"] == 0:
                        frozen = _assert_frozen_equal_v13(frozen_reference, model, optics, cfg["mode"])
                        _save_fine_checkpoint_v13(
                            resume_file, model, optics, cfg, progress, source_checkpoint,
                            selection, frozen, trainable_counts, optimizer,
                        )
                        hide_internal(resume_file)
                    if progress["step"] % cfg["print_every"] == 0 or progress["cursor"] == len(order):
                        print(
                            f"{cfg['mode']}: epoch {epoch + 1}/{cfg['epochs']}; "
                            f"images {progress['cursor']}/{len(order)}; loss={value:.6f}",
                            flush=True,
                        )
                if int(progress["epoch_loss_count"]) != len(order):
                    raise AssertionError("fine_tune epoch loss does not cover every pair")
                mean_loss = float(progress["epoch_loss_sum"]) / len(order)
                
                
                
                frozen = _assert_frozen_equal_v13(frozen_reference, model, optics, cfg["mode"])
                _save_fine_checkpoint_v13(
                    resume_file, model, optics, cfg, progress, source_checkpoint,
                    selection, frozen, trainable_counts, optimizer,
                )
                hide_internal(resume_file)
                epoch_commit_unsafe = True
                progress["history"].append({"epoch": epoch + 1, "mean_image_objective": mean_loss, "pairs": len(order)})
                progress.update(epoch=epoch + 1, cursor=0, epoch_loss_sum=0.0, epoch_loss_count=0)
                _save_fine_checkpoint_v13(
                    resume_file, model, optics, cfg, progress, source_checkpoint,
                    selection, frozen, trainable_counts, optimizer,
                )
                hide_internal(resume_file)
                epoch_commit_unsafe = False
        except BaseException:
            if not optimizer_step_entered and not epoch_commit_unsafe:
                frozen = _assert_frozen_equal_v13(frozen_reference, model, optics, cfg["mode"])
                _save_fine_checkpoint_v13(
                    resume_file, model, optics, cfg, progress, source_checkpoint,
                    selection, frozen, trainable_counts, optimizer,
                )
                hide_internal(resume_file)
            elif epoch_commit_unsafe:
                print(
                    "fine_tune epoch commit 被中断；保留最近的完整恢复边界。",
                    flush=True,
                )
            else:
                print(
                    "中断发生在 optimizer.step 内；为避免保存部分更新，未覆盖已有断点。",
                    flush=True,
                )
            raise
        finally:
            hide_internal(history_file)

        if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
            raise FloatingPointError("fine_tune final trainable weights are non-finite")
        progress["status"] = "completed"
        frozen = _assert_frozen_equal_v13(frozen_reference, model, optics, cfg["mode"])
        final = run_dir / "fine_tuned_v13.pt"
        _save_fine_checkpoint_v13(
            final, model, optics, cfg, progress, source_checkpoint,
            selection, frozen, trainable_counts,
        )
        write_json(
            run_dir / "completed_fine_tune_v13.json",
            {
                "status": "completed",
                "checkpoint": final.name,
                "mode": cfg["mode"],
                "pairs": len(selection.pairs),
                "epochs": cfg["epochs"],
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": source_sha256,
                "selected_data_signature": cfg["selected_data_signature"],
                "trainable_parameter_counts": trainable_counts,
                "frozen_verification": frozen,
                "loss": "image_objective only",
                "hardware_otl_executed": False,
                "simulation_gradient_only": True,
            },
            internal=False,
        )
        print(f"COMPLETE: {final}", flush=True)
        return run_dir


def _safe_stem_v13(pair: SelectedPairV13) -> str:
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", f"{pair.stage}_{pair.family}_{pair.domain}_{pair.condition.stem}")
    digest = hashlib.sha256(pair.key.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{digest}"


def _comparison_sheet_v13(rows, destination: Path, title: str) -> None:
    if not rows:
        return
    visual_sheet(rows, destination, title, reconstruction_label="Optical reconstruction")


def run_test_v13(settings: dict) -> Path:


    cfg = dict(settings)
    for name in ("print_every", "phase_examples"):
        if type(cfg.get(name)) is not int or cfg[name] < (1 if name == "print_every" else 0):
            raise ValueError(f"{name} 参数无效")
    if type(cfg.get("seed")) is not int or type(cfg.get("amp")) is not bool:
        raise ValueError("test 的 seed/amp 参数无效")
    selection = discover_selected_pairs_v13(cfg["dataset"], purpose="test")
    cfg["dataset"] = str(selection.selected_path)
    cfg["selected_data_signature"] = selected_data_signature_v13(selection)
    cfg["selected_pairs"] = len(selection.pairs)

    checkpoint = Path(cfg["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_home = Path(cfg["run_home"]).expanduser().resolve()
    run_home.mkdir(parents=True, exist_ok=True)
    device = transfer_device(cfg["device"])
    lock = run_home / ("gpu.lock" if device.type == "cuda" else "cpu_test.lock")
    with exclusive_run(lock):
        run_dir = run_home / f"{time.strftime('%Y%m%d_%H%M%S')}_test_v13"
        run_dir.mkdir(exist_ok=False)
        predictions = run_dir / "predictions"
        predictions.mkdir()
        comparisons = run_dir / "comparison previews"
        if selection.has_targets:
            comparisons.mkdir()
        support = run_dir / "support"
        support.mkdir()
        hide_internal(support)
        _write_selected_manifest_v13(support / "test_pairs_v13.csv", selection)
        write_json(run_dir / "settings_v13.json", cfg, internal=False)

        model, optics, payload = load_inference_checkpoint_v13(checkpoint, device)
        model.eval()
        optics.eval()
        if device.type == "cuda":
            optics.precompute_transfers(device)
        rows: list[dict[str, object]] = []
        previews = []
        preview_groups: set[tuple[str, str]] = set()
        exported_phases = 0
        with torch.inference_mode():
            for index, pair in enumerate(selection.pairs, 1):
                condition = load_rgb_tensor(pair.condition).unsqueeze(0).to(device)
                
                output, logits = predict_transfer(model, optics, condition, pair.key, cfg["seed"], cfg["amp"])
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError(f"test reconstruction is non-finite: {pair.key}")
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError(f"test phase logits are non-finite: {pair.key}")
                stem = _safe_stem_v13(pair)
                np.save(predictions / f"{stem}_reconstruction_v13.npy", output[0].detach().float().cpu().numpy())
                Image.fromarray(_rgb8(output[0])).save(predictions / f"{stem}_reconstruction_v13.png")
                if exported_phases < cfg["phase_examples"]:
                    np.save(
                        predictions / f"{stem}_cws_phase_v13.npy",
                        phase_from_logits(logits)[0].detach().float().cpu().numpy(),
                    )
                    exported_phases += 1

                if pair.target is not None:
                    
                    target = load_rgb_tensor(pair.target).unsqueeze(0).to(device)
                    values: dict[str, object] = {
                        "stage": pair.stage,
                        "family": pair.family,
                        "domain": pair.domain,
                        "key": pair.key,
                        "group": pair.group,
                    }
                    for label, image in (("input", condition), ("reconstruction", output)):
                        mse = (image.float() - target).square().mean().clamp_min(1e-12)
                        values[f"{label}_psnr"] = float(-10.0 * torch.log10(mse))
                        values[f"{label}_ms_ssim"] = float(ms_ssim_index(image, target))
                    rows.append(values)
                    identity = (pair.stage, pair.group)
                    if identity not in preview_groups and len(previews) < 24:
                        previews.append((pair.key, condition[0].cpu(), output[0].cpu(), target[0].cpu()))
                        preview_groups.add(identity)
                    del target
                if index % cfg["print_every"] == 0 or index == len(selection.pairs):
                    print(f"test: {index}/{len(selection.pairs)}", flush=True)

        np.save(run_dir / "shared_odr_phase_v13.npy", optics.export_odr_phase(cpu=True).numpy())
        means: dict[str, float] = {}
        if rows:
            metrics_path = run_dir / "metrics_v13.csv"
            with metrics_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            names = ("input_psnr", "input_ms_ssim", "reconstruction_psnr", "reconstruction_ms_ssim")
            means = {name: float(np.mean([float(row[name]) for row in rows])) for name in names}
            for start in range(0, len(previews), 4):
                _comparison_sheet_v13(
                    previews[start:start + 4],
                    comparisons / f"comparison_{start // 4 + 1:02d}_v13.png",
                    "Selected-data evaluation; condition / reconstruction / GT",
                )

        role = payload.get("role")
        same_as_fine_data = bool(
            role == "fine_tune"
            and payload.get("selected_data_signature") == cfg["selected_data_signature"]
        )
        write_json(
            run_dir / "test_summary_v13.json",
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "checkpoint_role": role,
                "checkpoint_workflow": payload.get("workflow"),
                "checkpoint_phase": payload.get("phase"),
                "conditions": len(selection.pairs),
                "targets_available": len(rows),
                "metrics": means,
                "selected_data": selection.summary(),
                "target_used_by_generator": False,
                "same_data_as_fine_tuning": same_as_fine_data,
                "independent_test_claimed": False,
                "hardware_inference_executed": False,
                "geometry": optics.spec.to_dict(),
            },
            internal=False,
        )
        print(f"TEST OUTPUT: {run_dir}", flush=True)
        return run_dir


if __name__ == "__main__":
    training_entry_v13("latent")
