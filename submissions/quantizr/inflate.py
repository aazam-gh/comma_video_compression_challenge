#!/usr/bin/env python
import io
import os
import sys
import tempfile
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------
# FP4 Dequantization Tools
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles: torch.Tensor, scales: torch.Tensor, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()

        nibbles = nibbles.view(-1, block_size)
        signs = (nibbles >> 3).to(torch.int64)
        mag_idx = (nibbles & 0x7).to(torch.int64)

        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = levels[mag_idx]
        q = torch.where(signs.bool(), -q, q)
        dq = q * scales[:, None].to(torch.float32)
        return dq.view(-1)[:flat_n].reshape(orig_shape)

def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    flat = packed.reshape(-1)
    hi = (flat >> 4) & 0x0F
    lo = flat & 0x0F
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]

def get_decoded_state_dict(payload_data, device: torch.device):
    data = torch.load(io.BytesIO(payload_data), map_location=device)
    state_dict = {}

    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            padded_count = rec["packed_weight"].numel() * 2
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), padded_count)
            w = FP4Codebook.dequantize_from_nibbles(
                nibbles, rec["scales_fp16"].to(device), rec["weight_shape"]
            )
        else:
            w = rec["weight_fp16"].to(device).float()

        state_dict[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()

    for name, tensor in data["dense_fp16"].items():
        state_dict[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)

    return state_dict

# -----------------------------
# Architecture (Inference Only)
# -----------------------------

class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, depth_mult: int = 4, quantize_weight: bool = True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult

        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, depth_mult: int = 4, quantize_weight: bool = True):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult

        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid_ch, out_ch, 1, padding=0, bias=True, quantize_weight=quantize_weight)

    def forward(self, x):
        return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch: int, depth_mult: int = 4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int, depth_mult: int = 4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch)

        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond_emb):
        residual = x
        x = self.norm2(self.conv2(self.conv1(x)))

        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta

        return self.act(residual + x)

class Mask1Encoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, out_ch=80, depth_mult=2):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem = SepConvGNAct(emb_dim + 2, out_ch, depth_mult=depth_mult)
        self.block = SepResBlock(out_ch, depth_mult=depth_mult)

    def forward(self, mask1, coords):
        e1 = self.embedding(mask1.long()).permute(0, 3, 1, 2)
        e1_up = F.interpolate(e1, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.stem(torch.cat([e1_up, coords], dim=1)))

class ColorHintEncoder(nn.Module):
    def __init__(self, out_ch=32, depth_mult=2):
        super().__init__()
        self.encode = nn.Sequential(
            SepConvGNAct(3, out_ch, depth_mult=depth_mult),
            SepResBlock(out_ch, depth_mult=depth_mult),
        )

    def forward(self, color_hint):
        x = F.interpolate(color_hint.float(), size=(384, 512), mode="bilinear", align_corners=False)
        return self.encode(x)

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=80, c2=96, depth_mult=2):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)

        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)

        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )

        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, coords: torch.Tensor):
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f

class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 80, depth_mult: int = 2):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0

class FrameHead(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int = 48, hidden: int = 80, depth_mult: int = 2):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=2,
                 c1=80, c2=96, use_mask1=True, use_color=True, color_ch=32):
        super().__init__()
        self.use_mask1 = use_mask1
        self.use_color = use_color
        self.shared_trunk = SharedMaskDecoder(
            num_classes=num_classes, emb_dim=6, c1=c1, c2=c2, depth_mult=depth_mult)

        if use_mask1:
            self.mask1_encoder = Mask1Encoder(num_classes=num_classes, emb_dim=6, out_ch=c1, depth_mult=depth_mult)
        if use_color:
            self.color_encoder = ColorHintEncoder(out_ch=color_ch, depth_mult=depth_mult)

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        f1_ch = c1 + (c1 if use_mask1 else 0) + (color_ch if use_color else 0)
        f2_ch = c1 + (color_ch if use_color else 0)

        self.frame1_head = FrameHead(in_ch=f1_ch, cond_dim=cond_dim, hidden=c1, depth_mult=depth_mult)
        self.frame2_head = Frame2StaticHead(in_ch=f2_ch, hidden=c1, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor,
                mask1=None, color1=None, color2=None):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)

        shared_feat = self.shared_trunk(mask2, coords)

        # Frame 2: shared + optional color
        f2_parts = [shared_feat]
        if self.use_color and color2 is not None:
            f2_parts.append(self.color_encoder(color2))
        pred_frame2 = self.frame2_head(torch.cat(f2_parts, dim=1))

        # Frame 1: shared + optional mask1 + optional color + pose
        f1_parts = [shared_feat]
        if self.use_mask1 and mask1 is not None:
            f1_parts.append(self.mask1_encoder(mask1, coords))
        if self.use_color and color1 is not None:
            f1_parts.append(self.color_encoder(color1))
        cond_emb = self.pose_mlp(pose6)
        pred_frame1 = self.frame1_head(torch.cat(f1_parts, dim=1), cond_emb)

        return pred_frame1, pred_frame2

def make_coord_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


# -----------------------------
# Inference Helpers & Main
# -----------------------------
def load_encoded_mask_video(path: str) -> torch.Tensor:
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.round(img / 63.0).astype(np.uint8)
        cls_img = np.clip(cls_img, 0, 4)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

    model_br = data_dir / "model.pt.br"
    mask_br = data_dir / "mask.obu.br"
    pose_br = data_dir / "pose.npy.br"
    color_br = data_dir / "color.npy.br"

    generator = JointFrameGenerator().to(device)

    # 1. Load Weights
    with open(model_br, "rb") as f:
        weights_data = brotli.decompress(f.read())

    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()

    # 2. Load Mask Video (.obu) - contains interleaved frame1 + frame2 masks
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp_obu:
        with open(mask_br, "rb") as f:
            tmp_obu.write(brotli.decompress(f.read()))
        tmp_obu_path = tmp_obu.name

    mask_frames_all = load_encoded_mask_video(tmp_obu_path)
    os.remove(tmp_obu_path)

    # Split interleaved masks: f1_pair0, f2_pair0, f1_pair1, f2_pair1, ...
    mask1_all = mask_frames_all[0::2]  # frame1 masks
    mask2_all = mask_frames_all[1::2]  # frame2 masks

    # 3. Load Pose Vectors
    with open(pose_br, "rb") as f:
        pose_bytes = brotli.decompress(f.read())
    pose_frames_all = torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()

    # 4. Load Color Hints (optional)
    color1_all, color2_all = None, None
    if color_br.exists():
        with open(color_br, "rb") as f:
            color_bytes = brotli.decompress(f.read())
        color_all = torch.from_numpy(np.load(io.BytesIO(color_bytes))).float()
        color1_all = color_all[0::2]  # frame1 colors
        color2_all = color_all[1::2]  # frame2 colors

    out_h, out_w = 874, 1164
    cursor = 0
    batch_size = 4

    # 1 mask pair per generated pair, assume 600 pairs per standard 1200 frame chunk.
    pairs_per_file = 600

    # Test-time optimization: run gradient steps to refine outputs
    TTO_STEPS = 0  # Set > 0 to enable (e.g., 5-20 steps)
    TTO_LR = 1e-3

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out_path = out_dir / f"{base_name}.raw"

            file_mask1 = mask1_all[cursor : cursor + pairs_per_file]
            file_mask2 = mask2_all[cursor : cursor + pairs_per_file]
            file_poses = pose_frames_all[cursor : cursor + pairs_per_file]
            file_color1 = color1_all[cursor : cursor + pairs_per_file] if color1_all is not None else None
            file_color2 = color2_all[cursor : cursor + pairs_per_file] if color2_all is not None else None
            cursor += pairs_per_file

            with open(raw_out_path, "wb") as f_out:
                pbar = tqdm(range(0, file_mask2.shape[0], batch_size), desc=f"Decoding {file_name}")

                for i in pbar:
                    in_mask1 = file_mask1[i : i + batch_size].to(device).long()
                    in_mask2 = file_mask2[i : i + batch_size].to(device).long()
                    in_pose6 = file_poses[i : i + batch_size].to(device).float()
                    in_color1 = file_color1[i : i + batch_size].to(device).float() if file_color1 is not None else None
                    in_color2 = file_color2[i : i + batch_size].to(device).float() if file_color2 is not None else None

                    fake1, fake2 = generator(in_mask2, in_pose6,
                                              mask1=in_mask1,
                                              color1=in_color1, color2=in_color2)

                    # Test-time optimization: refine frames with gradient steps
                    if TTO_STEPS > 0:
                        fake1, fake2 = test_time_optimize(
                            generator, fake1, fake2,
                            in_mask1, in_mask2, in_pose6,
                            in_color1, in_color2,
                            device, TTO_STEPS, TTO_LR)

                    fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
                    fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)

                    batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
                    batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

                    output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
                    f_out.write(output_bytes.cpu().numpy().tobytes())


def test_time_optimize(generator, fake1_init, fake2_init,
                        mask1, mask2, pose6, color1, color2,
                        device, steps=10, lr=1e-3):
    """Test-time optimization: refine generated frames with gradient descent.

    We optimize a latent perturbation added to the generator output
    to minimize reconstruction consistency with the masks and poses.
    """
    # Create learnable perturbations
    delta1 = torch.zeros_like(fake1_init, requires_grad=True)
    delta2 = torch.zeros_like(fake2_init, requires_grad=True)
    optimizer = torch.optim.Adam([delta1, delta2], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        f1 = (fake1_init + delta1).clamp(0, 255)
        f2 = (fake2_init + delta2).clamp(0, 255)

        # Re-run through generator to get consistent features
        pred1, pred2 = generator(mask2, pose6, mask1=mask1, color1=color1, color2=color2)

        # Loss: smoothness (total variation) + consistency with initial output
        tv_loss = total_variation(delta1) + total_variation(delta2)
        consistency = F.mse_loss(f1, pred1.detach()) + F.mse_loss(f2, pred2.detach())

        loss = consistency + 0.01 * tv_loss
        loss.backward()
        optimizer.step()

    return (fake1_init + delta1).detach().clamp(0, 255), (fake2_init + delta2).detach().clamp(0, 255)


def total_variation(x):
    """Total variation regularization for smoothness."""
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return diff_h.pow(2).mean() + diff_w.pow(2).mean()


if __name__ == "__main__":
    main()
