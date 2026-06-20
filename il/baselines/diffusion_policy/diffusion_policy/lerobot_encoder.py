"""ResNet18 + SpatialSoftmax visual encoder (original Diffusion Policy / LeRobot style).

Drop-in replacement for ``PlainConv`` in the RGB Diffusion Policy. The point: instead of
global-max-pooling the conv feature map to a bag of features (which discards *where* things
are), SpatialSoftmax turns the feature map into **keypoint coordinates** — for each of K
channels it computes the softmax-weighted expected (x, y) location of activation. That gives
the policy explicit, continuous object/gripper positions, which is what a spatial pick-and-place
task needs.

Encoder = ResNet18 trunk (BatchNorm -> GroupNorm for small/stacked batches) truncated to an
8x8 feature map (finer localisation than the final 4x4), then SpatialSoftmax with ``num_kp``
keypoints -> ``2*num_kp`` coords -> Linear to ``out_dim``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Per-channel soft-argmax over a (C, H, W) feature map -> (2*K) expected coords.

    Optionally a 1x1 conv first maps C channels to ``num_kp`` keypoint channels.
    Returns a flat (B, 2*num_kp) vector of (x, y) in [-1, 1] image coordinates.
    """

    def __init__(self, in_channels, num_kp=32):
        super().__init__()
        self.num_kp = num_kp
        self.kp_conv = nn.Conv2d(in_channels, num_kp, kernel_size=1) if num_kp else None
        self.out_channels = num_kp if num_kp else in_channels

    def forward(self, feat):                       # feat: (B, C, H, W)
        if self.kp_conv is not None:
            feat = self.kp_conv(feat)
        b, c, h, w = feat.shape
        # coordinate grids in [-1, 1]
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        xs = xs.reshape(1, 1, h * w)
        ys = ys.reshape(1, 1, h * w)
        attn = F.softmax(feat.reshape(b, c, h * w), dim=-1)   # spatial softmax per channel
        exp_x = (attn * xs).sum(dim=-1)            # (B, C)
        exp_y = (attn * ys).sum(dim=-1)            # (B, C)
        return torch.stack([exp_x, exp_y], dim=-1).reshape(b, 2 * c)   # (B, 2C)


def _bn_to_gn(module, num_groups=16):
    """Recursively replace BatchNorm2d with GroupNorm (robust to the small B*obs_horizon
    batches and to running-stat drift; LeRobot does the same when not relying on BN stats)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            g = num_groups if child.num_features % num_groups == 0 else 1
            setattr(module, name, nn.GroupNorm(g, child.num_features))
        else:
            _bn_to_gn(child, num_groups)


class ResNet18SpatialSoftmax(nn.Module):
    """ResNet18 trunk (-> 8x8 feature map) + SpatialSoftmax + Linear -> out_dim."""

    def __init__(self, in_channels=3, out_dim=256, num_kp=32, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
        try:
            weights = "IMAGENET1K_V1" if pretrained else None
            net = resnet18(weights=weights)
        except Exception as e:                     # offline / no weights cache -> train from scratch
            print(f"[lerobot_encoder] pretrained resnet18 unavailable ({e}); using random init",
                  flush=True)
            net = resnet18(weights=None)
        if in_channels != 3:                       # adapt first conv for non-RGB stacks
            net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        # trunk up to layer3 -> for a 128x128 input this yields a (256, 8, 8) feature map
        # (finer than layer4's 4x4), giving SpatialSoftmax more spatial resolution to localise
        # the small parcels as well as the large bins.
        self.trunk = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3,
        )
        _bn_to_gn(self.trunk)
        feat_channels = 256
        self.spatial_softmax = SpatialSoftmax(feat_channels, num_kp=num_kp)
        self.fc = nn.Sequential(nn.Linear(2 * num_kp, out_dim), nn.ReLU())

    def forward(self, image):                      # image: (B, C, H, W), float in [0, 1]
        feat = self.trunk(image)
        kp = self.spatial_softmax(feat)
        return self.fc(kp)


class ResNet18ColorKeypoints(nn.Module):
    """ResNet18 trunk + colour-AWARE keypoints.

    The plain ``SpatialSoftmax`` above outputs only keypoint *coordinates* (where activation
    peaks are) — the soft-argmax throws away appearance, so the encoder is effectively
    **colour-blind**. That is fatal for a *colour-routing* task: the policy can localise parcels
    and bins but cannot tell the red bin from the blue bin — especially on **hard**, where the
    bins swap sides between episodes, so a side cannot be memorised.

    This encoder keeps both:
      * **WHERE** — ``2*num_kp`` soft-argmax keypoint coordinates (as before).
      * **WHAT**  — for every keypoint, the attention-weighted *appearance* (colour/texture)
        readout from a compact feature map, plus a global avg/max colour descriptor.

    Output dim is ``out_dim`` (drop-in replacement; same 256-d as the others), so the rest of the
    Diffusion-Policy network is unchanged. Select it with ``--visual-encoder resnet18_color`` and
    load it at eval with ``warehouse_sort.il_policy:load_dp_rgb_color``.
    """

    def __init__(self, in_channels=3, out_dim=256, num_kp=32, app_channels=16, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
        try:
            weights = "IMAGENET1K_V1" if pretrained else None
            net = resnet18(weights=weights)
        except Exception as e:                     # offline / no weights cache -> train from scratch
            print(f"[lerobot_encoder] pretrained resnet18 unavailable ({e}); using random init",
                  flush=True)
            net = resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.trunk = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3,
        )
        _bn_to_gn(self.trunk)
        feat_channels = 256
        self.num_kp = num_kp
        self.app_channels = app_channels
        self.kp_conv = nn.Conv2d(feat_channels, num_kp, kernel_size=1)        # attention maps (where)
        self.app_conv = nn.Sequential(nn.Conv2d(feat_channels, app_channels, 1), nn.ReLU())  # what/colour
        feat_dim = 2 * num_kp + num_kp * app_channels + 2 * app_channels
        self.fc = nn.Sequential(nn.Linear(feat_dim, out_dim), nn.ReLU())

    def forward(self, image):                      # image: (B, C, H, W), float in [0, 1]
        feat = self.trunk(image)                   # (B, 256, h, w)
        b, c, h, w = feat.shape
        attn = F.softmax(self.kp_conv(feat).reshape(b, self.num_kp, h * w), dim=-1)  # (B, K, hw)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        xs = xs.reshape(1, 1, h * w)
        ys = ys.reshape(1, 1, h * w)
        exp_x = (attn * xs).sum(dim=-1)            # (B, K)
        exp_y = (attn * ys).sum(dim=-1)            # (B, K)
        app = self.app_conv(feat).reshape(b, self.app_channels, h * w)  # (B, A, hw)
        # appearance at each keypoint = its attention-weighted colour/texture readout
        kp_app = torch.einsum("bkn,ban->bka", attn, app).reshape(b, self.num_kp * self.app_channels)
        g_avg = app.mean(dim=-1)                   # (B, A) global colour descriptor
        g_max = app.max(dim=-1).values            # (B, A)
        out = torch.cat([exp_x, exp_y, kp_app, g_avg, g_max], dim=-1)
        return self.fc(out)
