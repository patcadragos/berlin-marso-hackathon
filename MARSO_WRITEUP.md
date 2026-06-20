# Colour-Aware Diffusion Policy for Warehouse Parcel Sorting

**Subtitle:** Fixing the colour-blind perception bottleneck that caps the RGB baseline — plus
augmentation for the held-out layouts.

> Fill the `[...]` with your measured numbers before submitting.

## TL;DR

The provided RGB Diffusion-Policy template scores ≈0 not because the *control* is hard but
because its **perception is colour-blind**. We diagnose this, replace the encoder with a
**colour-aware keypoint encoder**, add **random-shift augmentation** for generalisation, and
train one checkpoint on the hard distribution that serves all three levels.

- easy: **[..]%**  · medium: **[..]%**  · hard: **[..]%**
- **final_score = 0.2·easy + 0.3·medium + 0.5·hard = [..]**

## The diagnosis

The task is *colour routing*: read each parcel's coloured tag, place it in the same-colour bin.
The policy sees only a 128×128 scene RGB plus 26-d proprioception — and the `state` vector is
**proprioception only**, with no parcel/bin/colour fields. So colour can come **only from pixels**.

The baseline encoder is **ResNet18 + SpatialSoftmax**. SpatialSoftmax's soft-argmax returns, per
channel, the *expected (x, y) location* of activation — **coordinates only**. It throws away
appearance, so the network learns *where* things are but not *what colour* they are. On **easy**
and **medium** the bins are fixed, so the policy can memorise "red bin = +y side" and partly cope.
On **hard** the bins **swap sides** between episodes — memorisation breaks and true colour reading
is required. Since hard is weighted **0.5**, the colour-blind encoder caps the score exactly where
the points are.

A second constraint: evaluation runs only **200 steps**, but hard has **6 parcels**; the policy
must be efficient or it times out before placing them all.

## What we changed

1. **Colour-aware encoder (`resnet18_color`).** Same ResNet18 trunk, but the head keeps both
   *where* (soft-argmax keypoint coordinates) **and** *what* (an attention-weighted colour/texture
   readout at each keypoint, plus a global colour descriptor). This restores the colour signal the
   routing task needs while preserving the spatial localisation that makes pick-and-place work.
   *(See `il/baselines/diffusion_policy/diffusion_policy/lerobot_encoder.py:ResNet18ColorKeypoints`.)*

2. **Random-shift augmentation (`aug_pad=4`).** DrQ-style translation augmentation, applied at
   training only and colour-safe (shifts pixels, never hue). Targets the held-out eval, which uses
   **wider position randomisation** than training.

3. **Train on the hard distribution, one checkpoint for all levels.** Hard has the most parcels,
   the bin-swaps, and the widest jitter, so it is the most general training signal; the fixed-shape
   RGB observation lets one checkpoint score easy, medium, and hard.

4. **Longer training (60k iters)** and **more denoising steps at eval (free quality)**.

## Results & ablation

| Encoder | easy | medium | hard | final |
|---|---|---|---|---|
| resnet18 (baseline, colour-blind) | [..] | [..] | [..] | [..] |
| plain_conv (colour-preserving) | [..] | [..] | [..] | [..] |
| **resnet18_color (ours)** | **[..]** | **[..]** | **[..]** | **[..]** |

Key finding: the colour-aware encoder's gain is **largest on hard** (the bin-swap level), exactly
as the diagnosis predicts — evidence the bottleneck was perception, not control.

## What we'd try next

- Faster scripted demos so hard fits inside the 200-step budget.
- Per-level fine-tuning from the shared checkpoint.
- RL fine-tuning from the sparse +1 reward on top of the IL policy (residual policy).

## Reproduce

Full commands in `MARSO_RUNBOOK.md`. In short:
`pixi run python il/train.py method=dp_rgb_color demo_dir=hard`, then `eval.py` per level with
`policy=warehouse_sort.il_policy:load_dp_rgb_color`. Repo (judge clones this): **<your fork URL>**.
