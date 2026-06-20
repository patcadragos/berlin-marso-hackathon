# Marso Hack — Runbook & Strategy (do this in order)

> ⚠️ **These changes were written but NOT executed** (no local GPU/Linux). ManiSkill 3 needs a
> CUDA Linux GPU, so run everything below in **Google Colab (GPU runtime)** or Kaggle. Every
> change is **opt-in and non-breaking**: the stock `method=dp_rgb` path is untouched and always
> works as a fallback. Run the sanity check first, then layer improvements, keeping what scores.

---

## Why the baseline scores ~0 (the thesis)

This is a **colour-routing** task: read the parcel's coloured tag, drop it in the same-colour bin.

- The observation is only a **128×128 scene RGB** + **26-d proprioception**. The `state` vector
  has **no** parcel/bin/colour info (proprioception only), so **colour must come from pixels**.
- The provided `resnet18` encoder is **ResNet18 + SpatialSoftmax**. SpatialSoftmax's soft-argmax
  outputs only **keypoint coordinates (WHERE)** and **discards appearance (WHAT/colour)**.
- Result: the policy can localise parcels/bins but **cannot tell the red bin from the blue bin**.
  - On **easy/medium** the bins never move, so it can *memorise a side* and partly fake it.
  - On **hard** the bins **swap sides** between episodes — memorising fails, colour is required.
    Hard is weighted **0.5**, so the colour-blind encoder caps the score where it matters most.

Second structural constraint: **eval runs only `max_episode_steps=200`** (`conf/config.yaml`),
but hard has **6 parcels** and the scripted demos take up to ~420 steps. So hard is also
**time-capped** — the policy must be *efficient*, or it runs out of steps before placing all 6.

---

## The levers (highest expected value first)

| # | Lever | How | Risk |
|---|-------|-----|------|
| 1 | **Colour-aware perception** | `method=dp_rgb_color` (custom `resnet18_color` encoder: keypoints **+** colour readout). Zero-code fallback: `flags.visual_encoder=plain_conv`. | low (opt-in) |
| 2 | **Augmentation for generalisation** | `flags.aug_pad=4` random-shift (held-out widens positions). On by default in `dp_rgb_color`. | low |
| 3 | **Train on HARD demos, one ckpt for all** | `demo_dir=hard` — most parcels/swaps/jitter → most general; scores all 3 levels. | low |
| 4 | **Train longer** | `flags.total_iters=60000` (image policies are data-hungry). | none |
| 5 | **More denoising at eval** | `num_inference_steps` 16→24 in the loader (eval-only, free). | none |
| 6 | **Faster demos for hard** (advanced) | regenerate hard demos that finish 6 parcels in <200 steps (see bottom). | medium (uses GPU time) |

---

## Step 0 — Open Colab with a GPU

Open the repo's Colab badge (it loads `starter.ipynb`), **Runtime → Change runtime type → T4 GPU**.
Then work from **your fork** so you can commit checkpoints:

```bash
# In a Colab cell — replace <you> with your GitHub user (fork the repo first on github.com)
!git clone https://github.com/<you>/berlin-marso-hackathon
%cd berlin-marso-hackathon
!curl -fsSL https://pixi.sh/install.sh | bash
!~/.pixi/bin/pixi install && ~/.pixi/bin/pixi run install
```

> If `pixi` is awkward in Colab, install deps with pip per the `starter.ipynb` cells — the badge
> notebook is pre-wired. The commands below assume `pixi run` works; drop the prefix if not.

## Step 1 — Get the data + sanity-check the stock baseline

```bash
# Join the competition on Kaggle + set a Kaggle API token (kaggle.json), then:
!pixi run python il/download_demos.py            # -> il/demos/{easy,medium,hard}/

# Sanity: does the pipeline train+eval at all? (short run)
!pixi run python il/train.py method=dp_rgb demo_dir=easy flags.total_iters=5000 flags.eval_freq=2500
!pixi run python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_sort_accuracy.pt \
    eval_config=conf/eval/default.yaml
```
You should see a `SORT ACCURACY` line. Expect it low — that's the point. This confirms your env works.

## Step 2 — Train the colour-aware policy (the main run)

```bash
# ~60-90 min on a T4. Trains on HARD demos with the colour encoder + augmentation.
!pixi run python il/train.py method=dp_rgb_color demo_dir=hard
```

## Step 3 — Evaluate on ALL THREE levels (estimate your weighted score)

```bash
CKPT=il/baselines/diffusion_policy/runs/warehouse_rgb_dp_color/checkpoints/best_eval_sort_accuracy.pt
for L in easy medium hard; do
  !pixi run python eval.py difficulty=$L \
      policy=warehouse_sort.il_policy:load_dp_rgb_color \
      checkpoint=$CKPT eval_config=conf/eval/default.yaml
done
```
Record the three `SORT ACCURACY` numbers and compute:
`final = 0.2*easy + 0.3*medium + 0.5*hard`. Each eval also drops an **mp4** under the run's
`videos/` folder — grab the best one for your submission video.

## Step 4 — A/B the encoder (only if you have GPU time)

Compare colour-aware vs the zero-code colour fallback (`plain_conv`) vs the stock keypoint encoder:

```bash
# plain_conv (colour-preserving, no custom code)
!pixi run python il/train.py method=dp_rgb demo_dir=hard flags.visual_encoder=plain_conv flags.aug_pad=4 flags.exp_name=warehouse_rgb_dp_plain
# eval it with the matching loader: policy=warehouse_sort.il_policy:load_dp_rgb_plain
```
Keep whichever wins on the weighted score. Update `submission.yaml`'s `policy:` to the matching
loader (`load_dp_rgb_color` or `load_dp_rgb_plain`).

## Step 5 — Package & submit

1. Edit `submission.yaml`: set `team:` and confirm `policy:` matches your winning encoder.
2. Commit your checkpoint(s) to the fork:
   ```bash
   !git add il/baselines/diffusion_policy/runs/warehouse_rgb_dp_color/checkpoints/best_eval_sort_accuracy.pt submission.yaml
   !git commit -m "Colour-aware DP submission"
   !git push
   ```
   (checkpoints are NOT gitignored by this repo, so a plain `git add` picks them up. If the
   `.pt` exceeds GitHub's 100MB file limit, host it elsewhere and fetch it with a script the
   repo runs; the manifest just needs a path that exists after clone.)
3. **Verify a fresh clone runs** (Step 2's eval command on a clean checkout) — the judge does this.
4. On Kaggle → **Writeups → New Writeup**: paste `MARSO_WRITEUP.md`, attach cover image + the
   YouTube video (≤3 min, script in `MARSO_VIDEO.md`), set Project Link = your fork URL, **Save → Submit**.

---

## Advanced lever — faster demos for hard (if hard is step-capped)

If hard eval shows high accuracy-per-attempt but low total (policy runs out of 200 steps), make
the demos finish faster so the learned policy is faster too. In `examples/scripted_policy.py`
raise `SPEED` (0.7→0.9), the descent/drop speeds, and shrink the per-phase `phase_steps`
thresholds and settle counts, then regenerate and retrain on the faster demos:

```bash
!pixi run python il/gen_demos.py --difficulty hard --num-episodes 200 --base-seed 9000
!pixi run python il/train.py method=dp_rgb_color demo_dir=hard
```
(Generating data with the scripted policy is allowed; submitting a scripted policy is not.)
