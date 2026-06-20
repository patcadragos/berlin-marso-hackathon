# 3-minute video script (≤3:00, upload to YouTube, attach to Writeup)

Record a screen capture. Show the agent sorting (use the mp4s eval.py drops in each run's
`videos/`). Keep it tight — judges watch many of these.

**0:00–0:20 — Hook + task.**
"Marso's challenge: a Franka Panda sorts parcels into colour-matched bins, from a single camera —
no privileged coordinates. Here's the scripted demo of a solved episode." → play easy demo gif/mp4.

**0:20–0:55 — The diagnosis (the differentiator).**
"The starter policy scores near zero. Why? Its encoder, ResNet18 + SpatialSoftmax, outputs only
keypoint *coordinates* — where things are — and throws away *colour*. But this is a colour-routing
task. On easy and medium the bins are fixed, so it can fake it by memorising a side. On hard the
bins swap — and it's blind." → show a hard clip where bins are on opposite sides.

**0:55–1:40 — The fix.**
"We built a colour-aware encoder: it keeps the keypoint locations AND reads the colour at each
keypoint, plus a global colour descriptor. We add random-shift augmentation to generalise to the
held-out layouts, and train one checkpoint on the hard distribution." → show a code glimpse of
`ResNet18ColorKeypoints` and the `dp_rgb_color` config.

**1:40–2:30 — Results.**
"Sort accuracy: easy [..]%, medium [..]%, hard [..]% → final score [..]." → play the best hard
rollout showing correct colour routing even with bins swapped. Show the ablation table: the gain
is biggest on hard — proof the bottleneck was perception, not control.

**2:30–3:00 — Close.**
"One checkpoint, reproducible from a fresh clone with `pixi install` and one train command. Repo
linked below. Thanks to Marso Robotics." → show the repo URL + final score on screen.
