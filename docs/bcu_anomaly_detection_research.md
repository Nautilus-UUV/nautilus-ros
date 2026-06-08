# Deep-learning fault detection + ordinal severity staging for the BCU pump

Research synthesis for the `bcu_fault_dataset` (`sim_data/bcu_fault_dataset/`). Goal: deep-learning
(plus strong classical-baseline) approaches for **online** multivariate time-series fault detection
**and** ordinal fault-severity classification on the Nautilus buoyancy-control-unit pump.

Produced by a fan-out web-research pass: 23 sources fetched, 110 candidate claims extracted,
25 adversarially fact-checked (20 confirmed, 5 killed). Every surviving source is off-domain, so the
numbers below justify **method choices, not expected performance** — validate on our own data.

## Problem setting (what the model sees)

- **Inputs — 3 raw channels at ~10 Hz** (throttled rosbag topics):
  - commanded pump RPM — `/bcu/rpm` (the ROS setpoint, *before* fault degradation is applied; exogenous "actor" signal)
  - external/ambient pressure — `/external/pressure` (depth proxy)
  - internal tank/bladder pressure — `/bcu/pressure` (the "response" signal)
- **Output — 6 ordinal severity classes** = pump effectiveness `{100, 80, 60, 40, 20, 0}%`
  (level 0 = healthy … level 5 = dead). This is ordinal classification, not flat multiclass.
- **Data scale:** 125 runs, ~106 min each, ~6 400 samples/run, ~222 dive-hours. Class-imbalanced —
  higher-severity levels appear in fewer runs/hours (level 5 reached in only 65/125 runs).
- **Fault dynamics:** within a run the hidden severity only ever **ratchets up** (irreversible,
  step transitions governed by a per-run MTTF). **But** we want the model to stay responsive to
  *instantaneous* physical evidence and be *able* to predict a less-faulty state after a more-faulty
  one — i.e. **no sticky/latching predictor**, and **no temporal shortcut** where a sequence model
  cheats by learning "time-since-start in the run" instead of the true fault signature.
- The physical fault is a **loss of pump effectiveness**, so its signature is a **residual /
  discrepancy between commanded RPM and the observed pressure/flow response** (commanded actuation
  produces a smaller-than-expected tank-pressure change). Detection is fundamentally a model-residual
  problem.

## Bottom line

Build the model around the **commanded-vs-response residual**, not a raw-window deep net. From only
3 channels that one move buys: high signal, a natural ordinal severity axis, and — critically —
**automatic immunity to the monotone-label time-shortcut**, because a residual is a memoryless
physical quantity that rises and falls with the *current* plant state, not with elapsed time.

**Recommended stack, ranked:**

| # | Stage | What | Why |
|---|-------|------|-----|
| 0 | Physics residual (sanity baseline) | Predict expected tank-pressure rate from commanded RPM via `physics.py`/`robot_specs.py` pump curve; residual = observed − expected | Tells us if DL is even needed; uses signal we already have |
| 1 | **Classical baseline to beat** | Sliding-window features over [cmd RPM, ext pressure, tank pressure, **residual**] → XGBoost / Random Forest ordinal | Literature repeatedly shows this is competitive with deep models |
| 2 | **Primary deep detector** | Small **prediction-error** model: commanded RPM (exogenous) → predicted tank pressure (response); score normalized residual `r=(ŷ−y)/σ` | Detects partial effectiveness loss *immediately*, even inside the normal range |
| 3 | **Ordinal severity head** | Feed residual features into **CORN** or **CDW-CE** loss → 6 levels | Rank-consistent / distance-aware; plain cross-entropy ignores ordering |
| 4 | Escalate only if needed | LSTM / TCN over windows | Justified only if the simple model misses *gradual* degradation |

## 1. Start with the simplest deep model — don't reach for the LSTM yet  `[high confidence]`

On a real **hydraulic-pump** dataset, a plain dense/snapshot autoencoder essentially **tied** an
LSTM-autoencoder for detection: dense-AE+MSE F1 **0.964**, dense-AE+Mahalanobis **0.966**,
LSTM-AE+MSE **0.965** — within 0.002 of each other, all trained on healthy data only, no fault
labels. The authors conclude the snapshot AE "strikes the best balance between reliability and
deployment cost" and the LSTM gave "no tangible gain." Standard recipe: **anomaly score =
reconstruction error; threshold = an upper-tail quantile of the reconstruction-error distribution on
nominal data.**

- Sanchez et al., *LSTM vs Feed-Forward Autoencoders for Unsupervised Fault Detection in Hydraulic Pumps* — https://arxiv.org/pdf/2601.11163
- EHA LSTM-autoencoder recipe — https://arxiv.org/abs/2606.05274

> **Caveat:** that null result has a ceiling effect (~0.997 recall — task was easy) and the dataset
> has 52 channels at minute-level sampling, very unlike our 3-channel / 10 Hz / *gradual* regime. The
> same paper notes prior work where **LSTMs do win when faults develop gradually** — which is our
> case. Rule: *start simple, escalate if the snapshot AE underperforms on gradual degradation* — not
> "LSTMs never help."

## 2. The high-signal move: prediction-error residual (commanded → response)  `[high confidence]`

The single most important finding for a 3-channel setup. Eiteneuer & Niggemann formalize a model that
**partitions signals into "actor" (exogenous command, `x`) vs "sensor" (response, `y`)**, models only
`y` as `p(yₜ | sₜ₋₁, xₜ)`, and flags anomalies on the prediction residual — **structurally identical
to commanded-RPM → predicted-tank-pressure.** Key properties:

- Learns a per-channel noise variance σ; uses **normalized residuals** `rₜ=(ŷ−y)/σ ~ N(0,1)` with a
  σ-threshold (3/4/8 σ) calibrated to a target false-positive rate.
- **Trains with zero fault labels** (healthy data only).
- "**larger residual = more advanced fault**" — a built-in severity axis.
- Their water-tank case detects a **25 % output-flux reduction (a partial effectiveness loss — our
  exact fault type) immediately, while the signal is still inside the normal operating range** —
  something stateless threshold detectors cannot do.

— Eiteneuer & Niggemann, https://arxiv.org/pdf/2010.15680. Corroborated by HIPER-CHAD
(Sensors 2026, 26(1):171, https://doi.org/10.3390/s26010171) and an LSTM-AE multi-step-prediction
paper, both of which train the second-stage model on **residual vectors rather than raw data**.

> **Caveat:** all sources off-domain (water tank, environmental sensors); the authors note an
> anomalous input can occasionally still yield a contextually-plausible output, and Gaussian-NLL-learned
> σ can be overconfident (Seitzer et al., ICLR 2022). Calibrate σ empirically on our nominal runs.

**Physics bonus:** we already have the pump curve in `physics.py` (`q_to_rpm`) and `robot_specs.py`.
A *physics-based* residual (expected vs observed tank-pressure rate) is a near-free, interpretable
baseline — and adding **mechanism-constraint rules** to filter known-benign transients lifted F1 from
0.868 → 0.920 on oilfield pumps (Wang et al., *Sci Rep* 15:2020, 2025,
https://www.nature.com/articles/s41598-025-85436-x)  `[medium confidence, single source]`.

## 3. The 6-class ordinal head: use a rank-consistent or distance-weighted loss  `[high confidence]`

Treat severity as **ordinal regression**, not flat 6-way classification — confusing healthy (0) with
dead (5) must cost more than an adjacent slip.

- **CORN** (Shi, Cao, Raschka 2021/2023) is preferred over **CORAL**. Both guarantee rank consistency
  across the K−1 binary "is severity > level k?" outputs, but CORAL enforces it by **sharing weights**
  across the binary heads, which "may restrict the expressiveness and capacity of the network." CORN
  drops that constraint and instead enforces consistency through conditional probabilities
  `P(y>rₖ | y>rₖ₋₁)` + the chain rule. Official PyTorch impl: `coral-pytorch`.
  — https://arxiv.org/abs/2111.08851 , https://github.com/Raschka-research-group/coral-pytorch
- **CDW-CE** (Class Distance Weighted Cross-Entropy, MICCAI 2022) is a simpler drop-in:
  `−Σ |i−c|^α · log(1−ŷᵢ)` — penalizes distant misclassifications more. Plain categorical
  cross-entropy "is not optimal for ordinal regression" because it penalizes all errors equally.
  — https://arxiv.org/pdf/2202.05167
- **Encoding/loss choice is not neutral** and is **architecture- and metric-dependent** (one-hot best
  for accuracy; Gaussian/progress-bar best for minimizing severity deviation). Benchmark CORN vs
  CDW-CE vs regression-then-threshold vs plain-CE rather than assuming a winner.
  — Wienholt et al., https://arxiv.org/abs/2402.05685

> **Refuted (killed 0-3):** the specific claim that "CORN beats CORAL on accuracy (age-estimation
> MAE)" did **not** survive verification. CORN's verified benefit is *structural* (rank consistency
> without the expressiveness cap), **not** guaranteed lower error.

**Evaluation:** report **Quadratic-Weighted Cohen's κ + MAE-in-levels**, not flat accuracy — and
choose the metric deliberately, since assessing ordinal classifiers is "challenging under imbalanced
data" (Yilmaz & Demirhan, *Appl. Soft Comput.* 134, 2023,
https://www.sciencedirect.com/science/article/pii/S1568494623000388).

## 4. Defeating the monotone-label time-shortcut  `[medium confidence — reasoned synthesis]`

Directly answers the non-sticky requirement. **Scoring an instantaneous residual is itself the
defense:** the residual depends on the *current* physical discrepancy, has no dependence on run
elapsed time, and falls again if the plant transiently behaves less-faulty — so the model *cannot*
learn "time-since-start → severity," and is inherently non-latching. Reinforce with guardrails:

- **Run-level train/val/test splits** — never let windows from one run land in both train and test.
- **Drop any absolute-time / sample-index feature.**
- **Windowed, memoryless classification** — classify each window from its own evidence; avoid a hidden
  state that only ratchets up.
- If using an LSTM, keep windows short and consider **time-jittering** augmentation.

> **Honest gap:** no surviving source *directly* benchmarks the elapsed-time shortcut under monotone
> labels or these specific mitigations — this finding is logical synthesis from the residual-detector
> evidence (https://arxiv.org/pdf/2010.15680), not a dedicated cited experiment. The general
> phenomenon is well-established (*Shortcut learning in DNNs*, Geirhos et al.,
> https://www.researchgate.net/publication/346813818_Shortcut_learning_in_deep_neural_networks).
> Validate on our own data.

## 5. Imbalance + classical baseline

- **Imbalance** (level 5 in only 65/125 runs): class weighting / focal loss / resampling at the
  window level; QWK is more robust than accuracy here.
- **The baseline to beat:** hand-crafted time/frequency features (+ the residual) →
  **XGBoost / Random Forest**. The literature repeatedly shows gradient-boosted trees on good residual
  features are competitive with deep models on few-channel pump data — if a tree on the physics
  residual matches the AE, that's the answer. The relative deep ranking observed
  (LSTM-AE > VAE > Isolation Forest) is only for *binary* detection and rests on synthetic labels, so
  treat it as weak  `[medium confidence]` (HIPER-CHAD, https://doi.org/10.3390/s26010171).

## What this report is and isn't

**Every surviving source is off-domain** — hydraulic/oilfield pumps, water tanks, environmental
sensors, chest radiographs — none an underwater glider, most with 5–52 channels or minute-level
sampling. **All numbers (F1, κ, MAE) justify *method choices*, not expected performance on our data.**
Two of the strongest "start simple" results are very recent non-peer-reviewed preprints. Treat this as
a **methodology recommendation** to validate on our own 125-run rosbag dataset.

### Open questions (candidates for a follow-up research round or our own experiments)

1. Direct evidence that sequence models exploit elapsed-time under monotone labels — and quantified
   benefit of each mitigation (time-jittering, run-level splits, removing time features, windowed
   memoryless classification).
2. How CORN vs CDW-CE vs regression-then-threshold actually compare on **multivariate time-series**
   severity staging (not images).
3. Which residual generator is cleanest for *our* pump: LSTM forecast of pressure-from-RPM vs a
   **Kalman/observer** vs an explicit physics model of pump-bladder dynamics.
4. Strongest classical baseline specifically here: residual+XGBoost vs HMM degradation-staging vs
   matrix-profile vs PCA/Mahalanobis residual monitoring.

## Verified sources

| Source | Angle |
|--------|-------|
| Sanchez et al., *LSTM vs Feed-Forward AEs for Hydraulic Pumps* — https://arxiv.org/pdf/2601.11163 | simple deep models, few channels |
| EHA LSTM-AE recipe — https://arxiv.org/abs/2606.05274 | simple deep models |
| LSTM-AE multi-step sensor fault prediction — https://www.researchgate.net/publication/383730145 | simple deep models |
| HIPER-CHAD (Sensors 2026 26(1):171) — https://doi.org/10.3390/s26010171 | residual detection / baselines |
| Wang et al., oilfield-pump LSTMA-AE + mechanism rules (*Sci Rep* 2025) — https://www.nature.com/articles/s41598-025-85436-x | physics-informed residual |
| Eiteneuer & Niggemann, actor/sensor prediction-error LSTM — https://arxiv.org/pdf/2010.15680 | residual / model-based detection |
| CORN (Shi, Cao, Raschka) — https://arxiv.org/abs/2111.08851 | deep ordinal classification |
| coral-pytorch (CORAL/CORN impl) — https://github.com/Raschka-research-group/coral-pytorch | deep ordinal classification |
| Wienholt et al., ordinal encodings — https://arxiv.org/abs/2402.05685 | deep ordinal classification |
| CDW-CE (Polat et al., MICCAI 2022) — https://arxiv.org/pdf/2202.05167 | deep ordinal classification |
| Yilmaz & Demirhan, ordinal-classifier evaluation under imbalance — https://www.sciencedirect.com/science/article/pii/S1568494623000388 | evaluation metrics |
| Geirhos et al., *Shortcut Learning in DNNs* — https://www.researchgate.net/publication/346813818 | leakage / shortcut learning |

### Claims that were fact-checked and killed (do not rely on)

- "LSTM-AE achieved ~99 % accuracy / F1 0.931–0.998 on EHA signals" — refuted 1-2 (arXiv:2606.05274).
- "LSTM-AE residual+classifier hit ~93 %/97 % fault-detection accuracy on two sensor datasets" —
  refuted 1-2 (ResearchGate 383730145).
- "Training the reconstruction model on prediction residuals (vs raw) is justified by noise-separation"
  — refuted 1-2 (Sensors 26(1):171). *(The residual-detector recommendation still stands on
  arXiv:2010.15680; only this particular framing was killed.)*
- "Attention LSTMA-AE outperforms plain LSTM-AE, F1 0.833 → 0.868" — refuted 1-2 (Sci Rep 2025).
- "CORN outperforms CORAL on age-estimation MAE" — refuted 0-3 (arXiv:2111.08851).
