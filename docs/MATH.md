# AHAIF mathematics and current flow

This page is written for **GitHub and ordinary browsers**.

- Pretty math uses GitHub display blocks: a line with only `$$`, then TeX, then `$$`.
- Under every formula there is an **ASCII line** that still reads if math fails.
- Tables never put TeX inside cells (underscores break italic). Use the names in the symbol table.

```
x[n]  -- twin ocean ---->  surprise S = e^T Pi e
      -- if S large ---->  Mel + env-norm  phi_1
      -- Dirichlet ----->  C_wake, T(e)
      -- policy pi ----->  level L0 .. L4
      -- L3 ------------>  Dung WHY / WHY-NOT
      -- report -------->  Dijkstra on interferometric atlas to SINK
```

Code is the source of truth if this file drifts: `src/uaere/math/` and the module listed under each section.

---

## 0. Symbol names (use these in prose)

| Name in prose | Meaning |
|---------------|---------|
| `x` | 1-second hydrophone window (16 kHz, 16000 samples) |
| `fs` | sample rate = 16000 Hz |
| `s` | environment state: temperature, salinity, depth, sea-state, SNR, turbulence, sound speed |
| `h` | noisy sensor-health estimate |
| `S` | predictive-coding surprise (the wake, not energy) |
| `Pi` | precision, diagonal, high where the ocean should be quiet |
| `alpha` | Dirichlet strengths |
| `p_hat` | Dirichlet mean (class probabilities) |
| `C_wake` | 1 minus reject-class mass (detection score). Fish heads use 1 minus P(other). |
| `T(e)` | event-level gate score. The explanation gate reads it. It is not a calibrated probability that the label is correct. |
| `pi` | policy: tau1, tau2, tau3, collab band, device id |
| `Ghat` | interferometric Green peak between two nodes |

---

## 1. Twin ocean

Mackenzie 1981 nine-term sound speed, metres per second. Code:
`twin/environment.py` (`mackenzie_ssp`).

$$
c(T,S,z)=1448.96+4.591T-5.304\times 10^{-2}T^{2}+2.374\times 10^{-4}T^{3}+1.340(S-35)+1.630\times 10^{-2}z+1.675\times 10^{-7}z^{2}-1.025\times 10^{-2}T(S-35)-7.139\times 10^{-13}Tz^{3}
$$

```
c(T, S, z) = 1448.96 + 4.591 T - 0.05304 T^2 + 0.0002374 T^3
             + 1.340 (S - 35) + 0.01630 z + 1.675e-7 z^2
             - 0.01025 T (S - 35) - 7.139e-13 T z^3
```

Thorp absorption in dB/km, with `f` in kHz:

$$
\alpha(f)=0.11\frac{f^{2}}{1+f^{2}}+44\frac{f^{2}}{4100+f^{2}}+2.75\times 10^{-4}f^{2}+0.003
$$

```
alpha(f) = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2) + 0.000275 f^2 + 0.003
```

Received amplitude: spherical spreading + Thorp + one surface image with a
phase flip of pi. Code: `twin/propagate.py`.

Idle battery each 1 s tick: `idle_mw * 0.001` joules.

---

## 2. Predictive-coding wake (neuroscience)

The generative model is the Knudsen/Wenz ambient spectrum for the current
sea-state, with extra high-frequency lift from turbulence. Code applies
that lift to **both** the predicted spectrum and the precision.

Predicted ambient power at bin `k` (`ss` = sea-state, `turb` in `[0, 1]`):

$$
\hat{P}_{k}(s)=\mathrm{Knudsen}(f_{k},\mathrm{ss})\cdot\bigl(1+8\,\mathrm{turb}\,(f_{k}/f_{\max})^{2}\bigr)
$$

```
P_hat_k(s) = Knudsen(f_k, sea_state) * (1 + 8 * turb * (f_k / f_max)^2)
```

Prediction error and precision at frequency bin `k`:

$$
e_{k}=\log P_{k}(x)-\log \hat{P}_{k}(s)
$$

$$
\Pi_{k}=\frac{1}{\hat{P}_{k}(s)+\varepsilon}
$$

```
e_k   = log P_k(x) - log P_hat_k(s)
Pi_k  = 1 / (P_hat_k(s) + eps)
```

Surprise (this **is** the wake):

$$
S(x,s)=\frac{1}{K}\sum_{k}\Pi_{k}e_{k}^{2}+\log(\mathrm{peakiness}+\varepsilon)
$$

```
S(x, s) = (1/K) sum_k  Pi_k * e_k^2  +  log(peakiness + eps)
```

`peakiness` is max/mean power in 10–400 Hz (blade-rate band).

Admit L1 with a logistic of surprise, **never** of energy:

$$
a=\sigma\left(\frac{S-c}{\gamma}\right)
$$

```
a = sigmoid( (S - c) / gamma )
```

Code: `representation/predictive.py`.

Storms match `P_hat` (high energy, **low** `S`). A distant tug is a
harmonic residual (low energy, **high** `S`).

---

## 3. Dirichlet event trust

Logistic class probabilities lifted to a Dirichlet:

$$
\alpha=1+\tau\,\hat{p}_{\mathrm{logistic}}
$$

$$
\hat{p}_{i}=\frac{\alpha_{i}}{\sum_{j}\alpha_{j}}
$$

```
alpha_i = 1 + tau * p_logistic_i
p_hat_i = alpha_i / sum_j alpha_j
```

Uncertainties and wake confidence:

$$
u_{\mathrm{epistemic}}=\frac{K}{\sum_{i}\alpha_{i}}
$$

$$
u_{\mathrm{aleatoric}}=1-\max_{k}\hat{p}_{k}
$$

$$
C_{\mathrm{wake}}=1-\hat{p}_{\mathrm{reject}}
$$

```
u_epistemic = K / sum_i alpha_i
u_aleatoric = 1 - max_k p_hat_k
C_wake      = 1 - p_hat_reject
```

Event trust (the **gate** reads this, ROC uses `C_wake`):

$$
T(e)=\sigma\bigl(w_{c}\,C_{\mathrm{wake}}+w_{h}\,\kappa(h)+w_{s}\,\rho(s,x)-w_{u}(u_{a}+u_{e})+b\bigr)
$$

```
T(e) = sigmoid( w_c * C_wake + w_h * kappa(h) + w_s * rho(s,x)
                - w_u * (u_a + u_e) + b )
```

- `kappa(h)` down-weights clipping, a fault, fewer than 12 ADC bits, and drift above 5 ppm. A healthy sensor scores 1.
- `rho(s,x)` is a sigmoid of low-minus-high Mel contrast, reduced by sea state and turbulence.
- The knowledge graph is not an input of `T(e)`.
- The weights are hand-set in `ModelConfig` and do not sum to 1: wake 2.2, health 0.8, environment 1.1, uncertainty 1.4, bias −0.6. They are not learned.
- The concentration in `alpha = 1 + concentration * p_logistic` is the fixed value 20. It is not learned. Then `sum alpha = K + 20`, so epistemic uncertainty `u_e = K / (K + 20)` is the same on every window. For the two-class fish head that value is 2/22. Only the aleatoric term `u_a = 1 - max p_hat` varies.
- `T(e)` is not the convex sum `w_c C + w_e E + w_k K + w_u (1 - U)`. That sum is not implemented. Do not describe the sigmoid as a probability of a correct label.

On the Pamlico fish files the environment vector is the imputed constant (12 °C, 34 PSU, 40 m, sea state 2, turbulence 0.1) and the sensor is healthy, so `kappa` is 1. The Mel passed into `rho` has already been instance-normalised per bin and then shifted by a FiLM bias that depends only on that environment vector (`representation/env_norm.py`). After that shift the low-high contrast is the bias, so `rho` does not vary across windows. With `u_e` also constant, `T(e)` is a strictly increasing function of `C_wake` at these weights. A threshold on `T(e)` therefore ranks windows the same way as a threshold on `C_wake`. The three-seed review in `artifacts/paper/review_upgrade/` confirms that retuning the threshold makes those two gates, and the ablations that zero a constant term, select the same windows.

Code: `math/dirichlet.py`, `trust/trust_score.py`, `trust/evidential_head.py`.

---

## 4. Policy

$$
\pi=(\tau_{1},\tau_{2},\tau_{3},\tau_{\mathrm{collab,lo}},\tau_{\mathrm{collab,hi}},d)
$$

$$
\min_{\pi} F(\pi)=\bigl(f_{\mathrm{miss}},\;f_{\mathrm{fa}},\;E,\;L_{p99},\;\mathrm{ECE},\;-X\bigr)
$$

```
pi = (tau1, tau2, tau3, tau_collab_lo, tau_collab_hi, device)
minimise F(pi) = (miss, false-alarm, energy, p99 latency, ECE, -explanation_coverage)
```

Expected compute energy:

$$
E_{\mathrm{compute}}=c_{0}+p_{1}c_{1}+p_{2}c_{2}+p_{3}c_{3}
$$

```
E_compute = c0 + p1*c1 + p2*c2 + p3*c3
```

**Proposition 1.** If `p3 < 1` and every `c_k > 0`, then
`E(pi) < E(always-on L3)`. Code: `math/bounds.py`.

Runtime staircase on `T(e)` (`policy/runtime_gate.py`). The collab band
is tested **only after** `T >= tau3`; when it hits, L4 **replaces** L3
(the node does not run both).

| Condition | Level |
|-----------|-------|
| `T < tau1` | L0 sleep |
| `tau1 <= T < tau2` | L1 trust only |
| `tau2 <= T < tau3` | L2 TinyCNN |
| `T >= tau3` and T outside collab band | L3 KG + WHY/WHY-NOT |
| `T >= tau3` and `tau_collab_lo <= T <= tau_collab_hi` | L4 neighbour-wake |

That staircase is what the synthetic suite searches, and what `RuntimeGate` still implements. The fish paper comparison does not use it as the matched table. Searched policies do not enter L4. Neighbour confirmation is in-process, not an acoustic modem.

### Fish paper gate (one threshold)

The comparison in `paper/manuscript.md` is produced by `uaere eval --suite review`. For a score `s` the action on window `i` is

```
a_i = Level 3    if s_i >= tau
a_i = Level 1    otherwise
```

Level 3 here means the cumulative datasheet cost through Level 3. The logistic label is emitted on both branches. The gate changes whether the explanation is paid for. It does not change macro-F1. Abstention is a different function: refuse the label when `s_i` is below its own validation threshold, and report the abstention rate, the error among accepted labels, and the full-set macro-F1 with refusals counted as errors. Do not describe that mode as the classifier that produced the fish macro-F1.

`tau` is chosen on validation only. Candidates are the 5th, 10th, …, 95th percentiles of the validation scores. The three minimised objectives are mean modelled energy, the fraction of negative windows sent to Level 3, and one minus the fraction of positive windows sent to Level 3. The chosen point is the Chebyshev knee after min-max scaling. Ties break toward lower energy, then the earlier percentile. The same rule is applied to wake confidence, inverted uncertainty, a confidence–uncertainty logit, and full `T(e)`. The frozen threshold is scored once on test. Test labels do not enter the grid.

The optional three-threshold staircase may still be searched with NSGA-II on validation (population 12, six generations) and scored once on test. It can stop at Level 0, so its joules are not a row of the one-threshold table. The older fish suite searched that staircase on the test split. The knee stored in `artifacts/paper/fish_suite.json` is that test-tuned point. Do not quote it as the review result.

### Energy identity used by the review

Per window, with no double count of the front end:

```
E_i = E_a,i + E_g,i + E_p,i
E_bar = (1/N) * sum_i E_i
saving_percent = 100 * (E_bar_baseline - E_bar_proposed) / E_bar_baseline
```

`E_a` is `compute_energy_j` at Level 1 (surprise front end plus the logistic). `E_g` is 0: trust arithmetic has no separate MAC line and is folded into `E_a`. `E_p` is 0 when the gate stops at Level 1. When the gate pays for an explanation, `E_p = E(Level 3) - E(Level 1)`. That difference includes the Level-2 MAC budget, because `compute_energy_j` sums every level up to the selected one. Level 2 does not supply the fish label.

On the `stm32l476` profile at scale 1 the cumulative costs are Level 0 = 0.000208 J, Level 1 = 0.005408 J, Level 2 = 0.006448 J, Level 3 = 0.0064532 J. The graph-only increment, Level 3 minus Level 2, is 5.2e-6 J. About 0.995 of `E_p` is the unused Level-2 budget. The profile is 80 MHz, 8 mW active, 2.5 nJ per INT8 MAC. These joules are datasheet estimates. They are not a measured current and they are not an underwater deployment. An event arrival rate is not in the model. The same rate would cancel in the saving percentage. The synthetic suite’s factor-of-about-32 saving is a different policy on simulated windows. It is not this fish saving.

Code: `eval/gate_compare.py`, `math/energy.py`, `policy/runtime_gate.py`.

---

## 5. Dung argumentation (explainable KG)

Candidates = `caused_by` neighbours of the predicted event, plus
environmental alternatives (`rain`, `wind_waves`).

An attack `(A, B, unless_sea_state_ge = m)` **fires** when
`sea_state < m`. Grounded extension: drop every attacked candidate.

$$
\mathrm{accepted}=\{c:\text{no firing attack lands on }c\}
$$

```
accepted = { c  |  no fired attack points at c }
```

Example sentence:

```
WHY tug: cues match cause.vessel.tug in sea-state 2.
WHY-NOT: cause.env.rain attacked by event.tug_pass (sea-state 2 < 3)
```

Code: `causal/argumentation.py`, `kg/ontology.yaml` (`rel: attacks`).

Cue-mask counterfactual (not Pearl identification):

$$
\mathrm{CF}(c,\hat{y})=1\text{ if cue-band energy of }c\text{ drops after a 20:1 mask}
$$

```
CF(c, y_hat) = 1  if  energy in c's cue band drops after a 20:1 mask
```

---

## 6. Interferometric death-zone atlas (seismology)

On **non-event** frames, nodes `i` and `j`:

$$
\hat{G}_{ij}(\tau)=\mathcal{F}^{-1}\bigl(X_{i}(\omega)\,X_{j}^{*}(\omega)\bigr)
$$

$$
g_{ij}=\frac{\max_{\tau}|\hat{G}_{ij}(\tau)|}{\lVert x_{i}\rVert\,\lVert x_{j}\rVert}
$$

```
Ghat_ij(tau) = IFFT( X_i(omega) * conj(X_j(omega)) )
g_ij         = max|Ghat| / (||x_i|| ||x_j||)
```

EMA update: `g <- (1 - alpha) * g + alpha * g_new`.

Link `i--j` is **dead** if `g < g_min` (default 0.04). No Bellhop.

Code: `twin/interferometry.py`.

---

## 7. Report to the sink

After L3, Dijkstra on live nodes plus the surface sink.

Edge weight `w_ij`:

- `infinity` if the interferometric link is dead, or battery of `j` is 0
- otherwise `E_tx(range_ij) * (1 + 1 / Q_tilde_j)`

$$
w_{ij}=E_{\mathrm{tx}}(r_{ij})\left(1+\frac{1}{\tilde{Q}_{j}}\right)\quad\text{(or }\infty\text{ if dead)}
$$

```
w_ij = inf                         if interferometric-dead or Q_j = 0
w_ij = E_tx(r_ij) * (1 + 1/Q_j)    otherwise
```

Payload: class, `T(e)`, `C_wake`, WHY/WHY-NOT, HMAC
(`security/hal.py`). No finite path ⇒ `undeliverable`.

This is **not** Euclidean shortest path.

Code: `twin/routing.py`.

---

## 8. Complexity (measured)

| Stage | Big-O | Notes |
|-------|-------|--------|
| Surprise rFFT | O(N log N) | N = 16000 |
| Mel | O(frames × N_fft × bands) | 32 bands |
| TinyCNN | MAC tracer | 546 B INT8 |
| KG + Dung | O(V + E) | about 20 nodes |
| Dijkstra | O(V^2) here | V about 8 in the demo |

---

## 9. Algorithm (one second)

1. Drain `idle_mw` for 1 s.
2. Compute surprise `S`. If admit `a(S)` is tiny, stay at L0.
3. Else Mel, Dirichlet, `C_wake`, `T(e)`.
4. Gate on `T(e)`: L0 / L1 / L2 below `tau3`.
5. If `T >= tau3` and `T` is in the collab band: L4 neighbour-wake (replaces L3).
6. If `T >= tau3` and `T` is outside that band: L2 class, L3 Dung proof, then Dijkstra on the interferometric atlas to the sink.
7. On ambient frames: update `Ghat` for every node pair.
