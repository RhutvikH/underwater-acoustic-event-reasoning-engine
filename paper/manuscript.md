# Uncertainty-Aware Event-Level Trust Gating for Energy-Efficient Underwater Acoustic Event Classification

**Draft for revision.** The Springer Nature source of this revision is `paper/uaere_sn.tex`, with `paper/uaere.bib` and `paper/fig_architecture.pdf`. Numbers in the toadfish tables are copied from `artifacts/paper/review_upgrade/review_upgrade.json` (seeds 0, 1, and 2). The Watkins result is seed 0 of `artifacts/paper/species_suite.json`. The twin is seed 0 of `artifacts/paper/suite.json` and is an appendix, not a headline. Do not post this draft to arXiv or a public repository until a patent provisional has been filed.

Rhutvik Hegde<sup>1</sup>, Kalyani Manoj<sup>1</sup>, and Shashank Mouli Satapathy<sup>1,*</sup>

<sup>1</sup>School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu 632014, India  
<sup>*</sup>Corresponding author. ORCID: 0000-0002-1665-8101.

The corresponding author’s publicly listed ongoing project is “Marine Dataset Generation and Design of Expert System for Multi-aspects Analysis of Underwater Species: Indian Scenario,” Ministry of Earth Sciences, Government of India. That project is not the source of the measurements reported here.

---

## Abstract

This study examines whether uncertainty and contextual terms improve event-level explanation gating over a confidence threshold under an explicit energy-cost model. The label is a class-balanced logistic on a pre-segmented window and is emitted whether or not the explanation runs. A hand-set sigmoid, called event trust, combines wake confidence with sensor health, an environmental term, and a Dirichlet uncertainty. Trust is a gate score, not a calibrated probability that the label is correct. A validation percentile of that score chooses between stopping at Level 1 and paying for a Level-3 explanation.

On three recording-locked seeds of the public Pamlico oyster-toadfish candidates, macro-F1 is 0.844 ± 0.004 and does not depend on the gate. Health, environment, and epistemic uncertainty are constant on these files, so the trust score ranks windows in the same order as wake confidence and selects the same test windows. The uncertainty-only gate does not: it also explains confident negatives. Relative to always running Level 3, a datasheet model estimates a saving of 10.4% ± 0.7% only when the selected path is charged for a Level-2 computation that does not produce the label. The clips are pre-segmented candidates from one estuary. A 16-species Watkins tape hold-out reaches macro-F1 0.281 and misses a pre-registered bar of 0.70. That result is not generalisation. The energy figures are model estimates, not measured power and not an underwater deployment.

**Keywords.** Underwater acoustics; explanation gating; selective inference; confidence threshold; energy-cost model; oyster toadfish.

---

## 1. Introduction

### 1.1 Background and motivation

Passive acoustic monitoring spends its energy in the instrument that digitizes the sea, and only afterwards in any classifier. A fixed installation is either a cabled hydrophone or an autonomous recorder that writes to its own memory and is recovered later [1], [2]. The sensor is passive: it does not project sound at the animal [3]. Continuous broadband recording keeps short calls and fills disks. A programmed duty cycle stretches a battery and drops every call that falls in an off interval [11], [12]. A classifier that runs on shore, months later, never sees the joules that choice already spent.

The recorders that produced the literature’s long archives set the scale. A single-channel HARP at 200 kHz draws about 0.5 W while it writes [12]. Earlier HARP descriptions put the hydrophone near 50 mW and the logger near 250 mW, with a large extra cost while disks spin [11]. Compact recorders and click loggers occupy a different point on the same trade: less bandwidth, a smaller housing, and a detector that is part of the instrument rather than a later computer [13]–[15]. Turning a recording into a habitat metric is a separate measurement problem [16]. Ambient spectra are the background of that recording [18], [19], [17]. None of those watts is the millijoule cost of an explanation on a sample that has already been stored. Putting the two numbers in one sentence, as if a software gate had replaced the recorder, would describe the wrong device.

The call used for the main experiment is the oyster toadfish boatwhistle. The fundamental sits near 80–220 Hz and the call lasts a few hundred milliseconds [20]–[23]. The public candidate set that accompanies the ToadFishFinder catalog was recorded in Pamlico Sound, template-matched, and released as 1.35 s clips at 24 kHz [9], [24]. This paper scores the centre 1.0 s of those clips at 16 kHz. It does not discover events in a continuous stream.

### 1.2 Problem statement

The decision studied here is narrower than “classify the ocean.” For each already cut window, an energy-constrained node has a label from a cheap head and a more expensive explanation. It must decide whether the explanation is justified, using the evidence in that window, while keeping a check on explanations that fire on the wrong class. The quantities available to that decision are the acoustic evidence, the uncertainty of the prediction, a sensor-health term, an environmental term, and a computational cost. The implemented action is binary. If the score is at least τ, run Level 3. Otherwise stop at Level 1. The label is emitted in both cases.

Acceptance of a label, escalation to a heavier model, and abstention are different actions. This system implements the second only as a choice to run or skip the explanation. A separate abstention baseline, which refuses the label, is reported under its own name. It is not the source of the macro-F1 above.

### 1.3 Research gap

Three limits of a conventional confidence threshold matter for this decision.

First, a threshold on the top probability does not say what should happen when the node is unhealthy, when the sea state already explains the band, or when the probability is peaked on the negative class. Second, the same number is often treated as a calibrated probability that the label is correct. A gate score need not be that probability. Third, a threshold chosen on the test set, or a multi-threshold search that sees the test labels, reports an operating point the node did not have when the window arrived.

Published trust models for underwater acoustic networks answer a different question. They score whether a node is faulty, selfish, or compromised, and they reweight that node in routing or fusion [4]–[7], [29]. Ship and animal classifiers answer which class produced a clip that was already cut out [8]–[10]. Dirichlet evidential heads answer how diffuse a class distribution is [30]. A knowledge graph can state a cause [31]–[33]. None of these, by itself, is a per-window choice between a cheap label and a costly explanation under a stated cost model.

### 1.4 Research objectives and contributions

The objective is to run that comparison on the available recordings and to state what it shows, including where the proposed score adds nothing.

The contributions that the experiments support are three.

1. On the two-class toadfish corpus, event trust does not outperform a confidence threshold. After each score’s own validation threshold, the two gates select the same windows. An uncertainty-only gate explains confident negatives and is less selective. The equivalence is the finding. It is not a claim that the extra terms improve the decision.
2. A cost account of that decision. The reported saving against always running Level 3 exists only when the selected path is charged for a Level-2 budget the logistic does not execute. If that budget is omitted, the same windows save 0.062% ± 0.004% (5.410 mJ against 5.413 mJ for always attaching the graph walk of 0.005 mJ). The joules are a datasheet model, not a deployment measurement.
3. A matched protocol: one validation rule for every score, abstention reported separately from explanation gating, and intervals whose unit is the seed (three redraws inside one estuary, not a new site).

The hand-set score is the object of the comparison. It is not, on these files, a better gate. The knowledge graph explains a predicted class. It is not a contribution to detection or to the trust score. The weights were not fit on these recordings. Wake has the largest coefficient, uncertainty is subtracted, and health and environment are smaller positive terms for inputs that vary. Scales of 0.5 and 2 left the validation mask unchanged.

---

## 2. Related work

### 2.1 Underwater acoustic event classification

Ship-radiated-noise benchmarks such as DeepShip score vessel class on clips that have already been cut from a recording [8]. Bioacoustic catalogs score calls the same way. Bohnenstiehl’s ResNet-50 and SVM remain the specialist result on the Pamlico toadfish candidates [9]. This paper uses a class-balanced logistic on log-Mel statistics plus three band numbers, and it does not claim to beat that specialist. The Watkins Marine Mammal Sound Database is a library of isolated, identified cuts gathered over decades and instruments [26]. Random splits of the small best-of subset, as in BEANS [10], and date-and-location or tape hold-outs [25] are different tasks. Section 6.3 reports a tape hold-out and does not compare its macro-F1 with the toadfish macro-F1.

### 2.2 Energy-aware and selective inference

Duty-cycled recorders save energy by not acquiring [11], [12], [27], [28]. A compute gate on a window that has already been digitized saves a different, and much smaller, quantity. Selective classification refuses the label when confidence is low [37]. Early-exit networks skip later layers when an intermediate classifier is already confident [38]. Here the label is always the logistic, and the threshold decides only whether an explanation is attached. The confidence-threshold row in Section 6.1 is that narrower rule. The abstention baseline is the selective-classification action and is reported separately. An always-cheap path and an always-expensive path bound the table: the first never explains, the second always does. Both use the logistic label.

### 2.3 Uncertainty-aware classification and trust estimation

Network trust estimates a node or a link and is used for routing or fusion, not for a per-window explanation [4]–[7]. Sensoy, Kaplan, and Kandemir map evidence to a Dirichlet and read epistemic uncertainty from the total concentration [30]. This paper does not train that loss. It lifts a fitted logistic with a fixed concentration of 20, which makes the epistemic term constant on a two-class head. The hand-written graph can state a cause after the label is chosen [31]–[33]. It is not an input to the gate score.

### 2.4 What the comparison is

Table A places the implemented baselines next to the nearest published lines. Cells describe what those papers and this code actually do.

**Table A. What is and is not in the decision.**

| Approach | Uncertainty in the score | Health or environment in the score | What the threshold does | Unit |
|----------|--------------------------|------------------------------------|-------------------------|------|
| Node or channel trust [4]–[7] | No event-uncertainty model | Channel or neighbour, not this score | Routing or fusion | Node |
| Selective classification [37] | No | No | Refuses the label | Sample |
| Early exit [38] | No | No | Skips later layers of the classifier | Sample |
| Clip classifier [8], [9] | No | No | No threshold | Clip |
| Dirichlet head [30] | Yes, from a trained concentration | No | No threshold | Class distribution |
| Confidence threshold (implemented) | No separate term | No | Level 3 or Level 1; label kept | Window |
| Uncertainty threshold (implemented) | Yes, inverted | No | Level 3 or Level 1; label kept | Window |
| Confidence–uncertainty (implemented) | Yes | Weights set to 0 | Level 3 or Level 1; label kept | Window |
| Event trust (implemented) | Yes; epistemic part constant here | Yes, and both constant here. The graph is not an input | Level 3 or Level 1; label kept | Window |

The comparison is one validation-chosen threshold and one cost model, applied to each scalar. Section 6 shows that, on these files, event trust selects the same windows as the confidence threshold, and that the uncertainty threshold does not. The graph is not context inside the trust score.

---

## 3. Problem formulation and system framework

### 3.1 System model and assumptions

A window is 1.0 s at 16 kHz, taken from the centre of a longer clip. Integer PCM is divided by full scale, \(2^{31}\) for 32-bit and 32768 for 16-bit. IEEE-float samples are kept as stored. Training clips are not peak-normalised. The environment state carries temperature, salinity, depth, sea state, turbulence, and a sound speed [35]. The sensor state carries clipping, drift, ADC width, and a fault class. The Pamlico release has no conductivity-temperature-depth cast and no health log. Every fish window uses one imputed state: 12 °C, 34 PSU, 40 m, sea state 2, turbulence 0.1, and a healthy sensor. Sea state 2 is a constant.

Level 0 is an FFT surprise against a Knudsen/Wenz ambient [18], [19]. A logistic map with center 8 and scale 1.5 crosses an admit value of 0.15 at surprise 5.398. There is no test of the form “if energy exceeds a threshold.” A head that stores biological labels records surprise and does not sleep on that cutoff. A ship head, which has no label list, still returns a reject when admit is low. The fish experiments in Section 6 always run the logistic, so Level 0 does not change their labels.

The twin used in the appendix is a different generator: spherical spreading, Thorp absorption at a fixed 1 kHz [36], one surface bounce, and a delay clipped at 0.5 s. Its fish class is a 280 Hz pulse train. It is not a Pamlico recording.

### 3.2 Problem formulation

For window \(i\), the logistic produces a class distribution. From it the node computes wake confidence \(C_i\), aleatoric uncertainty \(U_i^{(a)}\), and epistemic uncertainty \(U_i^{(e)}\). Health consistency \(\kappa_i\) and environmental consistency \(\rho_i\) are the other two scalars defined in Section 4.2. The trust score \(T_i\) is a function of those scalars. The implemented decision is

\[
a_i = g(T_i) =
\begin{cases}
\text{Level 3 explanation} & \text{if } T_i \ge \tau, \\
\text{Level 1 label only} & \text{if } T_i < \tau.
\end{cases}
\]

The label \(\hat{y}_i\) does not depend on \(a_i\). The cost of window \(i\) does. Section 4.5 writes that cost as \(E_i = E_{a,i} + E_{g,i} + E_{p,i}\). The threshold \(\tau\) is chosen on validation only, by the rule in Section 4.4. The same rule is applied to every competing score, so the trust gate does not receive extra degrees of freedom.

### 3.3 Overview of the proposed framework

The path that is implemented, the path that is only a cost, and the path that is only a simulation are different. The Springer figure is `paper/fig_architecture.pdf`. Solid boxes run on the toadfish files. Joules printed beside the branches are the datasheet model.

```text
waveform
  -> Level 0 surprise (implemented; fish labels do not sleep on it)
  -> Level 1 log-Mel, logistic, Dirichlet lift, trust  (implemented)
  -> gate on one score against tau                      (implemented)
        |-- score < tau --> emit the label              (implemented)
        |-- score >= tau --> Level 3 graph explanation  (implemented)
Level 2 CNN budget is charged by the cost model when Level 3 is selected.
It does not supply the reported label.
The digital twin is a separate simulation.
Joules are the STM32L476 datasheet model, not a current measurement.
```

---

## 4. Proposed event-level trust-gating method

### 4.1 Overall architecture

The reported fish label is the argmax of a class-balanced logistic regression on 102 numbers: 99 log-Mel statistics (per-band mean, standard deviation, and maximum of 32 bands, plus three coarse band means) and three waveform numbers inside the band stored on the graph node `cause.taxon.opsanus_tau`, which is 80–500 Hz. Those three numbers are the in-band power fraction, the in-band spectral flatness, and the autocorrelation peak between lags \(f_s/500\) and \(f_s/80\). The bounds are read from the graph. They were not tuned on the test files. Features are standardised with the training mean and variance. The solver is lbfgs with `class_weight="balanced"`. A small convolutional budget of 546 parameters exists in the hierarchy for a MAC count. It does not supply this label.

Validation macro-F1 on the seed-0 recording-locked split was 0.799 without the three band numbers and 0.817 with them. A multilayer perceptron on the same vector scored 0.819 and was not used. The test split did not choose the features or the estimator. The Mel-only test macro-F1 of 0.827 is preserved in `artifacts/paper/fish_suite_melstats.json` and is not the number in Section 6.

### 4.2 Event-level trust estimation

Logistic probabilities \(\tilde{p}\) are lifted by

\[
\alpha_k = 1 + 20\,\tilde{p}_k, \qquad \hat{p}_k = \alpha_k / \sum_j \alpha_j.
\]

For a head with a negative class, wake confidence is \(C = 1 - \hat{p}_{\mathrm{neg}}\). For the toadfish head the negative class is `other`. Aleatoric uncertainty is \(u_a = 1 - \max_k \hat{p}_k\). Epistemic uncertainty is \(u_e = K / \sum_k \alpha_k\). With this fixed concentration, \(\sum_k \alpha_k = K + 20\), so \(u_e = K/(K+20)\) on every window. For two classes, \(u_e = 2/22 \approx 0.0909\).

Health consistency \(\kappa\) is 1 unless the sensor report says clipping, a fault, fewer than 12 ADC bits, or a drift above 5 ppm. Environmental consistency \(\rho\) is a sigmoid of the low-band minus high-band Mel contrast, reduced by sea state and turbulence. After per-bin instance normalisation, that contrast equals the FiLM bias of the environment state. When the state is the imputed constant, \(\rho\) is the same number on every window.

Event trust is

\[
T = \sigma\bigl(2.2\,C + 0.8\,\kappa + 1.1\,\rho - 1.4\,(u_a + u_e) - 0.6\bigr),
\]

where \(\sigma\) is the logistic sigmoid. The four weights and the bias are hand-set. They are not learned. They do not sum to 1. \(T\) is a gate score. It is not a calibrated probability that the label is correct, and it is not the convex combination \(w_C C + w_E E + w_K K + w_U(1-U)\). The knowledge graph is not an input of \(T\).

The coefficients are constants stored with the implementation. They were not estimated from these recordings, and no validation search selected them. The wake coefficient is the largest, so the score moves with confidence when the other inputs are fixed. Uncertainty is subtracted. Health and environment have smaller positive coefficients for the case in which the sensor report or the sea state changes. The bias shifts the sigmoid. On this corpus those roles are not exercised: \(\kappa = 1\), \(\rho = 0.259\), and \(u_e = 2/22\) on every window. Multiplying any one coefficient by 0.5 or by 2 left the validation percentile unchanged. Any strictly increasing function of wake confidence would have selected the same windows after its own threshold was retuned.

On a two-class head, \(u_a = 1 - \max(C, 1-C)\). With \(\kappa\), \(\rho\), and \(u_e\) constant, \(T\) is a strictly increasing function of \(C\) at the weights above. A threshold on \(T\) and a threshold on \(C\) therefore rank windows in the same order. That is why the trust gate cannot outperform confidence on this dataset. Section 6.2 is the empirical counterpart: after each score’s own validation threshold is applied, the test masks match. The fixed-concentration lift does not provide a window-varying epistemic uncertainty in this two-class experiment, and it is not evidence of superior uncertainty estimation.

### 4.3 Uncertainty estimation and calibration

The concentration 20 is a constant in the code. It is not a trained parameter. Changing it re-lifts the same logistic probabilities. It does not refit the logistic. Section 6.4 sweeps the values 5, 10, 20, and 40 on validation. The reported test rows use 20.

Two uncertainty scores are compared with the same validation rule. The Dirichlet score used as a gate is \(1 - (u_a + u_e)\), so a large uncertainty is the low side of the threshold. The alternative is one minus the normalised entropy of the Dirichlet mean, \(1 - H(\hat{p})/\log K\). Both are high when the mean is peaked on either class. A confident negative therefore looks like a confident positive to these two scores. Wake confidence does not make that mistake, because a confident negative has low \(C\).

Calibration of \(C\), of \(T\), of \(1-(u_a+u_e)\), and of the entropy score against the event indicator is reported with the baseline comparison in Section 6.1. A lower value means the scalar is closer to a probability of the event. It does not by itself make a better gate. \(T\) is not that probability.

### 4.4 Cost-sensitive gating policy

For a score \(s\), candidate thresholds are the 5th, 10th, …, 95th percentiles of the validation scores. At each candidate the validation objectives, all minimised, are the mean modelled energy, the fraction of negative windows that receive Level 3, and one minus the fraction of positive windows that receive Level 3. The chosen threshold is the Chebyshev knee of that grid after min-max scaling. Ties break toward lower energy, then the earlier percentile. The frozen threshold is applied once to the test scores. Test labels do not enter the grid.

The same rule defines four scores: wake confidence, the inverted uncertainty, the confidence–uncertainty logit with the health and environment weights set to zero, and full trust. Always-Level-1 and always-Level-3 have no threshold. An optional three-threshold staircase is searched with NSGA-II on validation only (population 12, six generations) and scored once on test. That staircase can stop at Level 0. It is not a row of the one-threshold table, and it is not the knee previously fit on the test split. That earlier knee, and the 3.82 mJ figure attached to it, is retired.

The abstention baseline uses the same percentile grid and a different action: if the score is below its own validation threshold, the label is refused. Its objectives are the abstention rate, the error rate among accepted labels, and one minus the full-set macro-F1 when a refusal matches no gold label. It is reported separately so that a lower error on the accepted subset is not mistaken for a better classifier.

### 4.5 Computational complexity and energy-cost model

The device row is the STM32L476 profile in the datasheet model: 80 MHz, 8 mW active, 2.5 nJ per INT8 MAC. No board current was measured. MAC counts by level are \(8\times 10^4\), \(2\times 10^6\), \(4\times 10^5\), \(2\times 10^3\), and \(2\times 10^3\) for Levels 0 through 4. Energy through a level is the cumulative MAC cost plus the active energy for the time those MACs would take. The resulting level costs are 0.208, 5.408, 6.448, and 6.453 mJ per window for Levels 0 through 3.

For a window that runs the logistic,

\[
E_i = E_{a,i} + E_{g,i} + E_{p,i}.
\]

\(E_{a,i}\) is the cost through Level 1, 5.408 mJ. \(E_{g,i}\) is 0. Trust is arithmetic on quantities Level 1 already computed, and the profile has no separate trust MAC count. \(E_{p,i}\) is the remainder when Level 3 is selected, 1.045 mJ, and 0 otherwise. Of that 1.045 mJ, 1.040 mJ is the Level-2 budget and 0.005 mJ is Level 3. The Level-2 share of the additional path is 0.995. The fish label does not execute that CNN. The cost model charges it because `compute_energy_j` sums every level up to the one selected. The graph walk is the small remainder.

Average energy is the mean of \(E_i\) over the \(N\) test windows. The saving against always-Level-3 is

\[
100 \times (\bar{E}_{\mathrm{L3}} - \bar{E}) / \bar{E}_{\mathrm{L3}}.
\]

An arrival rate is not in the model. The same rate would multiply every policy and cancel in the percentage. No arrival process was fitted.

---

## 5. Experimental setup

### 5.1 Datasets and preprocessing

**Toadfish.** Clips are ToadFishFinder v4, Dryad [24], folders of boatwhistles and other sounds, 1.35 s, 24 kHz, 32-bit integer PCM. The loader takes the centre 1.0 s and resamples to 16 kHz. At most 3,000 files per class are drawn, and a deployment parsed from the file name is kept whole. Both classes of one deployment stay on one side. The split is 60/20/20 by file mass inside `recording_locked_split`. It blocks same-recording leakage. It does not hold out a second estuary. Every station in the release is Pamlico Sound. Seed 0 uses 1,803/1,797, 615/585, and 582/618 files for other and boatwhistle, in 734, 257, and 309 groups. Seeds 1 and 2 redraw the groups. Group overlap across train, validation, and test was 0 on every seed. The events are pre-segmented. This is not end-to-end discovery in a continuous recording.

**Noise.** For seed 0 only, white noise is added to the test waveform before feature extraction, at 20, 10, 5, and 0 dB, with a fixed random stream per level. The logistic and the threshold stay frozen. The bounds 80 and 500 Hz are not retuned. These files are derived evaluation data. They are not new field recordings.

**Watkins.** The full-cuts archive of the Woods Hole Oceanographic Institution [26] is used under its academic terms and is not redistributed. A species enters only if it has at least 40 cuts and 8 tapes and if a 16 kHz front end keeps the call. Sixteen species remained. At most 500 files per species were kept, whole tapes, seed 0. The tape id is the leading five digits of the trailing token. This is a call-library task on isolated cuts. It is not the toadfish task, and the macro-F1 values are not compared.

**Twin.** The appendix uses the seed-0 busy-strait draw, 138/32/43 windows. It is a simulation.

### 5.2 Experimental protocol and parameter settings

Each fish seed fits the logistic on its own training windows (`max_iter` 2000; the fits stopped at iterations 64, 67, and 63). Every threshold in Section 4.4 is chosen on that seed’s validation windows and applied once to its test windows. The concentration used for the test rows is 20. The weight vector is the hand-set vector in Section 4.2. Seeds are 0, 1, and 2. The sample standard deviation uses \(R-1\) in the denominator. The 95% interval is a Student interval on those three seeds. Paired tests are two-sided paired \(t\) tests across seeds. Three seeds are a small sample. The intervals and the \(p\)-values are descriptive.

The threshold is not chosen on the test set. Bands are not retuned after the test score. Species are not dropped. Temperature scaling and Cohen’s kappa were not run.

### 5.3 Baseline methods

The one-threshold table contains always-Level-1, always-Level-3, a confidence threshold, an uncertainty threshold, a confidence–uncertainty threshold, and the trust threshold. Ablations set the environment weight or the uncertainty weight to zero and retune the threshold, or compute trust and then never run Level 3. The uncertainty comparison adds the entropy score. The abstention baseline is separate. The NSGA-II staircase is separate. The seed-0 energy-threshold ROC in Section 6.1 is the preserved detector from `artifacts/paper/fish_suite.json` on the same split. This upgrade did not refit it, and seeds 1 and 2 do not have that AUC.

### 5.4 Evaluation metrics

Classification uses macro-F1, per-class recall, and, for seed 0, the confusion matrix. The error rate among accepted predictions equals the logistic error rate on every explanation-gate row, because those rows accept every label. Abstention reports both the error on the accepted subset and the macro-F1 on all windows with refusals counted as errors. Energy uses joules per window, the saving against always-Level-3, and the fraction of windows that select Level 3. Explanation recall is the fraction of true boatwhistles that receive Level 3. Explanation false alarm is the fraction of `other` windows that receive Level 3. Calibration uses expected calibration error of a score against the event indicator.

---

## 6. Results and discussion

### 6.1 Comparative performance evaluation

Trust-based gating does not outperform confidence-based gating. Table 1 is that comparison. The confidence row and the trust row are identical. Macro-F1 is the logistic’s and is the same on every explanation-gate row, because the gate does not change the label. Gating does not improve classification accuracy. The mean over three seeds is 0.844 ± 0.004 (seeds 0.839, 0.848, and 0.844). The 95% interval is [0.833, 0.854]. Mean per-class recall is 0.816 ± 0.039 for the boatwhistle and 0.872 ± 0.041 for `other`. The mean error rate on all predictions is 0.156. Wake ROC-AUC is 0.908 ± 0.009. Wake ECE is 0.043 ± 0.017, with a 95% interval [0.000, 0.085] that is too wide to support a tight calibration claim.

**Table 1. One-threshold gates on the toadfish test windows. Mean ± sample standard deviation over seeds 0, 1, and 2. Energy is the datasheet model.**

| Method | Macro-F1 | Energy (mJ) | Saving vs always-L3 (%) | Level-3 rate | Explain recall | Explain false alarm | Error on accepted labels |
|--------|----------:|------------:|------------------------:|-------------:|---------------:|--------------------:|-------------------------:|
| Always Level 1 | 0.844 ± 0.004 | 5.408 | 16.2 | 0 | 0 | 0 | 0.156 |
| Always Level 3 | 0.844 ± 0.004 | 6.453 | 0 | 1 | 1 | 1 | 0.156 |
| Confidence threshold | 0.844 ± 0.004 | 5.779 | 10.4 ± 0.7 | 0.355 ± 0.041 | 0.662 | 0.039 | 0.156 |
| Uncertainty threshold | 0.844 ± 0.004 | 5.904 | 8.5 ± 0.7 | 0.474 | 0.501 | 0.446 | 0.156 |
| Confidence–uncertainty | 0.844 ± 0.004 | 5.779 | 10.4 ± 0.7 | 0.355 ± 0.041 | 0.662 | 0.039 | 0.156 |
| Trust threshold | 0.844 ± 0.004 | 5.779 | 10.4 ± 0.7 | 0.355 ± 0.041 | 0.662 | 0.039 | 0.156 |

The paired difference between trust and the confidence threshold is 0 on energy, on the Level-3 rate, on explanation recall, and on explanation false alarm. The paired \(t\) test is undefined for a zero difference. Trust versus the uncertainty threshold spends less energy (paired \(p = 0.026\)), explains more of the true boatwhistles (paired \(p = 0.032\)), and explains far fewer negatives (paired \(p = 0.0009\)). Those \(p\)-values use three seeds and are descriptive. They are not evidence of superiority on a new deployment. The entropy-of-Dirichlet-mean gate matched the uncertainty gate on every seed: the same Level-3 rate, the same recall, and the same false-alarm rate. Both scores are high when the distribution is peaked, including when it is peaked on `other`.

On seed 0, expected calibration error of wake confidence against the event indicator is 0.059. The same indicator gives 0.250 for inverted uncertainty, 0.160 for the entropy score, and 0.260 for trust. Trust is a worse probability of the event than wake confidence. Emitting an uncertainty value does not make a better explanation gate. Across seeds, wake ECE is 0.059, 0.044, and 0.025. Epistemic uncertainty was 2/22 on every window.

Seed 0 reproduces the earlier single-seed fit. Macro-F1 is 0.839, wake ROC-AUC is 0.898, ECE is 0.059, and Brier score is 0.120. Precision, recall, F1, and support are 0.880, 0.796, 0.836, and 618 for the boatwhistle, and 0.803, 0.885, 0.842, and 582 for `other`. The confusion matrix, rows true and columns predicted, in the order (boatwhistle, other), is [[492, 126], [67, 515]]. Seed 1 is [[472, 125], [57, 546]]. Seed 2 is [[522, 84], [103, 491]].

The validation rule chose the 65th percentile of each monotone score. On seed 0 the numeric thresholds differ (trust 0.848, confidence 0.767, confidence–uncertainty 0.653, environment weight removed 0.808, uncertainty weight removed 0.898). The selected windows do not differ.

The seed-0 energy-threshold detector, preserved from the earlier suite on this same split, has ROC-AUC 0.656 against 0.898 for wake confidence. That comparison is a ranking of the event, not a row of Table 1.

**Abstention, reported separately.** Refusing the label when trust is below a validation threshold does not reproduce Table 1. On seeds 0, 1, and 2 the abstention rates are 0.591, 0.627, and 0.488. The error rates among accepted labels are 0.075, 0.060, and 0.160, which are lower than the full-set error on seeds 0 and 1. Macro-F1 on the accepted subset is 0.480, 0.484, and 0.457. Macro-F1 on all windows, with refusals counted as errors, is 0.409, 0.403, and 0.423. On seed 0 the full-set recall of `other` under that scoring is 0. The rule keeps confident boatwhistle predictions and drops the negative class. The reduction in accepted error is the refusal, not a better classifier. The logistic macro-F1 on the same windows remains 0.839, 0.848, and 0.844.

### 6.2 Component-wise ablation study

Table 2 is the same test protocol with one component removed at a time. “Level 3 never” still computes trust and then withholds the graph.

**Table 2. Ablations. Test means. A blank change means the test mask matched the trust row.**

| Configuration | Trust score | Uncertainty in the score | Environment weight | Level 3 | Macro-F1 | Energy (mJ) | Level-3 rate |
|---------------|-------------|--------------------------|--------------------|---------|----------:|------------:|-------------:|
| No gate | unused | no | no | never | 0.844 | 5.408 | 0 |
| Wake only | \(C\) | no | no | if \(C \ge \tau\) | 0.844 | 5.779 | 0.355 |
| Wake and uncertainty | logit, health and environment weights 0 | yes | no | if score \(\ge \tau\) | 0.844 | 5.779 | 0.355 |
| Trust, environment weight 0 | yes | yes | 0 | if \(T \ge \tau\) | 0.844 | 5.779 | 0.355 |
| Trust, uncertainty weight 0 | yes | no | yes | if \(T \ge \tau\) | 0.844 | 5.779 | 0.355 |
| Full trust | yes | yes | yes | if \(T \ge \tau\) | 0.844 | 5.779 | 0.355 |
| Full trust, Level 3 never | computed | yes | yes | never | 0.844 | 5.408 | 0 |

The graph is not an input to \(T\). When the logistic says toadfish, the attached text names `cause.taxon.opsanus_tau` because that event has one `caused_by` edge. The match is the wiring, not a validation of the class. A label-free band comparison has ROC-AUC 0.630 ± 0.053 and is not used as a check that the predicted class is correct. The graph does not change the label.

Health consistency was 1 on every window. Environmental consistency was 0.259 on every window, to numerical noise, for the reason in Section 4.2. Zeroing either weight shifts the logit by a constant. The retuned percentile absorbs the shift. This ablation cannot show a field effect of sea state, because the sea state never changes, and it cannot show a spectral effect of \(\rho\), because instance normalisation has already removed the clip’s band means from that scalar.

### 6.3 Robustness and generalisation analysis

**Noise.** Table 3 is the frozen seed-0 head and the frozen clean-validation threshold, with white noise added before the features.

**Table 3. Derived white noise on the seed-0 test waveforms. Not a field recording.**

| SNR (dB) | Macro-F1 | Wake ROC-AUC | Wake ECE | Boatwhistle recall | Level-3 rate |
|---------:|---------:|-------------:|---------:|-------------------:|-------------:|
| clean | 0.839 | 0.898 | 0.059 | 0.796 | 0.362 |
| 20 | 0.820 | 0.878 | 0.081 | 0.735 | 0.309 |
| 10 | 0.758 | 0.831 | 0.147 | 0.595 | 0.223 |
| 5 | 0.711 | 0.810 | 0.208 | 0.487 | 0.166 |
| 0 | 0.608 | 0.787 | 0.280 | 0.327 | 0.077 |

Boatwhistle recall falls as the noise rises. Recall of `other` rises, because the head says `other` more often. The Level-3 rate falls because the scores fall under a threshold that was chosen on clean validation.

**Second dataset.** The Watkins tape hold-out is seed 0 only and is a negative result, not a successful generalisation. Test macro-F1 is 0.281 against a pre-registered bar of 0.70. Accuracy is 0.285. The majority class, humpback whale, has accuracy 0.062. Seeds 1 and 2 were not run. Fin whale F1 is 0.767 and killer whale F1 is 0.692. Striped dolphin and walrus are 0.000, each one test tape, and they stay in the average. The archive mixes equipment, decades, and sites inside a species folder [26], and the split holds the tape out [25]. It is not the BEANS random split [10], and 0.281 is not comparable to the toadfish macro-F1.

**Pre-segmentation.** Both acoustic evaluations score events that were already cut. Neither demonstrates detection in a continuous recording.

**Level 0 on the seed-0 fish split.** Test boatwhistles have median surprise 3.599, and all 618 sit below the twin admit point at 5.398. A pipeline that slept on that cutoff would reject every boatwhistle. The fish head records surprise and does not use it as the label. Those counts are `artifacts/paper/fish_l0_surprise.json`.

The digital twin is a different generator. Its large simulated saving is not a result of this section. The assumptions and the secondary numbers are in the appendix.

### 6.4 Estimated energy-cost and sensitivity analysis

The trust gate’s 10.4% ± 0.7% is the saving of explaining about 35% of windows instead of all of them, under the cumulative datasheet model that charges an unused Level-2 budget. If Level 2 is not charged, the same decisions save 0.062% ± 0.004%. That alternative is the published energy identity with the Level-2 term removed. It is not a new run. The twin’s factor of about 32 is not this number. Always stopping at Level 1 saves 16.2% and explains nothing. The knee trades those two objectives against explanation false alarms. On the seed-0 validation curve the 65th percentile, which is the chosen point, explains 35% of windows, recalls 0.646 of validation boatwhistles, and false-alarms on 0.068 of validation negatives, for a validation saving of 10.5%. The 5th percentile explains 95% of windows and saves 0.8%. The 95th percentile explains 5% and saves 15.4%. The test operating point was the validation knee, not a point read off the test curve.

Scaling the additional-path cost by 0.5, 1, 2, and 5, and retuning the threshold on validation, left the chosen percentile at 65. The validation saving moved from 5.7% to 10.5%, 18.1%, and 31.9%, because the same decisions were priced differently. Scaling only the Level-3 joules by the same factors moved the validation saving from 10.51% to 10.70%. The result tracks the Level-2 budget inside the cost model. It does not track the graph walk.

Scaling each hand-set weight by 0.5, 1, and 2, one at a time, left the validation percentile at 65 and left the validation explanation rates unchanged. The same was true of concentrations 5, 10, 20, and 40: epistemic uncertainty moved from 0.286 to 0.048, the numeric threshold moved, and the validation mask did not. These sweeps are validation surfaces. They were not used to pick a test-set champion. The reported test configuration remains concentration 20 and the hand-set weights.

The validation-chosen NSGA-II staircase is a different policy. On the three test sets its modelled energy is 6.07, 5.85, and 3.70 mJ per window, with detection misses 0.105, 0.104, and 0.070. Seed 2’s 3.70 mJ sits below the Level-1 cost of 5.41 mJ, so that policy is charging some windows as Level 0. Its energy is not comparable to Table 1. The retired test-tuned knee at 3.82 mJ is not this row and is not quoted as a result.

### 6.5 Discussion of key findings

This study examines whether uncertainty and contextual terms improve event-level explanation gating over a confidence threshold under an explicit energy-cost model. On the evaluated two-class toadfish corpus, the trust score ranks windows the same way as wake confidence, while the uncertainty-only gate is less selective. The estimated energy benefit depends on charging a Level-2 budget that does not produce the label.

The classifier that should be believed is the three-seed macro-F1 of 0.844 ± 0.004 on pre-segmented Pamlico candidates. The gate does not change it. The intervals are over three seed-level means inside one estuary. Paired contrasts on those three seeds are descriptive. They are not evidence of superiority on a new deployment. The trust-minus-confidence difference is zero on every test window, so it is zero inside every recording group.

Watkins macro-F1 of 0.281 missed the pre-registered bar of 0.70. It is not generalisation. The white-noise rows are a controlled perturbation of the frozen seed-0 head, not a sea trial. The twin remains an appendix.

---

## 7. Limitations and threats to validity

This study evaluates event-level trust gating using available underwater acoustic recordings and a model-based energy-cost framework. The reported energy savings are estimates derived from the specified computational assumptions and are not measurements from a physical underwater deployment. The evaluation therefore establishes the potential of the proposed decision mechanism under the tested conditions rather than demonstrating its operational performance in an underwater environment. Future work should examine the framework using embedded hardware and field measurements when suitable facilities become available.

Several threats sit inside that statement. The fish files are pre-segmented candidates, so the study does not show event discovery in a continuous stream. The environment and the sensor health do not vary, and the trust score collapses to wake confidence. A corpus with real sea state, real faults, or more than two classes could separate the terms. The cost model charges a Level-2 network the label does not run, so the 10.4% is sensitive to that assumption and insensitive to the graph. Three seeds give wide intervals, especially for ECE. The Watkins evaluation is one seed. The twin is a simulation, and its factor-of-32 saving is not the fish saving. Level 4 neighbour confirmation is not reached and is not an acoustic modem. Temperature scaling was not fit.

---

## 8. Conclusion and future work

Uncertainty and contextual terms do not improve event-level explanation gating over a confidence threshold on the two-class toadfish corpus studied here. The logistic macro-F1 is 0.844 ± 0.004 and is unchanged by the explanation decision. The trust score ranks windows as wake confidence does, because several of its components are constant. With concentration 20 fixed, epistemic uncertainty is 2/22 on every two-class window. The uncertainty-only gate is less selective. A datasheet model estimates a 10.4% ± 0.7% saving against always running Level 3 only when an unused Level-2 budget is charged. The clips are pre-segmented candidates from one estuary. A Watkins tape hold-out at macro-F1 0.281 does not meet a pre-registered bar of 0.70 and is not generalisation.

The useful next measurement is a cost that matches the code path that actually runs, on a device that can report current, and a recording in which sea state and sensor health are not constants. Until those exist, the energy numbers stay estimates, and the trust gate stays equivalent to a confidence threshold on the corpus studied here.

---

## Appendix. Digital-twin simulation

The twin is not the toadfish experiment. It uses spherical spreading, Thorp absorption at a fixed 1 kHz [36], one surface bounce, and a delay clipped at 0.5 s. Its fish class is a 280 Hz pulse train, not a Pamlico boatwhistle. The seed-0 busy-strait draw has 138, 32, and 43 windows. On the 43 test windows, wake ROC-AUC is 0.916, the energy-threshold AUC is 0.069, and wake ECE is 0.328. A gated policy in that run uses 0.208 mJ per window against 6.66 mJ for always-Level-3, about 32 times, because most simulated windows never reach the graph. A 10 Wh scaling of those joules is about 2,003 days against 63 days. Both figures are datasheet scaling of a different generator. They are not a battery measurement and they are not the toadfish saving of 10.4%. The graph loaded for that run had 20 nodes and 21 edges. The live ontology has 31 nodes and 65 edges. The synthetic hit rate was not measured on the larger graph.

---

## References

[1] D. K. Mellinger, K. M. Stafford, S. E. Moore, R. P. Dziak, and H. Matsumoto, “An overview of fixed passive acoustic observation methods for cetaceans,” *Oceanography*, vol. 20, no. 4, pp. 36–45, 2007, doi: 10.5670/oceanog.2007.03.

[2] R. S. Sousa-Lima, T. F. Norris, J. N. Oswald, and D. P. Fernandes, “A review and inventory of fixed autonomous recorders for passive acoustic monitoring of marine mammals,” *Aquatic Mammals*, vol. 39, no. 1, pp. 23–53, 2013, doi: 10.1578/AM.39.1.2013.23.

[3] W. M. X. Zimmer, *Passive Acoustic Monitoring of Cetaceans*. Cambridge, U.K.: Cambridge University Press, 2011.

[4] A. Signori, F. Campagnaro, I. Nissen, and M. Zorzi, “Channel-based trust model for security in underwater acoustic networks,” *IEEE Internet of Things Journal*, vol. 9, no. 20, pp. 20479–20491, 2022, doi: 10.1109/JIOT.2022.3176374.

[5] S. Song, B. Huangfu, Jiani Guo, J. Liu, J. Cui, and X. Shen, “A digital twin-based intelligent network architecture for underwater acoustic sensor networks,” *IEEE Transactions on Mobile Computing*, vol. 24, no. 9, pp. 8196–8213, 2025, doi: 10.1109/TMC.2025.3555640.

[6] Y. He, G. Han, A. Li, T. Taleb, C. Wang, and H. Yu, “A federated deep reinforcement learning-based trust model in underwater acoustic sensor networks,” *IEEE Transactions on Mobile Computing*, vol. 23, no. 5, pp. 5150–5161, 2024, doi: 10.1109/TMC.2023.3301825.

[7] N. Xia, S. Wei, M. Li, Y. Wang, and J. Wan, “Building trust for underwater wireless sensor networks via Riemannian variational autoencoders,” *IEEE Transactions on Mobile Computing*, vol. 25, no. 1, pp. 117–131, 2026, doi: 10.1109/TMC.2025.3590741.

[8] M. Irfan, Z. Jiangbin, S. Ali, M. Iqbal, Z. Masood, and U. Hamid, “DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification,” *Expert Systems with Applications*, vol. 183, 115270, 2021, doi: 10.1016/j.eswa.2021.115270.

[9] D. R. Bohnenstiehl, “Automated cataloging of oyster toadfish (*Opsanus tau*) boatwhistle calls using template matching and machine learning,” *Ecological Informatics*, vol. 77, 102268, 2023, doi: 10.1016/j.ecoinf.2023.102268.

[10] M. Hagiwara, B. Hoffman, J.-Y. Liu, M. Cusimano, F. Effenberger, and K. Zacarian, “BEANS: The benchmark of animal sounds,” arXiv:2210.12300, 2022.

[11] S. M. Wiggins and J. A. Hildebrand, “High-frequency Acoustic Recording Package (HARP) for broad-band, long-term marine mammal monitoring,” in *Proc. Int. Symp. Underwater Technology*, 2007, pp. 551–557, doi: 10.1109/UT.2007.370760.

[12] S. Wiggins, B. Thayre, K. Lenssen, J. Hurwitz, J. Hildebrand, and K. Frasier, “Low-power mass-storage upgrade for High-frequency Acoustic Recording Packages (HARPs),” Marine Physical Laboratory Technical Memorandum 666, Scripps Institution of Oceanography, La Jolla, CA, USA, Mar. 2023.

[13] R. T. Taylor, M. S. Ballard, C. W. Cushing, K. M. Lee, A. R. McNeese, P. S. Wilson, J. Bonnel, and D. P. Zitterbart, “Evaluating the directivity of compact underwater acoustic recording devices,” *Proc. Mtgs. Acoust.*, vol. 54, 070005, 2024, doi: 10.1121/2.0001995.

[14] N. R. E. Todd, A. S. Kavanagh, E. Rogan, and M. J. Jessopp, “What the F-POD? Comparing the F-POD and C-POD for monitoring of harbor porpoise (*Phocoena phocoena*),” *Ecology and Evolution*, vol. 13, e10186, 2023, doi: 10.1002/ece3.10186.

[15] J. Ivanchikova and N. Tregenza, “Validation of the F-POD—A fully automated cetacean monitoring system,” *PLoS ONE*, vol. 18, no. 11, e0293402, 2023, doi: 10.1371/journal.pone.0293402.

[16] N. D. Merchant, K. M. Fristrup, M. P. Johnson, P. L. Tyack, M. J. Witt, P. Blondel, and S. E. Parks, “Measuring acoustic habitats,” *Methods in Ecology and Evolution*, vol. 6, no. 3, pp. 257–265, 2015, doi: 10.1111/2041-210X.12330.

[17] J. A. Hildebrand, “Anthropogenic and natural sources of ambient noise in the ocean,” *Marine Ecology Progress Series*, vol. 395, pp. 5–20, 2009, doi: 10.3354/meps08353.

[18] V. O. Knudsen, R. S. Alford, and J. W. Emling, “Underwater ambient noise,” *Journal of Marine Research*, vol. 7, no. 3, pp. 410–429, 1948.

[19] G. M. Wenz, “Acoustic ambient noise in the ocean: Spectra and sources,” *Journal of the Acoustical Society of America*, vol. 34, no. 12, pp. 1936–1956, 1962, doi: 10.1121/1.1909155.

[20] G.-A. Gray and H. E. Winn, “Reproductive ecology and sound production of the toadfish, *Opsanus tau*,” *Ecology*, vol. 42, no. 2, pp. 274–282, 1961, doi: 10.2307/1932079.

[21] M. L. Fine, “Seasonal and geographical variation of the mating call of the oyster toadfish *Opsanus tau* L.,” *Oecologia*, vol. 36, pp. 45–57, 1978, doi: 10.1007/BF00344570.

[22] C. P. H. Elemans, A. F. Mensinger, and L. C. Rome, “Vocal production complexity correlates with neural instructions in the oyster toadfish (*Opsanus tau*),” *Journal of Experimental Biology*, vol. 217, pp. 1887–1893, 2014.

[23] A. F. Mensinger, “Disruptive communication: stealth signaling in the toadfish,” *Journal of Experimental Biology*, vol. 217, pp. 344–350, 2014, doi: 10.1242/jeb.090316.

[24] D. R. Bohnenstiehl, “Data for: ToadFishFinder classifier model v4,” Dryad, 2023, doi: 10.5061/dryad.gtht76hr9. CC0.

[25] R. Aguiar, G. Maguolo, L. Nanni, Y. Costa, and C. Silla Jr., “On the importance of passive acoustic monitoring filters,” *Journal of Marine Science and Engineering*, vol. 9, no. 7, 685, 2021, doi: 10.3390/jmse9070685.

[26] L. Sayigh, M. A. Daher, J. Allen, H. Gordon, K. Joyce, C. Stuhlmann, and P. Tyack, “The Watkins marine mammal sound database: An online, freely accessible resource,” *Proceedings of Meetings on Acoustics*, vol. 27, 040013, 2016, doi: 10.1121/2.0000358.

[27] S. M. Van Parijs, C. W. Clark, R. S. Sousa-Lima, S. E. Parks, S. Rankin, D. Risch, and I. C. Van Opzeeland, “Management and research applications of real-time and archival passive acoustic sensors over varying temporal and spatial scales,” *Marine Ecology Progress Series*, vol. 395, pp. 21–36, 2009, doi: 10.3354/meps08123.

[28] M. F. Baumgartner *et al.*, “Persistent near real-time passive acoustic monitoring for baleen whales from a moored buoy: System description and evaluation,” *Methods in Ecology and Evolution*, vol. 10, no. 9, pp. 1476–1489, 2019, doi: 10.1111/2041-210X.13244.

[29] I. F. Akyildiz, D. Pompili, and T. Melodia, “Underwater acoustic sensor networks: research challenges,” *Ad Hoc Networks*, vol. 3, no. 3, pp. 257–279, 2005, doi: 10.1016/j.adhoc.2005.01.004.

[30] M. Sensoy, L. Kaplan, and M. Kandemir, “Evidential deep learning to quantify classification uncertainty,” in *Advances in Neural Information Processing Systems*, 2018.

[31] J. Traer, S. V. Norman-Haignere, and J. H. McDermott, “Causal inference in environmental sound recognition,” *Cognition*, vol. 214, 104627, 2021.

[32] Michel Olvera, C. Wang, Paraskevas Stamatiadis, G. Richard, and S. Essid, “iKnow-audio: Integrating knowledge graphs with audio-language models,” in *Proc. EMNLP*, 2025, pp. 34683–34700, doi: 10.18653/v1/2025.emnlp-main.1759.

[33] P. M. Dung, “On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games,” *Artificial Intelligence*, vol. 77, no. 2, pp. 321–357, 1995.

[34] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182–197, 2002.

[35] K. V. Mackenzie, “Nine-term equation for sound speed in the oceans,” *Journal of the Acoustical Society of America*, vol. 70, no. 3, pp. 807–812, 1981, doi: 10.1121/1.386920.

[36] W. H. Thorp, “Analytic description of the low-frequency attenuation coefficient,” *Journal of the Acoustical Society of America*, vol. 42, no. 1, p. 270, 1967, doi: 10.1121/1.1910566.

[37] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in *Advances in Neural Information Processing Systems*, 2017.

[38] S. Teerapittayanon, B. McDanel, and H. T. Kung, “BranchyNet: Fast inference via early exiting from deep neural networks,” in *Proc. 23rd Int. Conf. Pattern Recognition*, 2016, pp. 2464–2469, doi: 10.1109/ICPR.2016.7900006.

---

## Data and reproducibility

The toadfish waveforms are CC0 on Dryad [24]. They are not in this git repository. Watkins audio is an academic WHOI release [26] and is not redistributed. The twin is generated from a seed.

```bash
uaere eval --suite review --root data/toadfish --out artifacts/paper/review_upgrade --max-per-class 3000 --min-files 40 --steps 2000
```

The review command runs seeds 0, 1, and 2. It does not overwrite `artifacts/paper/fish_suite.json` or `artifacts/models/fish_head.joblib`. Do not re-run `--suite fish`, `--suite species`, or `--suite paper` against those output paths. The fish-suite knee in the older file was selected on the test split. Do not quote its 3.82 mJ as the result of the review command. The Mel-only toadfish table, macro-F1 0.827, is `artifacts/paper/fish_suite_melstats.json`. The Watkins table, macro-F1 0.281, is `artifacts/paper/species_suite.json`. Operator notes are in `docs/USER_GUIDE.md`. The patent specification is a separate document, `paper/complete_specification.md`, and its claim language is not a result of this paper.
