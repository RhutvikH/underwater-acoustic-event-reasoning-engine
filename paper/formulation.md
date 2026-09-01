# AHAIF numbered equations

These identities are implemented in `src/uaere/math/`. If this file and the code
diverge, the code wins and this file is regenerated.

## Environment and representation

**(1)** Mackenzie sound speed \(c(T,S,z)\) — `mackenzie_ssp`.

**(2)** Thorp absorption \(\alpha(f)\) dB/km — `thorp_absorption_db_km`.

**(3)** Noise-PSD EMA on non-event frames
\[\hat{n}_t = (1-\alpha)\hat{n}_{t-1} + \alpha\,\mathrm{PSD}(x_t)\]

**(4)** Environment-conditioned instance-norm
\[\phi^{\mathrm{adapt}} = \gamma(s)\odot \mathrm{IN}(\phi) + \beta(s)\]

## Evidential trust (no fixed threshold)

**(5)** \(\alpha = f_\theta(\phi_1,h)+1 > 0\)

**(6)** \(\hat{p} = \alpha / \sum_i \alpha_i\)

**(7)** \(u_{\mathrm{epistemic}} = K / \sum_i \alpha_i\)

**(8)** \(u_{\mathrm{aleatoric}} = 1 - \max_k \hat{p}_k\)

**(9)** \(C_{\mathrm{wake}} = 1 - \hat{p}_{\mathrm{reject}}\)

**(10)**
\[T(e)=\sigma\big(w_c C_{\mathrm{wake}} + w_h \kappa(h) + w_s \rho(s,x) - w_u(u_a+u_e)\big)\]

## Multi-objective program

**(11)** \(\pi=(\tau_1,\tau_2,\tau_3,m,d)\)

**(12)** \(\min_\pi F(\pi)=(f_{\mathrm{miss}},f_{\mathrm{fa}},E,L,\mathrm{ECE},-X)\)

**(13)** \(E_{\mathrm{compute}}=c_0+p_1 c_1+p_2 c_2+p_3 c_3\)

**(14)** Weighted Chebyshev scalarisation on the Pareto set \(\Pi^\star\).

**Proposition 1.** If \(p_3<1\) and \(c_k>0\), then \(E(\pi)<E(\pi_{\mathrm{on}})\).
(`gated_energy_strictly_less`)

**Proposition 2.** Extra miss rate \(\varepsilon(\tau_3)\) is at most the posterior
mass between \(\tau_3\) and the Bayes threshold. (`gated_miss_bound`)

## Causal check

**(15)** \(\mathrm{CF}(c,\hat{y})=\mathbf{1}[f_\theta(\phi(x\setminus\mathrm{cues}(c)))\ne\hat{y}]\)
