# Provenance — vendored engines

The modules in this package are **vendored, not rewritten**. They are the
author's own verified engines, copied from his earlier open-source repos so
this desk is self-contained. If a number here ever disagrees with the source
repo, the source repo is the reference implementation.

| Module(s) | Vendored from | Notes |
|---|---|---|
| `chain.py`, `pricing.py`, `greeks.py`, `iv.py` | `greeks-lab` (`src/greekslab/`) | NSE bhavcopy parsing + option-chain analytics (IV, Greeks, GEX, gamma flip, max pain). Only the package path changed (`greekslab.*` → `nifty_desk.engines.*`); the GEX convention is byte-identical: `GEX_K = (Γ_CE·OI_CE − Γ_PE·OI_PE) × LOT × S² × 0.01`, OI in contracts, lot read from the bhavcopy (65 for NIFTY in Nov 2026), gamma at each strike's own IV. |
| `regimes.py` | `textbook-quant-models` (`src/textbook_quant/`) | Hamilton (1989) 2-state Gaussian Markov-switching model: EM (scaled forward–backward) from scratch, Hamilton filter, smoothed probabilities, Viterbi, expected durations. States are volatility-ordered: 0 = low-vol, 1 = high-vol. |
| `garch.py` | `textbook-quant-models` (`src/textbook_quant/`) | GARCH / GJR / EGARCH filters with Gaussian MLE fits, multi-step forecasts, QLIKE/MSE losses and a Diebold–Mariano test. The desk uses plain GARCH(1,1). |

Both source repos were verified end-to-end on real data by the author
(greeks-lab on the 3 Nov 2026 NSE bhavcopy; textbook-quant-models on NIFTY
2015–2026 and US Treasury data). This desk's test-suite cross-checks the
vendored chain engine against greeks-lab's published 3 Nov 2026 case
numbers rather than trusting the copy blindly.
