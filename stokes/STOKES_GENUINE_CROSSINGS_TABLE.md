# Stokes Crossings: Phase Equality vs Genuine Dominance-Relevant Events

This note records the computed crossing table for 10 values of \(\beta\) in
\((-\pi,0)\cup(0,\pi)\), with emphasis on the distinction between:

- **earliest phase crossing**: first \(\tilde\gamma\) where
  \(\Delta_{\Im}(\tilde\gamma;k)=0\),
- **earliest genuine crossing**: first crossing where
  \(|\Delta_{\Re}(\tilde\gamma;k)|<\varepsilon\), with \(\varepsilon=0.05\).

## Definitions

\[
\Delta_{\Re}(\tilde\gamma;k):=\Re\Phi(\theta_0^*)-\Re\Phi(\theta_k^*),
\qquad
\Delta_{\Im}(\tilde\gamma;k):=\Im\Phi(\theta_0^*)-\Im\Phi(\theta_k^*)
\]

Here \(\theta_0^*\) is the dominant saddle (largest \(\Re\Phi\) at fixed
\(\tilde\gamma,\beta\)), and \(k\) labels competitor saddles by nearest
singularity index (restricted to \(|k|\le 3\)).

## Final Table (\(\varepsilon=0.05\))

| \(\beta/\pi\) | Earliest phase crossing \(\tilde\gamma/\pi\) | \(k\) | \(\Delta_{\Re}\) there | Earliest genuine crossing \(\tilde\gamma/\pi\) (\(|\Delta_{\Re}|<0.05\)) | \(k\) |
|---:|---:|---:|---:|---:|---:|
| -0.950 | 0.51829 | +1 | +12.0843 | 2.96085 | -1 |
| -0.750 | 0.39875 | +1 | +4.1717 | 2.20238 | -1 |
| -0.550 | 0.45206 | +1 | +2.3879 | 3.07275 | -1 |
| -0.350 | 0.31150 | -1 | +5.3031 | 4.04259 | -1 |
| -0.150 | 0.31073 | -1 | +13.1059 | none found on \([0.3\pi,5\pi]\) | — |
| +0.150 | 0.44362 | -1 | +7.2182 | 3.35873 | -1 |
| +0.350 | 0.50135 | -1 | +1.4509 | 1.94956 | -1 |
| +0.550 | 1.45635 | -1 | +0.0000 | 1.45635 | -1 |
| +0.750 | 0.33567 | +1 | +8.3000 | none found on \([0.3\pi,5\pi]\) | — |
| +0.950 | 0.36217 | +1 | +0.0000 | 0.36217 | +1 |

## Main Interpretation

- For most sampled \(\beta\), the first \(\Im\)-equality crossing occurs while
  \(\Delta_{\Re}\) is still large and positive, so it is not dominance-relevant.
- Clear early dominance-relevant events appear at \(\beta/\pi\approx +0.55\)
  and \(+0.95\), where \(\Delta_{\Re}\approx 0\) at the first crossing.
- For \(\beta/\pi=-0.15\) and \(+0.75\), no genuine crossing was found in the
  scan window \([0.3\pi,5\pi]\).

## Numerical Protocol (as run)

- Model/solver: `qaoa_core.py` (`all_saddles`, `Phi`, `dPhi`, `d2Phi`,
  `singularities`).
- \(\beta\) grid (10 values):
  \[
  \beta/\pi \in \{-0.95,-0.75,-0.55,-0.35,-0.15,+0.15,+0.35,+0.55,+0.75,+0.95\}.
  \]
- \(\tilde\gamma\) sweep: \(0.3\pi\) to \(5\pi\), sampled on a uniform grid.
- Competitors restricted to \(|k|\le 3\), with \(k\) assigned by nearest
  singularity.
- Crossing location obtained by linear interpolation across sign changes of
  \(\Delta_{\Im}\).

## Caveats

- Results are numerical and can shift slightly with solver/grid settings.
- "None found" means no crossing satisfying \(|\Delta_{\Re}|<0.05\) in the
  stated window, not a proof of non-existence.

