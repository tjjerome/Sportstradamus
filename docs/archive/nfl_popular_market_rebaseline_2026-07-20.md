# NFL popular-market re-baseline (2026-07-20 evidence record)

> Status: RECORD — archived 2026-10-16 from
> [model_improvement_track.md](../handoffs/model_improvement_track.md) §6.6, where it sat as a
> dated block. `CONFIRM_EVIDENCE_CORNERS` (`training/model_strategy/specs.py`) cites it for
> the passing-TDs corner. The `/tmp` evidence roots and harness scripts it names are gone.

The old “efficient-book kill” conclusion for passing yards / passing TDs and the Mixture-first route for the yardage cells are
superseded. On 2026-07-20 all four popular markets earned isolated full-HPO artifacts with
independent heldout six-gate passes; that work changed no `shipped` state. It is a record, not the
current state: the passing-yards pass was superseded on 2026-08-27 (the track's §10), the other three ship
on later confirm recipes (its §6.9), and the `/tmp` quarantine files, evidence roots and harness scripts
named below no longer exist. `CONFIRM_EVIDENCE_CORNERS` (`model_strategy/specs.py`) still seeds
the passing-TDs corner from this record.

- **Passing yards — isolated full-HPO + independent heldout PASS.** The exact recipe was SkewNormal,
  `ratio_meanyr`, distribution CRPS, blend NLL, loss-selected HPO, direct SN, no stabilization,
  and no posthoc. On n=369: g1 Brier delta −0.0010, row CI [−0.0051, 0.0032] and
  player-clustered CI-high 0.0033; g2 0.1206; g3 0.1211; g4 PIT-KS 0.0572 ≤ 0.0707 (tail
  0.0332); g5 −0.0075; g6 CITL 0.9904 (CI-high 1.0697). Pickle version
  `20260720.ratio_meanyr.e59505d4`, temperature 1.1540011, dispersion 0.9908284, model weight
  0.1093457. Quarantine: `/tmp/NFL_passing-yards.full-hpo-pass-20260720.mdl` SHA-256
  `6108de3a8aa90a90f34e4c0b13de7b003c6568129cf58a23c8ca183a1c9ca3bc`; scorecard
  `/tmp/passing-yards-fresh-scorecard.csv` SHA-256
  `f5d76a0fee5dc1229411d6f40382ee838baaf6bf4b9dbeb6c023ae4664c3a652`.

  ```bash
  sportstradamus meditate --league NFL --market "passing yards" --force \
    --bypass-withholding --dist SkewNormal --target-normalization ratio_meanyr \
    --dist-training-loss crps --blending-loss-fn nll --hpo-selection loss \
    --sn-param direct --stabilization None
  ```

- **Passing TDs — isolated full-HPO + independent heldout PASS.** The evidence recipe was DPO,
  legacy-labeled `ratio_meanyr`,
  distribution NLL, blend NLL, loss-selected HPO, count-dispersion objective `pit_ks`, direct
  invocation flag `--sn-param direct`, no stabilization, and **`roe_mean` posthoc** (affine a=0.2841855,
  b=0.8736016). Canonical `stat_meta` now records `target_normalization: none`. This is an identity
  cleanup, not a model change: the DPO count branch never applies the continuous target transform,
  so `none` and legacy `ratio_meanyr` both train/decode the same raw count labels. Goldens
  `test_dpo_count_path_target_normalization_is_probability_inert` and
  `test_dpo_distribution_selection_accepts_canonical_none_without_transform` in
  `tests/golden/test_pipeline_artifact_parity.py` pin bitwise-equal mean/dispersion arrays and
  equal DPO CDF/PMF probabilities. The command and version below retain their historical
  `ratio_meanyr` evidence identity. On n=373: g1 −0.0187, row CI [−0.0284, −0.0087] and
  clustered CI-high −0.0087;
  g2 0.1780; g3 0.1499; g4 0.0498 ≤ 0.0703 (tail 0.0187); g5 0.0439; g6 CITL 1.0479
  (CI-high 1.2394) and over ratio 0.9103 (CI-low 0.7901). Pickle version
  `20260720.ratio_meanyr.afa8f007`, temperature 1.3753360, dispersion 1.1951890, model weight
  0.6822488. Quarantine: `/tmp/NFL_passing-tds.full-hpo-pass-20260720.mdl` SHA-256
  `e00646834f2646ba2c304a863755608862396691df5c32302209f96d8242ab6b`; scorecard
  `/tmp/passing-tds-fresh-scorecard.csv` SHA-256
  `e8b16b4e05efc712f58d8f48749cf5a06f329d9bd37f3ad99012ef2b063232b2`.

  ```bash
  sportstradamus meditate --league NFL --market "passing tds" --force \
    --bypass-withholding --dist DPO --target-normalization ratio_meanyr \
    --dist-training-loss nll --blending-loss-fn nll --hpo-selection loss \
    --count-dispersion-objective pit_ks --sn-param direct --stabilization None
  ```

- **Rushing yards — isolated cold full-HPO + independent heldout PASS.** Experiment
  `rushing-qb-rb-affine-groupcdf-bookpool-v1` adds an affine marginal, separate QB/RB endpoint
  CDF maps, line-only temperature, and a coherent model/book probability pool to the existing
  SkewNormal `ratio_meanyr` / CRPS / blend-NLL / loss-selected / direct / no-stabilization /
  no-posthoc path. The player-grouped conditional-OOF calibration audit passed on n=1010
  (g1 −0.00315, row CI-high 0.000735, player CI-high 0.000566; PIT-KS 0.04586; QB 0.03583,
  RB 0.05384). The isolated run completed 95 trials at 1:00:27, best trial 53 / validation loss
  0.765541, version `20260720.ratio_meanyr.b45d9dae`; full-fit head
  `4.4653484193 + 1.0039169446 × marginal`, QB/RB lambdas 1/1, temperature 1.6410711827,
  pool weight 0.4211669563. Its untouched heldout then passed on n=1009: g1 −0.0024, row CI
  [−0.0061, 0.0011], clustered CI-high 0.0011; g2 0.1913; g3 0.1098; g4 0.0368 ≤ 0.05
  (tail 0.0367); g5 0.0272; g6 recent ratio 0.9309 (CI-high 0.9656) and CITL 0.9621
  (CI-high 1.0473).

  ```bash
  MPLCONFIGDIR=/tmp/mpl-rushing-confirm \
    poetry run python /tmp/run_rushing_full_hpo_confirm.py
  ```

  Evidence root `/tmp/rushing-full-hpo-confirm-k9md4fgm/`: confirmation SHA-256
  `2e229934d8e7314e0872d16d8e91aa3a2ba46244713ea9f98c8d1d326a800aee`, scorecard
  `a7d12f4fa6de30dacfd63f171ca968460403cef9ed2c9be447058c5e6e8c11bb`, model
  `47d9576b78a43623d0cf5c614f7b8d75e4159948b8355590d5f99ed37d345479`, test CSV
  `0dd6f0e730f5d76a5fa72be780207c17ed6997656d54d53419e9353778bb552a`. All ten protected
  production targets retained their content hashes/sizes. The strict fingerprint is false only
  because an external process rewrote `stat_calibration.json` with identical content but a new
  mtime; the restoration-audit SHA-256 is
  `324d754f0dbe86ba29350a38f6c1ba09a87250efb12625df55822c69f73e7da6`.

- **Receiving yards — isolated cold full-HPO + independent heldout PASS.** Experiment
  `receiving-role-position-two-part-groupcdf-fixedlinear-v3` uses two role boundary intercepts
  plus an RB residual, six role×position positive CDF maps, two role nonpositive maps, shared
  positive/nonpositive lambda selection, line-only temperature, exact endpoint-first settlement,
  and a policy-fixed authentic-quote pool `0.8 × book + 0.2 × model` (missing/synthetic quotes
  remain unpooled). The exact base recipe was SkewNormal, `ratio_meanyr`, distribution CRPS,
  blend NLL, loss-selected HPO, direct SN, no stabilization, no posthoc, and the frozen cached
  matrix. The 60-minute cold search used no initial parameters, completed 146 of 300 configured
  trials, and selected trial 72 at the best logged output loss 0.629173; the final pickle version
  is `20260720.ratio_meanyr.b9a6cba1` and its line temperature is 1.4702421.

  Before heldout scoring, nested Player GroupKFold on the canonical 2,239 rows / 382 players
  passed all ten named support guards and all 12 validation gate guards. The exact validation
  audit was: g1 −0.0003677, row CI [−0.0020198, 0.0013740], player-clustered CI
  [−0.0020539, 0.0013343]; g2 0.0927752; g3 0.2569047; g4 global 0.0432616, role full
  high/low 0.0325552/0.0590294, role-positive 0.0344159/0.0715984, position full WR/RB/TE
  0.0420076/0.0516557/0.0520554, and position-positive 0.0464157/0.0692140/0.0540941;
  g5 debiased ECE 0.0173928 and rate error 0.0038382; g6 recent correlation 0.5605996 and
  CITL ratio 0.9593831, CI [0.8978878, 1.0292329]. Fold lambdas (+/−) were (1/.5),
  (1/.5), (1/.25), (1/.25), (1/.75); fold temperatures ranged 1.4551961–1.4956771.

  Its untouched heldout then passed all six gates on n=2,238: g1 −0.0005, row CI
  [−0.0022, 0.0012] and player-clustered CI-high 0.0012; g2 0.0239; g3 0.1932; g4
  PIT-KS 0.0340 ≤ 0.05 (tail 0.0340; central-50/80 coverage 0.4960/0.7976); g5 debiased
  ECE 0.0159 (raw 0.0423); g6 recent correlation 0.5508 and CITL ratio 0.9670
  (CI-high 1.0240).

  ```bash
  MPLCONFIGDIR=/tmp/mpl-receiving-confirm \
    poetry run python /tmp/run_receiving_full_hpo_confirm.py
  ```

  Canonical split fingerprint
  `9589cedab5e06ad318bb4e7b4c6e4e6c75bc776e87a03e480c8f32836b49f10a`. Evidence root
  `/tmp/receiving-full-hpo-confirm-k4ipe86a/`: confirmation SHA-256
  `a83c6684b9f283572f0f7fc722069d604356b366d4f709cc2be9325362458afb`, scorecard
  `9abd76d3e8e18793de4e3499746e288db1de76ee5ab1f5a170746feab63d4660`, model
  `5706a153b77464ffcb00036bd122f4eb00c1a6194660ddfa8a97e8f165703607`, and test CSV
  `8e22c48c12ad9cd80194946428426c1868b0038e9c32217f3230296ff43160bb`. The harness redirected
  model/test/scorecard output into its isolated root, suppressed matrix/config/report/SHAP writes,
  and verified all ten protected production targets content- and metadata-identical before/after;
  restoration-audit SHA-256
  `4b2200cbfc8103f28f3d9f665574ece0bbff66156e2aed04e0b04f05cab8998b`. The prior direct-read
  result remains retired because 1,118 purported validation rows overlapped the heldout.
