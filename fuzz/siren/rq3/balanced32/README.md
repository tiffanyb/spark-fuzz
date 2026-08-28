# RQ3 balanced32 non-custom subset plus augmentations

This folder contains the 32-case RQ3 subset used to avoid the custom pinned
insertion targets and to reduce the R-CBF imbalance in the original 65-case
result set, plus 8 R-SSS insertion targets added from `rq3/rsss_hunt` and
2 pair-hunt insertion targets added from `rq3/pair_hunt`.

Selection rule:
- Start from the original non-custom RQ3 insertion targets.
- Exclude every target with `pinned_scene: true`.
- Keep all non-R-CBF targets.
- Randomly keep 6 R-CBF targets from the original 39 using seed `20260824`.
- Add 8 R-SSS targets randomly selected from `rq3/rsss_hunt` using seed
  `20260825`.
- Add 1 pSSA and 1 SSS target selected from `rq3/pair_hunt/targets`, outside
  the existing set and with no obstacle at `[9.0, 9.0, 9.0]`, using seed
  `20260825`.

Contents:
- `targets/`: selected stage-3 target JSON files.
- `results/`: matching stage-3 fuzz result JSON files.
- `manifest.json`: selected files, counts, the exact R-CBF sample, the added
  R-SSS sample, and the added pair-hunt pSSA/SSS targets.

Picker coverage:
- The original 32 cases include `random`, `random_seeded`, `cem`,
  `cem_seeded`, `bo`, and `bo_seeded`.
- The added R-SSS cases include `random`, `random_seeded`, `cem`, and
  `cem_seeded`.
- The added pair-hunt pSSA/SSS cases include `random`, `random_seeded`, `cem`,
  and `cem_seeded`.

Distribution:
- CBF: 6
- pSSA: 5
- R-CBF: 6
- R-SSA: 6
- R-SSS: 8
- SSA: 6
- SSS: 5
- Total: 42
