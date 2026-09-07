Std = sample std over the eval seeds (ddof = 1); n = seeds. Arms with prefix 'fx_'.

| cell | arm | mode | mean ± std (n) | per-seed | block ms (amortized) | call ms (batch) | episode s |
|---|---|---|---|---|---|---|---|
| pusht | fx_fl192 | CEM | 59.3 ± 5.0 (3) | 0: 64, 1: 60, 42: 54 | 72.3 | 2433.6 | 0.51 ± 0.21 |
| pusht | fx_fl192 | reactive (goal-conditioned) | 66.0 ± 8.0 (3) | 0: 74, 1: 66, 42: 58 | 6.0 | 179.4 | 0.06 ± 0.03 |
| pusht | fx_fl192 | gradient | 61.3 ± 6.4 (3) | 0: 64, 1: 66, 42: 54 | 734.3 | 20605.4 | 4.15 ± 2.58 |
| pusht | fx_fl192 | gradient-TR | 62.7 ± 7.6 (3) | 0: 68, 1: 66, 42: 54 | 751.0 | 20315.0 | 4.09 ± 2.64 |
| pusht | fx_fl192 | best-of-K | 62.0 ± 6.9 (3) | 0: 66, 1: 66, 42: 54 | 30.9 | 1023.5 | 0.23 ± 0.10 |
| pusht | fx_fl192 | random-candidates | 38.0 ± 6.0 (3) | 0: 44, 1: 38, 42: 32 | 15.2 | 653.2 | 0.15 ± 0.04 |
| pusht | fx_fl192 | SteerMPC | 48.0 ± n/a (1) | 42: 48 | 9147.1 | 341237.9 | 68.27 ± 24.63 |
| pusht | fx_flsig192 | CEM | 76.7 ± 2.3 (3) | 0: 78, 1: 74, 42: 78 | 82.4 | 2309.3 | 0.48 ± 0.23 |
| pusht | fx_flsig192 | reactive (goal-conditioned) | 74.7 ± 4.2 (3) | 0: 76, 1: 78, 42: 70 | 6.9 | 182.6 | 0.06 ± 0.04 |
| pusht | fx_flsig192 | gradient | 72.0 ± 8.7 (3) | 0: 78, 1: 76, 42: 62 | 904.1 | 21093.9 | 4.24 ± 3.04 |
| pusht | fx_flsig192 | gradient-TR | 76.0 ± 5.3 (3) | 0: 78, 1: 80, 42: 70 | 967.5 | 20871.5 | 4.20 ± 3.18 |
| pusht | fx_flsig192 | best-of-K | 76.7 ± 4.2 (3) | 0: 80, 1: 78, 42: 72 | 35.3 | 981.8 | 0.22 ± 0.11 |
| pusht | fx_flsig192 | random-candidates | 34.7 ± 2.3 (3) | 0: 36, 1: 32, 42: 36 | 15.7 | 674.2 | 0.16 ± 0.04 |
| pusht | fx_nm192 | CEM | 80.7 ± 6.4 (3) | 0: 88, 1: 76, 42: 78 | 29.4 | 720.9 | 0.17 ± 0.10 |
| pusht | fx_nm192 | reactive (goal-conditioned) | 72.7 ± 5.0 (3) | 0: 78, 1: 68, 42: 72 | 4.9 | 143.1 | 0.05 ± 0.03 |
| pusht | fx_nm192 | gradient | 83.3 ± 5.8 (3) | 0: 90, 1: 80, 42: 80 | 193.3 | 2861.5 | 0.60 ± 0.60 |
| pusht | fx_nm192 | gradient-TR | 78.7 ± 2.3 (3) | 0: 80, 1: 76, 42: 80 | 150.9 | 2859.6 | 0.60 ± 0.52 |
| pusht | fx_nm192 | best-of-K | 78.0 ± 2.0 (3) | 0: 80, 1: 78, 42: 76 | 20.4 | 550.9 | 0.13 ± 0.07 |
| pusht | fx_nm192 | random-candidates | 37.3 ± 5.8 (3) | 0: 44, 1: 34, 42: 34 | 4.2 | 179.6 | 0.06 ± 0.02 |
| pusht | fx_nm192 | SteerMPC | 80.0 ± 4.0 (3) | 0: 84, 1: 76, 42: 80 | 1439.0 | 39497.1 | 7.93 ± 3.40 |
| pusht | fx_sig192 | CEM | 83.3 ± 8.3 (3) | 0: 90, 1: 86, 42: 74 | 34.3 | 717.8 | 0.17 ± 0.11 |
| pusht | fx_sig192 | reactive (goal-conditioned) | 74.0 ± 9.2 (3) | 0: 84, 1: 72, 42: 66 | 5.9 | 149.5 | 0.06 ± 0.04 |
| pusht | fx_sig192 | gradient | 86.0 ± 9.2 (3) | 0: 96, 1: 78, 42: 84 | 339.1 | 2976.9 | 0.62 ± 0.83 |
| pusht | fx_sig192 | gradient-TR | 80.7 ± 9.0 (3) | 0: 90, 1: 80, 42: 72 | 189.1 | 2943.4 | 0.61 ± 0.59 |
| pusht | fx_sig192 | best-of-K | 78.0 ± 10.0 (3) | 0: 88, 1: 78, 42: 68 | 22.5 | 573.9 | 0.14 ± 0.08 |
| pusht | fx_sig192 | random-candidates | 40.7 ± 2.3 (3) | 0: 42, 1: 38, 42: 42 | 4.4 | 188.9 | 0.06 ± 0.02 |
| pusht | fx_sig192 | SteerMPC | 82.7 ± 8.1 (3) | 0: 90, 1: 84, 42: 74 | 1321.3 | 35734.4 | 7.17 ± 2.77 |
| toolhang | fx_mfl192 | reactive TC (goal-match OR task; old protocol) | 82.0 ± 0.0 (3) | 0: 82, 1: 82, 42: 82 | 7.8 | 117.2 | 1.31 ± 1.47 |
| toolhang | fx_mfl192 | TC: env success only, start to finish | 80.0 ± 4.0 (3) | 0: 76, 1: 84, 42: 80 | 8.4 | 128.8 | 1.56 ± 1.81 |
| toolhang | fx_mflsig192 | reactive TC (goal-match OR task; old protocol) | 86.0 ± 8.0 (3) | 0: 78, 1: 94, 42: 86 | 12.2 | 121.1 | 1.25 ± 1.71 |
| toolhang | fx_mflsig192 | TC: env success only, start to finish | 84.7 ± 9.5 (3) | 0: 74, 1: 92, 42: 88 | 10.7 | 117.6 | 1.23 ± 1.63 |
| toolhang | fx_mnm192 | reactive TC (goal-match OR task; old protocol) | 84.7 ± 4.2 (3) | 0: 80, 1: 88, 42: 86 | 5.8 | 80.9 | 1.09 ± 1.25 |
| toolhang | fx_mnm192 | TC: env success only, start to finish | 84.0 ± 4.0 (3) | 0: 80, 1: 88, 42: 84 | 5.6 | 78.4 | 1.03 ± 1.17 |
| toolhang | fx_msig192 | reactive TC (goal-match OR task; old protocol) | 82.7 ± 7.6 (3) | 0: 88, 1: 74, 42: 86 | 5.5 | 80.8 | 1.05 ± 1.13 |
| toolhang | fx_msig192 | TC: env success only, start to finish | 80.7 ± 3.1 (3) | 0: 80, 1: 78, 42: 84 | 4.9 | 80.0 | 1.05 ± 1.11 |
