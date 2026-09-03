# Best Final Hard Validation Summary

Selection rule: within each group, choose the run with the lowest last-epoch `0_mse_meters` value on hard validation (`subset=val`). Test values are the final single test-set dots. Medium validation is also the last validation point from the selected run.

The normalized `c-const` tuning group is intentionally excluded.

| group | hard_val | hard_test | medium_val | medium_test | epoch | lr | experiment | hash |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `scratch` | 353.87 | 269.54 | 124.67 | 185.90 | 100 | 6e-5 | `scratch_1000_100_6e-5` | `c97e233c41924970a85075a5` |
| `b-const->a` | 331.28 | 217.89 | 93.23 | 161.25 | 40 | 6e-5 | `tune_bp-const_1000_40_6e-5` | `fa93e4d06bc745c1ae37bf71` |
| `b-unconst->a` | 302.34 | 306.81 | 191.26 | 206.46 | 10 | 6e-5 | `tune_bp-unconst_1000_10_6e-5` | `ccaa7e35a5e8483ca3634b28` |
| `c-const->a` | 316.14 | 372.29 | 176.39 | 160.77 | 10 | 2e-4 | `tune_c-const_1000_10_2e-4` | `ffecd51655ae401c94104385` |
| `c-unconst->a` | 293.68 | 302.28 | 197.60 | 148.77 | 10 | 2e-4 | `tune_c-unconst_1000_10_2e-4` | `7703e8d468954d12bf009065` |
