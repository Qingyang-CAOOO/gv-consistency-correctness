# External repositories

`external/rankalign/` is created at runtime by `scripts/bootstrap_rankalign.sh` and is intentionally gitignored.

Upstream:
- https://github.com/juand-r/rankalign

Do not edit upstream files in-place for project experiments. Add wrappers/adapters under `src/gvsmc/` so that the baseline and the extension remain cleanly separable.
