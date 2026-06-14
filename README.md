# hdmap-model-bench

Model integrations for HD map construction experiments.

## Models

### MapTR

See [models/maptr/README.md](models/maptr/README.md).

Current policy:

- use official MapTR configs directly
- keep checkpoints under the workspace-level `checkpoints/maptr/`, outside this repo
- use `models/maptr/predictor.py` only for raw inference
- use notebooks for inspection and visualization while the workflow is still changing
