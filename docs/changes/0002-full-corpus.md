# Train on the full parallel corpus

The original run used the first one million pairs. The full-corpus configuration removes that cap so the same pipeline
can train on the remainder. The existing validation and test sentences stay frozen, and the preparation stage excludes
their English sources from training.

The hypothesis is that a larger training set improves held-out chrF++ without degrading a length bucket. Score any
exported model against the current champion using the full frozen test set, retain the gate report, and compare the
tagged shadow and canary results before promotion.

The full training run and production rollout are pending. The workspace has CPU-only PyTorch, and the repository has
no configured Cloud Run release target. This record contains no before-and-after quality claim.
