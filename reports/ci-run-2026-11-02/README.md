# Archived outputs from the first neural CI run

This folder preserves outputs from the end-to-end GitHub Actions run completed on 2 November 2026. It records the code and environment used at that time; it is not the latest experiment result.

- Commit: `c54c5c5f94f220e4097d3b9d2be0b51da4675e85`
- Workflow run: [Neural framework checks](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/41379771316)
- Environment: Python 3.11, CPU PyTorch 2.15.0 and TensorFlow 2.21.0
- Training settings: seed 42, up to 100 epochs, patience 12 and batch size 32

In that run, the neural models did not outperform the conventional regressors. The saved files are useful for tracing that earlier experiment, but they are not evidence of clinical performance. The workflow artifact also included training histories, test predictions, scaler details and trained weights. GitHub artifact retention was configured for 14 days, so that original artifact may no longer be available; the files committed in this folder are the durable record.
