# Robustness Toolkit

Internal collection of adversarial-attack utilities and robustness metrics used
by the vision team for image classification models.

- `src/attacks.py` — PGD, FGSM and AutoAttack wrappers for image classifiers
- `src/metrics.py` — clean accuracy, robust accuracy and calibration metrics

This toolkit targets image classification models and does not contain NLP benchmark
code.
