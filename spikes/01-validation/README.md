# Spike 1: Model Validation Back-Test

Back-tests the wcsim match model (PRD §5.5) against the 2018 and 2022 FIFA World
Cups using pre-tournament Elo and FIFA ratings. Compares Elo / FIFA / Blend
rating modes under 90-min and post-ET scoring conventions, scoring predictions
with RPS (stored as `results/brier.json` for historical reasons) and producing
an Elo-mode calibration plot at `results/calibration.png`.

Conclusion paragraph filled in by Task 11.

## Reproduce

```bash
pip install -r requirements.txt
python validate.py
```
