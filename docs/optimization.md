# Controlled optimisation experiment

After accuracy and the baseline are established, change **one** thing and remeasure with the identical protocol.

Built-in knob: `LAB_CHANNELS_LAST=1` (channels-last memory format). Precision stays FP32, TF32 stays off.

```bash
# baseline summary already exists: artifacts/benchmark/<base>/summary.json
LAB_CHANNELS_LAST=1 make up-both
make benchmark EXPERIMENT=variant-channels-last
make validate                      # output checks must still pass with the change
python -m benchmarks.compare --base artifacts/benchmark/<base>/summary.json \
  --variant artifacts/benchmark/variant-channels-last/summary.json \
  --change LAB_CHANNELS_LAST=1 --variant-validation artifacts/validation/validation-cuda-<run>.json \
  --out artifacts/benchmark/compare-channels-last.json
```
Report improvements **and** regressions per cell; differences inside the run-to-run variation (cv in the comparison file) are not evidence. No speedup is assumed.
Other candidates (precision, batch scheduling, preprocessing) are separate, later experiments with their own correctness gate.
