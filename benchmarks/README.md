# Benchmark artifacts

```
benchmarks/
    results/    JSONL from benchmark runs
    reports/    text reports
```

Both are ignored by git. Real-model runs can produce large outputs and are the
user's data, not the repository's; committing them would also invite reading a
stale number as a current claim.

**No benchmark result is committed here, and none should be.** A number in this
repository that nobody can reproduce from the recorded fingerprint is worse than
no number.

Run one with:

```
python -m open_context_eval benchmark --out benchmarks/results/run.jsonl
```
