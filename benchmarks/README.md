# Benchmark artifacts

```
benchmarks/
    results/    JSONL from benchmark runs
    reports/    text reports
```

Local runs are ignored by git. Real-model runs can produce large outputs and are
the user's data, not the repository's; committing them would also invite reading
a stale number as a current claim.

One exception is committed deliberately:
`results/v4-adversarial-8b.jsonl` is the complete run behind the public charts
and benchmark claims. Keeping that exact input lets anyone audit and redraw the
published result without spending tokens or needing a credential. No other run
artifact should be committed or included in a source distribution.

Run one with:

```
python -m open_context_eval benchmark --out benchmarks/results/run.jsonl
```
