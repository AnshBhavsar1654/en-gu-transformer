# Cache decoder attention projections

The decoder previously recalculated every token in the generated prefix on each step. The candidate keeps projected
keys and values for the generated tokens and encoder memory. A beam reorder also reorders its cached history. This
keeps the original checkpoint and training format usable. Cache activation is an explicit checkpoint inference option.

### Local measurement

One CPU thread, beam 5, the first 60 frozen test pairs for quality, and the first 30 for single-sentence latency. RSS
is process memory after loading the model. These are development measurements for the existing v1 weights.

| Inference path | p50 | p95 | RSS after load | BLEU | chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-prefix decoding | 1,014 ms | 4,302 ms | 565 MiB | 22.52 | 46.98 |
| KV cache | 361 ms | 847 ms | 564 MiB | 22.52 | 46.98 |
| KV cache with dynamic int8 linears | 247 ms | 618 ms | 533 MiB | 23.80 | 47.46 |

The cache trial reduced p95 by about 80% on this sample. The int8 trial changed predictions and remains a benchmark
option. Reproduce the measurement against a trusted self-contained checkpoint.

```bash
python -m en_gu.benchmark best.pt --variant baseline --limit 60 --latency-sentences 30
python -m en_gu.benchmark best.pt --variant cached --limit 60 --latency-sentences 30 --candidate artifacts/cached.pt
python -m en_gu.benchmark best.pt --variant cached-int8 --limit 60 --latency-sentences 30
```

Use a checkpoint with an isolated evaluation fingerprint for promotion. The existing v1 weights predate the frozen-set
isolation and these subset scores are insufficient for release. Registry tests exercise two promotions and one
rejection with controlled scores. Production gate and rollout evidence are pending.
