"""Publish scheduled monitoring results as JSON and Prometheus/Influx samples."""

import json
import os
import tempfile
from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Gauge, generate_latest


@dataclass
class Sample:
    measurement: str
    field: str
    value: float
    labels: dict = field(default_factory=dict)

    @property
    def metric_name(self):
        return f"en_gu_{self.measurement}_{self.field}"


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(content)
    try:
        os.chmod(temp.name, 0o644)
        os.replace(temp.name, path)
    finally:
        if os.path.exists(temp.name):
            os.unlink(temp.name)


def escape_tag(value):
    return value.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def write_results(prefix, outcome, samples):
    registry = CollectorRegistry()
    gauges = {}
    lines = []
    for sample in samples:
        if sample.metric_name not in gauges:
            gauges[sample.metric_name] = Gauge(
                sample.metric_name, sample.metric_name, tuple(sample.labels), registry=registry
            )
        gauge = gauges[sample.metric_name]
        if sample.labels:
            gauge = gauge.labels(**sample.labels)
        gauge.set(sample.value)
        tags = "".join(f",{name}={escape_tag(value)}" for name, value in sample.labels.items())
        lines.append(f"en_gu_{sample.measurement}{tags} {sample.field}={float(sample.value)}")
    atomic_write(prefix.with_suffix(".json"), json.dumps(outcome, indent=2, allow_nan=False).encode() + b"\n")
    atomic_write(prefix.with_suffix(".prom"), generate_latest(registry))
    atomic_write(prefix.with_suffix(".influx"), ("\n".join(lines) + "\n").encode())
