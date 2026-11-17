"""Prometheus metrics of the inference API, served at GET /metrics.

Each app gets its own registry, so several apps (e.g. in tests) never clash. Paths are labelled with the route
template rather than the raw URL, which keeps the number of time series bounded.
"""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)


class ServiceMetrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self):
        self.registry = r = CollectorRegistry()
        self.requests = Counter(
            "en_gu_http_requests_total", "HTTP requests by route and status", ["method", "path", "status"], registry=r
        )
        self.latency = Histogram(
            "en_gu_http_request_duration_seconds",
            "End-to-end request latency",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.in_progress = Gauge("en_gu_http_requests_in_progress", "Requests being handled", registry=r)
        self.translate_seconds = Histogram(
            "en_gu_translate_seconds",
            "Model time per /translate call",
            ["beam_size"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.sentences = Histogram(
            "en_gu_translate_sentences", "Sentences per /translate call", buckets=(1, 2, 3, 5, 10, 20, 40), registry=r
        )
        self.chars = Histogram(
            "en_gu_translate_input_chars",
            "Input characters per /translate call",
            buckets=(10, 25, 50, 100, 200, 500, 1000, 2000),
            registry=r,
        )
        self.feedback = Counter("en_gu_feedback_total", "Feedback received", ["rating", "with_correction"], registry=r)
        self.model_ready = Gauge("en_gu_model_ready", "1 once the model is loaded", registry=r)
        self.model_load_seconds = Gauge("en_gu_model_load_seconds", "Time it took to load the model", registry=r)
        self.model_info = Gauge(
            "en_gu_model_info",
            "The model being served (value is always 1)",
            ["version", "source", "device"],
            registry=r,
        )

    def model_loaded(self, translator):
        self.model_ready.set(1)
        self.model_load_seconds.set(translator.load_seconds)
        self.model_info.labels(translator.version, translator.source, translator.device).set(1)

    def observe_request(self, method, path, status, seconds):
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(seconds)

    def observe_translation(self, beam_size, seconds, n_sentences, n_chars):
        self.translate_seconds.labels(str(beam_size)).observe(seconds)
        self.sentences.observe(n_sentences)
        self.chars.observe(n_chars)

    def render(self):
        return generate_latest(self.registry)
