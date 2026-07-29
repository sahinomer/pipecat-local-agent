"""Translate Pipecat MetricsFrame data into Prometheus instruments."""

from pipecat.frames.frames import ErrorFrame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from observability.metrics import METRICS, AgentMetrics, normalize_processor_name


class PrometheusMetricsObserver(BaseObserver):
    """Aggregate Pipecat performance, usage, turn, and error frames."""

    def __init__(self, *, metrics: AgentMetrics = METRICS):
        super().__init__()
        self._metrics = metrics
        self._frames_seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if not isinstance(frame, (MetricsFrame, ErrorFrame)):
            return
        if frame.id in self._frames_seen:
            return
        self._frames_seen.add(frame.id)

        if isinstance(frame, ErrorFrame):
            self._metrics.pipeline_errors_total.labels(
                processor=normalize_processor_name(str(data.source))
            ).inc()
            return

        for metric in frame.data:
            if isinstance(metric, TTFBMetricsData):
                self._observe_duration(metric, "ttfb", metric.value)
            elif isinstance(metric, TTFAMetricsData):
                self._observe_duration(metric, "ttfa", metric.ttfa)
                self._observe_duration(
                    metric,
                    "tts_leading_silence",
                    metric.leading_silence,
                )
            elif isinstance(metric, ProcessingMetricsData):
                self._observe_duration(metric, "processing", metric.value)
            elif isinstance(metric, TextAggregationMetricsData):
                self._observe_duration(metric, "text_aggregation", metric.value)
            elif isinstance(metric, LLMUsageMetricsData):
                self._record_llm_usage(metric)
            elif isinstance(metric, TTSUsageMetricsData):
                self._metrics.tts_characters_total.labels(
                    processor=normalize_processor_name(metric.processor),
                    model=metric.model or "",
                ).inc(metric.value)
            elif isinstance(metric, TurnMetricsData):
                processor = normalize_processor_name(metric.processor)
                result = "complete" if metric.is_complete else "incomplete"
                self._metrics.turn_predictions_total.labels(
                    processor=processor,
                    result=result,
                ).inc()
                self._metrics.observe_pipeline_duration(
                    processor=processor,
                    model=metric.model,
                    metric="turn_processing",
                    value=metric.e2e_processing_time_ms / 1000.0,
                )

    def _observe_duration(self, metric, name: str, value: float) -> None:
        self._metrics.observe_pipeline_duration(
            processor=metric.processor,
            model=metric.model,
            metric=name,
            value=value,
        )

    def _record_llm_usage(self, metric: LLMUsageMetricsData) -> None:
        processor = normalize_processor_name(metric.processor)
        model = metric.model or ""
        usage = metric.value
        values = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "cache_read": usage.cache_read_input_tokens,
            "cache_creation": usage.cache_creation_input_tokens,
            "reasoning": usage.reasoning_tokens,
        }
        for token_type, value in values.items():
            if value is not None:
                self._metrics.llm_tokens_total.labels(
                    processor=processor,
                    model=model,
                    type=token_type,
                ).inc(value)
