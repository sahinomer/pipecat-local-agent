import unittest
import urllib.request
from unittest.mock import Mock

from pipecat.frames.frames import ErrorFrame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from prometheus_client import CollectorRegistry

from observability.metrics import (
    AgentMetrics,
    MetricsHttpServer,
    normalize_processor_name,
)
from observability.pipecat_observer import PrometheusMetricsObserver


class PrometheusMetricsObserverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = CollectorRegistry()
        self.metrics = AgentMetrics(
            registry=self.registry,
            collect_system=False,
        )
        self.observer = PrometheusMetricsObserver(metrics=self.metrics)

    @staticmethod
    def _pushed(frame):
        return FramePushed(
            source=Mock(__str__=Mock(return_value="SourceProcessor#42")),
            destination=Mock(),
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=1,
        )

    async def test_records_and_deduplicates_pipecat_metrics(self):
        frame = MetricsFrame(
            data=[
                ProcessingMetricsData(
                    processor="OpenAILLMService#17",
                    model="local-model",
                    value=1.5,
                ),
                TTFAMetricsData(
                    processor="XTTSService#18",
                    model="xtts",
                    ttfa=0.4,
                    ttfb=0.25,
                    leading_silence=0.15,
                ),
                LLMUsageMetricsData(
                    processor="OpenAILLMService#17",
                    model="local-model",
                    value=LLMTokenUsage(
                        prompt_tokens=100,
                        completion_tokens=25,
                        total_tokens=125,
                    ),
                ),
                TTSUsageMetricsData(
                    processor="XTTSService#18",
                    model="xtts",
                    value=42,
                ),
                TurnMetricsData(
                    processor="LocalSmartTurnAnalyzerV3#3",
                    is_complete=True,
                    probability=0.9,
                    e2e_processing_time_ms=75,
                ),
            ]
        )

        pushed = self._pushed(frame)
        await self.observer.on_push_frame(pushed)
        await self.observer.on_push_frame(pushed)

        self.assertEqual(
            self.registry.get_sample_value(
                "local_agent_pipeline_duration_seconds_count",
                {
                    "processor": "OpenAILLMService",
                    "model": "local-model",
                    "metric": "processing",
                },
            ),
            1,
        )
        self.assertEqual(
            self.registry.get_sample_value(
                "local_agent_llm_tokens_total",
                {
                    "processor": "OpenAILLMService",
                    "model": "local-model",
                    "type": "completion",
                },
            ),
            25,
        )
        self.assertEqual(
            self.registry.get_sample_value(
                "local_agent_tts_characters_total",
                {
                    "processor": "XTTSService",
                    "model": "xtts",
                },
            ),
            42,
        )
        self.assertEqual(
            self.registry.get_sample_value(
                "local_agent_turn_predictions_total",
                {
                    "processor": "LocalSmartTurnAnalyzerV3",
                    "result": "complete",
                },
            ),
            1,
        )

    async def test_records_error_frame_once(self):
        frame = ErrorFrame(error="failure")
        pushed = self._pushed(frame)

        await self.observer.on_push_frame(pushed)
        await self.observer.on_push_frame(pushed)

        self.assertEqual(
            self.registry.get_sample_value(
                "local_agent_pipeline_errors_total",
                {"processor": "SourceProcessor"},
            ),
            1,
        )


class ProcessorNameTests(unittest.TestCase):
    def test_only_pipecat_instance_suffix_is_removed(self):
        self.assertEqual(
            normalize_processor_name("OpenAILLMService#123"),
            "OpenAILLMService",
        )
        self.assertEqual(
            normalize_processor_name("processor#custom"),
            "processor#custom",
        )


class MetricsHttpServerTests(unittest.TestCase):
    def test_serves_the_configured_registry(self):
        metrics = AgentMetrics(
            registry=CollectorRegistry(),
            collect_system=False,
        )
        server = MetricsHttpServer(
            host="127.0.0.1",
            port=0,
            registry=metrics.registry,
        )

        server.start()
        try:
            port = server._server.server_port
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics",
                timeout=5,
            ) as response:
                body = response.read().decode("utf-8")
        finally:
            server.stop()

        self.assertIn("local_agent_connections_active", body)


if __name__ == "__main__":
    unittest.main()
