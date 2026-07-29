import asyncio
import json
import unittest
from unittest.mock import Mock

from aiohttp import web
from loguru import logger
from prometheus_client import CollectorRegistry

from observability.loki import LokiLogSink
from observability.metrics import AgentMetrics


class LokiLogSinkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.response_status = 204

        async def receive_logs(request):
            self.requests.append(await request.json())
            return web.Response(status=self.response_status)

        app = web.Application()
        app.router.add_post("/loki/api/v1/push", receive_logs)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        socket = self.site._server.sockets[0]
        self.base_url = f"http://127.0.0.1:{socket.getsockname()[1]}"

        self.metrics = AgentMetrics(
            registry=CollectorRegistry(),
            collect_system=False,
        )

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_batches_structured_records_with_session_metadata(self):
        sink = LokiLogSink(
            base_url=self.base_url,
            metrics=self.metrics,
            batch_size=10,
            flush_interval=0.01,
        )
        await sink.start()
        handler_id = logger.add(sink, level="DEBUG", catch=False)
        try:
            async def log_for_session(session_id):
                with logger.contextualize(session_id=session_id):
                    logger.info(
                        "Session event",
                        outcome="completed",
                    )

            await asyncio.gather(
                log_for_session("session-a"),
                log_for_session("session-b"),
            )
        finally:
            logger.remove(handler_id)
            await sink.stop()

        self.assertEqual(len(self.requests), 1)
        stream = self.requests[0]["streams"][0]
        self.assertEqual(
            stream["stream"],
            {
                "service_name": "pipecat-local-agent",
                "level": "INFO",
            },
        )

        documents = [json.loads(value[1]) for value in stream["values"]]
        metadata = [value[2] for value in stream["values"]]
        self.assertEqual(
            {document["session_id"] for document in documents},
            {"session-a", "session-b"},
        )
        self.assertEqual(
            {document["outcome"] for document in documents},
            {"completed"},
        )
        self.assertEqual(
            {item["session_id"] for item in metadata},
            {"session-a", "session-b"},
        )
        self.assertEqual(
            self.metrics.registry.get_sample_value(
                "local_agent_log_records_sent_total"
            ),
            2,
        )
        self.assertEqual(
            self.metrics.registry.get_sample_value(
                "local_agent_loki_available"
            ),
            1,
        )

    async def test_failed_delivery_drops_batch_without_raising(self):
        self.response_status = 503
        sink = LokiLogSink(
            base_url=self.base_url,
            metrics=self.metrics,
            batch_size=1,
            flush_interval=0.01,
        )
        sink._warn_delivery_failure = Mock()
        await sink.start()
        handler_id = logger.add(sink, catch=False)
        try:
            with logger.contextualize(session_id="session-failure"):
                logger.error("Delivery should be dropped")
        finally:
            logger.remove(handler_id)
            await sink.stop()

        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            self.metrics.registry.get_sample_value(
                "local_agent_log_records_dropped_total",
                {"reason": "delivery_failed"},
            ),
            1,
        )
        self.assertEqual(
            self.metrics.registry.get_sample_value(
                "local_agent_loki_available"
            ),
            0,
        )
        sink._warn_delivery_failure.assert_called_once()

    def test_rejects_invalid_buffer_configuration(self):
        with self.assertRaisesRegex(ValueError, "queue_capacity"):
            LokiLogSink(base_url=self.base_url, queue_capacity=0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            LokiLogSink(base_url=self.base_url, batch_size=0)
        with self.assertRaisesRegex(ValueError, "flush_interval"):
            LokiLogSink(base_url=self.base_url, flush_interval=0)
        with self.assertRaisesRegex(ValueError, "request_timeout"):
            LokiLogSink(base_url=self.base_url, request_timeout=0)


if __name__ == "__main__":
    unittest.main()
