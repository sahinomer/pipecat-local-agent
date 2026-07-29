import re
import unittest
from unittest.mock import Mock, patch

from loguru import logger

import server


class ServerSessionContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_assigns_session_id_to_logs_and_pipeline_session(self):
        records = []
        handler_id = logger.add(
            lambda message: records.append(message.record.copy()),
            filter=lambda record: record["message"] == "inside agent session",
        )
        captured_session_id = None

        async def fake_run_agent_session(websocket, *, session_id):
            nonlocal captured_session_id
            captured_session_id = session_id
            logger.info("inside agent session")

        try:
            with patch.object(
                server,
                "run_agent_session",
                side_effect=fake_run_agent_session,
            ) as run_agent_session:
                websocket = Mock()
                await server.websocket_endpoint(websocket)
        finally:
            logger.remove(handler_id)

        self.assertIsNotNone(captured_session_id)
        self.assertRegex(captured_session_id, re.compile(r"^[0-9a-f]{32}$"))
        run_agent_session.assert_awaited_once_with(
            websocket,
            session_id=captured_session_id,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["extra"]["session_id"], captured_session_id)


if __name__ == "__main__":
    unittest.main()
