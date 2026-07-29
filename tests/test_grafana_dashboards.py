import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GrafanaDashboardTests(unittest.TestCase):
    def test_logs_dashboard_has_safe_default_filters(self):
        dashboard_path = (
            ROOT
            / "observability"
            / "grafana"
            / "dashboards"
            / "local-agent-logs.json"
        )
        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))

        variables = {
            variable["name"]: variable
            for variable in dashboard["templating"]["list"]
        }
        level = variables["level"]
        self.assertTrue(level["includeAll"])
        self.assertEqual(level["allValue"], ".*")

        targets = [
            target
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        ]
        expressions = [target["expr"] for target in targets]
        logs_expression = next(
            expression
            for expression in expressions
            if expression.startswith("{service_name=")
        )

        self.assertIn("${level:raw}", logs_expression)
        self.assertIn("${session_id:raw}", logs_expression)
        self.assertNotIn(":regex}", json.dumps(dashboard))
        self.assertNotIn("$__auto", json.dumps(dashboard))
        self.assertEqual(
            sum("$__range" in expression for expression in expressions),
            3,
        )


if __name__ == "__main__":
    unittest.main()
