import unittest
from datetime import datetime
from unittest.mock import patch

from app import build_research_brief


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 8, 30, 9, 15, 0, tzinfo=tz)


class ResearchBriefRuntimeContextTests(unittest.TestCase):
    def test_brief_contains_runtime_product_freshness_context(self):
        with patch("app.datetime", FixedDatetime):
            brief = build_research_brief(
                "Recommend a current consumer product.",
                "comprehensive",
                "general",
            )

        self.assertIn("AS_OF_DATE: 2026-08-30", brief)
        self.assertIn("SOURCE_WINDOW_START: 2025-08-30", brief)
        self.assertIn("MARKET: United States", brief)
        self.assertIn(
            "PRODUCT_REQUIREMENT: Latest model/version currently available for purchase",
            brief,
        )


if __name__ == "__main__":
    unittest.main()
