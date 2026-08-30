import unittest

from agent_files import (
    CAP_NOTICE,
    build_agent_file_state,
    cap_stage_artifact,
    get_agent_file_content,
)
from config import (
    ANALYSIS_ARTIFACT_MAX_CHARS,
    ANALYSIS_ARTIFACT_MAX_LINES,
    RESEARCH_ARTIFACT_MAX_CHARS,
    RESEARCH_ARTIFACT_MAX_LINES,
)


class AgentFileStateTests(unittest.TestCase):
    def test_caps_intermediate_artifacts_by_lines_and_characters(self):
        research = "\n".join(f"evidence {index} " + "x" * 300 for index in range(140))
        analysis = "\n".join(f"finding {index} " + "y" * 300 for index in range(100))

        capped_research = cap_stage_artifact("research.md", research)
        capped_analysis = cap_stage_artifact("/analysis.md", analysis)

        self.assertLessEqual(len(capped_research.splitlines()), RESEARCH_ARTIFACT_MAX_LINES)
        self.assertLessEqual(len(capped_research), RESEARCH_ARTIFACT_MAX_CHARS)
        self.assertLessEqual(len(capped_analysis.splitlines()), ANALYSIS_ARTIFACT_MAX_LINES)
        self.assertLessEqual(len(capped_analysis), ANALYSIS_ARTIFACT_MAX_CHARS)
        self.assertTrue(capped_research.endswith(CAP_NOTICE))
        self.assertTrue(capped_analysis.endswith(CAP_NOTICE))

    def test_builds_canonical_file_data_for_every_shared_path(self):
        files = build_agent_file_state(
            "original brief",
            {
                "research.md": "research",
                "analysis.md": "analysis",
                "final.md": "final",
            },
        )

        self.assertEqual(
            set(files),
            {"/brief.md", "/research.md", "/analysis.md", "/final.md"},
        )
        self.assertEqual(files["/brief.md"]["content"], "original brief")
        self.assertEqual(files["/research.md"]["content"], "research")
        self.assertEqual(files["/brief.md"]["encoding"], "utf-8")

    def test_extracts_current_file_data_and_legacy_strings(self):
        current = build_agent_file_state(
            "brief",
            {"research.md": "current research"},
        )
        self.assertEqual(
            get_agent_file_content(current, "research.md"),
            "current research",
        )
        self.assertEqual(
            get_agent_file_content({"research.md": "legacy research"}, "research.md"),
            "legacy research",
        )


if __name__ == "__main__":
    unittest.main()
