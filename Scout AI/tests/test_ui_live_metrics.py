import unittest
from pathlib import Path


UI_PATH = Path(__file__).resolve().parents[1] / "templates" / "index.html"


class LiveMetricsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = UI_PATH.read_text(encoding="utf-8")

    def test_refresh_starts_with_a_clean_dashboard(self):
        self.assertIn("resetDashboardState();", self.html)
        self.assertNotIn('loadTask("demo-enterprise-ai");', self.html)
        self.assertIn('id="research-results-section" class="hidden', self.html)
        self.assertIn("style=\"width: 0%;\"", self.html)

    def test_authoritative_web_is_the_default_research_focus(self):
        self.assertIn(
            '<option value="general" selected>🌐 All Authoritative Web</option>',
            self.html,
        )
        self.assertNotIn('<option value="finance" selected>', self.html)

    def test_empty_activity_metrics_hide_their_labels(self):
        for field_id in (
            "activity-model-call",
            "activity-attempt",
            "activity-prompt-tokens",
            "activity-last-seen",
        ):
            self.assertIn(f'id="{field_id}" class="hidden"></span>', self.html)
        self.assertIn("field.classList.toggle('hidden', !value);", self.html)
        self.assertNotIn("'Model call —'", self.html)
        self.assertNotIn("'Attempt —'", self.html)
        self.assertNotIn("'Prompt tokens —'", self.html)
        self.assertNotIn("Last activity: waiting", self.html)

    def test_main_content_has_no_header_or_footer_spacer(self):
        self.assertIn(
            '<main class="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-0 pb-0 space-y-0">',
            self.html,
        )
        self.assertNotIn('<main class="flex-1', self.html)

    def test_charts_have_no_fallback_placeholder_metrics(self):
        for placeholder in (
            "Primary Subject",
            "Secondary / Peer",
            "SEC Filings & Earnings",
            "Technical Papers & HELM",
            "[42, 28, 18, 12]",
        ):
            self.assertNotIn(placeholder, self.html)
        self.assertIn("renderDonutChart(data.metrics.evidence_rigor)", self.html)
        self.assertIn("rawValues: s.raw_values || []", self.html)

    def test_active_agent_paths_use_green_highlight(self):
        self.assertIn(".active-connector", self.html)
        self.assertIn("stroke: #22c55e !important", self.html)
        self.assertIn("pathMap", self.html)
        self.assertIn("replace('-agent', '')", self.html)

    def test_curated_query_starters_have_high_contrast_styles(self):
        self.assertIn("border border-cyan-500/50 bg-slate-950/85", self.html)
        self.assertIn("text-xs font-extrabold text-cyan-200", self.html)
        for color in ("teal", "cyan", "indigo", "emerald"):
            self.assertIn(f"bg-{color}-950/90", self.html)
            self.assertIn(f"border-{color}-500/70", self.html)
            self.assertIn(f"focus:ring-{color}-400", self.html)

    def test_topology_nodes_use_distinct_high_contrast_palettes(self):
        expected_gradients = {
            "grad-orchestrator": "#fb923c",
            "grad-research": "#22d3ee",
            "grad-analysis": "#a78bfa",
            "grad-writer": "#f472b6",
            "grad-filesystem": "#facc15",
        }
        for gradient_id, color in expected_gradients.items():
            self.assertIn(f'id="{gradient_id}"', self.html)
            self.assertIn(f'stop-color="{color}"', self.html)
        for glow in ("glow-orange", "glow-cyan", "glow-violet", "glow-pink", "glow-amber"):
            self.assertIn(f'id="{glow}"', self.html)

    def test_topology_titles_are_positioned_above_nodes(self):
        self.assertIn('y="-48" text-anchor="middle"', self.html)
        self.assertEqual(self.html.count('y="-45" text-anchor="middle"'), 3)
        self.assertEqual(self.html.count('y="-31" text-anchor="middle"'), 3)
        self.assertGreaterEqual(self.html.count('paint-order="stroke"'), 8)

    def test_specialist_row_is_separated_from_orchestrator(self):
        self.assertIn('id="node-orchestrator" transform="translate(450, 65)"', self.html)
        for x_position in (180, 450, 720):
            self.assertIn(f'transform="translate({x_position}, 205)"', self.html)
        self.assertIn('id="node-filesystem" transform="translate(450, 340)"', self.html)
        self.assertIn('id="path-orch-analysis" d="M 450 65 L 450 205"', self.html)
        self.assertIn('id="path-analysis-fs" d="M 450 205 L 450 340"', self.html)
        self.assertIn('viewBox="0 0 900 380"', self.html)


if __name__ == "__main__":
    unittest.main()
