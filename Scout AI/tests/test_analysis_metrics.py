import unittest

from app import extract_matrix_from_analysis, extract_metrics_from_text


CAR_ANALYSIS = """Original research question: Which car should I buy?

Decision summary
- Toyota Camry Hybrid is the balanced choice.

Comparison table
| Vehicle | Fuel Efficiency | Key Strength | Key Weakness |
|---|---|---|---|
| Toyota Camry Hybrid | 47 MPG | Balanced efficiency and resale | No complete TCO data |
| Tesla Model 3 | 117 MPGe | Lowest operating cost | High depreciation |

Risks and gaps
- Insurance costs are missing for all candidates. [Evidence IDs: 8, 9]
- Resale estimates differ by trim. [Evidence ID: 10]

Recommended actions
- Obtain dealer and insurance quotes. [Evidence IDs: 8, 9]
- Check local charging costs before selecting an EV. [Evidence ID: 6]
"""


STRUCTURED_ANALYSIS = """Decision summary
- A grounded decision.

Strategic SWOT & Horizon Matrix
Strengths
- Strong verified outcome. [Evidence ID: 1]
Weaknesses
- Limited sample. [Evidence ID: 2]
Opportunities
- Validate the leading option. [Evidence ID: 3]
Threats
- Input prices may change. [Evidence ID: 4]
Decision horizons
- Near term (0-6 months): Collect current quotes. [Evidence ID: 5]
- Medium term (6-24 months): Recheck depreciation. [Evidence ID: 6]
- Long term (24+ months): Review replacement cost. [Evidence ID: 7]
"""


METRIC_ANALYSIS = """Decision summary
- Toyota Camry Hybrid is the balanced choice.

Finding matrix
| Finding | Evidence IDs | Fact/Inference/Assumption | Decision implication |
|---|---|---|---|
| Camry returns 47 MPG | 1 | Fact | Efficient operation |
| Insurance varies | 2, 3 | Assumption | Confirm local pricing |

Comparison table
| Vehicle | Fuel Efficiency | Resale Score | TCO (5-Year) | Key Strength | Key Weakness |
|---|---|---|---|---|---|
| Toyota Camry Hybrid | 47 MPG | 55% | N/A | Balanced choice | TCO gap |
| Tesla Model 3 | 117 MPGe | 50% | $16,000 | Highest efficiency | Depreciation |

Risks and gaps
- Insurance costs vary. [Evidence ID: 2]
- Trim-level resale values differ. [Evidence ID: 3]

Recommended actions
- Obtain current quotes. [Evidence IDs: 1, 2]
"""


METRIC_RESEARCH = """Evidence Ledger
| ID | Claim or metric | Date/scope | Source URL | Source type | Confidence |
|---|---|---|---|---|---|
| 1 | EPA efficiency result | 2026 | https://www.epa.gov/example | Government | High |
| 2 | Independent vehicle test | 2026 | https://www.consumerreports.org/example | Consumer Reports | High |
| 3 | Resale estimate | 2026 | https://www.kbb.com/example | Industry publication | Medium |
"""


METRIC_FINAL = """The recommendation is supported by current evidence.

Sources:
- https://www.epa.gov/example
- https://www.consumerreports.org/example
"""


class AnalysisMatrixExtractionTests(unittest.TestCase):
    def test_existing_analysis_uses_comparison_and_decision_sections(self):
        matrix = extract_matrix_from_analysis(CAR_ANALYSIS)

        self.assertEqual(matrix["source"], "/analysis.md")
        self.assertIn(
            "Toyota Camry Hybrid: Balanced efficiency and resale",
            matrix["swot"]["strengths"],
        )
        self.assertIn(
            "Tesla Model 3: High depreciation",
            matrix["swot"]["weaknesses"],
        )
        self.assertEqual(
            matrix["swot"]["opportunities"][0],
            "Obtain dealer and insurance quotes. [Evidence IDs: 8, 9]",
        )
        self.assertEqual(
            matrix["swot"]["threats"][0],
            "Insurance costs are missing for all candidates. [Evidence IDs: 8, 9]",
        )
        self.assertEqual(
            matrix["horizon"]["near_term"],
            matrix["swot"]["opportunities"][:3],
        )
        self.assertEqual(matrix["horizon"]["medium_term"], [])
        self.assertEqual(matrix["horizon"]["long_term"], [])

    def test_explicit_swot_and_horizons_are_preserved(self):
        matrix = extract_matrix_from_analysis(STRUCTURED_ANALYSIS)

        self.assertEqual(
            matrix["swot"]["strengths"],
            ["Strong verified outcome. [Evidence ID: 1]"],
        )
        self.assertEqual(
            matrix["horizon"]["near_term"],
            ["Collect current quotes. [Evidence ID: 5]"],
        )
        self.assertEqual(
            matrix["horizon"]["medium_term"],
            ["Recheck depreciation. [Evidence ID: 6]"],
        )
        self.assertEqual(
            matrix["horizon"]["long_term"],
            ["Review replacement cost. [Evidence ID: 7]"],
        )

    def test_empty_analysis_never_returns_placeholder_content(self):
        matrix = extract_matrix_from_analysis("")

        for items in matrix["swot"].values():
            self.assertEqual(items, [])
        for items in matrix["horizon"].values():
            self.assertEqual(items, [])

    def test_dashboard_metrics_are_derived_from_all_live_artifacts(self):
        metrics = extract_metrics_from_text(
            METRIC_FINAL,
            METRIC_ANALYSIS,
            "Which car should I buy?",
            METRIC_RESEARCH,
        )

        self.assertEqual(
            metrics["radar"]["categories"],
            ["Fuel Efficiency", "Resale Score", "Tco (5-Year)"],
        )
        self.assertEqual(
            [series["name"] for series in metrics["radar"]["series"]],
            ["Toyota Camry Hybrid", "Tesla Model 3"],
        )
        self.assertEqual(metrics["radar"]["series"][0]["values"], [40, 55, 0])
        self.assertEqual(metrics["radar"]["series"][1]["values"], [100, 50, 100])
        self.assertEqual(
            metrics["evidence_rigor"]["labels"],
            [
                "Government & Regulatory",
                "Independent Research & Testing",
                "Industry & Trade Media",
            ],
        )
        self.assertEqual(metrics["evidence_rigor"]["values"], [1, 1, 1])
        self.assertEqual(metrics["strategic_confidence"]["score"], 89)
        self.assertEqual(metrics["logic_rigor_score"], 8.9)
        self.assertEqual(metrics["radar"]["source"], "/analysis.md")
        self.assertEqual(metrics["evidence_rigor"]["source"], "/research.md")

    def test_empty_artifacts_return_empty_charts_and_zero_confidence(self):
        metrics = extract_metrics_from_text("", "", "Empty query", "")

        self.assertEqual(metrics["radar"]["categories"], [])
        self.assertEqual(metrics["radar"]["series"], [])
        self.assertEqual(metrics["evidence_rigor"]["labels"], [])
        self.assertEqual(metrics["strategic_confidence"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
