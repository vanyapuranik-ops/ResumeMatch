import json
import tempfile
import unittest
from pathlib import Path

from src.role_recommender import evidence_for, match_role, recommend

class RoleRecommenderTests(unittest.TestCase):
    def test_sql_alone_is_not_database_system_evidence(self):
        self.assertIsNone(evidence_for("Used SQL for data analysis.", "databases"))

    def test_specific_database_technology_is_evidence(self):
        self.assertEqual(
            evidence_for("Built an application using PostgreSQL.", "databases"),
            "postgresql",
        )

    def test_machine_learning_alias_does_not_match_plain_ml_word(self):
        self.assertIsNone(evidence_for("Worked in an ML department.", "machine learning"))

    def test_role_score_is_bounded(self):
        role = {
            "role_id": "test_role",
            "title": "Test Role",
            "categories": ["INFORMATION-TECHNOLOGY"],
            "required_skills": ["python", "machine learning"],
            "preferred_skills": ["pandas"],
            "project_evidence": ["model evaluation"],
        }
        result = match_role(
            "Python machine learning pandas model evaluation",
            role,
        )
        self.assertGreaterEqual(result["fit_score"], 0)
        self.assertLessEqual(result["fit_score"], 100)

    def test_recommendation_returns_requested_count(self):
        catalog = {
            "roles": [
                {
                    "role_id": "python_role",
                    "title": "Python Role",
                    "categories": ["INFORMATION-TECHNOLOGY"],
                    "required_skills": ["python"],
                    "preferred_skills": [],
                    "project_evidence": [],
                },
                {
                    "role_id": "sql_role",
                    "title": "SQL Role",
                    "categories": ["FINANCE"],
                    "required_skills": ["sql"],
                    "preferred_skills": [],
                    "project_evidence": [],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            results = recommend("Python SQL", path, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Python Role")

if __name__ == "__main__":
    unittest.main()
