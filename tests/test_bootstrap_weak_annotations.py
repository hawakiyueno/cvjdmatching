import unittest

from bootstrap_weak_annotations import (
    extract_entities,
    metadata_role_terms,
    prune_entities,
    split_role_candidates,
)


class BootstrapWeakAnnotationsTests(unittest.TestCase):
    def test_split_role_candidates_drops_unsplit_full_phrase(self) -> None:
        candidates = split_role_candidates("Blockchain/Web3 marketer, Technical writer")
        self.assertIn("Blockchain", candidates)
        self.assertIn("Web3 marketer", candidates)
        self.assertIn("Technical writer", candidates)
        self.assertNotIn("Blockchain/Web3 marketer, Technical writer", candidates)

    def test_metadata_role_terms_skips_generic_single_word_roles(self) -> None:
        row = {"position": "Lead", "primary_keyword": "Support"}
        self.assertEqual(metadata_role_terms(row), set())

    def test_prune_entities_removes_generic_and_duplicate_noise(self) -> None:
        entities = [
            {"label": "JOB_ROLE", "text": "Lead", "start": 0, "end": 4, "normalized": "lead"},
            {"label": "JOB_ROLE", "text": "Technical writer", "start": 10, "end": 26, "normalized": "technical writer"},
            {"label": "JOB_ROLE", "text": "Blockchain/Web3 marketer, Technical writer", "start": 0, "end": 42, "normalized": "blockchain/web3 marketer, technical writer"},
            {"label": "SKILL", "text": "leadership", "start": 50, "end": 60, "normalized": "leadership"},
        ]
        pruned = prune_entities(entities)
        self.assertEqual([(entity["label"], entity["text"]) for entity in pruned], [("JOB_ROLE", "Technical writer"), ("SKILL", "leadership")])

    def test_extract_entities_uses_stricter_degree_rule(self) -> None:
        row = {
            "text": "Master of the Marketing team\nBachelor's degree in Computer Science\nPosition: Data Engineer",
            "position": "Data Engineer",
            "primary_keyword": "Data Engineer",
        }
        entities = extract_entities(row)
        texts = {(entity["label"], entity["text"]) for entity in entities}
        self.assertIn(("DEGREE", "Bachelor's degree in Computer Science"), texts)
        self.assertNotIn(("DEGREE", "Master of the Marketing team"), texts)


if __name__ == "__main__":
    unittest.main()
