import unittest
from market import is_allowed_non_meme_project


class NonMemeAllowlistTests(unittest.TestCase):
    def project(self, name, symbol, approved=True):
        return {
            "name": name, "symbol": symbol, "approved_non_meme": approved,
            "token_address": "ExampleMintAddress123"
        }

    def test_known_ecosystem_project_passes(self):
        self.assertTrue(is_allowed_non_meme_project(self.project("Jupiter", "JUP")))

    def test_unknown_symbol_fails_closed(self):
        self.assertFalse(is_allowed_non_meme_project(self.project("New Token", "NEWT")))

    def test_meme_name_is_rejected_even_if_symbol_is_on_allowlist(self):
        self.assertFalse(is_allowed_non_meme_project(self.project("Jupiter Meme Coin", "JUP")))

    def test_evidence_backed_new_category_project_passes(self):
        project = self.project("Example Lending Protocol", "EXL")
        project.update({
            "discovery_source": "dexscreener_category_search",
            "discovery_confidence": 0.8,
            "discovery_category": "Solana lending borrowing",
            "has_project_website": True,
        })
        self.assertTrue(is_allowed_non_meme_project(project))

    def test_dynamic_discovery_without_website_is_rejected(self):
        project = self.project("Example Lending Protocol", "EXL")
        project.update({
            "discovery_source": "dexscreener_category_search",
            "discovery_confidence": 0.8,
            "discovery_category": "Solana lending borrowing",
            "has_project_website": False,
        })
        self.assertFalse(is_allowed_non_meme_project(project))

    def test_meme_terms_are_rejected_for_dynamic_projects(self):
        project = self.project("Example Doge Protocol", "EXL")
        project.update({
            "discovery_source": "dexscreener_category_search",
            "discovery_confidence": 0.8,
            "discovery_category": "Solana lending borrowing",
            "has_project_website": True,
        })
        self.assertFalse(is_allowed_non_meme_project(project))

    def test_manual_approval_flag_is_required(self):
        self.assertFalse(is_allowed_non_meme_project(self.project("Jupiter", "JUP", False)))


if __name__ == "__main__":
    unittest.main()
