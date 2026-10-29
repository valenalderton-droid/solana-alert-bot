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

    def test_unknown_symbol_passes_without_curated_allowlist(self):
        self.assertTrue(is_allowed_non_meme_project(self.project("New Utility Protocol", "NEWU")))

    def test_meme_name_is_rejected_even_if_symbol_is_known(self):
        self.assertFalse(is_allowed_non_meme_project(self.project("Jupiter Meme Coin", "JUP")))

    def test_new_project_without_website_or_approval_passes(self):
        project = self.project("Example Lending Protocol", "EXL", approved=False)
        project["has_project_website"] = False
        self.assertTrue(is_allowed_non_meme_project(project))

    def test_meme_terms_are_rejected_for_unknown_projects(self):
        project = self.project("Example Doge Protocol", "EXL")
        self.assertFalse(is_allowed_non_meme_project(project))

    def test_missing_address_is_rejected(self):
        project = self.project("Utility Protocol", "UTIL")
        project["token_address"] = ""
        self.assertFalse(is_allowed_non_meme_project(project))

    def test_explicit_non_solana_chain_is_rejected(self):
        project = self.project("Utility Protocol", "UTIL")
        project["chain"] = "ethereum"
        self.assertFalse(is_allowed_non_meme_project(project))


if __name__ == "__main__":
    unittest.main()
