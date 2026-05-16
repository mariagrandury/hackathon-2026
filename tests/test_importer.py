"""Unit tests for the pure helpers in ``import_participants_info``.

The importer's job is to turn an Eventbrite registration CSV into the
shape the participants dataset expects. The interesting logic is in
three pure helpers that we can test without any CSV file:

  - ``clean_username``: strips URLs/quotes/@, blocks non-answers and
    blacklisted strings, rejects values with spaces or unsafe chars.
  - ``map_country``: substring match against ``COUNTRY_PATTERNS``,
    biased to the first comma-separated component so "Sevilla,
    Andalucía" → "es" via the early ``sevilla`` match rather than
    ``andaluc``.
  - ``_order_by_lang``: ranks Eventbrite columns ES > PT > EN so the
    Spanish answer wins when a registrant filled the form in more than
    one language.

Run from the repo root::

    python -m unittest tests.test_importer
"""

from __future__ import annotations

import unittest

from import_participants_info import (
    USERNAME_BLACKLIST,
    _order_by_lang,
    clean_username,
    map_country,
)


# ---------------------------------------------------------------------------
# clean_username
# ---------------------------------------------------------------------------


class CleanUsernameTrivial(unittest.TestCase):

    def test_none_returns_empty(self):
        # Defensive against pd.NA / None upstream — we coerce via ``or ""``.
        self.assertEqual(clean_username(None), "")

    def test_empty_string(self):
        self.assertEqual(clean_username(""), "")

    def test_whitespace_only(self):
        self.assertEqual(clean_username("   "), "")


class CleanUsernameStripsLeadingChars(unittest.TestCase):

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(clean_username("  alice  "), "alice")

    def test_strips_at_sign(self):
        self.assertEqual(clean_username("@alice"), "alice")

    def test_strips_quotes(self):
        self.assertEqual(clean_username("'alice'"), "alice")
        self.assertEqual(clean_username('"alice"'), "alice")

    def test_strips_trailing_punctuation(self):
        # ``.`` / ``,`` / ``;`` / ``:`` / ``/`` removed from the tail before
        # the regex check (so "alice." passes through, not as malformed).
        self.assertEqual(clean_username("alice."), "alice")
        self.assertEqual(clean_username("alice,"), "alice")
        self.assertEqual(clean_username("alice/"), "alice")


class CleanUsernameURLs(unittest.TestCase):
    """HF profile URLs get the username extracted (people paste these
    instead of bare handles)."""

    def test_full_https_url(self):
        self.assertEqual(
            clean_username("https://huggingface.co/alice"), "alice"
        )

    def test_short_hf_co_url(self):
        self.assertEqual(clean_username("https://hf.co/alice"), "alice")

    def test_protocol_less_url(self):
        self.assertEqual(clean_username("huggingface.co/alice"), "alice")

    def test_www_prefix(self):
        self.assertEqual(
            clean_username("https://www.huggingface.co/alice"), "alice"
        )

    def test_url_with_trailing_slash(self):
        self.assertEqual(
            clean_username("https://huggingface.co/alice/"), "alice"
        )

    def test_organisation_link_rejected(self):
        # The URL_PATTERNS use a negative lookahead for "organizations" so
        # an org link like the join-confirmation URL doesn't get its slug
        # turned into a fake username.
        self.assertEqual(
            clean_username("https://huggingface.co/organizations/somosnlp"),
            "",
        )


class CleanUsernameNonAnswers(unittest.TestCase):
    """Form non-answers stay out of the dataset."""

    def test_no_tengo(self):
        self.assertEqual(clean_username("no tengo"), "")
        self.assertEqual(clean_username("NO TENGO"), "")  # case-insensitive

    def test_n_a(self):
        self.assertEqual(clean_username("n/a"), "")
        self.assertEqual(clean_username("N/A"), "")

    def test_dot(self):
        # Single dot stripped to "" via rstrip, then NON_ANSWERS catches it.
        self.assertEqual(clean_username("."), "")


class CleanUsernameBlacklist(unittest.TestCase):

    def test_org_slug_blocked(self):
        # Confirms the constant + the filter both wired up.
        self.assertIn("somosnlp-hackathon-2026", USERNAME_BLACKLIST)
        self.assertEqual(clean_username("somosnlp-hackathon-2026"), "")

    def test_blacklist_is_case_insensitive(self):
        self.assertEqual(clean_username("SOMOSNLP-HACKATHON-2026"), "")


class CleanUsernameRejects(unittest.TestCase):

    def test_contains_space_after_strip(self):
        self.assertEqual(clean_username("alice smith"), "")

    def test_contains_residual_scheme(self):
        # A weird leftover like "http://" without HF.co host should reject.
        self.assertEqual(clean_username("alice://bob"), "")

    def test_unsafe_chars_rejected(self):
        # The regex allows [\w.\-] only.
        self.assertEqual(clean_username("alice!bob"), "")
        self.assertEqual(clean_username("alice?bob"), "")


class CleanUsernameAccepts(unittest.TestCase):

    def test_plain_handle(self):
        self.assertEqual(clean_username("alice"), "alice")

    def test_with_dots(self):
        self.assertEqual(clean_username("alice.smith"), "alice.smith")

    def test_with_dashes(self):
        self.assertEqual(clean_username("alice-smith"), "alice-smith")

    def test_with_underscores(self):
        self.assertEqual(clean_username("alice_smith"), "alice_smith")

    def test_mixed_case_preserved(self):
        # HF usernames are case-sensitive in practice; we don't lower() here.
        self.assertEqual(clean_username("AliceSmith"), "AliceSmith")


# ---------------------------------------------------------------------------
# map_country
# ---------------------------------------------------------------------------


class MapCountryBasic(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(map_country(""), "")
        self.assertEqual(map_country(None), "")
        self.assertEqual(map_country("   "), "")

    def test_unknown(self):
        # Documented behaviour: unlisted countries (e.g. France) import
        # with a blank country, with a console warning at import time.
        self.assertEqual(map_country("France"), "")

    def test_canonical_names(self):
        # Sample one from each region; the rest are covered by the patterns
        # table directly.
        self.assertEqual(map_country("España"), "es")
        self.assertEqual(map_country("Brasil"), "br")
        self.assertEqual(map_country("Brazil"), "br")
        self.assertEqual(map_country("Mexico"), "mx")
        self.assertEqual(map_country("México"), "mx")
        self.assertEqual(map_country("Spain"), "es")


class MapCountryCityAndRegion(unittest.TestCase):
    """COUNTRY_PATTERNS also lists cities/regions for ES + a few others —
    these tests lock that behaviour."""

    def test_spanish_cities(self):
        self.assertEqual(map_country("Sevilla"), "es")
        self.assertEqual(map_country("Madrid"), "es")
        self.assertEqual(map_country("Andalucía"), "es")

    def test_peruvian_cities(self):
        self.assertEqual(map_country("Lima"), "pe")
        self.assertEqual(map_country("Cusco"), "pe")
        self.assertEqual(map_country("Chiclayo"), "pe")

    def test_mexican_cities(self):
        self.assertEqual(map_country("CDMX"), "mx")
        self.assertEqual(map_country("Oaxaca"), "mx")


class MapCountryFirstComponentBias(unittest.TestCase):
    """When the value is comma-separated, the first component wins. This
    matters for entries like 'Sevilla, Andalucía' (both → ES, so the same
    result) but also 'Asunción, Paraguay' where the first match is the
    city not the country. The bias is documented in the helper's comment."""

    def test_comma_separated_first_match(self):
        # First component is "asunción" → matches Paraguay.
        self.assertEqual(map_country("Asunción, Paraguay"), "py")

    def test_first_component_wins_over_fallback(self):
        # If the first component doesn't match, the whole-string fallback
        # checks the substring against the full value.
        self.assertEqual(map_country("Some place, Lima"), "pe")

    def test_case_insensitive(self):
        self.assertEqual(map_country("ESPAÑA"), "es")
        self.assertEqual(map_country("brasil"), "br")


# ---------------------------------------------------------------------------
# _order_by_lang
# ---------------------------------------------------------------------------


class OrderByLang(unittest.TestCase):
    """Eventbrite emits one column per ticket-language; we want the ES
    answer first, then PT, then EN, so the first-non-empty pickup yields
    the right text when participants answered in more than one form."""

    def test_es_first(self):
        cols = [
            "Country or countries you represent",
            "¿Cuál es tu nombre de usuario en Hugging Face?",
        ]
        ordered = _order_by_lang(cols)
        # ES marker "cuál es tu nombre" should rank first.
        self.assertEqual(ordered[0], cols[1])
        self.assertEqual(ordered[1], cols[0])

    def test_full_three_lang_ordering(self):
        cols = [
            "Country or countries you represent",                    # EN
            "Qual é o seu nome de usuário no Hugging Face?",        # PT
            "¿Cuál es tu nombre de usuário en Hugging Face?",        # ES
        ]
        ordered = _order_by_lang(cols)
        # ES → PT → EN
        self.assertIn("cuál es tu nombre", ordered[0].lower())
        self.assertIn("qual é o seu nome", ordered[1].lower())
        self.assertIn("country or countries", ordered[2].lower())

    def test_unrecognized_sorts_last(self):
        cols = [
            "Random column",
            "¿Cuál es tu nombre de usuario?",
        ]
        ordered = _order_by_lang(cols)
        self.assertEqual(ordered[-1], "Random column")

    def test_empty_list(self):
        self.assertEqual(_order_by_lang([]), [])

    def test_country_markers_also_recognised(self):
        # Country columns use a different marker pair than HF-username
        # columns. The ranker should pick them up too.
        cols = [
            "Country or countries you represent",            # EN
            "Países que representen / Pais que representem", # ES (via "que representen")
        ]
        ordered = _order_by_lang(cols)
        self.assertIn("que representen", ordered[0].lower())


if __name__ == "__main__":
    unittest.main()
