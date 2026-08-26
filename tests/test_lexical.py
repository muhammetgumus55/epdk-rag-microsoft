"""Tests for BM25 indexing and Turkish diacritic folding.

The central property under test is SYMMETRY: fold_diacritics() is applied to
indexed text and to queries alike, so a question typed without Turkish letters
must retrieve exactly what the correctly-spelled one does. Step 4 measured that
ASCII-typed queries lost up to 0.23 cosine and flipped 8 of 21 gate decisions;
these tests pin the mechanism that fixes it.
"""
import pytest

from src import config, store
from src.extract import tr_lower, tr_upper
from src.lexical import (
    STOPWORDS,
    BM25Index,
    fold_diacritics,
    tokenize,
)

from tests.test_retrieval import make_chunk, unit  # noqa: F401


class TestFoldDiacritics:
    """Every Turkish-specific letter maps to its plain ASCII neighbour."""

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("ç", "c"), ("ğ", "g"), ("ı", "i"), ("ö", "o"), ("ş", "s"), ("ü", "u"),
            ("Ç", "C"), ("Ğ", "G"), ("İ", "I"), ("Ö", "O"), ("Ş", "S"), ("Ü", "U"),
        ],
    )
    def test_each_letter_folds_to_its_ascii_neighbour(self, source, expected):
        assert fold_diacritics(source) == expected

    def test_the_full_alphabet_folds_in_one_pass(self):
        assert fold_diacritics("çğıöşüÇĞİÖŞÜ") == "cgiosuCGIOSU"

    def test_circumflex_vowels_fold_too(self):
        # "hâl" and "kâr" appear in older mevzuat and must not split into
        # separate tokens from their plain spellings.
        assert fold_diacritics("hâl kâr") == "hal kar"

    def test_ascii_text_is_left_alone(self):
        assert fold_diacritics("elektrik piyasasi 6446") == "elektrik piyasasi 6446"

    def test_non_turkish_characters_pass_through(self):
        assert fold_diacritics("MADDE 12 – (1) %50") == "MADDE 12 – (1) %50"

    def test_it_is_idempotent(self):
        once = fold_diacritics("Önlisans süresi")
        assert fold_diacritics(once) == once


class TestFoldingRoundTrip:
    """All spellings of the same word must produce one index key."""

    @pytest.mark.parametrize(
        "variant",
        ["önlisans", "Önlisans", "onlisans", "ONLISANS", "ÖNLİSANS", "OnLiSaNs"],
    )
    def test_every_spelling_of_onlisans_yields_the_same_token(self, variant):
        assert tokenize(variant) == ["onlisans"]

    def test_a_whole_question_tokenizes_identically_either_way(self):
        typed = "Önlisans süresi ne kadardır?"
        folded = "onlisans suresi ne kadardir?"
        assert tokenize(typed) == tokenize(folded)
        assert tokenize(typed) == ["onlisans", "suresi", "kadardir"]

    def test_uppercase_turkish_i_collapses_with_dotless_i(self):
        # İ -> I -> i and ı -> i must converge, or "İLETİM" and "iletim" split.
        assert tokenize("İLETİM") == tokenize("iletim") == ["iletim"]

    def test_case_and_diacritics_together(self):
        assert tokenize("DAĞITIM ŞİRKETİ") == tokenize("dagitim sirketi")


class TestSeparationFromTurkishCasing:
    """fold_diacritics() and tr_lower()/tr_upper() must never be composed.

    tr_lower/tr_upper implement correct Turkish casing on real Turkish text.
    fold_diacritics deliberately destroys that information to make a lexical
    key. Chaining them silently corrupts text -- these tests document and pin
    the separation so a future "cleanup" cannot quietly merge them.
    """

    def test_they_are_distinct_functions_with_distinct_results(self):
        source = "IŞIK Işık ışık"
        assert tr_lower(source) != fold_diacritics(source)
        assert tr_upper(source) != fold_diacritics(source)

    def test_tr_lower_preserves_turkish_letters_that_folding_destroys(self):
        # The contract: casing keeps ı/ş/ğ, folding removes them.
        assert tr_lower("IŞIK") == "ışık"
        assert "ş" in tr_lower("IŞIK")
        assert "ş" not in fold_diacritics("IŞIK")

    def test_folding_then_tr_lower_reintroduces_a_turkish_letter(self):
        """Composing them corrupts the key -- the concrete reason for the ban.

        fold_diacritics("IŞIK") == "ISIK", pure ASCII as intended. Feeding that
        to tr_lower() then applies Turkish casing to it, and Turkish maps I -> ı,
        putting a non-ASCII letter *back* into a string whose whole purpose was
        to have none. The index key silently stops matching the ASCII query it
        was built to match.
        """
        assert fold_diacritics("IŞIK") == "ISIK"
        composed = tr_lower(fold_diacritics("IŞIK"))
        assert composed == "ısık"          # Turkish dotless ı is back
        assert not composed.isascii()      # ... so the key is no longer ASCII
        # What tokenize() actually does -- plain ASCII lower() -- stays clean.
        assert fold_diacritics("IŞIK").lower() == "isik"
        assert fold_diacritics("IŞIK").lower().isascii()

    def test_tr_lower_then_folding_is_not_what_tokenize_does(self):
        """tokenize() folds FIRST, then uses plain ASCII lower().

        Going the other way happens to agree here, but relies on tr_lower's
        Turkish rules to produce something folding can still handle. The
        pipeline does not depend on that coincidence; this test records which
        order is the real one.
        """
        source = "İLETİM"
        assert tokenize(source) == [fold_diacritics(source).lower()]

    def test_lexical_module_does_not_import_turkish_casing(self):
        """Structural guard: src/lexical.py must not reach for tr_lower/tr_upper."""
        import src.lexical as lexical

        assert not hasattr(lexical, "tr_lower")
        assert not hasattr(lexical, "tr_upper")

    def test_displayed_text_is_never_folded(self, tmp_path):
        """Folding is index-only: stored/returned text keeps real Turkish."""
        conn = store.connect(tmp_path / "t.db")
        text = "MADDE 6 – (1) Önlisans süresi yirmi dört ayı geçemez."
        store.insert_chunks(
            conn, source_path="p.doc", file_sha256="h", document_title="ELEKTRİK",
            chunks=[make_chunk(text=text)],
            embeddings=[unit(0)], quality_flag=None,
        )
        index = BM25Index.from_connection(conn)
        # The query is ASCII; the retrieved text is still fully Turkish.
        hits = index.search("onlisans suresi", 1)
        assert hits
        meta = store.fetch_chunk_metadata(conn, hits[0][0])
        assert meta["text"] == text
        assert "Ö" in meta["text"] and "ü" in meta["text"]
        conn.close()


class TestTokenize:
    def test_stopwords_are_removed(self):
        assert "ve" not in tokenize("dağıtım ve iletim")
        assert tokenize("dağıtım ve iletim") == ["dagitim", "iletim"]

    def test_stopword_list_is_stored_folded(self):
        # Otherwise "için" would never match the folded token "icin".
        assert "icin" in STOPWORDS
        assert "için" not in STOPWORDS

    def test_punctuation_and_single_characters_are_dropped(self):
        assert tokenize("MADDE 12 – (1) a) b)") == ["madde", "12"]

    def test_numbers_survive_because_law_numbers_matter(self):
        assert "6446" in tokenize("6446 sayılı Kanun")

    def test_multi_word_legal_terms_stay_intact(self):
        # No stemming: these must not be truncated to shared stems.
        assert tokenize("dağıtım bedeli") == ["dagitim", "bedeli"]
        assert tokenize("iletim tarifesi") == ["iletim", "tarifesi"]
        assert tokenize("serbest tüketici") == ["serbest", "tuketici"]

    def test_singular_and_plural_remain_distinct_without_stemming(self):
        assert tokenize("bedeli") != tokenize("bedelleri")

    def test_empty_and_stopword_only_queries_yield_nothing(self):
        assert tokenize("") == []
        assert tokenize("ve veya ile bu bir") == []


class TestBM25Index:
    @pytest.fixture
    def index(self):
        return BM25Index.build([
            (10, "Önlisans süresi mücbir sebep hâlleri hariç yirmi dört ayı geçemez."),
            (20, "Doğal gaz dağıtım şirketleri abone bağlantı bedeli tahsil eder."),
            (30, "Elektrik dağıtım bedeli dağıtım şirketi tarafından hesaplanır."),
            (40, "Yan hizmetler kapsamında primer frekans kontrol hizmeti sunulur."),
        ])

    def test_index_reports_its_size_and_vocabulary(self, index):
        assert len(index) == 4
        assert index.vocabulary_size > 0
        assert index.ids == [10, 20, 30, 40]

    def test_doc_lengths_and_average_are_computed(self, index):
        assert len(index.doc_lengths) == 4
        assert all(length > 0 for length in index.doc_lengths)
        assert index.avg_doc_length == pytest.approx(
            sum(index.doc_lengths) / 4
        )

    def test_search_returns_chunk_ids_not_positions(self, index):
        hits = index.search("önlisans", 5)
        assert hits[0][0] == 10  # the id, not index 0

    def test_search_ranks_the_relevant_document_first(self, index):
        assert index.search("primer frekans kontrol", 5)[0][0] == 40

    def test_ascii_query_matches_turkish_document(self, index):
        """The whole point of the folding: these must be identical."""
        assert index.search("onlisans suresi", 5) == index.search("önlisans süresi", 5)

    def test_query_with_no_corpus_terms_returns_nothing(self, index):
        # The domain-mismatch signal: no lexical evidence at all.
        assert index.search("balıkçılık avcılık denizcilik", 5) == []

    def test_stopword_only_query_returns_nothing(self, index):
        assert index.search("ve veya ile", 5) == []

    def test_top_k_is_respected(self, index):
        assert len(index.search("dağıtım", 1)) == 1

    def test_top_k_zero_returns_nothing(self, index):
        assert index.search("dağıtım", 0) == []

    def test_empty_index_is_safe_to_query(self):
        empty = BM25Index.build([])
        assert len(empty) == 0
        assert empty.search("herhangi bir sorgu", 5) == []

    def test_scores_are_descending(self, index):
        hits = index.search("dağıtım bedeli şirketi", 5)
        assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    def test_rarer_term_outweighs_common_one(self, index):
        # "dağıtım" appears in 2 of 4 docs, "frekans" in 1 -- IDF must reflect that.
        assert index.idf("frekans") > index.idf("dagitim")

    def test_unknown_term_has_zero_idf(self, index):
        assert index.idf("balikcilik") == 0.0

    def test_idf_is_never_negative(self, index):
        # A term in every document must not push scores below documents lacking it.
        universal = BM25Index.build([(1, "aynı kelime"), (2, "aynı kelime")])
        assert universal.idf("ayni") >= 0.0

    def test_build_from_connection_covers_only_active_chunks(self, tmp_path):
        conn = store.connect(tmp_path / "t.db")
        for i, (path, text) in enumerate([
            ("a.doc", "birinci belge dağıtım"),
            ("b.doc", "ikinci belge iletim"),
        ]):
            store.insert_chunks(
                conn, source_path=path, file_sha256=f"h{i}", document_title="X",
                chunks=[make_chunk(index=0, text=text)],
                embeddings=[unit(0)], quality_flag=None,
            )
        assert len(BM25Index.from_connection(conn)) == 2

        store.mark_document_inactive(conn, "a.doc")
        rebuilt = BM25Index.from_connection(conn)
        assert len(rebuilt) == 1
        assert rebuilt.search("dağıtım", 5) == []  # retired content is gone
        conn.close()

    def test_index_order_matches_the_dense_matrix_order(self, tmp_path):
        """Both are ordered by id, which downstream code relies on."""
        conn = store.connect(tmp_path / "t.db")
        for i in range(3):
            store.insert_chunks(
                conn, source_path=f"{i}.doc", file_sha256=f"h{i}", document_title="X",
                chunks=[make_chunk(index=0, text=f"belge {i} dağıtım")],
                embeddings=[unit(0)], quality_flag=None,
            )
        _, ids = store.fetch_active_embeddings(conn)
        assert BM25Index.from_connection(conn).ids == ids
        conn.close()

    def test_parameters_come_from_config_by_default(self):
        index = BM25Index.build([(1, "test belgesi")])
        assert index.k1 == config.BM25_K1
        assert index.b == config.BM25_B
