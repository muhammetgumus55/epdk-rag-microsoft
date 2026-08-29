"""Tests for domain-scope classification and the exclusion of out-of-scope text.

Two layers are covered, and they answer different questions:

  * `TestOmnibusDetection` / `TestArticleClassification` / `TestManualExclusions`
    -- does src.scope reach the right verdict on real corpus text?
  * `TestExcludedTextIsNotRetrievable`
    -- is an excluded chunk genuinely absent from the search structures, rather
    than merely scoring badly at query time? That is the guarantee the whole
    design rests on, so it is asserted against the store and both rankers.

Text samples are shortened excerpts of real documents in the corpus; the
docstring on each names where it came from.
"""
import numpy as np
import pytest

from src import config, lexical, scope, store
from src.chunk import ArticleRef, Chunk


# --------------------------------------------------------------------------
# Real corpus excerpts
# --------------------------------------------------------------------------

# From guncel-6446-...-degisiklik.docx, "BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA
# DAİR KANUN" -- the chunk retrieved at rank 3 for "Trafik cezası itiraz süresi".
TRAFIK_LEAK = (
    "ceza zamanaşımı işlemez. (3) Bu madde uyarınca infazı durdurulan kişi hakkında "
    "mahkemece Ceza Muhakemesi Kanununun 109 uncu maddesinin üçüncü fıkrasının (a) "
    "bendinde yer alan adlî kontrol tedbirine karar verilebilir. (5) Bu madde uyarınca "
    "verilecek kararlara karşı itiraz kanun yoluna gidilebilir. İtirazın incelenmesinde "
    "İcra ve İflas Kanununun 353 üncü maddesinin birinci fıkrasında belirlenen itiraz "
    "usulü uygulanır. MADDE 50 – 18/10/2012 tarihli ve 6356 sayılı Sendikalar ve Toplu "
    "İş Sözleşmesi Kanununun 26 ncı maddesinin onuncu fıkrasında yer alan “yüzde yirmi "
    "beşini” ibaresi “yüzde otuz beşini” şeklinde değiştirilmiştir."
)

# From guncel-6446-...-degisiklik-7103_v8_9-1docx.docx, "VERGİ KANUNLARI İLE BAZI
# KANUN VE KHK'LERDE DEĞİŞİKLİK" -- rank 2 for "Vergi levhası nereye asılır".
VERGI_LEAK = (
    "sayılı Vergi Usul Kanunu hükümlerine göre vergi ziyaı cezası uygulanarak gecikme "
    "faizi ile birlikte tahsil edilir. (7) Bakanlar Kurulu, birinci fıkrada yer alan "
    "tutarı; sıfıra kadar indirmeye, aracın hurdaya çıkarılması veya ihraç edilmesi "
    "durumuna göre farklılaştırmaya yetkilidir."
)

# From the same omnibus act -- a genuine electricity article that MUST survive.
ELECTRICITY_ARTICLE = (
    "MADDE 8- 6446 sayılı Kanunun 4 üncü maddesinin birinci fıkrasına aşağıdaki bent "
    "eklenmiştir. “ğ) Toplayıcılık faaliyeti” Dağıtım şirketi, tüketicilere ait "
    "elektrik enerjisi tüketimini ölçmek üzere sayaç okuma yapar."
)

# Opening of guncel-6446-...-degisiklik-5.pdf, filed under an electricity filename.
NUCLEAR_OPENING = (
    "Resmî Gazete Sayı : 31772 KANUN NÜKLEER DÜZENLEME KANUNU Kanun No. 7381 "
    "Kabul Tarihi: 5/3/2022 BİRİNCİ BÖLÜM Amaç, Kapsam ve Tanımlar"
)


class TestOmnibusDetection:
    @pytest.mark.parametrize("title", [
        "BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN",
        "VERGİ KANUNLARI İLE BAZI KANUN VE KANUN HÜKMÜNDE KARARNAMELERDE DEĞİŞİKLİK YAPILMASI HAKKINDA KANUN",
        "DEVLET MEMURLARI KANUNU İLE BAZI KANUNLARDA VE 375 SAYILI KANUN HÜKMÜNDE KARARNAMEDE DEĞİŞİKLİK YAPILMASINA DAİR KANUN",
        "BAZI VERGİ KANUNLARI İLE DİĞER BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN",
        "MADEN KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN",
        "BÜTÇE KANUNLARINDA YER ALAN BAZI HÜKÜMLERİN İLGİLİ KANUN VE KANUN HÜKMÜNDE KARARNAMELERE EKLENMESİNE DAİR KANUN",
    ])
    def test_real_omnibus_titles_are_detected(self, title):
        assert scope.is_omnibus_document(title) is True

    @pytest.mark.parametrize("title", [
        "ELEKTRİK PİYASASI KANUNU",
        "ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ BİRİNCİ BÖLÜM",
        "ELEKTRİK PİYASASI DENGELEME VE UZLAŞTIRMA YÖNETMELİĞİ BİRİNCİ KISIM",
        # The single-instrument amendment: names one law, so NOT omnibus.
        "ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİNDE DEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK",
        "YENİLENEBİLİR ENERJİ KAYNAKLARININ ELEKTRİK ENERJİSİ ÜRETİMİ AMAÇLI KULLANIMINA İLİŞKİN KANUN",
    ])
    def test_single_subject_electricity_titles_are_not_omnibus(self, title):
        assert scope.is_omnibus_document(title) is False

    def test_detected_from_body_when_title_extraction_failed(self):
        """Several omnibus files' titles come back as a Resmî Gazete banner."""
        body = "5 Temmuz 2012 PERŞEMBE Resmî Gazete Sayı : 28344 KANUN BAZI KANUNLARDA " \
               "DEĞİŞİKLİK YAPILMASI HAKKINDA KANUN Kanun No. 6352"
        assert scope.is_omnibus_document(None, body) is True

    def test_mention_deep_in_a_body_does_not_make_a_document_omnibus(self):
        """Only the opening is consulted, so a passing reference cannot mislabel."""
        body = "ELEKTRİK PİYASASI KANUNU " + ("dağıtım lisansı hükümleri. " * 200) + \
               "bazı kanunlarda değişiklik yapılmasına dair"
        assert scope.is_omnibus_document(None, body) is False


class TestArticleClassification:
    def test_traffic_leak_article_is_off_domain(self):
        assert scope.classify_text(TRAFIK_LEAK).label == "OFF_DOMAIN"

    def test_tax_leak_article_is_off_domain(self):
        assert scope.classify_text(VERGI_LEAK).label == "OFF_DOMAIN"

    def test_electricity_article_in_an_omnibus_act_is_kept(self):
        verdict = scope.classify_text(ELECTRICITY_ARTICLE)
        assert verdict.label == "ELECTRICITY"
        assert verdict.indexable is True

    def test_amended_code_decides_over_incidental_vocabulary(self):
        """A mining article stays off-domain even while mentioning electricity."""
        text = (
            "MADDE 11- 3213 sayılı Kanuna aşağıdaki geçici madde eklenmiştir. "
            "“GEÇİCİ MADDE 45- ruhsat sahibi veya rödövansçı olan gerçek veya tüzel "
            "kişiler tarafından ülkenin elektrik ihtiyacını karşılamak üzere yürütülen "
            "madencilik faaliyetlerinin tapuda zeytinlik olarak kayıtlı alanlarda"
        )
        assert scope.classify_text(text).label == "OFF_DOMAIN"

    def test_article_with_no_electricity_marker_at_all_is_off_domain(self):
        """In a grab-bag act, absence of any electricity signal is itself evidence."""
        text = (
            "MADDE 29- 18/1/1972 tarihli ve 1512 sayılı Noterlik Kanununun 27 nci "
            "maddesine aşağıdaki fıkra eklenmiştir. “İki defa yapılan ilana rağmen, "
            "birinci fıkra uyarınca atama yapılamayan bir noterliğe atama yapılır.”"
        )
        assert scope.classify_text(text).label == "OFF_DOMAIN"

    def test_genuinely_mixed_article_is_flagged_not_guessed(self):
        """Balanced evidence on both sides is left for a human, not decided."""
        text = (
            "MADDE 12- 6446 sayılı Kanunun dağıtım şirketi ve elektrik tarifesi "
            "hükmü ile 5510 sayılı Kanunun sosyal sigortalar hükmü birlikte "
            "değiştirilmiştir. Beyanname verilir."
        )
        verdict = scope.classify_text(text)
        assert verdict.label == "AMBIGUOUS"
        assert verdict.indexable is True  # kept, and reported for review

    def test_ambiguous_is_kept_and_never_silently_dropped(self):
        assert scope.ScopeVerdict("AMBIGUOUS", (), ()).indexable is True
        assert scope.ScopeVerdict("OFF_DOMAIN", (), ()).indexable is False

    def test_short_abbreviations_do_not_match_inside_words(self):
        """'kv' and 'mw' are electricity units, not substrings to hunt for."""
        verdict = scope.classify_text("Noterlik atama işlemleri hakkında hüküm.")
        assert "kv" not in verdict.electricity_hits
        assert "mw" not in verdict.electricity_hits


class TestArticleGrouping:
    def test_continuation_fragment_inherits_its_article_verdict(self):
        """A fragment loses the amendment header; the article it belongs to has it."""
        items = [
            ("MADDE 5", "MADDE 5- 3213 sayılı Maden Kanununun 13 üncü maddesi "
                        "ruhsat sahibi için değiştirilmiştir."),
            ("MADDE 5", "ait olmayan hammadde üretim izinlerine ilişkin hükümler "
                        "uygulanır. Rehabilitasyon uygulamaları izin süresince sürer."),
        ]
        verdicts = scope.classify_chunks(items)
        assert [v.label for v in verdicts] == ["OFF_DOMAIN", "OFF_DOMAIN"]

    def test_documents_with_no_article_refs_are_judged_per_chunk(self):
        """Otherwise one verdict would swallow a whole 56-chunk document."""
        items = [
            (None, "MADDE 1- 6446 sayılı Kanunun dağıtım lisansı hükmü değiştirilmiştir. "
                   "Elektrik dağıtım şirketi tarifeleri uygular."),
            (None, "MADDE 2- 2918 sayılı Karayolları Trafik Kanununda sürücü belgesi "
                   "hükmü değiştirilmiştir. Trafik cezası itiraz süresi düzenlenir."),
        ]
        verdicts = scope.classify_chunks(items)
        assert [v.label for v in verdicts] == ["ELECTRICITY", "OFF_DOMAIN"]


class TestManualExclusions:
    def test_nuclear_regulation_law_is_excluded_whole(self):
        decision = scope.document_scope("DÜZENLEME KANUNU", NUCLEAR_OPENING)
        assert decision.disposition == "EXCLUDED"
        assert decision.excluded_entirely is True
        assert "NDK" in decision.reason

    def test_exclusion_matches_the_body_not_the_misleading_filename(self):
        """The file is named for the Electricity Market Law it has nothing to do with."""
        assert scope.manual_exclusion(None, NUCLEAR_OPENING) is not None

    def test_every_chunk_of_an_excluded_document_is_off_domain(self):
        items = [(None, NUCLEAR_OPENING), ("MADDE 3", "Tanımlar"), ("MADDE 4", "Kapsam")]
        decision, verdicts = scope.scope_chunks("DÜZENLEME KANUNU", items)
        assert decision.disposition == "EXCLUDED"
        assert all(v.label == "OFF_DOMAIN" for v in verdicts)

    def test_electricity_document_mentioning_nuclear_is_not_excluded(self):
        """The pattern keys on the act's own name, not the word 'nükleer'."""
        decision = scope.document_scope(
            "ELEKTRİK PİYASASI KANUNU",
            "Nükleer santrallerden üretilen elektrik enerjisi bu Kanuna tabidir.",
        )
        assert decision.disposition == "IN_SCOPE"


class TestSingleSubjectDocumentsAreUntouched:
    def test_in_scope_document_chunks_are_never_classified(self):
        items = [("MADDE 1", "Elektrik piyasasında dağıtım tarifeleri.")] * 3
        decision, verdicts = scope.scope_chunks("ELEKTRİK PİYASASI TARİFELER YÖNETMELİĞİ", items)
        assert decision.disposition == "IN_SCOPE"
        # None, not a verdict: no judgement was made, and the store records NULL.
        assert verdicts == [None, None, None]


class TestExcludedTextIsNotRetrievable:
    """The core guarantee: excluded chunks are absent, not merely down-ranked."""

    @pytest.fixture
    def conn(self, tmp_path):
        c = store.connect(tmp_path / "scope.db")
        yield c
        c.close()

    @staticmethod
    def _chunk(index, text):
        return Chunk(
            doc_id="omnibus-test", text=text, strategy="article", index=index,
            article=ArticleRef("MADDE", str(index + 1)), document_title="BAZI KANUNLARDA DEĞİŞİKLİK",
        )

    @pytest.fixture
    def populated(self, conn):
        rng = np.random.default_rng(0)
        chunks = [self._chunk(0, ELECTRICITY_ARTICLE), self._chunk(1, TRAFIK_LEAK)]
        store.insert_chunks(
            conn,
            source_path="Kanunlar/omnibus.docx",
            file_sha256="omnibus-hash",
            document_title="BAZI KANUNLARDA DEĞİŞİKLİK",
            chunks=chunks,
            # The off-domain chunk is inserted the way ingest inserts it: no vector.
            embeddings=[rng.random(config.EMBEDDING_DIM, dtype=np.float32), None],
            quality_flag=None,
            scope_labels=["ELECTRICITY", "OFF_DOMAIN"],
        )
        return conn

    def test_excluded_chunk_is_stored_with_no_embedding(self, populated):
        row = populated.execute(
            "SELECT embedding, embedded_at, indexable, scope_label, text FROM chunks "
            "WHERE chunk_index = 1"
        ).fetchone()
        embedding, embedded_at, indexable, label, text = row
        assert embedding is None
        assert embedded_at is None
        assert indexable == 0
        assert label == "OFF_DOMAIN"
        assert text  # retained in full, for provenance

    def test_excluded_chunk_is_absent_from_the_dense_matrix(self, populated):
        matrix, ids = store.fetch_active_embeddings(populated)
        assert matrix.shape[0] == 1
        kept = store.fetch_chunk_metadata(populated, ids[0])
        assert kept["chunk_index"] == 0

    def test_excluded_chunk_is_absent_from_the_bm25_index(self, populated):
        """BM25 indexes text, and the excluded row still holds text -- so this
        filter is the one that could realistically be forgotten."""
        index = lexical.BM25Index.from_connection(populated)
        assert len(index) == 1
        # Terms unique to the excluded article must not exist in the vocabulary.
        for term in lexical.tokenize("sendikalar toplu iş sözleşmesi adlî kontrol"):
            assert term not in index.postings

    def test_query_matching_only_excluded_text_retrieves_nothing(self, populated):
        index = lexical.BM25Index.from_connection(populated)
        assert index.search("Sendikalar ve Toplu İş Sözleşmesi Kanunu itiraz", 5) == []
