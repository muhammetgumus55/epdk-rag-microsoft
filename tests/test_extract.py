"""Tests for Turkish-safe extraction, title derivation and article chunking."""
from pathlib import Path

import pytest

from src import config
from src.chunk import ArticleRef, chunk_document, estimate_tokens, find_articles
from src.extract import (
    ExtractedDoc,
    Page,
    fix_hyphenation,
    normalize_text,
    quality_flags,
    strip_repeating_lines,
    tr_lower,
    tr_upper,
)
from src.titles import extract_title


def make_doc(text: str, pages: list[Page] | None = None, **kw) -> ExtractedDoc:
    doc = ExtractedDoc(
        path=Path("fake.docx"),
        original_filename="fake.docx",
        doc_id="deadbeef-fake",
        file_sha256="deadbeef",
        detected_type=kw.pop("detected_type", "docx"),
        page_number_note=kw.pop("page_number_note", "docx has no fixed page model"),
    )
    doc.pages = pages if pages is not None else [Page(number=None, text=text)]
    return doc


# ---------------------------------------------------------------- casing


class TestTurkishCasing:
    def test_dotted_capital_i_lowercases_to_dotted_i(self):
        # Python's default gives "i̇stanbul" (i + combining dot), which breaks
        # string equality against the correctly-spelled Turkish word.
        assert tr_lower("İSTANBUL") == "istanbul"
        assert "İSTANBUL".lower() != "istanbul"

    def test_dotless_i_uppercases_to_plain_I(self):
        assert tr_upper("ırmak") == "IRMAK"
        assert tr_lower("IRMAK") == "ırmak"

    def test_plain_I_lowercases_to_dotless(self):
        assert tr_lower("I") == "ı"
        assert tr_upper("i") == "İ"

    def test_round_trip_preserves_both_i_forms(self):
        for word in ("İLİŞKİN", "ışık", "İSTİKLAL", "sığır"):
            assert tr_upper(tr_lower(word)) == tr_upper(word)

    def test_other_turkish_letters_survive(self):
        assert tr_lower("ŞEHİR ÇAĞ ÖĞÜT") == "şehir çağ öğüt"


# ---------------------------------------------------------------- hyphenation


class TestHyphenation:
    def test_rejoins_word_split_across_lines(self):
        assert fix_hyphenation("yönet-\nmelik") == "yönetmelik"

    def test_rejoins_with_indentation_after_break(self):
        assert fix_hyphenation("elektrik-\n   piyasası") == "elektrikpiyasası"

    def test_keeps_real_hyphen_not_at_line_break(self):
        assert fix_hyphenation("ön-koşul") == "ön-koşul"

    def test_keeps_hyphen_when_continuation_is_uppercase(self):
        # An uppercase continuation signals a genuine compound, not a split word.
        assert fix_hyphenation("Türkiye-\nAvrupa") == "Türkiye-\nAvrupa"

    def test_keeps_turkish_lowercase_continuation_joined(self):
        assert fix_hyphenation("başla-\nığı") == "başlaığı"

    def test_does_not_join_across_blank_line(self):
        out = fix_hyphenation("madde-\n\nsonraki")
        assert "madde-" in out


# ---------------------------------------------------------------- cleanup


class TestNormalization:
    def test_nfc_normalization(self):
        decomposed = "i̇ş"  # combining marks
        assert normalize_text(decomposed) == normalize_text(decomposed)
        import unicodedata

        assert normalize_text("é") == unicodedata.normalize("NFC", "é")

    def test_collapses_excess_blank_lines(self):
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_nbsp(self):
        assert " " not in normalize_text("a b")


class TestRepeatingLines:
    def test_removes_running_header_across_pages(self):
        pages = [Page(number=i, text=f"EPDK BAŞLIK\nGövde {i}\n{i}") for i in range(1, 6)]
        cleaned = strip_repeating_lines(pages)
        assert all("EPDK BAŞLIK" not in p.text for p in cleaned)
        assert all(f"Gövde {p.number}" in p.text for p in cleaned)

    def test_removes_standalone_page_numbers(self):
        pages = [Page(number=1, text="Gövde metni\n7")]
        assert strip_repeating_lines(pages)[0].text == "Gövde metni"

    def test_keeps_unique_content(self):
        pages = [Page(number=i, text=f"Benzersiz {i}") for i in range(1, 5)]
        cleaned = strip_repeating_lines(pages)
        assert all(p.text.strip() for p in cleaned)


# ---------------------------------------------------------------- quality


class TestQualityChecks:
    def test_flags_empty_document(self):
        flags = quality_flags(make_doc(""))
        assert any(f.startswith("empty") for f in flags)

    def test_flags_scanned_pdf_as_needing_ocr(self):
        doc = make_doc("", pages=[Page(number=1, text="", needs_ocr=True)])
        flags = quality_flags(doc)
        assert any(f.startswith("needs-ocr") for f in flags), flags

    def test_flags_replacement_characters(self):
        doc = make_doc("Elektrik piyasası " + "�" * 50 + " yönetmeliği " * 20)
        assert any(f.startswith("replacement-chars") for f in quality_flags(doc))

    def test_flags_mojibake(self):
        doc = make_doc("YÃ¶netmelik " * 60)
        assert any(f.startswith("mojibake") for f in quality_flags(doc))

    def test_flags_document_with_no_turkish_characters(self):
        doc = make_doc("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. " * 20)
        assert any(f.startswith("no-turkish-chars") for f in quality_flags(doc))

    def test_clean_turkish_document_has_no_decode_flags(self):
        text = (
            "Elektrik Piyasası Kanunu kapsamında yürütülen faaliyetlerin "
            "düzenlenmesine ilişkin usul ve esaslar şöyledir. " * 8
        )
        flags = quality_flags(make_doc(text))
        assert not [f for f in flags if f.split(":")[0] in
                    {"empty", "near-empty", "replacement-chars", "mojibake", "no-turkish-chars"}], flags

    def test_flags_gibberish_run(self):
        doc = make_doc("Yönetmelik " + "x" * 120 + " devamı metin " * 20)
        assert any(f.startswith("gibberish-run") for f in quality_flags(doc))


# ---------------------------------------------------------------- articles


class TestArticleDetection:
    def test_madde_gecici_and_ek_are_distinct_namespaces(self):
        text = (
            "MADDE 5 – Birinci hüküm burada yer alır.\n"
            "GEÇİCİ MADDE 5 – Tamamen farklı bir geçici hüküm.\n"
            "EK MADDE 5 – Yine farklı bir ek hüküm.\n"
        )
        refs = [ref for ref, _, _ in find_articles(text)]
        assert refs == [
            ArticleRef("MADDE", "5"),
            ArticleRef("GEÇİCİ MADDE", "5"),
            ArticleRef("EK MADDE", "5"),
        ]
        # Same number, different provisions - must not collapse.
        assert len(set(refs)) == 3

    def test_accepts_en_dash_hyphen_and_dot_separators(self):
        text = "MADDE 1 – Bir.\nMADDE 2 - İki.\nMADDE 3. Üç.\n"
        assert len(find_articles(text)) == 3

    def test_handles_letter_suffixed_article_numbers(self):
        refs = [r for r, _, _ in find_articles("MADDE 5/A – Eklenen madde metni.\n")]
        assert refs == [ArticleRef("MADDE", "5/A")]

    def test_ignores_inline_cross_references(self):
        text = "Bu Kanunun 5 inci maddesinde belirtilen esaslar uygulanır.\n"
        assert find_articles(text) == []

    def test_article_ref_str_includes_kind(self):
        assert str(ArticleRef("GEÇİCİ MADDE", "3")) == "GEÇİCİ MADDE 3"


class TestChunking:
    def test_one_chunk_per_article(self):
        text = "\n".join(f"MADDE {i} – Kısa bir hüküm metni burada." for i in range(1, 6))
        chunks = chunk_document(make_doc(text))
        article_chunks = [c for c in chunks if c.strategy == "article"]
        assert len(article_chunks) == 5
        assert [c.article.number for c in article_chunks] == ["1", "2", "3", "4", "5"]

    def test_oversized_article_falls_back_to_subitems(self):
        filler = "hüküm metni yönetmelik kapsamında uygulanır " * 40
        text = "MADDE 1 – Giriş cümlesi.\n" + "\n".join(
            f"({i}) {filler}" for i in range(1, 5)
        )
        chunks = chunk_document(make_doc(text), chunk_size=200, overlap=20)
        assert any(c.strategy == "article-sub" for c in chunks), [c.strategy for c in chunks]
        assert all(c.article == ArticleRef("MADDE", "1") for c in chunks if c.article)

    def test_subitem_split_handles_letter_markers(self):
        filler = "elektrik piyasası faaliyetleri düzenlenir " * 40
        text = "MADDE 2 – Giriş.\n" + "\n".join(f"{ch}) {filler}" for ch in "abcd")
        chunks = chunk_document(make_doc(text), chunk_size=200, overlap=20)
        assert any(c.strategy == "article-sub" for c in chunks)

    def test_document_without_articles_uses_token_windows(self):
        text = "Bu belgede madde yapısı bulunmamaktadır. " * 200
        chunks = chunk_document(make_doc(text), chunk_size=120, overlap=10)
        assert chunks
        assert {c.strategy for c in chunks} == {"token-window"}
        assert all(c.article is None for c in chunks)

    def test_chunk_carries_document_title(self):
        from src.titles import TitleInfo

        info = TitleInfo(title="ELEKTRİK PİYASASI KANUNU")
        chunks = chunk_document(make_doc("MADDE 1 – Hüküm."), info)
        assert chunks[0].document_title == "ELEKTRİK PİYASASI KANUNU"
        assert "ELEKTRİK PİYASASI KANUNU" in chunks[0].citation

    def test_page_numbers_present_for_pdf_pages(self):
        pages = [Page(number=1, text="MADDE 1 – Birinci madde metni."),
                 Page(number=2, text="MADDE 2 – İkinci madde metni.")]
        doc = make_doc("", pages=pages, detected_type="pdf", page_number_note=None)
        chunks = chunk_document(doc)
        assert all(c.page_start is not None for c in chunks), [c.page_start for c in chunks]
        assert all(c.page_note is None for c in chunks)

    def test_page_note_explains_absence_for_docx(self):
        chunks = chunk_document(make_doc("MADDE 1 – Hüküm metni."))
        assert chunks[0].page_start is None
        assert chunks[0].page_note  # never silently omitted

    def test_reads_sizes_from_config(self, monkeypatch):
        text = "Madde yapısı yok burada. " * 300
        monkeypatch.setattr(config, "CHUNK_SIZE", 100)
        monkeypatch.setattr(config, "CHUNK_OVERLAP", 10)
        small = chunk_document(make_doc(text))
        monkeypatch.setattr(config, "CHUNK_SIZE", 400)
        large = chunk_document(make_doc(text))
        assert len(small) > len(large)

    def test_empty_document_yields_no_chunks(self):
        assert chunk_document(make_doc("")) == []


# ---------------------------------------------------------------- titles


class TestTitleExtraction:
    def test_extracts_law_title_and_number(self):
        text = (
            "ELEKTRİK PİYASASI KANUNU\n"
            "Kanun No.\t: 4628\n"
            "Kabul Tarihi\t: 20/2/2001\n"
            "Yayımlandığı R. Gazete\t: Tarih : 3/3/2001 Sayı : 24335\n"
        )
        info = extract_title(make_doc(text))
        assert info.title == "ELEKTRİK PİYASASI KANUNU"
        assert info.mevzuat_type == "Kanun"
        assert info.number == "4628"
        assert info.rg_date == "2001-03-03"
        assert info.rg_number == "24335"
        assert info.confidence == "high"

    def test_extracts_title_wrapped_across_two_lines(self):
        text = (
            "YENİLENEBİLİR ENERJİ KAYNAKLARININ ELEKTRİK ENERJİSİ\n"
            "ÜRETİMİ AMAÇLI KULLANIMINA İLİŞKİN KANUN\n"
            "Kanun Numarası\t: 5346\n"
        )
        info = extract_title(make_doc(text))
        assert "YENİLENEBİLİR" in info.title and "KANUN" in info.title
        assert info.number == "5346"

    def test_parses_resmi_gazete_banner_with_turkish_month(self):
        text = (
            "5 Temmuz 2012 PERŞEMBE\tResmî Gazete\tSayı : 28344\n"
            "KANUN\n"
            "BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN\n"
        )
        info = extract_title(make_doc(text))
        assert info.rg_date == "2012-07-05"
        assert info.rg_number == "28344"

    def test_skips_preamble_line_as_title(self):
        text = (
            "Enerji Piyasası Düzenleme Kurumundan:\n"
            "KURUL KARARI\n"
            "Karar No: 10695\n"
            "ENERJİ PİYASASI BİLDİRİM SİSTEMİ KULLANIM TALİMATI\n"
        )
        info = extract_title(make_doc(text))
        assert info.title and "Kurumundan" not in info.title
        assert info.mevzuat_type == "Kurul Kararı"
        assert info.number == "10695"

    @pytest.mark.parametrize(
        "title_line, expected",
        [
            # Turkish agglutination: the type word carries suffixes, and
            # yönetmelik mutates k -> ğ before a vowel.
            ("ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ", "Yönetmelik"),
            ("LİSANS YÖNETMELİĞİNDE DEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK", "Yönetmelik"),
            ("PARA CEZALARI HAKKINDA TEBLİĞİ", "Tebliğ"),
            ("6446 SAYILI ELEKTRİK PİYASASI KANUNU", "Kanun"),
            # "KARARNAME" merely starts like "KARAR" and must not classify as a decision.
            ("KANUN HÜKMÜNDE KARARNAMELERE EKLENMESİNE DAİR KANUN", "Kanun"),
        ],
    )
    def test_type_classification_handles_turkish_suffixes(self, title_line, expected):
        assert extract_title(make_doc(title_line + "\n")).mevzuat_type == expected

    def test_multiline_title_keeps_short_final_line(self):
        # Dropping the short "DAİR KANUN" tail truncates the title *and* hides
        # the type word that classification depends on.
        text = (
            "BÜTÇE KANUNLARINDA YER ALAN BAZI HÜKÜMLERİN İLGİLİ KANUN VE\n"
            "KANUN HÜKMÜNDE KARARNAMELERE EKLENMESİNE\n"
            "DAİR KANUN\n"
            "Kanun No. 6338\n"
        )
        info = extract_title(make_doc(text))
        assert info.title.endswith("DAİR KANUN")
        assert info.mevzuat_type == "Kanun"
        assert info.number == "6338"

    def test_classifies_yonetmelik_over_kanun_when_both_appear(self):
        text = (
            "ELEKTRİK PİYASASI KANUNUNDA DEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK\n"
            "MADDE 1 – Hüküm.\n"
        )
        assert extract_title(make_doc(text)).mevzuat_type == "Yönetmelik"

    def test_recovers_title_glued_to_trailing_metadata(self):
        # Real corpus shape: heading runs straight into Kanun No / Kabul Tarihi / MADDE.
        text = (
            "26 Mart 2020 PERŞEMBE\tResmî Gazete\tSayı : 31080 (Mükerrer)\n"
            "KANUN\n"
            "BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN Kanun No. 7226 "
            "Kabul Tarihi: 25/3/2020 MADDE 1 – 10/6/1949 tarihli...\n"
        )
        info = extract_title(make_doc(text))
        assert info.title == "BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN"
        assert info.number == "7226"
        assert info.rg_date == "2020-03-26"

    def test_strips_issuing_body_prefix_from_title(self):
        text = (
            "12 Aralık 2014 CUMA\tResmî Gazete\tSayı : 29203\n"
            "TEBLİĞ\n"
            "Enerji Piyasası Düzenleme Kurumundan: 6446 SAYILI ELEKTRİK PİYASASI "
            "KANUNUNUN 16 NCI MADDESİ UYARINCA 2015 YILINDA UYGULANACAK PARA "
            "CEZALARI HAKKINDA TEBLİĞ\n"
        )
        info = extract_title(make_doc(text))
        assert info.title.startswith("6446 SAYILI")
        assert "Kurumundan" not in info.title
        assert info.mevzuat_type == "Tebliğ"

    def test_resmi_gazete_banner_is_never_used_as_title(self):
        text = (
            "5 Temmuz 2012 PERŞEMBE\tResmî Gazete\tSayı : 28344\n"
            "KANUN\n"
            "BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN\n"
        )
        info = extract_title(make_doc(text))
        assert "Resmî Gazete" not in (info.title or "")
        assert "PERŞEMBE" not in (info.title or "")

    def test_unidentifiable_document_returns_none_and_flags(self):
        # Lowercase prose with no heading, no number, no RG reference.
        text = "bu belge herhangi bir başlık taşımamaktadır ve düz metinden ibarettir. " * 6
        info = extract_title(make_doc(text))
        assert info.title is None
        assert info.mevzuat_type is None
        assert info.number is None
        assert info.confidence == "none"
        assert any(f.startswith("title-missing") for f in info.flags)
        assert any(f.startswith("number-missing") for f in info.flags)

    def test_empty_document_flags_rather_than_guesses(self):
        info = extract_title(make_doc(""))
        assert info.title is None
        assert info.flags

    def test_never_invents_rg_date_from_unrelated_numbers(self):
        text = "TEBLİĞ\nBu tebliğ 12345 sayılı belgeye atıf yapar.\n"
        info = extract_title(make_doc(text))
        assert info.rg_date is None


class TestTokenEstimate:
    def test_scales_with_word_count(self):
        assert estimate_tokens("bir iki üç") > 0
        assert estimate_tokens("bir iki üç dört beş altı") > estimate_tokens("bir iki üç")

    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0
