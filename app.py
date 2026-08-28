"""Streamlit chat UI over src/answer.py's RAG pipeline.

UI concern only. Retrieval, the confidence gate and generation all live in
src/answer.py, src/retrieval.py and src/session.py untouched -- this module's
job is to load the pipeline once, serialize access to it (Foundry Local
serves exactly one request at a time), gate the page behind a shared
password, and render what the pipeline already produces: gate state,
citations assembled in code, and the measured per-answer timings.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time

import streamlit as st

from src.answer import Answerer, GeneratedAnswer
from src.llm import FoundryUnavailable
from src.session import Session

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
# The OpenAI client and its transport log one INFO line per HTTP call, which
# would otherwise bury the pipeline's own rewrite / hallucination log lines.
for _noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("epdk_ui")

st.set_page_config(
    page_title="EPDK Mevzuat Asistanı",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Visual direction: dense, calm, restrained neutral palette. No default
# Streamlit theming, no purple gradients, no chat-bubble/AI-chrome look.
# Body copy uses a serif stack (no external font fetches -- this runs
# air-gapped-capable, on local Foundry Local hardware) to read like a
# document rather than a messaging app; interactive chrome stays sans.
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
footer {visibility: hidden;}

.block-container {
    padding-top: 2.5rem;
    max-width: 760px;
}

[data-testid="stChatMessageContent"], .stChatMessage {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", "PT Serif", serif;
    font-size: 1.02rem;
    line-height: 1.55;
}

.stChatMessage {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}

.epdk-caption {
    color: #6b6a63;
    font-size: 0.88rem;
    margin-bottom: 1.4rem;
}

.epdk-low-confidence {
    border-left: 4px solid #b8860b;
    background: #fdf6e3;
    color: #6b4e00;
    padding: 0.7rem 1rem;
    border-radius: 3px;
    margin-bottom: 0.8rem;
    font-size: 0.92rem;
    font-family: -apple-system, "Segoe UI", sans-serif;
}

.epdk-not-found {
    border-left: 4px solid #8a8a82;
    background: #f2f1ec;
    padding: 1rem 1.15rem;
    border-radius: 3px;
    white-space: pre-line;
}

.epdk-error {
    border-left: 4px solid #b3402a;
    background: #fbede9;
    color: #7a2a18;
    padding: 1rem 1.15rem;
    border-radius: 3px;
    white-space: pre-line;
}

.epdk-kaynakca-label {
    font-family: -apple-system, "Segoe UI", sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #7a7a72;
    margin-top: 1.1rem;
    margin-bottom: 0.35rem;
}

.epdk-stage {
    color: #8a8a82;
    font-family: -apple-system, "Segoe UI", sans-serif;
    font-size: 0.92rem;
}

.epdk-waiting {
    color: #b8860b;
    font-family: -apple-system, "Segoe UI", sans-serif;
    font-size: 0.92rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Password gate -- single shared password, hmac-compared, never hardcoded.
# --------------------------------------------------------------------------


def _configured_password() -> str | None:
    pw = os.environ.get("EPDK_UI_PASSWORD")
    if pw:
        return pw
    try:
        secret = st.secrets["EPDK_UI_PASSWORD"]
    except Exception:  # noqa: BLE001 - no secrets.toml, or key absent
        return None
    return str(secret) if secret else None


def _check_password(entered: str, configured: str) -> bool:
    return hmac.compare_digest(entered.encode("utf-8"), configured.encode("utf-8"))


def require_login() -> None:
    if st.session_state.get("authenticated"):
        return

    st.title("EPDK Mevzuat Asistanı")
    st.markdown(
        '<p class="epdk-caption">Elektrik piyasası mevzuatı üzerine iç kullanım '
        "asistanı. Devam etmek için parola girin.</p>",
        unsafe_allow_html=True,
    )

    configured = _configured_password()
    with st.form("login_form"):
        entered = st.text_input("Parola", type="password")
        submitted = st.form_submit_button("Giriş")

    if submitted:
        if configured is None:
            st.error(
                "Parola yapılandırılmamış. EPDK_UI_PASSWORD ortam değişkenini "
                "tanımlayıp uygulamayı yeniden başlatın."
            )
            logger.error("login attempted with no EPDK_UI_PASSWORD configured")
        elif _check_password(entered, configured):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Parola yanlış.")

    st.stop()


# --------------------------------------------------------------------------
# Model loading -- once per process, shared across every session, via
# st.cache_resource. A bare module-level lock would NOT be shared this way:
# Streamlit re-executes this whole script on every rerun of every session, so
# only objects handed back from a cached function are actually singletons.
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_answerer() -> Answerer:
    logger.info("loading retriever + chat client (once per process)")
    started = time.perf_counter()
    answerer = Answerer.open()
    logger.info("pipeline ready in %.2fs", time.perf_counter() - started)
    return answerer


@st.cache_resource(show_spinner=False)
def get_generation_lock() -> threading.Lock:
    return threading.Lock()


# --------------------------------------------------------------------------
# Pipeline call: serialized behind the shared lock (Foundry Local serves one
# request at a time -- retrieval's query embedding and generation both hit
# it), with an honest "sırada bekliyor" state instead of a silent queue, and
# a staged progress indicator while the request is in flight.
# --------------------------------------------------------------------------


# A static, all-three-stages label rather than a live animation: the pipeline
# call below runs synchronously, deliberately NOT on a worker thread spawned
# just to animate this. Streamlit runs every script rerun -- including the
# one triggered by submitting this chat message -- on its own OS thread, even
# within the same browser session. Answerer/Retriever are built once via
# st.cache_resource and reused across all of those threads, including the
# sqlite3.Connection inside Retriever: it is opened with
# check_same_thread=False (see store.connect()) specifically so it can be
# shared this way, which is safe without extra locking because
# GENERATION_LOCK already serializes every call -- no two threads ever touch
# the connection at once.
STAGE_LABEL = "\U0001f50d Aranıyor · ⚖️ Değerlendiriliyor · ✍️ Yanıt üretiliyor..."


def run_pipeline(session: Session, question: str) -> GeneratedAnswer:
    lock = get_generation_lock()
    stage_ph = st.empty()
    waited = False
    while not lock.acquire(timeout=0.25):
        waited = True
        stage_ph.markdown(
            '<span class="epdk-waiting">⏳ Sırada bekliyor — başka bir soru '
            "işleniyor...</span>",
            unsafe_allow_html=True,
        )
    if waited:
        logger.info("request %r waited for the generation lock", question[:60])

    try:
        stage_ph.markdown(f'<span class="epdk-stage">{STAGE_LABEL}</span>', unsafe_allow_html=True)
        answer = session.ask(question)
    finally:
        lock.release()

    stage_ph.empty()
    return answer


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_timings(answer: GeneratedAnswer) -> None:
    t = answer.timings
    if not t:
        return
    dense, lexical = t.get("retrieve_dense"), t.get("retrieve_lexical")
    parts = [f"alma+kapı {t.get('retrieve', 0) * 1000:.0f} ms"]
    if dense is not None or lexical is not None:
        parts.append(
            f"(yoğun {(dense or 0) * 1000:.0f} ms / bm25 {(lexical or 0) * 1000:.0f} ms)"
        )
    if "budget" in t:
        parts.append(f"bütçe {t['budget'] * 1000:.1f} ms")
    if "generate" in t:
        parts.append(f"üretim {t['generate'] * 1000:.0f} ms")
    parts.append(f"toplam {t.get('total', 0) * 1000:.0f} ms")
    st.caption(" · ".join(parts))


def render_citations(answer: GeneratedAnswer) -> None:
    st.markdown('<div class="epdk-kaynakca-label">KAYNAKÇA</div>', unsafe_allow_html=True)
    for citation in answer.citations:
        idx = citation.label - 1
        result = answer.results[idx] if 0 <= idx < len(answer.results) else None
        with st.expander(citation.render()):
            if citation.quality_flag:
                st.caption(f"⚠️ Çıkarım uyarısı: {citation.quality_flag}")
            st.caption(citation.source_path)
            with st.container(border=True):
                st.text(result.text if result is not None else "Kaynak metni bulunamadı.")


def render_answer(answer: GeneratedAnswer) -> None:
    if answer.rewritten_query and answer.rewritten_query != answer.question:
        st.caption(f"Takip sorusu olarak yeniden yazıldı: _{answer.rewritten_query}_")

    if answer.decision == "NOT_FOUND":
        st.markdown(f'<div class="epdk-not-found">{answer.text}</div>', unsafe_allow_html=True)
    elif not answer.generated:
        # Gate said to generate, but something downstream failed (context
        # exhausted, server unreachable, empty completion). Distinct from
        # NOT_FOUND: this is a technical failure, not "no provision found".
        st.markdown(f'<div class="epdk-error">{answer.text}</div>', unsafe_allow_html=True)
    else:
        if answer.low_confidence:
            st.markdown(
                '<div class="epdk-low-confidence">⚠️ Düşük güven: bulunan '
                "kaynakların soruyla ilgisi zayıf olabilir. Cevabı dikkatle "
                "değerlendirin.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(answer.text)
        if answer.hallucinated_references:
            st.caption(
                f"⚠️ Model, sağlanmayan {answer.hallucinated_references} "
                "kaynağa atıf yaptı; bu atıflar listeden çıkarıldı."
            )
        if answer.citations:
            render_citations(answer)
        else:
            st.caption("Model hiçbir kaynağa atıf yapmadı.")

    render_timings(answer)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar(answerer: Answerer) -> None:
    with st.sidebar:
        st.markdown("**EPDK Mevzuat Asistanı**")
        st.caption("Elektrik piyasası mevzuatı — iç kullanım")

        if st.button("Yeni sohbet", use_container_width=True):
            st.session_state.session = Session(answerer=answerer)
            st.session_state.messages = []
            st.rerun()

        if st.button("Çıkış yap", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

        with st.expander("Sistem"):
            retriever = answerer.retriever
            st.caption(f"Vektör: {len(retriever):,} parça, {retriever.load_seconds:.2f}s yükleme")
            st.caption(f"BM25: {len(retriever.bm25):,} belge")
            st.caption(f"Sohbet modeli: {answerer.client.model_id}")

        with st.expander("Bilinen sınırlamalar"):
            st.markdown(
                "- **Alan karışıklığı olabilir.** Sistem yalnızca elektrik "
                "piyasası mevzuatını kapsar; doğal gaz, LPG veya petrol "
                "hakkında sorulan bazı sorular yanlışlıkla elektrik "
                "mevzuatından bir cevap döndürebilir.\n"
                "- **Çok parçalı takip soruları eksik cevaplanabilir.** "
                "Aynı anda birden fazla alt soru içeren nüanslı takip "
                "sorularında yalnızca bir kısmına cevap verilebilir."
            )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    require_login()

    try:
        with st.spinner("Modeller yükleniyor (chat + gömme, VRAM'e alınıyor)..."):
            answerer = load_answerer()
    except FoundryUnavailable as exc:
        st.error(f"Foundry Local sunucusuna erişilemedi: {exc}")
        logger.error("Foundry Local unavailable: %s", exc)
        st.stop()
        return

    if "session" not in st.session_state:
        st.session_state.session = Session(answerer=answerer)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_sidebar(answerer)

    st.title("EPDK Mevzuat Asistanı")
    st.markdown(
        '<p class="epdk-caption">Cevaplar yalnızca indekslenmiş elektrik piyasası '
        "mevzuatına dayanır ve kod tarafından derlenen kaynaklarla desteklenir.</p>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                render_answer(msg["answer"])

    question = st.chat_input("Sorunuzu yazın...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                answer = run_pipeline(st.session_state.session, question)
            except Exception as exc:  # noqa: BLE001 - user gets Turkish, log gets detail
                logger.error("pipeline failed for %r: %s", question[:60], exc)
                st.markdown(
                    '<div class="epdk-error">Beklenmeyen bir hata oluştu. Foundry '
                    "Local sunucusunun çalıştığını kontrol edip tekrar deneyin."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                render_answer(answer)
                st.session_state.messages.append({"role": "assistant", "answer": answer})


if __name__ == "__main__":
    main()
