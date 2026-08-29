"""Generate the three demo-video overlay diagrams in docs/presentation/.

Standalone, run-once visual asset generator -- not part of the RAG pipeline.
Renders with matplotlib at exact 1920x1080px so the PNGs drop directly into
video editing software.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.path import Path
import matplotlib.patches as mpatches
from pathlib import Path as FsPath

OUT_DIR = FsPath(__file__).resolve().parent.parent / "docs" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080

# ---------------------------------------------------------------------------
# Palette -- restrained blues/grays for a regulatory/energy-sector context.
# ---------------------------------------------------------------------------
NAVY = "#1B2A41"
BLUE = "#2E5C8A"
BLUE_MID = "#4A7BAA"
BLUE_LIGHT = "#DCE6F0"
BLUE_PALE = "#EEF3F8"
GRAY = "#5B6B7C"
GRAY_LIGHT = "#E3E8ED"
GRAY_LINE = "#9AA7B4"
TEXT_DARK = "#1B2A41"
TEXT_MUTED = "#5B6B7C"
WHITE = "#FFFFFF"
GREEN = "#3D7A5C"
GREEN_LIGHT = "#E4F0EA"
AMBER = "#A9782E"
AMBER_LIGHT = "#F3EADA"
RED = "#A6473F"
RED_LIGHT = "#F3E2E0"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def new_canvas(transparent: bool = False):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    if not transparent:
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(WHITE)
    else:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    return fig, ax


def box(
    ax,
    cx,
    cy,
    w,
    h,
    title,
    subtitle=None,
    facecolor=WHITE,
    edgecolor=BLUE,
    textcolor=TEXT_DARK,
    subcolor=TEXT_MUTED,
    linewidth=2.4,
    title_size=25,
    sub_size=18,
    dashed=False,
    rounding=16,
    zorder=3,
):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=(0, (7, 5)) if dashed else "solid",
        zorder=zorder,
    )
    ax.add_patch(patch)
    n_title_lines = title.count("\n") + 1
    if subtitle:
        ax.text(
            cx, cy + h * 0.16, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=textcolor, zorder=zorder + 1,
            multialignment="center", linespacing=1.25,
        )
        ax.text(
            cx, cy - h * (0.20 + 0.09 * (n_title_lines - 1)), subtitle, ha="center", va="center",
            fontsize=sub_size, color=subcolor, zorder=zorder + 1,
        )
    else:
        ax.text(
            cx, cy, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=textcolor, zorder=zorder + 1,
            multialignment="center", linespacing=1.25,
        )
    return patch


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=3.0, dashed=False, rad=0.0, zorder=2):
    style = "-|>"
    connectionstyle = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    fa = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=26,
        linewidth=lw,
        color=color,
        linestyle=(0, (6, 4)) if dashed else "solid",
        connectionstyle=connectionstyle,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(fa)


def section_label(ax, x, y, text, color=TEXT_MUTED, size=20, ha="left"):
    ax.text(
        x, y, text, ha=ha, va="center", fontsize=size,
        color=color, fontweight="bold", family="Segoe UI",
    )


# ---------------------------------------------------------------------------
# Diagram 1 -- pipeline_overview.png
# ---------------------------------------------------------------------------
def make_pipeline_overview():
    fig, ax = new_canvas()

    # Row 1: ingestion (top)
    row1_y = 800
    row1_h = 190
    row1_labels = [
        ("EPDK.gov.tr", "kaynak"),
        ("Otomatik\nİndirme", "675 belge"),
        ("Extraction", "PDF / Word"),
        ("Madde Bazlı\nChunking", "27.047 chunk"),
        ("Embedding", None),
        ("SQLite\nStorage", None),
    ]
    n1 = len(row1_labels)
    margin = 60
    gap1 = 34
    bw1 = (W - 2 * margin - (n1 - 1) * gap1) / n1
    centers1 = [margin + bw1 / 2 + i * (bw1 + gap1) for i in range(n1)]

    section_label(ax, margin, row1_y + row1_h / 2 + 55,
                  "VERİ HAZIRLIĞI  ·  bir kez çalışır", color=BLUE, size=22)

    for i, (cx, (title, sub)) in enumerate(zip(centers1, row1_labels)):
        is_source = i == 0
        is_storage = i == n1 - 1
        fc = NAVY if is_storage else (BLUE_PALE if is_source else WHITE)
        ec = NAVY if is_storage else BLUE
        tc = WHITE if is_storage else TEXT_DARK
        sc = "#C7D3E0" if is_storage else TEXT_MUTED
        box(
            ax, cx, row1_y, bw1, row1_h,
            title, sub,
            facecolor=fc, edgecolor=ec, textcolor=tc, subcolor=sc,
            dashed=is_source, title_size=22, sub_size=17,
        )

    for i in range(n1 - 1):
        x1 = centers1[i] + bw1 / 2
        x2 = centers1[i + 1] - bw1 / 2
        arrow(ax, x1, row1_y, x2, row1_y, color=BLUE_MID, lw=3.2)

    # Row 2: query (bottom)
    row2_y = 260
    row2_h = 190
    row2_labels = [
        ("Kullanıcı\nSorusu", None),
        ("Hybrid\nRetrieval", "Dense + BM25"),
        ("Confidence\nGate", None),
        ("Foundry Local\nGeneration", None),
        ("Cited\nAnswer", None),
    ]
    row2_title_sizes = [23, 23, 23, 20, 23]
    n2 = len(row2_labels)
    gap2 = 40
    bw2 = (W - 2 * margin - (n2 - 1) * gap2) / n2
    centers2 = [margin + bw2 / 2 + i * (bw2 + gap2) for i in range(n2)]

    section_label(ax, margin, row2_y + row2_h / 2 + 55,
                  "SORGU  ·  her soru için çalışır", color=BLUE, size=22)

    for i, (cx, (title, sub)) in enumerate(zip(centers2, row2_labels)):
        is_start = i == 0
        is_end = i == n2 - 1
        fc = NAVY if is_end else WHITE
        ec = NAVY if is_end else BLUE
        tc = WHITE if is_end else TEXT_DARK
        sc = "#C7D3E0" if is_end else TEXT_MUTED
        box(
            ax, cx, row2_y, bw2, row2_h,
            title, sub,
            facecolor=fc, edgecolor=ec, textcolor=tc, subcolor=sc,
            dashed=is_start, title_size=row2_title_sizes[i], sub_size=17,
        )

    for i in range(n2 - 1):
        x1 = centers2[i] + bw2 / 2
        x2 = centers2[i + 1] - bw2 / 2
        arrow(ax, x1, row2_y, x2, row2_y, color=BLUE_MID, lw=3.2)

    # Connector: Storage (row1 last) -> Hybrid Retrieval (row2 second)
    sx, sy = centers1[-1], row1_y - row1_h / 2
    tx, ty = centers2[1], row2_y + row2_h / 2
    mid_y = (sy + ty) / 2
    ax.plot([sx, sx], [sy, mid_y], color=GRAY_LINE, lw=2.6,
            linestyle=(0, (7, 5)), zorder=1)
    ax.plot([sx, tx], [mid_y, mid_y], color=GRAY_LINE, lw=2.6,
            linestyle=(0, (7, 5)), zorder=1)
    arrow(ax, tx, mid_y, tx, ty, color=GRAY_LINE, lw=2.6, dashed=True)
    ax.text((sx + tx) / 2, mid_y + 26, "sorgulanır", ha="center", va="center",
            fontsize=17, color=TEXT_MUTED, style="italic")

    fig.savefig(OUT_DIR / "pipeline_overview.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagram 2 -- hybrid_search.png
# ---------------------------------------------------------------------------
def make_hybrid_search():
    fig, ax = new_canvas()

    query_y = 970
    paths_y = 760
    fusion_y = 545
    gate_y = 335
    branch_y = 110

    # Query box
    box(ax, W / 2, query_y, 380, 110, "Sorgu", facecolor=BLUE_PALE, edgecolor=BLUE,
        title_size=26, dashed=True)

    # Dense / BM25 paths
    dense_cx, bm25_cx = 560, 1360
    path_w, path_h = 620, 190
    box(ax, dense_cx, paths_y, path_w, path_h, "Dense Search (Semantic)",
        "embedding tabanlı benzerlik", facecolor=WHITE, edgecolor=BLUE,
        title_size=24, sub_size=18)
    box(ax, bm25_cx, paths_y, path_w, path_h, "BM25 Search (Keyword)",
        "birebir terim eşleşmesi", facecolor=WHITE, edgecolor=BLUE,
        title_size=24, sub_size=18)

    arrow(ax, W / 2 - 70, query_y - 55, dense_cx + 60, paths_y + path_h / 2 + 6,
          color=BLUE_MID, lw=3.0)
    arrow(ax, W / 2 + 70, query_y - 55, bm25_cx - 60, paths_y + path_h / 2 + 6,
          color=BLUE_MID, lw=3.0)

    # RRF Fusion
    fusion_w, fusion_h = 460, 150
    box(ax, W / 2, fusion_y, fusion_w, fusion_h, "RRF Fusion",
        facecolor=NAVY, edgecolor=NAVY, textcolor=WHITE, title_size=26)

    arrow(ax, dense_cx, paths_y - path_h / 2, W / 2 - 40, fusion_y + fusion_h / 2 + 8,
          color=BLUE_MID, lw=3.0, rad=0.12)
    arrow(ax, bm25_cx, paths_y - path_h / 2, W / 2 + 40, fusion_y + fusion_h / 2 + 8,
          color=BLUE_MID, lw=3.0, rad=-0.12)

    # Confidence Gate
    gate_w, gate_h = 480, 150
    box(ax, W / 2, gate_y, gate_w, gate_h, "Confidence Gate",
        facecolor=BLUE, edgecolor=BLUE, textcolor=WHITE, title_size=26)
    arrow(ax, W / 2, fusion_y - fusion_h / 2, W / 2, gate_y + gate_h / 2,
          color=BLUE_MID, lw=3.2)

    # Branches
    branch_w, branch_h = 430, 150
    branches = [
        (300, "ANSWER", GREEN, GREEN_LIGHT),
        (W / 2, "ANSWER_WEAK", AMBER, AMBER_LIGHT),
        (1620, "NOT_FOUND", GRAY, GRAY_LIGHT),
    ]
    for bx, label, edge, fill in branches:
        box(ax, bx, branch_y, branch_w, branch_h, label,
            facecolor=fill, edgecolor=edge, textcolor=edge, title_size=24)
        rad = 0.0
        if bx < W / 2 - 5:
            rad = 0.18
        elif bx > W / 2 + 5:
            rad = -0.18
        arrow(ax, W / 2, gate_y - gate_h / 2, bx, branch_y + branch_h / 2 + 6,
              color=edge, lw=2.8, rad=rad)

    fig.savefig(OUT_DIR / "hybrid_search.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagram 3 -- why_hybrid.png
# ---------------------------------------------------------------------------
def draw_x_mark(ax, cx, cy, r, color):
    circ = Circle((cx, cy), r, facecolor=WHITE, edgecolor=color, linewidth=3.4, zorder=5)
    ax.add_patch(circ)
    k = r * 0.42
    ax.plot([cx - k, cx + k], [cy - k, cy + k], color=color, lw=5, zorder=6, solid_capstyle="round")
    ax.plot([cx - k, cx + k], [cy + k, cy - k], color=color, lw=5, zorder=6, solid_capstyle="round")


def draw_check_mark(ax, cx, cy, r, color):
    circ = Circle((cx, cy), r, facecolor=WHITE, edgecolor=color, linewidth=3.4, zorder=5)
    ax.add_patch(circ)
    ax.plot(
        [cx - r * 0.42, cx - r * 0.05, cx + r * 0.5],
        [cy - r * 0.02, cy - r * 0.4, cy + r * 0.35],
        color=color, lw=5, zorder=6, solid_capstyle="round", solid_joinstyle="round",
    )


def make_why_hybrid():
    fig, ax = new_canvas()

    query_text = '"Doğal gaz dağıtım bedeli\nnasıl hesaplanır?"'

    panel_w = 860
    panel_h = 860
    panel_y = W  # unused placeholder
    gap = 40
    left_cx = 60 + panel_w / 2
    right_cx = W - 60 - panel_w / 2
    panel_top = 980
    panel_bottom = 60

    def panel_bg(cx, edge):
        FsPathUnused = None
        rect = FancyBboxPatch(
            (cx - panel_w / 2, panel_bottom), panel_w, panel_top - panel_bottom,
            boxstyle="round,pad=0,rounding_size=22",
            linewidth=2.0, edgecolor=GRAY_LIGHT, facecolor=BLUE_PALE, zorder=0,
        )
        ax.add_patch(rect)

    panel_bg(left_cx, RED)
    panel_bg(right_cx, GREEN)

    header_y = 900
    ax.text(left_cx, header_y, "Sadece Dense Search", ha="center", va="center",
            fontsize=30, fontweight="bold", color=TEXT_DARK)
    ax.text(right_cx, header_y, "Hybrid Search (Dense + BM25)", ha="center", va="center",
            fontsize=30, fontweight="bold", color=TEXT_DARK)

    query_y = 760
    for cx in (left_cx, right_cx):
        box(ax, cx, query_y, 700, 170, query_text, facecolor=WHITE, edgecolor=BLUE,
            title_size=22)

    arrow(ax, left_cx, query_y - 90, left_cx, 470, color=GRAY_LINE, lw=3.2)
    arrow(ax, right_cx, query_y - 90, right_cx, 470, color=GRAY_LINE, lw=3.2)

    # Left: wrong result
    result_y = 360
    box(ax, left_cx, result_y, 720, 190,
        "Elektrik Dağıtım Bağlantı\nBedelleri Tebliği",
        facecolor=RED_LIGHT, edgecolor=RED, textcolor=RED, title_size=22)
    draw_x_mark(ax, left_cx, 165, 42, RED)
    ax.text(left_cx, 100, "Yanlış eşleşme", ha="center", va="center",
            fontsize=22, fontweight="bold", color=RED)

    # Right: correct outcome
    box(ax, right_cx, result_y, 720, 190,
        "NOT_FOUND",
        "doğru şekilde reddedildi",
        facecolor=GREEN_LIGHT, edgecolor=GREEN, textcolor=GREEN, subcolor=GREEN,
        title_size=26, sub_size=20)
    draw_check_mark(ax, right_cx, 165, 42, GREEN)
    ax.text(right_cx, 100, "Doğru sonuç", ha="center", va="center",
            fontsize=22, fontweight="bold", color=GREEN)

    fig.savefig(OUT_DIR / "why_hybrid.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    make_pipeline_overview()
    make_hybrid_search()
    make_why_hybrid()
    print(f"Saved 3 diagrams to {OUT_DIR}")
