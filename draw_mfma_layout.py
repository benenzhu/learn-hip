#!/usr/bin/env python3
"""Generate small SVG diagrams for MFMA FP4 layout notes."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Style:
    stroke: str = "#333"
    text: str = "#222"
    blue: str = "#1f5f99"
    red: str = "#b33"
    kg_colors: tuple[str, ...] = ("#ffe6e6", "#e8f4ff", "#e7ffe7", "#fff3d9")


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>",
            ".title{font:700 28px Arial,sans-serif;fill:#1f5f99}",
            ".label{font:700 24px Arial,sans-serif;fill:#b33}",
            ".dim{font:18px Arial,sans-serif;fill:#222}",
            ".text{font:15px Arial,sans-serif;fill:#222}",
            ".small{font:12px Arial,sans-serif;fill:#333}",
            ".tiny{font:10px Arial,sans-serif;fill:#333}",
            ".callout{font:14px Arial,sans-serif;fill:#222}",
            "</style>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
        ]

    def rect(self, x, y, w, h, fill="#fff", stroke="#333", sw=1.5):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke="#777", sw=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def arrow(self, x1, y1, x2, y2, stroke="#333", sw=1.4):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#arrow)"/>'
        )

    def text(self, x, y, s, cls="text", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>')

    def multiline(self, x, y, lines, cls="callout", line_h=18):
        for i, line in enumerate(lines):
            self.text(x, y + i * line_h, line, cls)

    def save(self, path: Path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def draw_split_matrix(
    svg: Svg,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    horizontal: bool,
    labels: list[str],
    colors: tuple[str, ...],
    title: str,
    dim_top: str | None = None,
    dim_side: str | None = None,
):
    n = len(labels)
    if horizontal:
        seg_w = w / n
        for i, label in enumerate(labels):
            sx = x + i * seg_w
            svg.rect(sx, y, seg_w, h, colors[i % len(colors)], "#777", 1)
            svg.text(sx + 10, y + 24, label, "small")
            if i:
                svg.line(sx, y, sx, y + h)
    else:
        seg_h = h / n
        for i, label in enumerate(labels):
            sy = y + i * seg_h
            svg.rect(x, sy, w, seg_h, colors[i % len(colors)], "#777", 1)
            svg.text(x + 12, sy + seg_h / 2 + 4, label, "small")
            if i:
                svg.line(x, sy, x + w, sy)
    svg.rect(x, y, w, h, "none", "#333", 1.5)
    svg.text(x - 42, y + h / 2 + 8, title, "label")
    if dim_top:
        svg.text(x + w / 2, y - 14, dim_top, "dim", "middle")
    if dim_side:
        svg.text(x + w + 15, y + h / 2, dim_side, "dim")


def draw_scale_matrix(
    svg: Svg,
    x: int,
    y: int,
    *,
    title: str,
    row_label: str,
    col_label: str,
    colors: tuple[str, ...],
):
    w, h = 320, 80
    seg_h = h / 4
    for i in range(4):
        sy = y + i * seg_h
        svg.rect(x, sy, w, seg_h, colors[i], "#777", 1)
        svg.text(x + w + 10, sy + seg_h / 2 + 4, f"kg{i}", "tiny")
    svg.rect(x, y, w, h, "none", "#333", 1.5)
    svg.text(x - 45, y + h / 2 + 8, title, "label")
    svg.text(x + w / 2, y - 12, col_label, "dim", "middle")
    svg.text(x + 10, y + h + 24, row_label, "small")


def draw_16x16x128_original(path: Path):
    style = Style()
    svg = Svg(1180, 820)
    svg.text(55, 60, "v_mfma_scale_f32_16x16x128_f4_f4 original layout", "title")

    draw_scale_matrix(
        svg,
        95,
        110,
        title="Ax",
        row_label="Scale[lane_k, lane_mn] = [4 K-groups, 16 rows]",
        col_label="16 rows",
        colors=style.kg_colors,
    )

    draw_split_matrix(
        svg,
        95,
        340,
        520,
        160,
        horizontal=True,
        labels=["K 0..31\nlanes 0..15", "K 32..63\nlanes 16..31", "K 64..95\nlanes 32..47", "K 96..127\nlanes 48..63"],
        colors=style.kg_colors,
        title="A",
        dim_top="K:128",
        dim_side="M:16",
    )
    svg.text(165, 530, "A lane reads A[row=lane_mn, K=lane_k*32 : +32]", "small")

    draw_scale_matrix(
        svg,
        750,
        70,
        title="Bx",
        row_label="Scale[lane_k, lane_mn] = [4 K-groups, 16 cols]",
        col_label="16 cols",
        colors=style.kg_colors,
    )

    draw_split_matrix(
        svg,
        750,
        230,
        320,
        360,
        horizontal=False,
        labels=["K 0..31 / lanes 0..15", "K 32..63 / lanes 16..31", "K 64..95 / lanes 32..47", "K 96..127 / lanes 48..63"],
        colors=style.kg_colors,
        title="B",
        dim_top="N:16",
        dim_side="K:128",
    )
    svg.text(780, 620, "B lane reads B[K=lane_k*32 : +32, col=lane_mn]", "small")

    draw_split_matrix(
        svg,
        750,
        670,
        320,
        120,
        horizontal=False,
        labels=["rows 0..3 by lane_k=0", "rows 4..7 by lane_k=1", "rows 8..11 by lane_k=2", "rows 12..15 by lane_k=3"],
        colors=style.kg_colors,
        title="C/D",
        dim_top="N:16",
        dim_side="M:16",
    )

    # svg.text(95, 640, "lane_mn = threadIdx.x % 16", "text")
    # svg.text(95, 665, "lane_k  = threadIdx.x / 16", "text")
    # svg.text(95, 690, "Each lane: 32 fp4 A values, 32 fp4 B values, 4 fp32 accumulators.", "text")
    # svg.text(95, 715, "E8M0 scale: one byte per 32 K values; lane_k selects one of four K groups.", "text")

    svg.save(path)


def draw_16x16x128_a_c_annotated(path: Path):
    style = Style()
    svg = Svg(1180, 860)
    svg.text(45, 55, "v_mfma_scale_f32_16x16x128_f4_f4: A / Ax / C layout", "title")
    cell = 16

    # Ax: 4 K-groups x 16 rows, drawn as a compact vertical scale panel.
    svg.text(50, 195, "Ax", "label")
    ax_x, ax_y, ax_w, ax_h = 105, 110, 4 * cell, 16 * cell
    row_h = ax_h / 16
    col_w = ax_w / 4
    for kg in range(4):
        svg.rect(ax_x + kg * col_w, ax_y, col_w, ax_h, style.kg_colors[kg], "#ddd", 0.5)
    for kg in range(1, 4):
        svg.line(ax_x + kg * col_w, ax_y, ax_x + kg * col_w, ax_y + ax_h)
    for r in range(1, 16):
        svg.line(ax_x, ax_y + r * row_h, ax_x + ax_w, ax_y + r * row_h, "#bbb", 0.5)
    svg.rect(ax_x, ax_y, ax_w, ax_h, "none", "#333", 1.5)
    svg.text(ax_x + ax_w / 2, ax_y - 12, "K//32:4", "dim", "middle")
    svg.text(ax_x + ax_w + 12, ax_y + ax_h / 2, "M:16", "dim")
    for kg in range(4):
        svg.text(ax_x + kg * col_w + col_w / 2, ax_y + ax_h + 12, f"{kg}", "tiny", "middle")
    for row in (0, 1, 14, 15):
        for kg in range(4):
            svg.text(
                ax_x + (kg + 0.5) * col_w,
                ax_y + (row + 0.72) * row_h,
                f"{kg * 16 + row}",
                "tiny",
                "middle",
            )
    svg.text(ax_x - 36, ax_y + 0.72 * row_h, "row0", "tiny")
    svg.text(ax_x - 36, ax_y + 1.72 * row_h, "row1", "tiny")
    svg.text(ax_x - 28, ax_y + 8.5 * row_h, "...", "small")
    svg.text(ax_x - 42, ax_y + 14.72 * row_h, "row14", "tiny")
    svg.text(ax_x - 42, ax_y + 15.72 * row_h, "row15", "tiny")

    svg.multiline(
        ax_x,
        ax_y + ax_h + 45,
        [
            # "8 Bit (E8M0)",
            # "lane_mn = tid % 16",
            # "lane_k  = tid / 16",
            # "Ax[lane_k, lane_mn]",
        ],
    )

    # A matrix. This is an abstract register view: each visible cell is one
    # packed 32-bit register, i.e. 8 fp4 values.
    svg.text(355, 210, "A", "label")
    a_x, a_y, a_w, a_h = 410, 130, 32 * cell, 16 * cell
    visible_cols_per_kg = 8
    total_visible_cols = 4 * visible_cols_per_kg
    cell_w = a_w / total_visible_cols
    row_h_a = a_h / 16
    for kg in range(4):
        sx = a_x + kg * visible_cols_per_kg * cell_w
        sw = visible_cols_per_kg * cell_w
        svg.rect(sx, a_y, sw, a_h, style.kg_colors[kg], "#ddd", 0.5)
        for local_col in range(visible_cols_per_kg):
            col_x = sx + local_col * cell_w
            svg.line(col_x, a_y, col_x, a_y + a_h, "#cfcfcf", 0.45)
            svg.text(col_x + cell_w / 2, a_y + row_h_a * 0.72, f"{kg*16}", "tiny", "middle")
            svg.text(col_x + cell_w / 2, a_y + row_h_a * 1.72, f"{kg*16 + 1}", "tiny", "middle")
            svg.text(col_x + cell_w / 2, a_y + row_h_a * 14.72, f"{kg*16 + 14}", "tiny", "middle")
            svg.text(col_x + cell_w / 2, a_y + row_h_a * 15.72, f"{kg*16 + 15}", "tiny", "middle")
    for kg in range(1, 4):
        x = a_x + kg * visible_cols_per_kg * cell_w
        svg.line(x, a_y, x, a_y + a_h, "#555", 1.2)
    svg.line(a_x + a_w, a_y, a_x + a_w, a_y + a_h, "#cfcfcf", 0.45)
    for r in range(1, 16):
        dash = None if r in (1, 2) else "4 4"
        svg.line(a_x, a_y + r * row_h_a, a_x + a_w, a_y + r * row_h_a, "#aaa", 0.7, dash)
    svg.rect(a_x, a_y, a_w, a_h, "none", "#333", 1.5)
    svg.text(a_x + a_w / 2, a_y - 14, "K:128", "dim", "middle")
    svg.text(a_x + a_w + 14, a_y + a_h / 2, "M:16", "dim")
    svg.text(a_x - 34, a_y + row_h_a * 0.72, "row0", "tiny")
    svg.text(a_x - 34, a_y + row_h_a * 1.72, "row1", "tiny")
    svg.text(a_x - 28, a_y + row_h_a * 8.5, "...", "small")
    svg.text(a_x - 40, a_y + row_h_a * 14.72, "row14", "tiny")
    svg.text(a_x - 40, a_y + row_h_a * 15.72, "row15", "tiny")

    # Zoom one lane-0 register to show it contains 8 fp4 values.
    zoom_x, zoom_y = a_x - 60, a_y - 35
    zoom_cell_w, zoom_h = 18, 18
    for i in range(8):
        svg.rect(zoom_x + i * zoom_cell_w, zoom_y, zoom_cell_w, zoom_h, "#ffe6e6", "#777", 0.8)
        svg.text(zoom_x + (i + 0.5) * zoom_cell_w, zoom_y + 13, "0", "tiny", "middle")
    svg.text(zoom_x + 4 * zoom_cell_w, zoom_y - 10, "8 fp4", "dim", "middle")
    svg.arrow(zoom_x + 4 * zoom_cell_w, zoom_y + zoom_h, a_x + 0.5 * cell_w, a_y)# + 0.5 * row_h_a)

    # C matrix.
    svg.text(355, 500, "C/D", "label")
    c_x, c_y, c_w, c_h = 410, 500, 16 * cell, 16 * cell
    c_cell_w = c_w / 16
    c_cell_h = c_h / 16
    for lk in range(4):
        svg.rect(c_x, c_y + lk * 4 * c_cell_h, c_w, 4 * c_cell_h, style.kg_colors[lk], "#ddd", 0.5)
    for col in range(17):
        svg.line(c_x + col * c_cell_w, c_y, c_x + col * c_cell_w, c_y + c_h, "#cfcfcf", 0.45)
    for row in range(17):
        dash = None if row in (1, 2, 14, 15) else "4 4"
        svg.line(c_x, c_y + row * c_cell_h, c_x + c_w, c_y + row * c_cell_h, "#aaa", 0.7, dash)
    visible_c_cols = (0, 1, 14, 15)
    for row in range(16):
        lane_k_for_row = row // 4
        for col in visible_c_cols:
            svg.text(
                c_x + (col + 0.5) * c_cell_w,
                c_y + (row + 0.72) * c_cell_h,
                f"{lane_k_for_row * 16 + col}",
                "tiny",
                "middle",
            )
    svg.rect(c_x, c_y, c_w, c_h, "none", "#333", 1.5)
    svg.text(c_x + c_w / 2, c_y - 14, "N:16", "dim", "middle")
    svg.text(c_x + c_w + 14, c_y + c_h / 2, "M:16", "dim")
    svg.text(c_x - 34, c_y + c_cell_h * 0.72, "row0", "tiny")
    svg.text(c_x - 34, c_y + c_cell_h * 1.72, "row1", "tiny")
    svg.text(c_x - 28, c_y + c_cell_h * 8.5, "...", "small")
    svg.text(c_x - 40, c_y + c_cell_h * 14.72, "row14", "tiny")
    svg.text(c_x - 40, c_y + c_cell_h * 15.72, "row15", "tiny")
    svg.text(c_x + 0.5 * c_cell_w, c_y + c_h + 16, "c0", "tiny", "middle")
    svg.text(c_x + 1.5 * c_cell_w, c_y + c_h + 16, "c1", "tiny", "middle")
    svg.text(c_x + 14.5 * c_cell_w, c_y + c_h + 16, "c14", "tiny", "middle")
    svg.text(c_x + 15.5 * c_cell_w, c_y + c_h + 16, "c15", "tiny", "middle")
    # svg.text(60, 865, "B/Bx omitted here: same lane_k/lane_mn idea, but lane_mn indexes N columns instead of A rows.", "text")
    svg.save(path)


def main():
    base = Path(__file__).parent
    overview = base / "mfma_16x16x128_fp4_layout_generated.svg"
    annotated = base / "mfma_16x16x128_fp4_a_c_annotated.svg"
    draw_16x16x128_original(overview)
    draw_16x16x128_a_c_annotated(annotated)
    print(overview)
    print(annotated)


if __name__ == "__main__":
    main()
