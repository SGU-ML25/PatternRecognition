#!/usr/bin/env python3
"""Ghép presentation SkyPose thành 1 file HTML standalone (chạy offline).

Tái sử dụng CSS + thư viện mermaid + viewer sơ đồ của file mẫu (build/vendor/),
thay nội dung, bảng màu và nhúng biểu đồ huấn luyện dạng base64.

Chạy: python3 report/presentation/build/assemble.py
"""
import base64
import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

BUILD = Path(__file__).resolve().parent
REPO = BUILD.parents[2]
OUT = BUILD.parent / "skypose-presentation.html"
FIG_DIR = REPO / "runs" / "20261002-1405_resnet18_kaggle" / "figures"
VENDOR = BUILD / "vendor"

TITLE = "SkyPose — Presentation"
CONTENT = ["c00_intro.html", "c01_04_model.html", "c05_09_system.html", "c10_11_compare.html", "c12_end.html"]

# Bảng màu mẫu (cam) -> xanh bầu trời
PALETTE = {
    "--primary: #e65100;": "--primary: #0369a1;",
    "--primary-light: #ff833a;": "--primary-light: #38bdf8;",
    "--primary-dark: #ac1900;": "--primary-dark: #0c4a6e;",
    "--accent: #00838f;": "--accent: #0d9488;",
    "--accent-light: #4fb3bf;": "--accent-light: #5eead4;",
}

EXTRA_CSS = """
        /* ===== OVERRIDE cho presentation SkyPose ===== */
        .code-block { white-space: pre; tab-size: 4; }
        nav { height: auto; min-height: 56px; padding: .35rem 2rem; }
        nav .logo { white-space: nowrap; font-size: 1rem; }
        nav .nav-links { row-gap: .25rem; justify-content: flex-end; }
        .hero { margin-top: 56px; }
        tr:hover td { background: #e0f2fe; }
        .flow-step { white-space: normal; max-width: 260px; text-align: center; line-height: 1.25; }
        .highlight.warning { background: #fff8e1; }
        .dm-body .vp { display: block; width: max-content; }
        .card, .arch-module, .card-grid > *, .arch-grid > *, .tier-item, .timeline-item { min-width: 0; }
        code { overflow-wrap: anywhere; }
        li, p, td, th { overflow-wrap: break-word; }
        .arch-grid { margin: 1.5rem 0; }

        /* Hình nhúng */
        .fig-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.2rem; margin: 1.2rem 0; }
        .fig-grid.one { grid-template-columns: minmax(0, 760px); justify-content: center; }
        .fig { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .8rem; margin: 0; min-width: 0; }
        .fig img { width: 100%; height: auto; display: block; border-radius: 6px; }
        .fig figcaption { font-size: .82rem; color: var(--text-secondary); margin-top: .4rem; text-align: center; }

        /* Nhãn nguồn số liệu */
        .tag { display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .5px; padding: .1rem .5rem;
               border-radius: 10px; vertical-align: middle; }
        .tag.measured { background: #dcfce7; color: #166534; }
        .tag.planned { background: #fef3c7; color: #92400e; }
        .tag.ref { background: #e0e7ff; color: #3730a3; }
        .note { font-size: .85rem; color: var(--text-secondary); margin-bottom: .4rem; }
        td.up { color: var(--success); font-weight: 700; }

        @media (max-width: 768px) {
            .fig-grid { grid-template-columns: minmax(0, 1fr); }
            nav { padding: .35rem 1rem; }
        }
"""

NAV_FIX_JS = """    <script>
        // Giữ hero không bị nav (có thể cao hơn 56px) che mất
        (function () {
            function fixNavOffset() {
                var nav = document.querySelector('nav');
                var hero = document.querySelector('.hero');
                if (nav && hero) hero.style.marginTop = nav.offsetHeight + 'px';
            }
            window.addEventListener('load', fixNavOffset);
            window.addEventListener('resize', fixNavOffset);
            fixNavOffset();
        })();
    </script>"""

SVG_FIX = """
                var svgEl = vp.querySelector('svg');
                if (svgEl) {
                    var vb = svgEl.viewBox && svgEl.viewBox.baseVal;
                    if (vb && vb.width) {
                        svgEl.setAttribute('width', Math.round(vb.width));
                        svgEl.setAttribute('height', Math.round(vb.height));
                        svgEl.style.maxWidth = 'none';
                    }
                }"""


def diagram_js():
    spec = importlib.util.spec_from_file_location("diagram_data", BUILD / "diagram_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main()
    return buf.getvalue()


def embed_figures(html):
    def repl(m):
        path = FIG_DIR / f"{m.group(1)}.png"
        if not path.exists():
            raise SystemExit(f"Thiếu hình: {path}")
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return re.sub(r"\{\{FIG:([a-z0-9_]+)\}\}", repl, html)


def main():
    head = (VENDOR / "part_head.html").read_text(encoding="utf-8")
    overlay = (VENDOR / "part_overlay.html").read_text(encoding="utf-8")
    lib = (VENDOR / "part_mermaid_lib.html").read_text(encoding="utf-8")
    tail = (VENDOR / "part_tail.html").read_text(encoding="utf-8")

    old_render = "                vp.innerHTML = result.svg;"
    if old_render not in tail:
        raise SystemExit("Không tìm thấy đoạn render SVG trong part_tail.html")
    tail = tail.replace(old_render, old_render + SVG_FIX, 1)

    head = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", head, count=1)
    for old, new in PALETTE.items():
        if old not in head:
            raise SystemExit(f"Không tìm thấy khai báo màu: {old}")
        head = head.replace(old, new)
    idx = head.rindex("    </style>")
    head = head[:idx] + EXTRA_CSS + head[idx:]
    if "    <!-- NAV -->" in head:
        head = head[:head.rindex("    <!-- NAV -->")]

    content = embed_figures("".join((BUILD / n).read_text(encoding="utf-8") for n in CONTENT))
    html = "\n".join([head.rstrip("\n"), "", content.rstrip("\n"), "", NAV_FIX_JS, "",
                      overlay.rstrip("\n"), "", diagram_js().rstrip("\n"), "", lib, tail])
    OUT.write_text(html, encoding="utf-8")
    print(f"Đã ghi {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
