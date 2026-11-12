"""Small helper to draw the workshop diagrams as SVG with official AWS Architecture Icons.

The style follows img/deployment-project-architecture-aws.svg (notebook 05):
white nodes with a colored accent bar, a 64 px icon on top, a bold label, and a gray detail.
The icons in ./icons come from the AWS Architecture Icons asset package
(https://aws.amazon.com/architecture/icons/) and are embedded as base64 data URIs,
so each generated SVG is a single self-contained file.
"""
import base64
from html import escape
from pathlib import Path

ICONS = Path(__file__).parent / "icons"

# Accent colors, same as the notebook 05 diagram
SAGEMAKER = "#00A4A6"
STORAGE = "#7AA116"
APP_INTEGRATION = "#DD344C"
DEVELOPER_TOOLS = "#0073BB"
CONTAINERS = "#ED7100"
GENERAL = "#232F3E"
HIGHLIGHT = "#E07B00"

STYLE = """
    text { font-family: 'Amazon Ember', Arial, Helvetica, sans-serif; fill: #37475A; }
    .heading { font-size: 31px; font-weight: 700; fill: #232F3E; }
    .eyebrow { font-size: 16px; font-weight: 700; letter-spacing: 1.2px; fill: #687078; }
    .group-label { font-size: 24px; font-weight: 700; fill: #232F3E; }
    .node { fill: #FFFFFF; stroke: #AAB7B8; stroke-width: 2; }
    .node-optional { fill: #FFFFFF; stroke: #AAB7B8; stroke-width: 2; stroke-dasharray: 8 6; }
    .node-dark { fill: #232F3E; stroke: #232F3E; stroke-width: 2; }
    .node-required { fill: #FFFFFF; stroke: #232F3E; stroke-width: 3.5; }
    .node-label { font-size: 21px; font-weight: 700; fill: #232F3E; }
    .node-label-light { font-size: 21px; font-weight: 700; fill: #FFFFFF; }
    .node-detail { font-size: 16px; font-weight: 400; fill: #687078; }
    .node-detail-light { font-size: 16px; font-weight: 400; fill: #D5DBDB; }
    .code { font-family: 'Courier New', monospace; font-size: 17px; font-weight: 700; fill: #232F3E; }
    .code-light { font-family: 'Courier New', monospace; font-size: 17px; font-weight: 700; fill: #FFFFFF; }
    .flow-label { font-size: 16px; font-weight: 600; fill: #687078; paint-order: stroke; stroke: #FFFFFF; stroke-width: 7px; stroke-linejoin: round; }
    .highlight-label { font-size: 17px; font-weight: 700; fill: #E07B00; paint-order: stroke; stroke: #FFFFFF; stroke-width: 7px; stroke-linejoin: round; }
    .legend { font-size: 17px; fill: #37475A; }
    .step { font-size: 17px; font-weight: 700; fill: #FFFFFF; }
    .badge { font-size: 16px; font-weight: 700; fill: #FFFFFF; }
    .arrow { fill: none; stroke: #687078; stroke-width: 3; marker-end: url(#arrow); }
    .arrow-dashed { fill: none; stroke: #687078; stroke-width: 3; stroke-dasharray: 9 7; marker-end: url(#arrow); }
    .arrow-highlight { fill: none; stroke: #E07B00; stroke-width: 3; stroke-dasharray: 10 7; marker-end: url(#arrow-highlight); }
    .line-dashed { fill: none; stroke: #687078; stroke-width: 3; stroke-dasharray: 9 7; }
    .group { fill: #F9FAFB; stroke: #7D8998; stroke-width: 3; }
    .subgroup { fill: #FFFFFF; stroke: #0073BB; stroke-width: 3; }
    .note { fill: #FFF7E6; stroke: #FF9900; stroke-width: 2; }
    .provided { fill: #E6F6F6; stroke: #00A4A6; stroke-width: 2; }
"""


def _icon_data(name):
    data = (ICONS / f"{name}.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


class Diagram:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.title, self.desc = title, desc
        self.parts = [f'<rect width="100%" height="100%" fill="#FFFFFF"/>']

    # ----- primitives -----
    def add(self, markup):
        self.parts.append(markup)

    def text(self, x, y, text, cls="node-detail", anchor="start"):
        self.add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>')

    def icon(self, name, x, y, size=64):
        self.add(f'<image href="{_icon_data(name)}" x="{x}" y="{y}" width="{size}" height="{size}" preserveAspectRatio="xMidYMid meet"/>')

    def heading(self, text, x=40, y=58):
        self.text(x, y, text, "heading")

    def group(self, x, y, w, h, label=None, icon=None, cls="group", stroke=None):
        extra = f' style="stroke: {stroke}"' if stroke else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" class="{cls}"{extra}/>')
        if icon:
            self.icon(icon, x + 24, y + 20, 48)
        if label:
            self.text(x + (88 if icon else 24), y + 53, label, "group-label")

    def step(self, x, y, number):
        """Orange numbered circle, centered on (x, y)."""
        self.add(f'<circle cx="{x}" cy="{y}" r="17" fill="{HIGHLIGHT}"/>')
        self.text(x, y + 6, str(number), "step", "middle")

    def badge(self, x, y, text, width=170):
        self.add(f'<rect x="{x}" y="{y}" width="{width}" height="32" rx="16" fill="{HIGHLIGHT}"/>')
        self.text(x + width / 2, y + 22, text, "badge", "middle")

    def node(self, x, y, w, h, label, detail=(), icon=None, accent=SAGEMAKER, cls="node",
             code=(), eyebrow=None, step=None, icon_size=64, dark=False):
        """A node in the notebook 05 style: accent bar, icon on top, label, code lines, detail."""
        box_cls = "node-dark" if dark else cls
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" class="{box_cls}"/>')
        if accent and not dark:
            self.add(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>')
        cx = x + w / 2
        cursor = y + 24
        if eyebrow:
            self.text(cx, cursor + 8, eyebrow, "eyebrow", "middle")
            cursor += 24
        if icon:
            self.icon(icon, cx - icon_size / 2, cursor, icon_size)
            cursor += icon_size + 30
        else:
            cursor += 22
        label_lines = [label] if isinstance(label, str) else list(label)
        label_lines = [line for line in label_lines if line]
        if not label_lines:
            cursor -= 22 if not icon else 0
        for i, line in enumerate(label_lines):
            self.text(cx, cursor + 25 * i, line, "node-label-light" if dark else "node-label", "middle")
        cursor += 25 * max(len(label_lines) - 1, 0)
        for line in code:
            cursor += 26
            self.text(cx, cursor, line, "code-light" if dark else "code", "middle")
        if detail:
            cursor += 30
            for i, line in enumerate(detail):
                self.text(cx, cursor + 20 * i, line, "node-detail-light" if dark else "node-detail", "middle")
        if step is not None:
            self.step(x + 24, y, step)

    def arrow(self, d, cls="arrow", label=None, lx=None, ly=None, anchor="middle", label_cls="flow-label"):
        self.add(f'<path d="{d}" class="{cls}"/>')
        if label:
            self.text(lx, ly, label, label_cls, anchor)

    # ----- output -----
    def save(self, path):
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">\n'
            f'<title id="title">{escape(self.title)}</title>\n'
            f'<desc id="desc">{escape(self.desc)}</desc>\n'
            '<defs>\n'
            '  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#687078"/></marker>\n'
            f'  <marker id="arrow-highlight" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{HIGHLIGHT}"/></marker>\n'
            f'  <style>{STYLE}  </style>\n'
            '</defs>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )
        Path(path).write_text(svg, encoding="utf-8")
