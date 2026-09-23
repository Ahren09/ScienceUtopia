#!/usr/bin/env python3
"""Render the paper's corrected framework and teaser as green vector PDFs.

Run from the isolated code worktree:
    python -B src/render_paper_revision_figures.py
    python -B src/render_paper_revision_figures.py --preview_dir outputs/visual/james_revision_existing/previews

Dependencies: matplotlib and PyMuPDF (fitz). No simulator imports, GPU calls,
external model calls, or modifications to the paper/source inputs occur.

The wording is grounded in the revised method and existing implementation:
  paper/src/2_method.tex: phases 0-5 and persistent researcher state.
  run_simulation.py: run_one_round, _run_phase_0_resubmissions,
      _run_phase_1_5_collaboration_formation, _run_phase_5_update_funding.
  utopia/agents/conference.py: make_acceptance_decisions.
  utopia/agents/base_agent.py: state serialization and inactive-agent handling.

The drawings deliberately omit a claim of reputation evolution, a universal
prior-work comparison, and action-cost details. Decisions are described as
ranked venue quotas, avoiding the old erroneous ceiling formula. Resubmission
selects a paper and venue and copies the existing artifact, without revising
its content. Annual project costs are charged before peer review, and any
resource update that would make the balance negative immediately deactivates
the researcher. Phase 5 adds a low-resource attrition check, so its label
"Check attrition" does not imply that attrition occurs only at that phase.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs' / 'visual' / 'james_revision_existing'
GREEN = '#008000'
WHITE = '#FFFFFF'
GREEN_RGB = (0.0, 128 / 255, 0.0)

PHASES = (
    ('0', 'Resubmission', ('From year 2', 'Resubmit to a venue', 'or abandon')),
    ('1', 'Directions &\ncollaboration',
     ('Select a project', 'Collaboration in', 'network runs')),
    ('2', 'Submission &\ncitations',
     ('Retrieve a paper', 'Choose a venue', 'Select citations')),
    ('3', 'Peer review',
     ('Assigned reviewers', 'Recommendations', 'Written reasons')),
    ('4', 'Decisions &\nmemory',
     ('Rank mean scores', 'Apply venue quota', 'Retain decisions')),
    ('5', 'Funding &\nattrition',
     ('Allocate funding', 'Update resources', 'Check attrition')),
)


def source_evidence(method):
    """Read and hash the specific inputs; reject a stale phase/decision description."""
    files = [
        ROOT / 'run_simulation.py',
        ROOT / 'utopia/agents/conference.py',
        ROOT / 'utopia/agents/base_agent.py',
        method,
    ]
    contents = {str(path): path.read_text() for path in files}
    simulation = contents[str(files[0])]
    for phase in range(6):
        if f'_run_phase_{phase}_' not in simulation:
            raise ValueError(f'Expected implementation of phase {phase} is missing')
        if f'Phase {phase}:' not in contents[str(method)]:
            raise ValueError(f'Revised method does not describe phase {phase}')
    if 'max(1, round(len(paper_scores) * self.acceptance_rate))' not in contents[str(files[1])]:
        raise ValueError('Venue quota implementation changed; review the figure wording')
    if 'agent.resources <= 10' not in simulation:
        raise ValueError('Attrition implementation changed; review the figure wording')
    return {name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in contents.items()}


def configure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.0,
        'text.color': GREEN,
        'axes.labelcolor': GREEN,
        'axes.edgecolor': GREEN,
        'axes.facecolor': WHITE,
        'figure.facecolor': WHITE,
        'savefig.facecolor': WHITE,
        'savefig.edgecolor': WHITE,
        'xtick.color': GREEN,
        'ytick.color': GREEN,
        'lines.color': GREEN,
        'patch.edgecolor': GREEN,
        'pdf.fonttype': 42,
        'pdf.use14corefonts': False,
        'path.simplify': False,
        'text.usetex': False,
    })


def canvas(width, height):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width, height), dpi=160, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, value, size=8, weight='normal', align='left', **kwargs):
    return ax.text(x, y, value, color=GREEN, fontsize=size, fontweight=weight,
                   ha=align, va='center', linespacing=1.18, **kwargs)


def box(ax, x, y, width, height, linewidth=0.85, rounding=0.05):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f'round,pad=0,rounding_size={rounding}',
        linewidth=linewidth, edgecolor=GREEN, facecolor=WHITE,
        joinstyle='round', zorder=2)
    ax.add_patch(patch)
    return patch


def line(ax, points, linewidth=0.75):
    ax.plot(*zip(*points), color=GREEN, linewidth=linewidth,
            solid_capstyle='round', solid_joinstyle='round', zorder=1)


def arrow(ax, points, linewidth=0.8, head=6.5):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as PlotPath
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head,
                            linewidth=linewidth, edgecolor=GREEN, facecolor=GREEN,
                            joinstyle='round', capstyle='round', zorder=3)
    ax.add_patch(patch)


def researcher_icon(ax, x, y, scale=1):
    """A small original line glyph, entirely vector strokes."""
    from matplotlib.patches import Arc, Circle
    ax.add_patch(Circle((x, y + 0.074 * scale), 0.045 * scale,
                        edgecolor=GREEN, facecolor=WHITE, linewidth=0.8, zorder=3))
    ax.add_patch(Arc((x, y - 0.066 * scale), 0.21 * scale, 0.19 * scale,
                     theta1=0, theta2=180, edgecolor=GREEN, linewidth=0.8, zorder=3))
    line(ax, [(x - 0.105 * scale, y - 0.066 * scale),
              (x - 0.105 * scale, y - 0.096 * scale)], 0.8)
    line(ax, [(x + 0.105 * scale, y - 0.066 * scale),
              (x + 0.105 * scale, y - 0.096 * scale)], 0.8)


def ensure_text_fits(fig, constraints):
    """Check physical text extents before export, including every card's labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, bounds in constraints:
        extent = artist.get_window_extent(renderer)
        ax = artist.axes
        (left, bottom), (right, top) = ax.transData.transform(
            [(bounds[0], bounds[1]), (bounds[0] + bounds[2], bounds[1] + bounds[3])])
        if (extent.x0 < left - 1 or extent.x1 > right + 1
                or extent.y0 < bottom - 1 or extent.y1 > top + 1):
            raise ValueError(f'Text does not fit its box: {artist.get_text()!r}')
    for ax in fig.axes:
        for artist in ax.texts:
            if not fig.bbox.contains(*artist.get_window_extent(renderer).get_points()[0]):
                raise ValueError(f'Text leaves the page: {artist.get_text()!r}')
            if not fig.bbox.contains(*artist.get_window_extent(renderer).get_points()[1]):
                raise ValueError(f'Text leaves the page: {artist.get_text()!r}')


def framework():
    """A numbered serpentine loop, sized for ICLR's 5.5-inch text width."""
    fig, ax = canvas(5.50, 3.55)
    constraints = []
    left, right = 0.35, 5.35
    text(ax, left, 3.38, 'ONE SIMULATED YEAR', size=9.0, weight='bold')
    text(ax, right, 3.38, 'Fixed order: phases 0-5', size=7.8, align='right')

    box(ax, left, 2.94, right - left, 0.30, rounding=0.035)
    text(ax, left + 0.10, 3.09, 'Inherited state', size=7.8, weight='bold')
    text(ax, 1.69, 3.09, 'Expertise, projects, resources, memory, conflicts', size=7.6)

    gap = 0.20
    width = (right - left - 2 * gap) / 3
    height = 0.94
    centers = {}
    for index, (number, title, details) in enumerate(PHASES):
        column = index if index < 3 else 5 - index
        x = left + column * (width + gap)
        bottom = 1.80 if index < 3 else 0.66
        center = x + width / 2
        centers[index] = (center, bottom)
        box(ax, x, bottom, width, height)
        text(ax, center, bottom + 0.82, f'PHASE {number}',
             size=7.7, weight='bold', align='center')
        heading = text(ax, center, bottom + 0.60, title,
                       size=8.6, weight='bold', align='center')
        body = text(ax, center, bottom + 0.225, '\n'.join(details),
                    size=7.8, align='center')
        constraints += [(heading, (x + 0.06, bottom + 0.45, width - 0.12, 0.29)),
                        (body, (x + 0.06, bottom + 0.045, width - 0.12, 0.37))]
        if index in (1, 2):
            arrow(ax, [(x - gap + 0.025, bottom + height / 2),
                       (x - 0.025, bottom + height / 2)], head=6)
        elif index in (4, 5):
            arrow(ax, [(x + width + gap - 0.025, bottom + height / 2),
                       (x + width + 0.025, bottom + height / 2)], head=6)
    arrow(ax, [(centers[0][0], 2.94), (centers[0][0], 2.765)])
    arrow(ax, [(centers[2][0], 1.80), (centers[3][0], 1.625)])

    box(ax, left, 0.17, right - left, 0.30, rounding=0.035)
    text(ax, left + 0.10, 0.32,
         'Retained records: reviews, publications, citations and funding', size=7.8)
    arrow(ax, [(centers[5][0], 0.66), (centers[5][0], 0.495)])
    arrow(ax, [(left, 0.32), (0.15, 0.32), (0.15, 3.09), (left - 0.025, 3.09)])
    text(ax, 0.145, 1.68, 'Next year', size=7.5, align='center', rotation=90,
         bbox={'facecolor': WHITE, 'edgecolor': 'none', 'pad': 3}, zorder=4)
    ensure_text_fits(fig, constraints)
    return fig


def teaser():
    """Persistent identity and records connect a yearly sequence of outcomes."""
    fig, ax = canvas(5.50, 2.36)
    constraints = []
    state_y, state_h = 1.24, 0.62
    left, right, width = 0.15, 3.15, 2.20
    for x, heading in ((left, 'Year t: inherited state'),
                       (right, 'Year t + 1: updated state')):
        box(ax, x, state_y, width, state_h, linewidth=1.0, rounding=0.06)
        researcher_icon(ax, x + 0.19, state_y + 0.31, scale=0.9)
        title = text(ax, x + 0.36, state_y + 0.455, heading, size=8.4, weight='bold')
        body = text(ax, x + 0.36, state_y + 0.205,
                    'Expertise, projects, resources,\nmemory and conflict sets', size=7.5)
        constraints += [(title, (x + 0.33, state_y + 0.32, width - 0.39, 0.27)),
                        (body, (x + 0.33, state_y + 0.045, width - 0.39, 0.30))]

    labels = (
        ('Research', 'Directions and\nprojects'),
        ('Review', 'Scores and\njustifications'),
        ('Publication', 'Decisions and\ncitations'),
        ('Funding', 'Awards and\nresources'),
    )
    card_y, card_h, gap = 0.30, 0.64, 0.14
    card_w = (5.20 - 3 * gap) / 4
    centers = []
    for index, (title, body) in enumerate(labels):
        x = left + index * (card_w + gap)
        center = x + card_w / 2
        centers.append(center)
        box(ax, x, card_y, card_w, card_h, rounding=0.045)
        heading = text(ax, center, card_y + 0.455, title, size=9.0,
                       weight='bold', align='center')
        detail = text(ax, center, card_y + 0.205, body, size=7.8, align='center')
        constraints += [(heading, (x + 0.045, card_y + 0.33, card_w - 0.09, 0.26)),
                        (detail, (x + 0.045, card_y + 0.045, card_w - 0.09, 0.32))]
        if index:
            arrow(ax, [(x - gap + 0.018, card_y + card_h / 2),
                       (x - 0.018, card_y + card_h / 2)], head=5.5)
    arrow(ax, [(centers[0], state_y), (centers[0], card_y + card_h + 0.025)])
    arrow(ax, [(centers[-1], card_y + card_h), (centers[-1], state_y - 0.025)])
    text(ax, 2.75, 0.105, 'Retrieval uses temporally eligible literature', size=7.7, align='center')

    arrow(ax, [(right + width / 2, state_y + state_h),
               (right + width / 2, 2.14), (left + width / 2, 2.14),
               (left + width / 2, state_y + state_h + 0.025)], head=6.5)
    text(ax, 2.75, 2.14, 'Persistent state carries into the next year', size=8.2,
         align='center', bbox={'facecolor': WHITE, 'edgecolor': 'none', 'pad': 3.0},
         zorder=4)
    ensure_text_fits(fig, constraints)
    return fig


def verify_pdf(path, preview_dir=None):
    import fitz
    with fitz.open(path) as document:
        if len(document) != 1:
            raise ValueError(f'Expected one figure page: {path}')
        page = document[0]
        if page.get_images(full=True):
            raise ValueError(f'Raster image embedded in vector figure: {path}')
        text_content = page.get_text()
        for forbidden in ('?', 'ceil', 'phase 6', 'reputation', 'revise'):
            if forbidden in text_content.lower():
                raise ValueError(f'Unexpected text {forbidden!r} in {path}')
        spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                 for line_ in block['lines'] for span in line_['spans']]
        if not spans or any(span['color'] != 0x008000 for span in spans):
            raise ValueError(f'All text must be exactly #008000: {path}')
        for drawing in page.get_drawings():
            for role in ('color', 'fill'):
                color = drawing.get(role)
                if color is not None and not any(
                        len(color) == len(expected)
                        and all(abs(a - b) < 1e-5 for a, b in zip(color, expected))
                        for expected in (GREEN_RGB, (1.0, 1.0, 1.0))):
                    raise ValueError(f'Unexpected {role} {color} in {path}')
        for span in spans:
            if not page.rect.contains(fitz.Rect(span['bbox'])):
                raise ValueError(f"PDF text is clipped: {span['text']}")
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(dpi=200, alpha=False).save(preview_dir / (path.stem + '.png'))
        return {'file': str(path), 'page_points': list(page.rect),
                'embedded_images': 0, 'text_spans': len(spans),
                'color': GREEN, 'vector_paths': len(page.get_drawings()),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--method', type=Path,
                        default=ROOT.parent / 'paper' / 'src' / '2_method.tex')
    parser.add_argument('--preview_dir', type=Path)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if not output.is_relative_to((ROOT / 'outputs').resolve()):
        parser.error('Figure outputs must remain under the isolated worktree outputs/')
    if args.preview_dir and not args.preview_dir.resolve().is_relative_to(output):
        parser.error('QA previews must remain inside --out_dir')
    evidence = source_evidence(args.method.resolve())
    output.mkdir(parents=True, exist_ok=True)
    # Keep font-cache writes local. No screenshot or image assets are used.
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs' / 'logs' / 'paper_figure_matplotlib'))
    configure_matplotlib()
    import matplotlib.pyplot as plt
    results = []
    for stem, build in (('fig_framework_green', framework), ('fig_teaser_green', teaser)):
        fig = build()
        path = output / (stem + '.pdf')
        fig.savefig(path, format='pdf', facecolor=WHITE, edgecolor=WHITE,
                    metadata={'Title': stem.replace('_', ' '),
                              'Subject': 'Science Utopia paper revision; vector diagram in #008000',
                              'Creator': 'render_paper_revision_figures.py',
                              'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        results.append(verify_pdf(path, args.preview_dir))
    print(json.dumps({'figures': results, 'source_sha256': evidence}, indent=2))


if __name__ == '__main__':
    main()
