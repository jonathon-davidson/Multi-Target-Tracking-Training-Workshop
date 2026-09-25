# Multi-Target Tracking Training Workshop

A three-day technical training course, authored once in [Quarto](https://quarto.org)
and rendered to three deliverables per module from a single source file:

- an **instructor slide deck** (Beamer PDF),
- a **student guide** (LaTeX PDF, the same figures with full prose), and
- an **editable PowerPoint** deck (`.pptx`, same slide content with editable text and
  native Office Math equations).

Each module also ships an **interactive companion app** (Streamlit) that the students
drive to explore the module's ideas. The Python figures execute at render time, so the
slides, guide, and PowerPoint always agree with the apps.

The course is 13 modules across three days (single-target estimation, association and
track management, then multi-sensor fusion and performance). This README is written for
an instructor setting the project up on a **fresh machine** with none of the author's
tools preinstalled.

---

## What you need to install

Three separate pieces, none of which is assumed to be present:

1. **Python 3.10+** and the packages in `requirements.txt` (figures, the render
   engine, and the exercise apps).
2. **Quarto 1.4+** (1.6.x recommended), the tool that renders the modules.
3. **A LaTeX distribution** with a handful of packages, needed only for the two **PDF**
   outputs (slides + guide). The PowerPoint output does **not** need LaTeX.

Do the three sections below once, then jump to *Rendering the course*.

### 1. Python and the Python packages

Use an isolated environment so nothing collides with your system Python. Either a
virtual environment (shown here) or a conda/mamba environment works; the project does
**not** require conda.

```bash
cd "Multi-Target Tracking"           # the project folder (this file's directory)
python3 -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

That installs the scientific stack (numpy, scipy, matplotlib, streamlit) **and** the
Jupyter engine Quarto uses to run each module's code cells. Keep this environment
activated for every step below.

> The rendered student guides tell students to start an app with `conda activate radar`.
> `radar` is just the author's environment name. Activate **whatever environment you
> installed these requirements into** (for example `source .venv/bin/activate`) in its
> place; the `streamlit run ...` command itself is unchanged.

### 2. Quarto

Install Quarto from <https://quarto.org/docs/get-started/> (there are installers for
macOS, Windows, and Linux). Verify:

```bash
quarto --version
```

### 3. LaTeX (for the PDF slides and guide)

The two PDF outputs are built with **xelatex**. The simplest route is Quarto's bundled
TinyTeX, which Quarto can also auto-install missing packages into on the fly:

```bash
quarto install tinytex
```

This course uses a few LaTeX packages beyond the TinyTeX defaults (the Metropolis
Beamer theme, `tcolorbox`, `fontawesome5` for the callout boxes, KOMA-Script for the
guide, and their dependencies). On the first render TinyTeX will try to fetch any that
are missing automatically; if that is blocked on your network, install them once
explicitly:

```bash
tlmgr install metropolis pgfopts tcolorbox environ trimspaces \
              fontawesome5 koma-script caption booktabs enumitem \
              microtype helvetic pgf xcolor hyperref
```

Alternatively, a **full TeX Live** (Linux/Windows) or **MacTeX** (macOS) install
already contains all of these, and needs no `tlmgr install` step.

Confirm the whole toolchain is visible to Quarto:

```bash
quarto check
```

You want it to report the Python (Jupyter) engine **and** a working LaTeX install. If
either is missing, revisit the matching step above.

---

## Quick start

With the three pieces installed and the Python environment active, render everything:

```bash
bash render.sh all
```

The finished files land in `_output/`, three per module:

```
_output/01-intro-slides.pdf          # instructor Beamer deck
_output/01-intro-guide.pdf           # student guide
_output/01-intro-instructor.pptx     # editable PowerPoint
... (through module 13)
```

To start an interactive exercise for the students (from the project root, environment
active):

```bash
streamlit run exercises/ex05_maneuvering.py
```

---

## Rendering the course

`render.sh` is the one command you need. Its first argument selects the module (a
filename prefix, or empty for all modules); its second selects which outputs to build
(`pdf` = slides + guide, `pptx`, or `all`).

```bash
bash render.sh                 # every module: slides + guide (the two PDFs)
bash render.sh 05              # module 05 only: slides + guide
bash render.sh 05 all          # module 05 only: slides + guide + PowerPoint
bash render.sh 05 pptx         # module 05 only: the editable PowerPoint
bash render.sh all             # every module: all three outputs
bash render.sh pptx            # every module: PowerPoint only
```

After each PowerPoint render, `render.sh` automatically runs
`scripts/pptx_fixup.py` on the deck (this positions each figure's caption box and pins
it to a readable size so PowerPoint never shrinks it).

**Always look at the output before shipping it.** Rasterize the PDFs and page through
them:

```bash
pdftoppm -jpeg -r 110 _output/05-maneuvering-slides.pdf slidepg
pdftoppm -jpeg -r 110 _output/05-maneuvering-guide.pdf  guidepg
```

Check that equations rendered (not raw `$...$`), figures are present and unclipped,
each figure carries a numbered "Figure N" caption, and no slide overflows its frame.
For the PowerPoint, open it in **real PowerPoint** for the final check: LibreOffice and
most previewers do not render the native Office Math equations or recompute text
autofit, so they misrepresent math slides and caption sizing.

---

## Running the interactive exercises

Each module has a Streamlit companion app in `exercises/`. Run one from the **project
root** with the Python environment active:

```bash
streamlit run exercises/ex08_multitarget.py
```

Streamlit opens the app in a browser. The apps have on-screen controls (sliders,
scenario selectors, toggles) and live metrics; the students learn by changing values
and watching the result, so **no coding is required of them**. The apps share their
computation with the printed figures (both import `exercises/appcommon.py`), so what a
student sees in the app matches the slides and guide.

The full set: `ex01_motion_models.py`, `ex02_linear_filters.py`,
`ex03_nonlinear_filters.py`, `ex04_angle_only.py`, `ex05_maneuvering.py`,
`ex06_association.py`, `ex07_track_mgmt.py`, `ex08_multitarget.py`,
`ex09_conversion.py`, `ex10_fusion.py`, `ex11_fusion.py`, `ex12_registration.py`,
`ex13_attribute_metrics.py`.

---

## Project layout

```
Multi-Target Tracking/
├── README.md                  # this file
├── requirements.txt           # Python dependencies (pip install -r)
├── render.sh                  # render modules -> _output/ (the one command)
├── _quarto.yml                # shared Quarto project config
├── _quarto-slides.yml         # profile: instructor Beamer deck
├── _quarto-guide.yml          # profile: student LaTeX guide
├── _quarto-pptx.yml           # profile: editable PowerPoint
├── modules/                   # one .qmd per module (the single source)
│   ├── 01-intro.qmd
│   └── ... through 13-attribute-metrics.qmd
├── figures/                   # Python figure code (one modNN.py per module + common.py)
├── exercises/                 # Streamlit apps (ex*.py) + shared appcommon.py
├── theme/                     # look & feel: preambles, reference.pptx, person photos
│   ├── beamer-preamble.tex    # slide styling (colors, caption sizing, person cards)
│   ├── guide-preamble.tex     # guide styling (title band, headers, person cards)
│   ├── reference.pptx         # PowerPoint template (styles the .pptx output)
│   └── person-*.jpg|png       # "Person Behind It" portraits
├── scripts/
│   └── pptx_fixup.py          # post-processes each .pptx (figure captions)
└── _output/                   # rendered deliverables (created by render.sh)
```

---

## Editing or adding a module

Each module is a single `modules/NN-name.qmd`. A few rules keep the three outputs
consistent:

- **Front matter.** Every module begins with a YAML `title:` and `subtitle:`, plus the
  per-module identity macros in `header-includes`:

  ```yaml
  ---
  title: "Multi-Target Tracking Training Workshop"
  subtitle: "Module N · <Module Title>"
  header-includes:
    - |
      \def\ModuleFooter{Module N: <Short Title>}
      \def\ModuleNumber{N}
      \def\ModuleTitle{<Module Title>}
  ---

  # <Module Title>        <- the section-divider slide; do NOT prefix "Module N:"
  ## Topic one            <- numbered guide section N.1
  ## A figure {.unnumbered}
  ```

  Without the front-matter `title:`, Quarto would promote the leading `#` to the
  document title and turn every `##` into its own repeated "Module N" band.

- **One source, three targets.** Select slide-only versus guide-only content by
  **profile**, never by format (Beamer and the guide are both the `pdf` format, so a
  format test cannot separate them from the `pptx` output):
  - slide bullets / equations: `::: {.content-visible unless-profile="guide"}` (shows on
    slides **and** PowerPoint, hidden in the guide),
  - guide prose: `::: {.content-visible when-profile="guide"}`,
  - PowerPoint-only blocks (e.g. a person-photo column): `::: {.content-visible when-profile="pptx"}`.

- **Figures** are `{python}` cells that import from `figures/`, with a `#| label:
  fig-...` and a `#| fig-cap:` so all three outputs get the same numbered caption. Never
  paste a pre-rendered image.

- **Person cards** use a shared `\personbox{}` / `\personboxtwo{}` block (raw LaTeX for
  the slides and guide) plus a `when-profile="pptx"` two-column block for PowerPoint.
  Drop a `theme/person-<name>.jpg` (or `.png`) to fill the photo; a missing file shows a
  placeholder, so the card renders before the photo is added.

---

## Troubleshooting

- **`quarto check` shows no Jupyter/Python** — the Python environment is not active or
  the packages are not installed. `source .venv/bin/activate` and re-run
  `pip install -r requirements.txt`.
- **`ModuleNotFoundError: No module named 'figures'`** — cells must run from the project
  root. `_quarto.yml` sets `execute-dir: project`; render with `render.sh` (or run
  `quarto render` from the project root), not from inside `modules/`.
- **A LaTeX package is missing** (e.g. `metropolis.sty`, `fontawesome5.sty`,
  `tcolorbox.sty` not found) — install it into TinyTeX with the `tlmgr install` line in
  *step 3* above, or use a full TeX Live / MacTeX.
- **Raw `$...$` in a PDF** — a math delimiter problem in that module's source; use
  `$...$` for inline and `$$...$$` for display math.
- **A figure slide overflows / a caption is clipped (Beamer)** — the shared
  `theme/beamer-preamble.tex` already caps figure height on the 16:9 slides so the
  caption fits; if you add an unusually long caption, shorten it or lower the figure's
  `#| fig-height`.
- **PowerPoint caption looks tiny or clipped** — re-run the render so
  `scripts/pptx_fixup.py` runs, and check in real PowerPoint (previewers misrender
  autofit). The caption size is a one-line constant (`CAPPT`) at the top of that script.
- **Slow first render** — the Python cells execute and are cached
  (`execute: cache: true`); the first pass is the slow one, later renders reuse the
  cache.

---

## Environment summary

| Piece | What / version | Installed by |
|---|---|---|
| Python | 3.10+ with numpy, scipy, matplotlib, streamlit, jupyter, jupyter-cache, nbclient, nbformat, ipykernel | `pip install -r requirements.txt` |
| Quarto | 1.4+ (1.6.x recommended) | quarto.org installer |
| LaTeX | xelatex + metropolis, tcolorbox, fontawesome5, koma-script, caption, and dependencies | `quarto install tinytex` (+ `tlmgr install ...`) or full TeX Live / MacTeX |
| PowerPoint fixup | Python standard library only (no extra package) | included (`scripts/pptx_fixup.py`) |
