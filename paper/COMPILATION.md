# Compiling the paper

## Overleaf (recommended)

1. Go to [overleaf.com](https://overleaf.com) and sign in.
2. **New Project → Upload Project** — zip the `paper/` folder, or create a blank project and paste `main.tex`.
3. Download `neurips_2026.sty` from the [NeurIPS 2026 style files](https://neurips.cc/Conferences/2026/CallForPapers) and upload it to your project.
4. Upload these files:
   - `main.tex`
   - `references.bib`
   - `neurips_2026.sty`
   - `figures/fig1_mvp_el_gzip.png`
   - `figures/fig1_v2_sg_ctw.png`
5. Set compiler to **pdfLaTeX**. Click **Compile**.
6. Export → **Download as PDF** → save as `paper/Preprint.pdf`.

## Local (pdflatex)

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a TeX distribution (TeX Live or MiKTeX) with `neurips_2026.sty` on the path.

## arXiv submission

Upload `main.tex`, `references.bib`, and the figures folder. Select cs.AI + cs.IT as primary/cross-list categories. First-time authors may need an endorsement — the Zenodo DOI (`10.5281/zenodo.21884224`) covers citation in the interim.
