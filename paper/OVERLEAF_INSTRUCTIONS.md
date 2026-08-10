# How to compile on Overleaf

1. Go to https://overleaf.com and sign in (free account works).

2. Click **New Project → Upload Project** and zip this `paper/` folder, OR:
   - New Project → Blank Project → paste `main.tex` content.

3. You need `neurips_2025.sty`. Download it from:
   - Overleaf template gallery: search "NeurIPS 2025"
   - Or directly from NeurIPS: https://neurips.cc/Conferences/2025/PaperInformation/StyleFiles

4. Upload to your Overleaf project:
   - `main.tex` (this file)
   - `references.bib`
   - `neurips_2025.sty` (downloaded in step 3)
   - `figures/fig1_mvp_el_gzip.png`

5. Set compiler to **pdfLaTeX**. Click **Compile**.

6. Export PDF → **Download as PDF**.

7. Upload the PDF to the GitHub repo:
   ```
   git add paper/
   git add paper/preprint.pdf   # after export from Overleaf
   git commit -m "feat: add Day 5 pilot preprint source and PDF"
   git push
   ```

## arXiv submission (first-time authors)

- First-time authors may need an endorsement from an existing arXiv user.
- If blocked: release PDF on OSF (https://osf.io) or Zenodo (https://zenodo.org) for a DOI.
- In the fellowship letter, write:
  "Preprint PDF and code are publicly released; arXiv submission pending endorsement."
