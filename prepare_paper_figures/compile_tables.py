"""Compile all standalone .tex table files from main_table_experiments/ into PDFs in tables/."""

import subprocess
import shutil
from pathlib import Path

SRC = Path(__file__).parent / "main_table_experiments"
OUT = SRC / "tables"

OUT.mkdir(exist_ok=True)

DOC_PREAMBLE = r"""\documentclass{article}
\usepackage[landscape,margin=0.5in]{geometry}
\usepackage{makecell}
\usepackage{amsmath}
\begin{document}
"""

DOC_POSTAMBLE = r"""
\end{document}
"""

for tex_file in sorted(SRC.glob("*.tex")):
    stem = tex_file.stem
    wrapped = OUT / f"{stem}.tex"

    with open(tex_file) as f:
        content = f.read()

    with open(wrapped, "w") as f:
        f.write(DOC_PREAMBLE)
        f.write(content)
        f.write(DOC_POSTAMBLE)

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(OUT), wrapped.name],
        cwd=str(OUT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"FAILED: {stem}")
        # print last 20 lines of log on failure
        log = OUT / f"{stem}.log"
        if log.exists():
            lines = log.read_text().splitlines()
            for line in lines[-20:]:
                print(f"  {line}")
    else:
        print(f"  OK: {stem}")

    # cleanup aux/log files
    for ext in ["aux", "log", "out"]:
        (OUT / f"{stem}.{ext}").unlink(missing_ok=True)
    wrapped.unlink(missing_ok=True)

print(f"\nDone. PDFs in {OUT}/")
