"""Compile all standalone .tex table files from main_table_experiments/ into PDFs in tables/."""

import subprocess
from pathlib import Path

SRC = Path(__file__).parent / "main_table_experiments"
OUT = SRC / "tables"

DOC_PREAMBLE = r"""\documentclass{article}
\usepackage[landscape,margin=0.5in]{geometry}
\usepackage{makecell}
\usepackage{amsmath}
\usepackage{float}
\begin{document}
"""

DOC_POSTAMBLE = r"""
\end{document}
"""


def compile_all():
    OUT.mkdir(exist_ok=True)

    for tex_file in sorted(SRC.glob("*.tex")):
        stem = tex_file.stem
        wrapped = OUT / f"{stem}.tex"

        content = tex_file.read_text()
        wrapped.write_text(DOC_PREAMBLE + content + DOC_POSTAMBLE)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(OUT), wrapped.name],
            cwd=str(OUT), capture_output=True, text=True,
        )

        if result.returncode != 0:
            print(f"FAILED: {stem}")
            log = OUT / f"{stem}.log"
            if log.exists():
                for line in log.read_text().splitlines()[-20:]:
                    print(f"  {line}")
        else:
            print(f"  OK: {stem}")

        for ext in ["aux", "log", "out"]:
            (OUT / f"{stem}.{ext}").unlink(missing_ok=True)
        wrapped.unlink(missing_ok=True)

    print(f"\nDone. PDFs in {OUT}/")


if __name__ == "__main__":
    compile_all()
