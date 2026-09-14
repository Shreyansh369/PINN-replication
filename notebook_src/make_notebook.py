"""Assemble beam_pinn_research.ipynb from the section modules."""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
import sec_a, sec_b, sec_c, sec_d   # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "beam_pinn_research.ipynb"


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = sec_a.cells() + sec_b.cells() + sec_c.cells() + sec_d.cells()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(nb, OUT)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {OUT}  ({len(nb.cells)} cells: {n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
