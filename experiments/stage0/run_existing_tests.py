"""Execute the notebook's non-training cells (setup, physics, unit tests) in an isolated cwd."""
import os, tempfile
# The legacy notebook cells write figures/tables relative to the CWD ("results/..."). Always run
# them inside a throw-away directory so the committed legacy results/ tree can never be touched.
os.chdir(tempfile.mkdtemp(prefix="legacy_cells_"))
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebook_src"))
import sec_a, sec_b, sec_c
cells = [c for c in sec_a.cells() + sec_b.cells() if c.cell_type == "code"]
causal = [c for c in sec_c.cells() if c.cell_type == "code" and "def causal_weights" in c.source]
g = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(cells + causal):
    exec(compile(c.source, f"cell{i}", "exec"), g)
print(f"\n=== all {len(g['_TEST_LOG'])} checks passed; {time.time()-t0:.1f}s ===")
