"""Display authority regressions; execute the real TS helpers with Vue's refs."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')


@pytest.mark.skipif(not NODE or not (ROOT / 'web/node_modules/typescript').exists(),
                    reason='frontend source checks require the existing web dependencies and Node')
def test_current_nonpurchase_ai_and_purchase_revocation_are_separate():
    result = subprocess.run([NODE, str(ROOT / 'tests/frontend/nonpurchase_ai_authority.cjs')],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
