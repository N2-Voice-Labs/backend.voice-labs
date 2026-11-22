import subprocess
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[1]


def test_api_imports_without_the_voice_extra() -> None:
    """Pipecat is only needed to build a pipeline, not to import the HTTP layer."""

    code = (
        "import sys\n"
        "sys.modules['pipecat'] = None  # any 'import pipecat...' now raises ImportError\n"
        "import src.runtime.api, src.runtime.pipeline, src.runtime.session\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=RUNTIME_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
