"""The training logger must never import TensorFlow.

TensorFlow reserves almost all GPU memory on import-and-use, which starved
PyTorch and slowed one training run by 8x. The fake ``tensorflow`` module
below records whether anything imported it.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("tensorboard")


def test_logger_does_not_import_tensorflow(tmp_path):
    fake = tmp_path / "fake_site"
    fake.mkdir()
    (fake / "tensorflow.py").write_text(
        "import os\nopen(os.environ['TF_MARKER'], 'w').write('imported')\n"
    )
    marker = tmp_path / "tf_was_imported"
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(fake)!r})
        from pathlib import Path
        from chessnet.train import _Logger
        logger = _Logger(Path({str(tmp_path / "tb")!r}))
        assert logger.writer is not None, "logger should still work via the stub"
        logger.scalars("train", {{"x": 1.0}}, 1)
        logger.close()
    """)
    env = {**__import__("os").environ, "TF_MARKER": str(marker)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "the training logger imported TensorFlow"


def test_the_fake_tensorflow_is_detectable_without_the_block(tmp_path):
    """Sanity check: without the block, importing SummaryWriter imports it.

    Without this, the test above could pass vacuously.
    """
    fake = tmp_path / "fake_site"
    fake.mkdir()
    (fake / "tensorflow.py").write_text(
        "import os\nopen(os.environ['TF_MARKER'], 'w').write('imported')\n"
    )
    marker = tmp_path / "tf_was_imported"
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(fake)!r})
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except Exception:
            pass  # the fake module is empty; we only care that it was imported
    """)
    env = {**__import__("os").environ, "TF_MARKER": str(marker)}
    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert marker.exists(), "the check proves nothing if TF is never imported"
