"""Guard the inference image's dependency boundary.

The runtime Dockerfile installs a deliberately small set of packages: no
gymnasium, no PyYAML, no matplotlib, and the CPU wheel of torch rather than the
CUDA one. That only works while `gmai.api` and everything it reaches stay clear
of the training stack.

Nothing in the code enforces that. One convenience import inside `gmai.agent`
would break the image, and the failure would show up as a container that
crashes on startup rather than as a failing test. These tests make the boundary
explicit so it breaks here instead.
"""

import subprocess
import sys
import textwrap

import pytest

# Packages the runtime Dockerfile does NOT install.
#
# `tqdm` is deliberately absent from this list: torch imports it itself, via
# torch.hub and torch._dynamo, so it arrives as a transitive dependency whether
# we ask for it or not. Blocking it tested nothing about our own code and only
# produced a false failure.
TRAINING_ONLY = ("gymnasium", "matplotlib", "pytest")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestServingDependencyBoundary:
    def test_api_imports_without_training_packages(self):
        """Import gmai.api with the training-only modules blocked."""
        result = _run(f"""
            import sys

            class Blocker:
                blocked = {TRAINING_ONLY!r}
                def find_spec(self, name, path=None, target=None):
                    if name.split('.')[0] in self.blocked:
                        raise ImportError(f'{{name}} is not in the runtime image')
                    return None

            for _mod in {TRAINING_ONLY!r}:
                sys.modules.pop(_mod, None)
            sys.meta_path.insert(0, Blocker())
            import gmai.api  # noqa: F401
            print('ok')
        """)
        assert result.returncode == 0, (
            f"gmai.api pulled in a training-only package:\n{result.stderr}"
        )
        assert "ok" in result.stdout

    def test_serving_a_move_needs_no_training_packages(self):
        """Load a checkpoint and answer a request with the stack blocked."""
        result = _run(f"""
            import sys

            class Blocker:
                blocked = {TRAINING_ONLY!r}
                def find_spec(self, name, path=None, target=None):
                    if name.split('.')[0] in self.blocked:
                        raise ImportError(f'{{name}} is not in the runtime image')
                    return None

            for _mod in {TRAINING_ONLY!r}:
                sys.modules.pop(_mod, None)
            sys.meta_path.insert(0, Blocker())

            import tempfile, pathlib, torch
            from gmai.agent import DQNAgent
            from gmai import api

            torch.manual_seed(0)
            agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device='cpu')
            path = pathlib.Path(tempfile.mkdtemp()) / 'm.pt'
            agent.save(path)

            api.load_agent(path)
            response = api.move(api.MoveRequest(fen='4k3/8/8/8/8/8/Q7/4K3 w - - 0 1'))
            assert response.uci and response.in_scope
            print('ok')
        """)
        assert result.returncode == 0, (
            f"serving a move needs a training-only package:\n{result.stderr}"
        )
        assert "ok" in result.stdout

    @pytest.mark.parametrize("module", ["gmai.environment", "gmai.train"])
    def test_training_modules_do_use_the_training_stack(self, module):
        """Sanity check: the blocker works, and training modules do need it."""
        result = _run(f"""
            import sys

            class Blocker:
                blocked = {TRAINING_ONLY!r}
                def find_spec(self, name, path=None, target=None):
                    if name.split('.')[0] in self.blocked:
                        raise ImportError(f'{{name}} is not in the runtime image')
                    return None

            for _mod in {TRAINING_ONLY!r}:
                sys.modules.pop(_mod, None)
            sys.meta_path.insert(0, Blocker())
            import {module}
        """)
        assert result.returncode != 0, (
            f"{module} imported cleanly — the blocker is not working, so the "
            "other tests in this file prove nothing"
        )


class TestDockerfileMatchesTheCode:
    """Cheap static checks so the image and the code cannot drift apart."""

    @pytest.fixture
    def dockerfile(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "Dockerfile"
        if not path.exists():
            pytest.skip("Dockerfile not present")
        return path.read_text()

    def test_uses_cpu_torch_wheel(self, dockerfile):
        assert "download.pytorch.org/whl/cpu" in dockerfile

    def test_installs_every_runtime_dependency(self, dockerfile):
        for package in (
            "fastapi",
            "uvicorn",
            "prometheus-client",
            "python-chess",
            "numpy",
            "pydantic",
        ):
            assert package in dockerfile, f"{package} missing from the image"

    def test_does_not_install_training_packages(self, dockerfile):
        """Only inspect install commands — prose in comments is fine."""
        build_stage = dockerfile.split("FROM python:3.11-slim AS runtime")[0]
        install_lines = [
            line
            for line in build_stage.splitlines()
            if "pip install" in line or (line.startswith("      ") and '"' in line)
        ]
        installs = "\n".join(install_lines)
        for package in ("gymnasium", "matplotlib", "PyYAML"):
            assert package not in installs, (
                f"{package} is training-only and should not be in the runtime image"
            )

    def test_runs_as_non_root(self, dockerfile):
        assert "USER gmai" in dockerfile
        assert "useradd" in dockerfile

    def test_declares_a_healthcheck(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile
        assert "/health" in dockerfile
