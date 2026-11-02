"""Security guards: credentials must never be committed or exposed.

These tests never contain a real credential.  When a real ``backend/.env``
exists, its password is loaded at test time and searched for in every file
that could be committed.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from smartnest.database import DatabaseConfigError, load_settings

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent

# user:password@ inside a mongodb URI, where neither part is a <placeholder>.
CREDENTIAL_URI = re.compile(r"mongodb(?:\+srv)?://[^\s<>@/\"']+:[^\s<>@/\"']+@")

_SCANNED_SUFFIXES = {".py", ".md", ".toml", ".txt", ".example", ".cfg", ".ini", ".json", ".yml", ".yaml"}


def _committable_files():
    """Source, tests and docs: everything except .env and caches."""
    for path in BACKEND.rglob("*"):
        if not path.is_file() or path.name == ".env":
            continue
        parts = set(path.parts)
        if parts & {"__pycache__", ".pytest_cache", ".venv", "venv", "smartnest.egg-info", ".git"}:
            continue
        if path.suffix in _SCANNED_SUFFIXES or path.name == ".env.example":
            yield path


class TestIgnoreRules:
    def test_gitignore_lists_env(self):
        lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in [line.strip() for line in lines]

    @pytest.mark.skipif(shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
                        reason="needs a git checkout")
    def test_git_ignores_backend_env(self):
        result = subprocess.run(["git", "check-ignore", "-q", "backend/.env"],
                                cwd=REPO_ROOT, capture_output=True)
        assert result.returncode == 0, "backend/.env must be ignored by git"

    @pytest.mark.skipif(shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
                        reason="needs a git checkout")
    def test_env_is_not_tracked(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        assert "backend/.env\n" not in tracked + "\n"


class TestNoCredentialsInFiles:
    def test_env_example_contains_only_placeholders(self):
        text = (BACKEND / ".env.example").read_text(encoding="utf-8")
        assert "MONGODB_URI=" in text and "MONGODB_DATABASE=smartnest" in text
        assert "<db_password>" in text
        assert not CREDENTIAL_URI.search(text)

    def test_example_file_is_rejected_as_unconfigured(self, tmp_path):
        """Copying .env.example without editing it must fail loudly."""
        copy = tmp_path / ".env"
        copy.write_text((BACKEND / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(DatabaseConfigError, match="placeholder"):
            load_settings(env_file=copy, environ={})

    def test_no_credential_shaped_uri_in_committable_files(self):
        offenders = []
        for path in _committable_files():
            if path.name == "test_database_security.py":
                continue
            if CREDENTIAL_URI.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(BACKEND)))
        # Test files use clearly fake credentials; only production files and docs are checked.
        offenders = [o for o in offenders if not o.startswith("tests")]
        assert offenders == [], f"credential-shaped URIs found in: {offenders}"

    def test_real_password_from_env_appears_in_no_committable_file(self):
        env_file = BACKEND / ".env"
        if not env_file.exists():
            pytest.skip("no backend/.env to check against")
        try:
            settings = load_settings(env_file=env_file, environ={})
        except DatabaseConfigError:
            pytest.skip("backend/.env has no usable MONGODB_URI")
        secrets = [s for s in settings.secret_fragments if len(s) >= 6]
        assert secrets, "expected credentials in the URI"
        leaks = []
        for path in _committable_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(secret in text for secret in secrets):
                leaks.append(str(path.relative_to(BACKEND)))
        assert leaks == [], f"credentials leaked into: {leaks}"


class TestDriverIndependence:
    def test_agents_and_cnp_do_not_import_the_database(self):
        """Database access stays behind repositories; agent logic is storage-agnostic."""
        for package in ("agents", "cnp", "optimization", "reasoning", "workers", "communication"):
            for path in (BACKEND / "src" / "smartnest" / package).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                assert "smartnest.database" not in text and "pymongo" not in text, str(path)

    def test_database_layer_imports_nothing_from_agents_llms_or_other_features(self):
        forbidden = ("smartnest.agents", "smartnest.cnp", "smartnest.optimization",
                     "smartnest.workers", "smartnest.communication", "smartnest.reasoning",
                     "smartnest.simulation", "google", "genai", "gemini", "anthropic", "openai")
        for path in (BACKEND / "src" / "smartnest" / "database").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    assert not module.lower().startswith(forbidden), f"{path.name} imports {module}"
