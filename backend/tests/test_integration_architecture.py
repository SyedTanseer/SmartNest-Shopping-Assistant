"""Architecture, security and demo guards for Assignment 4.2."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import gemini_check
from smartnest.integrations.gemini import GeminiConfigError, GeminiSettings
from smartnest.integrations.gemini.testing import ScriptedLLMClient, grounded_response

BACKEND = Path(__file__).resolve().parents[1]
SRC = BACKEND / "src" / "smartnest"

KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv", "smartnest.egg-info", ".git"}
_SUFFIXES = {".py", ".md", ".toml", ".txt", ".example", ".json", ".yml", ".yaml", ".cfg", ".ini"}


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
    return found


def package_imports(*parts: str) -> dict[str, set[str]]:
    return {str(p.relative_to(SRC)): imports_of(p) for p in (SRC.joinpath(*parts)).rglob("*.py")}


def committable_files():
    for path in BACKEND.rglob("*"):
        if path.is_file() and path.name != ".env" and not (set(path.parts) & _SKIP_DIRS) \
                and (path.suffix in _SUFFIXES or path.name == ".env.example"):
            yield path


def assert_no_imports(package: str, forbidden: tuple[str, ...]) -> None:
    for module, imports in package_imports(*package.split("/")).items():
        for name in imports:
            assert not name.lower().startswith(forbidden), f"{module} imports {name}"


# ══ Dependency direction ════════════════════════════════════════════════════

class TestImportDirection:
    def test_knowledge_is_independent_of_storage_drivers_llms_and_agents(self):
        assert_no_imports("knowledge", ("pymongo", "bson", "smartnest.database", "google", "smartnest.integrations",
                                        "smartnest.bdi", "smartnest.cnp", "smartnest.optimization", "smartnest.agents",
                                        "smartnest.adaptive"))

    def test_gemini_is_separate_from_bdi_cnp_the_optimizer_and_storage(self):
        assert_no_imports("integrations/gemini", ("smartnest.bdi", "smartnest.cnp", "smartnest.optimization",
                                                  "smartnest.database", "smartnest.knowledge", "smartnest.agents",
                                                  "smartnest.workers", "smartnest.adaptive", "pymongo",
                                                  "smartnest.integrations.products"))

    def test_product_discovery_does_not_depend_on_bdi_cnp_or_storage(self):
        assert_no_imports("integrations/products", ("smartnest.bdi", "smartnest.cnp", "smartnest.database",
                                                    "smartnest.knowledge", "pymongo", "smartnest.adaptive"))

    def test_product_records_are_validated_by_the_optimizers_own_validator(self):
        imports = set().union(*package_imports("integrations", "products").values())
        assert "smartnest.optimization" in imports

    @pytest.mark.parametrize("package", ["bdi", "cnp", "optimization", "agents", "workers", "communication",
                                         "reasoning", "simulation", "core", "database"])
    def test_existing_code_does_not_import_the_new_layers(self, package):
        assert_no_imports(package, ("smartnest.integrations", "smartnest.knowledge", "smartnest.adaptive", "google"))

    def test_only_the_adaptive_layer_combines_everything(self):
        for package in ("knowledge", "integrations"):
            for module, imports in package_imports(package).items():
                assert not any(n.startswith("smartnest.adaptive") for n in imports), module

    def test_importing_the_gemini_layer_does_not_load_the_sdk_or_need_credentials(self):
        code = ("import sys; import smartnest.integrations.gemini, smartnest.integrations.products, smartnest.knowledge; "
                "print('google.genai' in sys.modules)")
        env = {**os.environ, "PYTHONPATH": str(BACKEND / "src")}
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            env.pop(name, None)
        out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False"

    def test_the_bdi_layer_still_has_no_new_dependencies(self):
        assert_no_imports("bdi", ("pymongo", "smartnest.database", "smartnest.integrations", "smartnest.knowledge",
                                  "google", "socket", "smartnest.communication"))


# ══ Secrets ═════════════════════════════════════════════════════════════════

class TestSecrets:
    def test_env_example_has_only_placeholders(self):
        text = (BACKEND / ".env.example").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY=<your_gemini_api_key>" in text and "GEMINI_MODEL=" in text
        assert not KEY_SHAPE.search(text)

    def test_the_unedited_example_is_rejected_as_unconfigured(self, tmp_path):
        from smartnest.integrations.gemini import load_gemini_settings
        copy = tmp_path / ".env"
        copy.write_text((BACKEND / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(GeminiConfigError, match="placeholder"):
            load_gemini_settings(env_file=copy, environ={})

    def test_no_api_key_shaped_string_in_any_committable_file(self):
        offenders = [str(p.relative_to(BACKEND)) for p in committable_files()
                     if KEY_SHAPE.search(p.read_text(encoding="utf-8", errors="ignore"))]
        # The tests use obviously fake keys; they must not look like real ones.
        assert offenders == [], f"key-shaped strings found in: {offenders}"

    def test_the_real_key_if_configured_is_in_no_committable_file(self):
        from smartnest.integrations.gemini import load_gemini_settings
        try:
            key = load_gemini_settings(environ={}).api_key
        except GeminiConfigError:
            pytest.skip("no GEMINI_API_KEY in backend/.env")
        leaks = [str(p.relative_to(BACKEND)) for p in committable_files()
                 if key in p.read_text(encoding="utf-8", errors="ignore")]
        assert leaks == [], f"the Gemini key leaked into: {leaks}"

    def test_the_env_file_remains_git_ignored(self):
        lines = (BACKEND.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in [l.strip() for l in lines]

    def test_no_logging_call_formats_the_api_key(self):
        for module, _ in package_imports("integrations", "gemini").items():
            text = (SRC / module).read_text(encoding="utf-8")
            assert not re.search(r"(log|logger)\.\w+\([^)]*api_key", text), module


# ══ The live-check script (verified without a key) ══════════════════════════

FAKE_SETTINGS = GeminiSettings(api_key="AIza" + "SyFAKE_KEY_FOR_TESTS_0123456789abcdef", model="gemini-3.5-flash-lite")
REQ = {"sofa_budget": 30000, "min_seating": 3, "preferred_style": "minimalist", "confidence": 0.9}
URL = "https://shop.example.com/p/sofa"
TEXT = "Cozy Sofa: Rs 24,999, 200 x 90 cm"
LISTING = {"name": "Cozy Sofa", "price": 24999, "currency": "INR", "source_url": URL, "width_cm": 200, "depth_cm": 90,
           "dimensions_unit": "cm", "seating_capacity": 3, "material": "fabric", "colour": "grey", "style": "minimalist",
           "availability": "in_stock", "rating": 4.3, "review_count": 12}


class TestGeminiCheckScript:
    def run(self, monkeypatch, capsys, client, search=False, configured=True):
        def settings():
            if not configured:
                raise GeminiConfigError("GEMINI_API_KEY is not set; add it to backend/.env")
            return FAKE_SETTINGS
        monkeypatch.setattr(gemini_check, "load_gemini_settings", settings)
        monkeypatch.setattr(gemini_check, "GenaiClient", lambda s: client)
        monkeypatch.setattr(gemini_check, "RedirectResolver", lambda: type("R", (), {"resolve": lambda s, u: u})())
        monkeypatch.setattr(gemini_check, "HttpImageVerifier", lambda: type("V", (), {"verify": lambda s, u: False})())
        code = gemini_check.main(search=search)
        return code, capsys.readouterr().out

    def test_not_configured_exits_2_with_instructions(self, monkeypatch, capsys):
        code, out = self.run(monkeypatch, capsys, ScriptedLLMClient(), configured=False)
        assert code == 2 and "NOT CONFIGURED" in out and ".env" in out

    def test_extraction_check_succeeds_and_hides_the_key(self, monkeypatch, capsys):
        code, out = self.run(monkeypatch, capsys, ScriptedLLMClient(REQ))
        assert code == 0 and "Verified" in out and "sofa_budget = 30000" in out
        assert FAKE_SETTINGS.api_key not in out and "AIza" not in out

    def test_a_wrong_extraction_is_flagged(self, monkeypatch, capsys):
        code, out = self.run(monkeypatch, capsys, ScriptedLLMClient({**REQ, "sofa_budget": 3000}))
        assert "WARNING" in out

    def test_api_failure_exits_1(self, monkeypatch, capsys):
        from smartnest.integrations.gemini import GeminiAPIError
        code, out = self.run(monkeypatch, capsys, ScriptedLLMClient(GeminiAPIError("quota", 429, False)))
        assert code == 1 and "FAIL" in out

    def test_search_check(self, monkeypatch, capsys):
        client = ScriptedLLMClient(REQ, grounded_response(TEXT, [URL], queries=["sofas"]), {"products": [LISTING]})
        code, out = self.run(monkeypatch, capsys, client, search=True)
        assert code == 0 and f"source: {URL}" in out and "ACCEPTED Cozy Sofa" in out and "1 accepted" in out

    def test_search_without_sources_is_reported_as_partial(self, monkeypatch, capsys):
        client = ScriptedLLMClient(REQ, grounded_response("nothing", []))
        code, out = self.run(monkeypatch, capsys, client, search=True)
        assert code == 1 and "no sources" in out


# ══ The demonstration ═══════════════════════════════════════════════════════

class TestDemo:
    def test_simulated_demo_passes_every_check_and_is_honest_about_being_simulated(self, capsys):
        import adaptive_run
        assert adaptive_run.main(live=False, atlas=False) == 0
        out = capsys.readouterr().out
        assert "FAIL" not in out and "MODE: SIMULATED" in out and "Nothing below is real data" in out
        for expected in (
            "[OK] the agent adopted a recommendation",
            "verified by running the UtilityOptimizer independently",
            "[OK] the agent recognised its intention was invalid",
            "[OK] a different, feasible product was selected automatically",
            "[OK] the style belief changed from minimalist to contemporary",
            "[OK] the user's requirements record was updated in place",
            "All checks passed.",
        ):
            assert expected in out, expected
        assert "https://simfurnish.invalid/p/" in out and "SIMULATED" in out

    def test_simulated_links_cannot_be_mistaken_for_real_ones(self):
        import adaptive_run
        market = adaptive_run.SimulatedMarket()
        assert all(l["source_url"].startswith("https://simfurnish.invalid/") for l in market.listings)
        assert all(l["name"].startswith("SIMULATED") for l in market.listings)

    def test_the_demo_cleans_up_what_it_created(self, capsys):
        import adaptive_run
        from smartnest.database import InMemoryRepository
        repos = []
        original = InMemoryRepository.__init__

        def spy(self, *a, **k):
            original(self, *a, **k)
            repos.append(self)

        InMemoryRepository.__init__ = spy
        try:
            adaptive_run.main(live=False, atlas=False)
        finally:
            InMemoryRepository.__init__ = original
        capsys.readouterr()
        assert repos and all(r.count() == 0 for r in repos)
