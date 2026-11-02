"""Architecture guards for Assignment 4.1: independence and compatibility."""

from __future__ import annotations

import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SRC = BACKEND / "src" / "smartnest"


def imports_of(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


def package_imports(package: str) -> dict[str, set[str]]:
    return {str(p.relative_to(SRC)): imports_of(p) for p in (SRC / package).rglob("*.py")}


class TestBdiIsIndependent:
    FORBIDDEN = (
        "pymongo", "bson", "smartnest.database",          # storage
        "socket", "smartnest.communication",              # networking
        "smartnest.cnp", "smartnest.workers",             # other features
        "google", "genai", "gemini", "anthropic", "openai", "requests", "httpx", "urllib3",  # external services
    )

    def test_bdi_imports_no_database_network_or_llm(self):
        for module, imports in package_imports("bdi").items():
            for name in imports:
                assert not name.lower().startswith(self.FORBIDDEN), f"{module} imports {name}"

    def test_bdi_reuses_the_existing_framework(self):
        all_imports = set().union(*package_imports("bdi").values())
        for expected in ("smartnest.agents.base_agent", "smartnest.core.events", "smartnest.core.models",
                         "smartnest.optimization"):
            assert expected in all_imports, expected

    def test_bdi_has_no_randomness(self):
        """Decisions must be deterministic."""
        for module, imports in package_imports("bdi").items():
            assert "random" not in imports, module

    def test_database_layer_does_not_know_about_bdi(self):
        for module, imports in package_imports("database").items():
            assert not any(name.startswith("smartnest.bdi") for name in imports), module


class TestExistingCodeIsUntouchedByBdi:
    @pytest.mark.parametrize("package", ["cnp", "optimization", "agents", "workers", "communication",
                                         "reasoning", "simulation", "core"])
    def test_nothing_else_imports_bdi(self, package):
        for module, imports in package_imports(package).items():
            assert not any(name.startswith("smartnest.bdi") for name in imports), module

    def test_cnp_agents_still_do_not_inherit_base_agent(self):
        text = (SRC / "cnp" / "bidding_agent.py").read_text(encoding="utf-8")
        assert "BaseAgent" not in text

    def test_only_the_two_new_event_types_were_added(self):
        from smartnest.core.models import EventType
        assert [e.value for e in EventType][:6] == [
            "budget_update", "furniture_compatibility", "product_availability",
            "missing_information", "product_validation", "price_change",
        ]
        assert {"room_update", "preference_update"} <= {e.value for e in EventType}
        assert len(EventType) == 8

    def test_existing_reflex_agent_ignores_the_new_event_types(self):
        from smartnest.agents.reflex_agent import ReflexAgent
        from smartnest.core.models import EnvironmentalEvent, EventType
        from smartnest.reasoning.rule_engine import RuleEngine
        from smartnest.reasoning.rules import all_smartnest_rules

        engine = RuleEngine()
        for rule in all_smartnest_rules():
            engine.register(rule)
        agent = ReflexAgent("reflex", engine)
        for kind in (EventType.ROOM_UPDATE, EventType.PREFERENCE_UPDATE):
            assert agent.handle_event(EnvironmentalEvent(event_type=kind, payload={"max_width_cm": 1})) is None


class TestDemo:
    def test_demo_runs_to_success_offline(self, capsys):
        import bdi_run

        assert bdi_run.main(use_atlas=False) == 0
        out = capsys.readouterr().out
        assert "FAIL" not in out
        for expected in (
            "[OK] the optimizer's first choice is SOFA_A2",
            "[OK] the new intention (SOFA_B3) is the optimizer's highest-utility feasible option",
            "[OK] no feasible product -> no recommendation",
            "[OK] state was restored from the store",
            "[OK] the restored agent reconsidered",
            "All checks passed",
        ):
            assert expected in out, expected
        assert out.count("[OK] no replanning, intention unchanged") == 4

    def test_demo_leaves_no_record_behind(self, capsys):
        import bdi_run
        from smartnest.database import InMemoryRepository

        created = []
        original = InMemoryRepository.__init__

        def spy(self, *a, **kw):
            original(self, *a, **kw)
            created.append(self)

        InMemoryRepository.__init__ = spy
        try:
            bdi_run.main(use_atlas=False)
        finally:
            InMemoryRepository.__init__ = original
        capsys.readouterr()
        assert created and created[0].count() == 0
