#!/usr/bin/env python3
"""SmartNest — Assignments 1.2 & 2.1.

Assignment 1.2: Rule-Based Environmental Reasoner
    A reflex agent processes simulated environmental events and triggers
    condition-action rules.

Assignment 2.1: Concurrent Task Daemonization
    Background workers monitor product prices, availability, and budget
    concurrently, publishing events to the shared EventStream.

Run from the ``backend`` directory:

    python src/main.py

Or after installing the package:

    smartnest
"""

from __future__ import annotations

import logging
import threading
from time import sleep, monotonic

from smartnest.agents.reflex_agent import ReflexAgent
from smartnest.core.events import EventStream
from smartnest.core.models import ActionLog, EnvironmentalEvent
from smartnest.reasoning.rule_engine import RuleEngine
from smartnest.reasoning.rules import all_smartnest_rules
from smartnest.simulation.environment import MockEnvironment
from smartnest.workers.budget_monitor_worker import BudgetMonitorWorker, BudgetStateProvider
from smartnest.workers.catalog_provider import SimulatedCatalogProvider
from smartnest.workers.product_monitor_worker import ProductMonitorWorker
from smartnest.workers.worker_manager import WorkerManager


# ── Logging configuration ───────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    """Configure logging with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] [%(threadName)-16s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Console printer ────────────────────────────────────────────────────────

def _print_separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_event(event: EnvironmentalEvent) -> None:
    print(f"\n  📡 Event received:  {event.event_id}")
    print(f"     Type:           {event.event_type}")
    print(f"     Source:         {event.source}")
    print(f"     Timestamp:      {event.timestamp.isoformat()}")
    print(f"     Payload:        {event.payload}")


def _print_action(log: ActionLog | None) -> None:
    if log is None:
        print("     ⚪ No rule triggered.")
        return
    print(f"     ✅ Rule triggered: {log.rule_name} (id={log.rule_id})")
    print(f"     Action:          {log.action_description}")
    print(f"     Result type:     {log.result_type}")
    print(f"     Result message:  {log.result_message}")
    if log.result_details:
        print(f"     Details:         {log.result_details}")


# ── Part 1: Assignment 1.2 Simulation ───────────────────────────────────────

def _run_assignment_1_2(stream: EventStream, agent: ReflexAgent) -> None:
    """Run the original Assignment 1.2 five-event simulation."""
    _print_separator("Assignment 1.2: Rule-Based Environmental Reasoner")
    print("  Simulating a two-bedroom apartment furnishing scenario.")
    print("  Budget: ₹2,00,000  |  Agent: ReflexAgent  |  Rules: 9")

    env = MockEnvironment(stream)
    env.emit_all()

    _print_separator("Processing Assignment 1.2 Event Stream")
    event_count = 0
    for event in stream.iterate(timeout=1.0):
        event_count += 1
        _print_event(event)
        log = agent.handle_event(event)
        _print_action(log)

    print(f"\n  Assignment 1.2 complete: {event_count} events processed, "
          f"{len(agent.action_log)} actions logged.")


# ── Part 2: Assignment 2.1 Concurrent Workers ───────────────────────────────

def _run_assignment_2_1(stream: EventStream, agent: ReflexAgent) -> None:
    """Demonstrate concurrent background workers monitoring a shared state."""
    _print_separator("Assignment 2.1: Concurrent Task Daemonization")
    print("  Starting ProductMonitorWorker and BudgetMonitorWorker concurrently.")

    # ── Setup simulated catalogue ───────────────────────────────────────
    initial_products = [
        {"product_name": "Sofa",           "price": 24999, "availability": "in_stock"},
        {"product_name": "Refrigerator",     "price": 32000, "availability": "in_stock"},
        {"product_name": "Washing machine",  "price": 18500, "availability": "in_stock"},
        {"product_name": "Study table",      "price":  7000, "availability": "in_stock"},
        {"product_name": "Office chair",     "price":  5500, "availability": "in_stock"},
    ]
    catalog = SimulatedCatalogProvider(initial_products)

    # ── Setup budget provider ───────────────────────────────────────────
    budget_provider = BudgetStateProvider(
        total_budget=200000,
        estimated_expenditure=175000,
    )

    # ── Create workers via WorkerManager ─────────────────────────────────
    manager = WorkerManager()
    product_worker = ProductMonitorWorker(
        catalog=catalog,
        event_stream=stream,
        poll_interval=1.0,
    )
    budget_worker = BudgetMonitorWorker(
        budget_provider=budget_provider,
        event_stream=stream,
        poll_interval=1.0,
    )
    manager.register(product_worker)
    manager.register(budget_worker)

    # ── Event consumer thread ────────────────────────────────────────────
    # Reads events from the stream and feeds them to the ReflexAgent.
    # This demonstrates that the main application stays responsive while
    # background workers publish events.
    consumer_stop = threading.Event()
    processed_events: list[EnvironmentalEvent] = []
    processed_logs: list[ActionLog] = []

    def event_consumer():
        while not consumer_stop.is_set():
            event = stream.next_event(timeout=0.5)
            if event is not None:
                processed_events.append(event)
                log = agent.handle_event(event)
                if log:
                    processed_logs.append(log)

    # ── Start everything ─────────────────────────────────────────────────
    # Allow workers to take initial snapshots
    manager.start_all()
    consumer_thread = threading.Thread(
        target=event_consumer, name="event-consumer", daemon=True
    )
    consumer_thread.start()
    sleep(1.5)  # Let workers take initial snapshots

    _print_separator("Simulating Catalogue & Budget Changes")

    # ── Simulation: Scenario 1 — Refrigerator price change ───────────────
    print("\n  📌 Simulating: Refrigerator price increase ₹32,000 → ₹34,500")
    catalog.update_price("Refrigerator", 34500)
    sleep(1.5)

    # ── Simulation: Scenario 2 — Product becoming unavailable ───────────
    print("\n  📌 Simulating: Washing machine going out of stock")
    catalog.update_availability("Washing machine", "out_of_stock")
    sleep(1.5)

    # ── Simulation: Scenario 3 — Budget exceeded ─────────────────────────
    print("\n  📌 Simulating: Expenditure rising from ₹1,75,000 to ₹2,18,000")
    budget_provider.set_estimated_expenditure(218000)
    sleep(1.5)

    # ── Simulation: Scenario 4 — Price decrease ──────────────────────────
    print("\n  📌 Simulating: Sofa price drop ₹24,999 → ₹22,000")
    catalog.update_price("Sofa", 22000)
    sleep(1.5)

    # ── Simulation: Scenario 5 — Product back in stock ───────────────────
    print("\n  📌 Simulating: Washing machine back in stock")
    catalog.update_availability("Washing machine", "in_stock")
    sleep(1.5)

    # ── Stop everything gracefully ───────────────────────────────────────
    _print_separator("Graceful Shutdown")
    consumer_stop.set()
    manager.stop_all(timeout=5.0)
    consumer_thread.join(timeout=3.0)

    # ── Print worker status ──────────────────────────────────────────────
    _print_separator("Worker Status")
    for wid, status in manager.get_status().items():
        print(f"  {wid}: {status}")

    # ── Print metrics ─────────────────────────────────────────────────────
    _print_separator("Performance Metrics")
    for snap in manager.get_metrics_snapshot():
        print(f"\n  Worker: {snap['worker_name']} ({snap['worker_id']})")
        print(f"    Execution count:      {snap['execution_count']}")
        print(f"    Poll errors:          {snap['poll_errors']}")
        print(f"    Avg poll duration:    {snap['average_poll_duration_s']:.6f}s")
        print(f"    Max poll duration:    {snap['max_poll_duration_s']:.6f}s")
        print(f"    Events generated:     {snap['events_generated']}")
        print(f"    Events published:     {snap['events_published']}")
        print(f"    Events lost:          {snap['events_lost']}")
        print(f"    Detection latency:    avg {snap['avg_detection_latency_s']:.3f}s, "
              f"max {snap['max_detection_latency_s']:.3f}s "
              f"(change → event published)")
        print(f"    Uptime:               {snap['uptime_s']:.2f}s")

    # ── Print processed events summary ───────────────────────────────────
    _print_separator("Concurrent Worker Events Processed by ReflexAgent")
    print(f"  Total events from workers: {len(processed_events)}")
    print(f"  Actions triggered:         {len(processed_logs)}")
    for i, log in enumerate(processed_logs, 1):
        print(f"    {i}. [{log.result_type}] {log.rule_name}: {log.result_message}")

    # ── Export metrics to CSV ────────────────────────────────────────────
    csv_path = manager.export_metrics_csv("worker_metrics.csv")
    print(f"\n  Metrics exported to: {csv_path}")


# ── Main entry point ───────────────────────────────────────────────────────

def run() -> None:
    """Execute the SmartNest simulation (Assignments 1.2 + 2.1)."""
    _configure_logging()

    _print_separator("SmartNest — Assignments 1.2 & 2.1")
    print("  Rule-Based Environmental Reasoner + Concurrent Task Daemonization")
    print()

    # Shared event stream and agent
    stream = EventStream()
    engine = RuleEngine()
    for rule in all_smartnest_rules():
        engine.register(rule)
    agent = ReflexAgent(agent_id="reflex_001", rule_engine=engine, name="SmartNestReflex")

    # Part 1: Assignment 1.2 (single-threaded reflex agent)
    _run_assignment_1_2(stream, agent)

    # Part 2: Assignment 2.1 (concurrent background workers)
    _run_assignment_2_1(stream, agent)

    _print_separator("Simulation Complete")
    print("  All assignments executed successfully.")
    print()


if __name__ == "__main__":
    run()
