# SmartNest — Assignments 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2 & 5.1 (+ SerpApi product discovery, optional dimensions and attributes)

SmartNest is an autonomous Home Planning and Product Selection Assistant. This repository contains the backend foundation, a rule-based environmental reasoner (Assignment 1.2), concurrent background worker infrastructure (Assignment 2.1), distributed agent messaging over TCP sockets (Assignment 2.2), multi-agent coordination using Contract Net Protocol (Assignment 3.1), explicit mathematical utility optimization for product selection (Assignment 3.2), a MongoDB Atlas persistence layer, a BDI agent architecture (Assignment 4.1), and knowledge representation with adaptive Gemini and live-product integration (Assignment 4.2).

## Project Structure

```
backend/
├── src/
│   ├── main.py                              # Assignments 1.2 & 2.1 simulation entry point
│   ├── agent_b_server.py                    # Assignment 2.2: Agent B server (Terminal 1)
│   ├── agent_a_client.py                    # Assignment 2.2: Agent A client (Terminal 2)
│   ├── cnp_run.py                           # Assignment 3.1: CNP negotiation demo
│   ├── utility_optimization_run.py          # Assignment 3.2: utility optimization demo
│   ├── mongodb_check.py                     # MongoDB Atlas connectivity check
│   ├── mongodb_smoke_test.py                # store a value, read it back (PASS/FAIL)
│   ├── bdi_run.py                           # Assignment 4.1: BDI demonstration
│   ├── adaptive_run.py                      # Assignment 4.2: adaptive integration demo (simulated or --live)
│   ├── gemini_check.py                      # Assignment 4.2: verify your Gemini key / live search
│   └── smartnest/
│       ├── core/
│       │   ├── models.py                    # Pydantic models (events, logs, message envelope)
│       │   └── events.py                    # Thread-safe EventStream (queue.Queue)
│       ├── agents/
│       │   ├── base_agent.py                # Abstract BaseAgent
│       │   └── reflex_agent.py              # ReflexAgent (condition → action)
│       ├── reasoning/
│       │   ├── rule_engine.py               # Generic, reusable RuleEngine + Rule
│       │   └── rules.py                     # SmartNest domain rules (9 rules)
│       ├── simulation/
│       │   ├── environment.py               # MockEnvironment emitting simulated events
│       │   └── event_stream.py              # EventStream re-export
│       ├── workers/
│       │   ├── background_worker.py         # Reusable daemon worker abstraction
│       │   ├── product_monitor_worker.py    # Monitors product prices & availability
│       │   ├── budget_monitor_worker.py    # Monitors budget thresholds
│       │   ├── worker_manager.py            # Centralized lifecycle management
│       │   ├── catalog_provider.py          # Product catalogue abstraction
│       │   └── metrics.py                   # Thread-safe performance metrics
│       ├── cnp/                             # Assignment 3.1: Contract Net Protocol (see below)
│       ├── optimization/                    # Assignment 3.2: utility model and optimizer (see below)
│       ├── database/                        # MongoDB Atlas persistence layer (see below)
│       ├── bdi/                             # Assignment 4.1: BDI architecture (see below)
│       ├── knowledge/                       # Assignment 4.2: structured knowledge records and store
│       ├── integrations/                    # Assignment 4.2: gemini/ and products/ (live discovery)
│       ├── adaptive/                        # Assignment 4.2: AdaptiveSession (glues it all together)
│       └── communication/
│           ├── message_codec.py             # JSON serialization for MessageEnvelope
│           ├── socket_transport.py          # TCP transport with length-prefixed framing
│           ├── message_router.py            # Dispatch messages to registered handlers
│           ├── remote_agent_server.py       # TCP server accepting connections
│           └── task_handlers.py             # STATE_SYNC, TASK_REQUEST handlers + evaluators
├── tests/
│   ├── conftest.py
│   ├── test_models.py                       # Model serialization, validation, message types
│   ├── test_rule_engine.py                  # Engine mechanics (priority, enable, errors)
│   ├── test_rules.py                        # Each rule: positive, negative, malformed
│   ├── test_agent.py                        # ReflexAgent end-to-end scenarios
│   ├── test_environment.py                 # MockEnvironment & EventStream
│   ├── test_background_worker.py            # Worker lifecycle, status, exceptions, metrics
│   ├── test_product_monitor_worker.py       # Price/availability change detection
│   ├── test_budget_monitor_worker.py        # Budget threshold & exceeded detection
│   ├── test_worker_manager.py              # Registration, concurrent start/stop, CSV export
│   ├── test_message_codec.py               # Encode/decode, invalid JSON, type validation
│   ├── test_socket_transport.py            # Framing, partial reads, errors, context manager
│   ├── test_message_router.py              # Handler dispatch, unsupported types, exceptions
│   ├── test_task_handlers.py               # STATE_SYNC, TASK_REQUEST, evaluate_furniture_fit
│   ├── test_server_integration.py          # Full client-server round-trip over TCP
│   ├── test_negotiation_state.py, test_proposal_evaluator.py,
│   │   test_product_catalog.py, test_cnp_integration.py      # Assignment 3.1
│   └── test_utility_model.py, test_constraint_validator.py,
│       test_utility_optimizer.py, test_utility_cnp_integration.py  # Assignment 3.2
│   test_database_*.py, test_base_repository.py, test_mongodb_check.py  # MongoDB layer
│   fake_mongo.py                              # fake collections for unit tests
│   bdi_support.py                             # helpers + independent optimizer oracle for BDI tests
│   test_bdi_*.py, test_in_memory_repository.py  # Assignment 4.1
│   test_knowledge*.py, test_gemini_*.py, test_product_discovery.py,
│   test_adaptive_session.py, test_integration_architecture.py   # Assignment 4.2
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e ".[dev]"
```

MongoDB Atlas (optional until a feature needs it): see [MongoDB Atlas Integration](#mongodb-atlas-integration-persistence-infrastructure) for creating `backend/.env`.

## Running the Simulation (Assignments 1.2 & 2.1)

```bash
cd backend
python src/main.py
```

## Two-Terminal Demonstration (Assignment 2.2)

### Terminal 1 — Agent B Server

```bash
cd backend
python src/agent_b_server.py
```

The server listens on `127.0.0.1:8765` and waits for Agent A to connect.

### Terminal 2 — Agent A Client

```bash
cd backend
python src/agent_a_client.py
```

Agent A connects to Agent B, transfers apartment configuration, receives an ACK, sends a furniture evaluation task request, and displays the result.

### What Happens

1. **State Sync**: Agent A sends its apartment state (project_id, budget, rooms, preferences) to Agent B. Agent B validates it, stores it locally, and returns an ACK.

2. **Task Request**: Agent A sends a `TASK_REQUEST` with a sofa product (width 230cm, price ₹24,999) to be evaluated against the living room (width 210cm).

3. **Autonomous Evaluation**: Agent B independently evaluates the furniture fit using the synchronized state — it does not hardcode the result. It checks:
   - Width: 230cm > 210cm → sofa exceeds room width by 20cm
   - Budget: ₹24,999 ≤ ₹30,000 → within budget
   - Result: `fits: false`, `recommendation: "Reject SOFA_001..."`

4. **Response**: Agent B returns an `INFORM_RESULT` message with the evaluation result. Task IDs correlate request to response.

### Configurable Options

```bash
python src/agent_b_server.py --host 127.0.0.1 --port 8765
python src/agent_a_client.py --host 127.0.0.1 --port 8765
```

## Running Tests

```bash
cd backend
python -m pytest
```

2551 tests covering all modules (302 from Assignments 1.2–3.1, 167 for Assignment 3.2, 156 for the MongoDB layer, 340 for Assignment 4.1, 386 for Assignment 4.2, 176 for SerpApi, 130 net for optional dimensions, 110 net for optional attributes, 139 for agent security, 635 for the furnishing layer, chat, API, cart, briefs, live search and real-catalog end-to-end), plus 28 optional tests that need live services and are skipped by default (`SMARTNEST_RUN_ATLAS_TESTS=1` for Atlas, `SMARTNEST_RUN_LIVE_TESTS=1` for Gemini, `SMARTNEST_RUN_SERPAPI_TESTS=1` for SerpApi).

## Architecture

### Data Models (`core/models.py`)

All data structures use Pydantic v2 for validation and JSON serialization:

- **`EnvironmentalEvent`** — Structured observation with `event_id`, `event_type`, `source`, `timestamp`, `payload`. Describes the SmartNest environment.
- **`ActionLog`** — Records what rule fired and the result of its action.
- **`MessageEnvelope`** — Inter-agent communication with `sender`, `receiver`, `message_type`, `task_id`, `payload`, `timestamp`, `schema_version`, `signature`. Separate from `EnvironmentalEvent`.

Message types include:
- Basic: `INFORM`, `REQUEST`, `QUERY`
- Assignment 2.2: `STATE_SYNC`, `TASK_REQUEST`, `ACK`
- Contract Net Protocol: `CFP`, `PROPOSAL`, `REJECT_PROPOSAL`, `ACCEPT_PROPOSAL`, `AWARD`, `INFORM_DONE`, `INFORM_RESULT`
- Failure: `FAILURE`, `CANCEL`

### Communication Layer (`communication/`)

#### MessageCodec (`message_codec.py`)

Serializes `MessageEnvelope` to/from JSON bytes. Handles invalid JSON, malformed payloads, and missing required fields with `CodecError`.

#### SocketTransport (`socket_transport.py`)

TCP client with length-prefixed framing (4-byte big-endian length + payload). Features:
- Reliable message send/receive with partial read handling
- Configurable timeouts: a timeout before any byte of a frame arrives raises the retryable `TransportTimeoutError`; once a header has arrived, the rest of the frame uses a separate, longer `frame_timeout`
- Oversized message rejection (`MAX_MESSAGE_SIZE = 16 MiB`)
- Connection close detection
- Context manager support

#### MessageRouter (`message_router.py`)

Dispatches `MessageEnvelope` objects to registered handlers by `MessageType`. Returns `FAILURE` for unsupported types. Handler exceptions are caught and returned as `FAILURE` rather than crashing the server.

#### RemoteAgentServer (`remote_agent_server.py`)

TCP server that:
- Listens on a configurable host/port
- Accepts connections in a background daemon thread
- Handles each client in its own thread
- Reads framed messages, routes them, sends responses
- Replies `FAILURE` (`malformed_message`) to a frame that isn't a valid `MessageEnvelope` and keeps the connection open
- Keeps idle connections open (optional `idle_timeout` to drop silent clients)
- `port` reports the OS-assigned port when started with `port=0`
- Graceful shutdown via `threading.Event`; `stop()` also closes client sockets so it returns promptly

#### TaskHandlers (`task_handlers.py`)

- **`TaskHandlerRegistry`** — Holds synchronized apartment state and registered task evaluators. Evaluators receive a deep copy, so they can't modify the stored state.
- **`handle_state_sync`** — Validates `project_id` and the shape of `rooms` (a mapping of room → numeric `width_cm`/`length_cm`), stores the state, returns ACK, or FAILURE with `invalid_rooms`
- **`handle_task_request`** — Dispatches to registered evaluators, returns INFORM_RESULT
- **`evaluate_furniture_fit`** — Checks product width vs room width, depth vs room length (when both are given), and price vs remaining budget. An unknown `target_room` returns `error: unknown_target_room` rather than silently using another room. Results are data-dependent, not hardcoded.

### EventStream vs MessageEnvelope

These two remain strictly separate:
- **`EventStream`** carries `EnvironmentalEvent` objects (observations about the environment)
- **`MessageEnvelope`** carries inter-agent messages over TCP sockets

Workers continue publishing to `EventStream` without modification. The socket transport operates exclusively on `MessageEnvelope` objects. Future integration between the two would use an explicit adapter, not schema mixing.

### Rule Engine (`reasoning/`)

9 rules evaluated in priority order. See Assignment 1.2 documentation in the code.

### Background Workers (`workers/`)

Concurrent daemon threads monitoring product prices, availability (including products removed from the catalogue), and budget. See Assignment 2.1 documentation in the code.

Metric definitions:
- **Detection latency** — time from the moment the monitored source changed (providers record change timestamps) to the moment the event was published. With a 1 s poll interval it averages around 0.5 s.
- **Poll errors** — polls that raised an exception. The worker keeps running (status stays `RUNNING`) and exposes the exception as `last_error`; status becomes `ERROR` only if the loop itself crashes.
- **Events lost** — events that could not be published (e.g. a bounded `EventStream` stayed full past the publish timeout).
- A `stop()` that times out leaves the status at `STOPPING` and refuses to restart until the old thread has exited, so two polling loops can never run at once.

### Agent (`agents/`)

- **`BaseAgent`** — Abstract base with `agent_id`, `name`, `state`, `action_log`, `logger`.
- **`ReflexAgent`** — Delegates event evaluation to a `RuleEngine`. No internal memory — pure reflex.

## Future Compatibility

- **Assignment 3.1 (Multi-agent)**: Implemented — see below. It reuses the router, server and transport unchanged.
- **Assignment 3.2 (Utility optimization)**: Implemented — see below. The socket transport and task handlers did not need changes.
- **Assignment 4.1 (BDI)**: Implemented — see below. It is a new agent type on `BaseAgent`; Agent B's `apartment_state` is unchanged.
- **Assignment 4.2 (Real APIs)**: External product research will use separate provider interfaces, not the communication layer.
- **Assignment 5.1 (Security)**: `MessageEnvelope.signature` field is an Ed25519 placeholder. No crypto implemented yet.
- **Assignment 5.2 (FastAPI)**: `RemoteAgentServer` runs in daemon threads and doesn't block HTTP request handling.

## Assignment 3.1: Multi-Agent Coordination (Contract Net Protocol)

### CNP Package (`cnp/`)

- **`negotiation_state.py`** — State machine with 10 states (CREATED → CFP_SENT → COLLECTING_PROPOSALS → EVALUATING → NEGOTIATING → AWARDED → EXECUTING → COMPLETED → TERMINATED, with FAILED as alternative). AWARDED → NEGOTIATING is allowed so a refused award can go to the runner-up. Invalid transitions raise `InvalidStateTransition`. All transitions are timestamped in `history`.
- **`proposal_evaluator.py`** — Weighted utility scoring isolated from the negotiation protocol (see below). Ready for Assignment 3.2 to replace with a more sophisticated optimizer.
- **`product_catalog.py`** — Thread-safe simulated catalogues for two independent bidding agents, 3 sofas each. All entries are clearly simulated — no real retailer URLs.
- **`coordinator_agent.py`** — Owns the negotiation policy. `run_negotiation()` runs CFP rounds, gives bidders constraint and offer feedback, decides when to stop, awards the best offer from any round (falling back to the runner-up if the winner refuses), and notifies everyone.
- **`bidding_agent.py`** — Independent product research agent with its own `RemoteAgentServer` and catalogue. Picks a product for the constraints announced so far, concedes on price when it is not leading (each agent has its own maximum discount), keeps separate state per negotiation, and checks an award against its own offers (negotiation ID, task ID, product, price) before reporting INFORM_DONE.
- **`cnp_demo.py`** — Sets up the scenario and prints the round-by-round summary, timeline and state history.

### Running the CNP Demonstration

```bash
cd backend
python src/cnp_run.py
```

### CNP Negotiation Lifecycle

1. **CFP (Round 1)** — The coordinator announces what the buyer wants: a three-seater minimalist sofa, budget ₹30,000. The 210 cm room width is the coordinator's private site knowledge and is **not** announced yet (`reveal_constraints_progressively=True`; set it to `False` to announce everything up front).
2. **Proposals** — Each agent independently picks a product from its own catalogue. Without a width, each offers its best-value sofa.
3. **Evaluation** — The coordinator applies hard constraints and utility scoring.
4. **Constraint feedback** — Proposals rejected for a constraint the bidders weren't told about cause the next CFP to reveal it (`updated_constraints: {"max_width": 210}`). Bidders then switch to products that fit.
5. **Offer feedback** — Once a feasible offer exists, each bidder is told whether it is leading and asked to improve. Trailing bidders concede on price (2.5% per round, up to their own limit); the leader holds.
6. **Stopping rule** — At least `min_rounds` (2) rounds always run. After that the coordinator stops when the best offer stops improving, when there is nothing new to reveal and no feasible offer, or at `max_rounds`.
7. **Award** — The best offer from any round gets an AWARD (product + price). If the winner refuses, the next-best offer is tried.
8. **Execution & consensus** — The winner verifies the award against its own offers and sends INFORM_DONE. Losers get REJECT_PROPOSAL, everyone gets INFORM_RESULT, and the negotiation terminates. With no feasible offer, everyone gets CANCEL and the state goes FAILED → TERMINATED.

CFPs go to every bidder before any reply is read, so bidders work in parallel. A bidder that disconnects is dropped and the negotiation continues with the others.

### Utility Calculation (3.1 scoring — still the CNP default)

| Component | Weight | Formula |
|-----------|--------|---------|
| Spatial | 0.35 | `product_width / room_width` |
| Price | 0.30 | `1.0 - (proposed_price / max_budget)` |
| Style | 0.20 | exact match = 1.0, partial (one name contains the other) = 0.5, otherwise or missing style = 0.0; no preference = 1.0 |
| Availability | 0.15 | in_stock = 1.0, low_stock = 0.7, backorder/preorder = 0.4 |

Hard constraints reject before scoring: not purchasable (out of stock / unavailable / discontinued), price over budget, unknown width, width over room width, too few seats. Weights are validated (known keys, non-negative, positive sum) and normalised to sum to 1. Ties are broken by lower price, then earlier round.

### Sample Negotiation (actual output of `python src/cnp_run.py`)

```
ROUND 1   (width not announced)
   research_agent_a   SOFA_A1   230 cm  ₹24,999          REJECTED: Width 230 cm exceeds room width 210 cm
   research_agent_b   SOFA_B1   225 cm  ₹22,999          REJECTED: Width 225 cm exceeds room width 210 cm

ROUND 2   (CFP reveals max_width = 210)
   research_agent_a   SOFA_A3   205 cm  ₹29,499          utility 0.6967
   research_agent_b   SOFA_B2   200 cm  ₹28,499          utility 0.6983   ← leading

ROUND 3   (offer feedback: A trailing, B leading)
   research_agent_a   SOFA_A3   205 cm  ₹28,762 (-2.5%)  utility 0.7040   ← new leader
   research_agent_b   SOFA_B2   200 cm  ₹28,499          utility 0.6983

[AWARD]      → research_agent_a: SOFA_A3 at ₹28,762 (round 3)
[EXECUTION]  research_agent_a → INFORM_DONE
[REJECT]     → research_agent_b
[CONSENSUS]  → both agents: INFORM_RESULT
Negotiation successfully terminated.
```

### Tests

CNP tests:
- `test_negotiation_state.py` — transitions, invalid transitions, history, award-refusal path
- `test_proposal_evaluator.py` — hard constraints (incl. seating, missing width), scoring, style/availability rules, weight validation, ranking tie-breaks
- `test_product_catalog.py` — construction, filtering, unknown width, availability states, independence, no URLs
- `test_cnp_integration.py` — over real TCP: round-1 width rejections, constraint reveal and product switching, concessions and discount limits, the coordinator's stopping rule, best-across-rounds award, refusal fallback, bidder drop-out, no-proposal failure, timeline labels/timestamps, ID correlation, award verification

## Assignment 3.2: Self-Interested Agent Utility Optimization

### Running the Demonstration

```bash
cd backend
python src/utility_optimization_run.py
```

It evaluates 14 simulated sofas from Agent A's and Agent B's catalogues, prints every rejection with its reason, the normalised component scores, the ranking, the selected product with its substituted calculation, each agent's own best option, a hand-checkable example, boundary probes, and finally runs the same decision through the Contract Net Protocol over TCP. Nothing in the output is hardcoded: the winner and ranking are computed from the data.

### Utility Function

```
U(p) = 100 × (0.30·P + 0.25·Q + 0.20·F + 0.15·S + 0.10·C)        U ∈ [0, 100]

P  price affordability   clamp(1 − price / budget, 0, 1)
Q  quality               0.80 × rating/5 + 0.20 × min(review_count/100, 1)
F  spatial utilisation   0.50 × width/max_width + 0.50 × depth/max_depth
S  style compatibility   1.0 exact · 0.5 recognised-compatible · 0.0 otherwise
C  colour compatibility  1.0 exact · 0.5 same colour family · 0.0 otherwise
```

All five components are normalised to [0, 1]. Weights, sub-weights (0.80/0.20, 0.50/0.50), the review saturation (100), the match scores and the style/colour tables are all in `ScoringConfig` and can be changed. A configuration is validated when it is created: the five weights must be finite, non-negative and sum to exactly 1 (as must each pair of sub-weights), and strings or booleans used as numbers are refused. A bad weight vector raises `ScoringConfigError` before it can reach the optimizer; `ScoringConfig.normalised()` turns relative importances into valid weights.

**Spatial score assumptions.** The limits are the free space available for the sofa (wall length × floor depth). A sofa that uses more of that space without exceeding it fills the area better and scores higher, up to 1.0 at an exact fit; width and depth matter equally by default. F is only defined for products within the limits: hard constraints are checked first, and the scorer refuses an oversized product instead of returning a value above 1. Walkway clearance is the user's choice: they enter the clearance-adjusted space as the limit.

**Style and colour.** Deterministic lookup tables (e.g. minimalist ↔ scandinavian / contemporary / modern / japandi; neutral ↔ beige / cream / grey / white / ivory / taupe / charcoal). An unknown style or colour scores 0. No natural-language interpretation or LLM is involved.

### Hard Constraints

A candidate is eligible only if **price ≤ budget**, **width ≤ max width**, **depth ≤ max depth**, **seats ≥ minimum** and **availability is confirmed** (`in_stock` or `low_stock`, configurable). Limits are inclusive. Every violated constraint is reported (e.g. `price ₹30,001 exceeds budget ₹30,000 by ₹1`). Utility is computed only after a candidate passes, so a high score can never override a constraint.

Before the constraints are checked, each record is validated. A missing or malformed mandatory field (missing dimensions, negative or zero price or size, rating outside 0–5, NaN or infinite values, non-integer review counts, empty style or colour, non-mapping records, duplicates) makes the record **invalid** with an explicit reason. It is never given a default value. Invalid requirements (e.g. a zero budget, which would divide by zero) raise `InvalidRequirementsError`.

### Architecture (`smartnest/optimization/`)

| Module | Responsibility |
|---|---|
| `scoring_config.py` | `ScoringConfig`: validated, JSON-serialisable weights and compatibility tables |
| `constraint_validator.py` | `UserRequirements`, `ProductCandidate`, `validate_record`, `ConstraintValidator` (record validation and hard constraints with reasons) |
| `utility_model.py` | `UtilityModel`: the five components, the weighted total and a `UtilityBreakdown` (components, weights, points contributed) |
| `utility_optimizer.py` | `UtilityOptimizer`: exhaustive evaluation, ranking, selection, explanation; `optimize_async` for non-blocking use |

The package has no dependency on CNP, sockets, databases, LLMs or the UI. Every result is a Pydantic model that round-trips through JSON.

**Algorithm.** Exhaustive and deterministic, which is exact for catalogue sizes like these: inspect every candidate → validate → check hard constraints → score each eligible one → sort by **U ↓, Q ↓, price ↑, product ID ↑, source agent ↑** → select rank 1. The result reports how many candidates were inspected, eligible, rejected and invalid, every breakdown, the ranking, the selection, an explanation of how it beat the runner-up (component by component, including tie-break rules), and `verified_optimal`, an explicit check that the selection's U is ≥ every feasible U. With no feasible candidate, `selected` is `None`: there is no recommendation.

### CNP Integration

- `ProposalEvaluator()` is unchanged (3.1 scoring is still the default, so all 3.1 behaviour and tests are preserved). `ProposalEvaluator.utility_based()` (or `ProposalEvaluator(utility_model=UtilityModel(...))`) delegates `evaluate()` to the optimizer. `utility_score` is U/100, so both modes share the [0, 1] scale, and the full 0–100 breakdown is in `utility_breakdown`. `rank()` uses the same tie-break order as the optimizer.
- **The coordinator always recalculates.** It scores the product attributes at the **offered** price (so concessions are re-scored) and attributes the offer to the bidder it is actually talking to. A bidder's `self_reported_utility` is never used for ranking. It is kept in `reported_utility`, and a mismatch is logged as a `utility_discrepancy` event.
- **Bidders evaluate their own catalogues.** `BiddingAgent(..., utility_optimizer=UtilityOptimizer())` proposes its own highest-utility feasible product whenever the CFP states every constraint the model needs, and attaches its own estimate and breakdown at the offered price. `evaluate_own_catalog()` returns the agent's full optimisation result. If the width hasn't been announced yet (round 1 with progressive disclosure), the agent falls back to its 3.1 heuristic.
- `CoordinatorAgent.set_requirements()` accepts optional `room_depth` and `preferred_colour`.
- Negotiation protocol, awards, discount policies and consensus handling are unchanged.

### Simulated Candidates

`catalog_agent_a()` / `catalog_agent_b()` still return the three sofas each that the 3.1 scenario uses. `extended=True` adds four more per agent (14 in total), chosen to exercise each constraint and its boundary. All products now carry depth, rating, review count and source agent. All data is simulated: no real listings, prices, reviews or URLs.

### Sample Output (`python src/utility_optimization_run.py`)

Requirements: budget ₹30,000, width ≤ 210 cm, depth ≤ 110 cm, ≥ 3 seats, minimalist, neutral.

```
Rank  ID           Price        P       Q       F       S       C         U
   1  SOFA_A2    ₹27,499   0.0834  0.8960  0.8825  1.0000  1.0000     67.55
   2  SOFA_B3    ₹26,999   0.1000  0.7940  0.8524  1.0000  1.0000     64.90
   3  SOFA_A3    ₹29,499   0.0167  0.9520  0.9335  1.0000  0.5000     62.97
   4  SOFA_B2    ₹28,499   0.0500  0.8880  0.9307  1.0000  0.5000     62.32
   5  SOFA_A6    ₹30,000   0.0000  0.8560  1.0000  1.0000  0.5000     61.40
   6  SOFA_B4    ₹27,999   0.0667  0.9580  0.9725  0.5000  0.5000     57.90
   7  SOFA_A7    ₹26,499   0.1167  0.8400  0.9426  0.0000  0.0000     43.35

Rejected:
  SOFA_A1  width 230 cm exceeds maximum width 210 cm by 20 cm
  SOFA_A4  depth 115 cm exceeds maximum depth 110 cm by 5 cm
  SOFA_A5  seating capacity 2 is below the minimum 3
  SOFA_B1  width 225 cm exceeds maximum width 210 cm by 15 cm
  SOFA_B5  price ₹30,001 exceeds budget ₹30,000 by ₹1
  SOFA_B6  availability not confirmed (out_of_stock)
  SOFA_B7  width 211 cm exceeds maximum width 210 cm by 1 cm

★ SOFA_A2 — Compact Minimalist Sofa A2 (research_agent_a) at ₹27,499
  U = 100 × (0.30×0.0834 + 0.25×0.8960 + 0.20×0.8825 + 0.15×1.0000 + 0.10×1.0000)
    = 2.50 + 22.40 + 17.65 + 15.00 + 10.00 = 67.55
  Beats runner-up SOFA_B3 (64.90) by 2.65: quality +2.55, spatial +0.60; trails on price −0.50.
  verified_optimal = True

CNP over TCP (utility mode): R1 A1/B1 rejected for width → R2 A2 67.55 vs B3 64.90
  → R3 B concedes to ₹26,324 (65.57) → winner SOFA_A2, same as the exhaustive optimizer.
```

**Hand-checkable example** (also run by the demo and the tests): budget ₹10,000, room 200 × 100 cm.
- X: ₹5,000, 200 × 100, rating 5 / 100 reviews, exact style and colour → P = 0.5, Q = F = S = C = 1 → U = 100(0.15 + 0.25 + 0.20 + 0.15 + 0.10) = **85.00**.
- Y: ₹2,000, 100 × 50, rating 4, compatible style, other colour → U = 100(0.24 + 0.21 + 0.10 + 0.075) = **62.50**.
- Z: cheapest and top-rated but 1 cm too wide → **rejected**.

Selected: X.

### Tests

The 3.2 tests run without network access or external services and give the same result on every run:
- `test_utility_model.py`: each component against its formula, clamping, rating/dimension/budget guards, weight and config validation, determinism, serialisation
- `test_constraint_validator.py`: every boundary (= limit eligible, +1 rejected), unavailable states, all malformed-field cases, zero/negative/NaN requirements
- `test_utility_optimizer.py`: the 14-candidate scenario checked against an independent formula implementation, the hand-checked example, empty / all-infeasible / all-invalid sets, duplicates, tie-breaking, input-order independence, repeated runs, async, JSON round-trip, the demo
- `test_utility_cnp_integration.py`: evaluator modes, re-scoring at the offered price, ignored and audited self-reports, a lying bidder, bidder self-evaluation, full TCP negotiations that must agree with the exhaustive optimizer

### Design Notes for Later Assignments

- **4.1 (BDI):** The optimizer is a pure function of (candidates, requirements, config) with no agent state, so a BDI agent can call it while choosing intentions (beliefs → `UserRequirements`, options → candidates) without touching the scoring logic.
- **4.2 (Real APIs):** Evaluation depends only on the record schema (`validate_record` / `ProductCandidate`), not on where products come from. An API-backed catalogue only has to produce compatible dicts, and incomplete API data is rejected as invalid, never defaulted.
- **5.1 (Signing):** No security logic was added. `MessageEnvelope.signature` is untouched.
- **5.2 (FastAPI):** `UtilityOptimizer.optimize_async()` runs in a worker thread, so the event loop stays free.
- **Gemini / MongoDB:** An LLM may later turn natural language into `UserRequirements`, but all scores are computed by Python. Every model is a JSON-serialisable Pydantic model, ready for document storage.

## MongoDB Atlas Integration (Persistence Infrastructure)

SmartNest now has a persistence layer for MongoDB Atlas. This milestone is **infrastructure only**: it connects, pings, and provides repositories. No existing application data is stored or migrated yet, and nothing in the agents, workers, CNP or optimizer imports it.

### Setup

1. **Create a database user and allow your IP in Atlas** (Database Access and Network Access).
2. **Install the dependencies** (`pymongo` and `python-dotenv` are now part of `pip install -e ".[dev]"`).
3. **Create `backend/.env`** from the template and fill in the real values:

   ```bash
   cd backend
   cp .env.example .env        # Windows: copy .env.example .env
   ```

   ```ini
   MONGODB_URI=mongodb+srv://<db_username>:<db_password>@<cluster>.mongodb.net/
   MONGODB_DATABASE=smartnest
   ```

   If the password contains special characters (`@ : / ? # %`), URL-encode them (`@` becomes `%40`).
4. **Verify the connection:**

   ```bash
   python src/mongodb_check.py
   ```

   It loads the configuration, connects, pings, inserts a temporary document, reads it back and verifies it, deletes it, and closes the connection. It exits with status 0 on success. The temporary document lives in a scratch collection (`_smartnest_connectivity_check`) that is removed again if the script created it, so the database is left as it was found.

Optional settings: `MONGODB_SERVER_SELECTION_TIMEOUT_MS` (default 5000) and `MONGODB_MAX_POOL_SIZE` (default 50). Real environment variables override `.env`.

### Security

- `backend/.env` is listed in `.gitignore` and is not tracked. **Never commit credentials.** `.env.example` holds placeholders only.
- Credentials never appear in `repr()`, `str()`, log lines, error messages or output: the URI is stored with `repr=False`, error messages pass through redaction (`mongodb://***@host`), and driver exceptions are not chained into tracebacks. `mongodb_check.py` prints only the host and database name.
- Tests guard this: `.env` must be git-ignored and untracked, `.env.example` must contain only placeholders, no credential-shaped URI may appear in source or docs, and (when `backend/.env` exists) the real password must appear in no committable file.
- If a password was ever shared in a chat, a ticket or a commit, rotate it in Atlas (Database Access → Edit user).

### Package (`smartnest/database/`)

| Module | Responsibility |
|---|---|
| `settings.py` | `DatabaseSettings`, `load_settings()`: reads `backend/.env` and the environment, validates the URI and database name |
| `connection.py` | `MongoConnection`: one shared, thread-safe `MongoClient` with connection pooling, `ping()`, `check_health()`, `close()`; `get_connection()` / `close_connection()` for the process-wide instance |
| `collections.py` | `Collections`: `users`, `home_projects`, `products`, `agent_states`, `tasks`, `messages`, `recommendations`, `negotiations` |
| `repositories/base_repository.py` | `Repository` (abstract interface) and `BaseRepository` (reusable CRUD on one collection) |
| `errors.py` | `DatabaseError` hierarchy and credential redaction; no raw PyMongo exception escapes |

```python
from smartnest.database import BaseRepository, Collections, get_connection, close_connection

conn = get_connection()                       # built lazily from backend/.env
conn.connect()                                # create the client and ping (fail fast at startup)
products = BaseRepository(conn.collection(Collections.PRODUCTS), id_prefix="prd")

doc_id = products.create({"name": "Sofa", "price": 27499})
products.get(doc_id)
products.update(doc_id, {"price": 25000})
products.find({"name": "Sofa"}, sort=[("price", 1)], limit=10)
products.delete(doc_id)

close_connection()                            # at shutdown
```

`BaseRepository` operations: `create`, `create_many`, `get`, `find_one`, `find`, `count`, `exists`, `update`, `update_many`, `upsert`, `delete`, `delete_many`. Conventions:
- `_id` values are strings (generated as `<prefix>_<16 hex>` unless you supply one, e.g. a model's `event_id`).
- `created_at` / `updated_at` are maintained automatically as timezone-aware UTC datetimes.
- Pydantic models are accepted; input documents are never mutated.
- `delete_many({})` is refused unless `allow_all=True`; negative `limit` / `skip` are rejected.
- Failures are `DatabaseConnectionError`, `DatabaseOperationError`, `DuplicateDocumentError`, `InvalidDocumentError` or `DatabaseConfigError`.

### Design Notes for Later Assignments

- **4.1 (BDI):** Agents should depend on the `Repository` interface, never on PyMongo. Beliefs, desires and intentions can be stored in `agent_states` with `upsert(agent_id, {...})`, which keeps `created_at` and bumps `updated_at`; a BDI agent under test can be given an in-memory `Repository`. Because the optimizer is a pure function and the repository holds no agent logic, neither needs to change.
- **Catalogues (4.2):** A products repository can sit behind the existing `CatalogProvider` / `ProductCatalog` abstractions as an implementation detail; the workers and bidding agents keep calling the same interfaces.
- **Background workers:** `MongoClient` is thread-safe and pooled, so all workers should share the one client from `get_connection()`; repositories hold no state and are safe to share. Do not create a client per thread or per request.
- **FastAPI (5.2):** Call `get_connection().connect()` in the app's startup (lifespan) and `close_connection()` at shutdown. PyMongo is synchronous, so from `async` endpoints run repository calls with `await asyncio.to_thread(...)` (or Starlette's `run_in_threadpool`) so the event loop stays free. `check_health()` never raises and suits a `/health` endpoint.
- **Gemini:** The database layer imports nothing from the agents, the LLM or any other feature (enforced by a test).

### Database Tests

```bash
python -m pytest                                  # unit tests; Atlas tests are skipped
$env:SMARTNEST_RUN_ATLAS_TESTS = "1"; python -m pytest -m atlas     # PowerShell: live Atlas tests
SMARTNEST_RUN_ATLAS_TESTS=1 python -m pytest -m atlas               # bash
```

The unit tests need no network: a fake PyMongo client and an in-memory fake collection stand in for Atlas (`tests/fake_mongo.py`). The optional Atlas tests run a CRUD round trip, duplicate-key handling, sorting and paging, and concurrent writes from several threads, each in a uniquely named scratch collection that is dropped afterwards.

- `test_database_settings.py`: loading, precedence, validation, redaction, collection names
- `test_database_connection.py`: lazy shared client, pool options, ping and health, close and reopen, error sanitising, thread-safe shared connection
- `test_base_repository.py`: the interface, every CRUD operation, validation, error translation, concurrent use
- `test_mongodb_check.py`: the connectivity script with a fake client, including failure and cleanup paths
- `test_database_security.py`: git-ignore and no-credential guards, layer independence
- `test_database_atlas_integration.py`: optional live tests

## Assignment 4.1: BDI Architecture

A deterministic belief-desire-intention agent that keeps a model of the world, decides which goals to pursue, commits to plans, and revises them when the world changes. It extends the existing `BaseAgent` and reads the existing `EventStream`; the CNP agents, the optimizer and all earlier code are unchanged. All decisions are plain Python (no LLM); Gemini will only be used later for conversation.

### Running the Demonstration

```bash
cd backend
python src/bdi_run.py                 # state kept in an in-memory repository (works offline)
python src/bdi_run.py --atlas         # state kept in MongoDB Atlas (agent_states), record removed afterwards
python src/bdi_run.py --atlas --keep  # ... and leave the record in Atlas so you can look at it
```

The demo (all figures are computed from the data, nothing is scripted):
1. The user wants a sofa: total budget ₹2,00,000, sofa budget ₹30,000, maximum width 210 cm, minimalist (room depth 110 cm, 3 seats and a neutral colour come from the Assignment 3.2 scenario). The agent adopts an intention using the existing `UtilityOptimizer`: **SOFA_A2**.
2. An event reports that **SOFA_A2 is out of stock**. The agent updates the availability belief, recognises that its intention is invalid, reconsiders its desires, re-runs the optimizer and commits to the highest-utility feasible alternative (**SOFA_B3**). The full before/after transition is printed, and the demo independently re-runs the optimizer to prove the choice is the true best.
3. Unrelated updates cause **no replanning**.
4. Real changes (style, room width, budget) do; when nothing is feasible the agent recommends nothing, and it recovers when the budget is restored.
5. The state is saved, the agent is "restarted", and the new instance restores its beliefs, desires, intentions and history and carries on.

### Architecture (`smartnest/bdi/`)

| Module | Responsibility |
|---|---|
| `beliefs.py` | `BeliefBase`: facts (`subject`, `predicate`, `value`, `source`, `timestamp`). An update that changes nothing returns `None`; a real change returns a `BeliefChange` with old and new value. Beliefs can be **invalidated** (they then count as unknown). |
| `conditions.py` | `Condition`: a serialisable test over beliefs (`==`, `in`, `<=`, exists, ...; the right side may be another belief). Used for activation conditions and preconditions. An unknown fact makes a condition **fail**, never pass. |
| `desires.py` | `DesireManager`: goals with id, description, **priority** (0–100), activation conditions and **status** (active / blocked / suspended / achieved / dropped). Several goals run at once; conflicting goals are resolved by priority and the loser is reactivated when the winner goes. |
| `intentions.py` | `IntentionManager`: committed plans with **preconditions** and ordered **actions**. Lifecycle: pending → active → completed, with **suspend / resume / cancel / replace**. Steps run through an injected executor; a `manual` step waits for outside confirmation. |
| `planner.py` | `BDIPlanner` with goal handlers. `select_product` reads requirements from beliefs, builds candidates from the catalogues, overlays what the agent learned since (availability and price from events), runs the `UtilityOptimizer`, and turns the winner into an intention whose preconditions are the optimizer's hard constraints expressed over beliefs. `maintain_budget` monitors spend against the total budget. |
| `bdi_agent.py` | `BDIAgent(BaseAgent)`: the deliberation cycle, event handling, persistence. |
| `transitions.py` | `Transition`: every step with the state **before**, the state **after**, and a human-readable **explanation**. |
| `state_store.py` | `AgentStateStore` interface, `InMemoryAgentStateStore`, and `RepositoryAgentStateStore` (any repository with `get` / `upsert` / `delete`). |

### Deliberation Cycle

1. **Perceive**: map the event to belief updates (`product_availability`, `price_change`, `budget_update`, plus the new `room_update` and `preference_update` event types). Updates that change nothing are dropped; malformed ones (negative price, NaN, missing product, ...) are rejected and recorded.
2. **Filter**: ask each wanted goal whether the change can affect it. If nothing is affected and no intention became invalid, the agent **does not replan** and records why.
3. **Check intentions**: evaluate every live intention's preconditions against the new beliefs. A failing one is recognised as invalid and suspended.
4. **Reconsider desires**: only the affected goals are re-planned, in priority order (maintenance goals are re-checked last, against the final commitments).
5. **Revise**: adopt a new intention, **keep** the current one (re-confirmed, figures refreshed), or **replace** it.
6. **Act**: automatic steps run (check constraints, recommend); the final step (reserve) waits for `agent.confirm()`.
7. **Persist**: save beliefs, desires, intentions and history if anything changed.

**What counts as unrelated** (no replanning, optimizer not called): an update to a belief with the value it already has; a product that is not a candidate; another product becoming unavailable or more expensive (it cannot beat the current choice); a total-budget change that does not move the effective sofa budget (the smaller of the sofa and total budgets); event types the agent does not track. **What does replan**: any change to the room, seats, style, colour or effective budget; any change to the chosen product; another product becoming available or cheaper.

### Persistence

State goes through an injected `AgentStateStore`; the BDI package imports no database code and no driver. With `RepositoryAgentStateStore` over the MongoDB `BaseRepository` on `agent_states`, each agent has one document keyed by its `agent_id` containing identity, beliefs, desires, intentions, transition history and timestamps (`created_at` is kept, `updated_at` advances).

```python
from smartnest.bdi import BDIAgent, RepositoryAgentStateStore
from smartnest.database import BaseRepository, Collections, get_connection

repo = BaseRepository(get_connection().collection(Collections.AGENT_STATES), id_prefix="agent")
agent = BDIAgent("home_planner_001", planner, state_store=RepositoryAgentStateStore(repo))  # restores if a record exists
```

- **Load on start** (`auto_load`), **save after meaningful changes** (no-op updates are not saved), **restore after restart**.
- **Missing record**: the agent starts fresh and begins saving.
- **Database failure on save**: recorded as a `persistence_error` transition; the agent keeps working and saves again with the next change.
- **Database failure or corrupted record on load**: the agent runs in memory and **saving is disabled**, so an existing record can never be overwritten by a blank state. `agent.load_state()` retries.
- For tests and offline work use `InMemoryRepository` (the real `BaseRepository` rules on a dictionary) or `InMemoryAgentStateStore`.

### Integration and Compatibility

- Reuses `BaseAgent` (identity, `state`, `action_log`, `handle_event`), `EnvironmentalEvent` / `EventStream` (`agent.process_events(stream)`), the `UtilityOptimizer` and its constraint validator, and the `ProductCatalog` classes (via `CatalogCandidateSource`).
- Only addition outside `bdi/`, `database/repositories/in_memory.py` and the demo: two `EventType` members (`room_update`, `preference_update`). The reflex agent ignores them.
- **Assignment 4.2 (real APIs):** products enter only through the `CandidateSource` protocol (`get_candidates()` returning records in the optimizer's schema). A retailer-API source replaces `CatalogCandidateSource` without touching beliefs, desires, the planner or the agent. Incomplete API records are rejected by the optimizer's validation, never defaulted.
- **Goal types are pluggable:** add a `GoalHandler` (`plan` + `is_affected_by`) for a new kind of goal (for example, choosing a table) and register it with `BDIPlanner`.

### Tests

340 new tests (965 in total, plus 13 optional live-Atlas tests):
- `test_bdi_beliefs.py`: updates, no-op detection, invalidation, `ensure`, conditions
- `test_bdi_desires_intentions.py`: prioritisation, conditions, statuses, conflicts; intention lifecycle, validation, execution, failure handling
- `test_bdi_planner.py`: requirements from beliefs, planning always agrees with running the optimizer directly, preconditions detect each invalid choice, relevance rules
- `test_bdi_agent.py`: the SOFA_A2 scenario, unrelated events (with a spy proving the optimizer is not called), budget / room / preference / price changes, multiple goals, confirmation, malformed input, framework integration
- `test_bdi_persistence.py`: save, restore, missing record, load and save failures, corrupted records, no overwrite after a failed load
- `test_in_memory_repository.py`, `test_bdi_architecture.py` (BDI imports no database, network or LLM; nothing else imports BDI; the demo runs), `test_bdi_atlas_integration.py` (opt-in: `SMARTNEST_RUN_ATLAS_TESTS=1 python -m pytest -m atlas`)

## Assignment 4.2: Knowledge Representation and Adaptive API Integration

SmartNest now records what it learns as structured, provenance-tracked **knowledge**, uses **Gemini** to turn language into validated structure, discovers **real products** (originally through Google Search grounding, now through SerpApi, see the SerpApi section), and feeds new information through the existing BDI agent and `UtilityOptimizer`. Gemini never decides: it proposes, the deterministic validators and the optimizer judge.

> **Verification status (31 Oct 2026, model `gemini-3.5-flash-lite`).**
> - **Verified live:** real Gemini requests succeed, including structured requirement extraction, preference and constraint extraction, change-request interpretation, product extraction from text, and explanations (3 live tests pass; the full scenario passes with `--hybrid`).
> - **Not verified live: Google Search grounding.** With the key tested, every call that enables the `google_search` tool was rejected with **HTTP 429 `RESOURCE_EXHAUSTED`** on `gemini-3.5-flash-lite`, `gemini-3.6-flash` and `gemini-3.8-flash` (plain calls on the same key work; `gemini-2.5-flash` returns 404 for new keys). That points to a project with no grounding quota, which Google meters separately and which usually needs a billing-enabled project, but the error text does not say so. The grounded-search code is therefore tested only against mocks and real SDK response objects. **No real product listing or URL has been retrieved yet.** Once the key's project has grounding access, run `python src/gemini_check.py --search`.

### Configuration

Add to `backend/.env` (git-ignored; never commit it):

```ini
GEMINI_API_KEY=<your key from https://aistudio.google.com/apikey>
GEMINI_MODEL=gemini-3.5-flash-lite      # optional; models are retired over time, see https://ai.google.dev/gemini-api/docs/models
GEMINI_TIMEOUT_SECONDS=60               # optional
```

`.env.example` holds placeholders only. A missing or placeholder key raises a clear `GeminiConfigError`; the key never appears in `repr()`, logs or error messages (all SDK errors are redacted). The ordinary test suite needs no key and makes no network calls.

### Running It

```bash
cd backend
python src/gemini_check.py              # one REAL structured-extraction request (needs the key)
python src/gemini_check.py --search     # + REAL Google-Search-grounded product discovery
python src/adaptive_run.py              # full scenario, SIMULATED Gemini and market (no key, no network)
python src/adaptive_run.py --live       # full scenario with REAL Gemini and REAL products (needs grounding quota)
python src/adaptive_run.py --hybrid     # REAL Gemini for every language task, SIMULATED market (works without grounding quota)
python src/adaptive_run.py --atlas      # persist knowledge and agent state in MongoDB Atlas (add --live for real data)
```

`gemini_check.py` exits 0 (verified), 1 (the API call failed) or 2 (not configured). **Simulated and hybrid modes are labelled as such on every run**: the products are invented and their links end in `.invalid`, so they cannot be mistaken for real listings. In simulated mode the "Gemini" is also a scripted stand-in; in hybrid mode every language task is a real Gemini call and only the search results are invented. Only `--live` shows real products. In `--live` mode, live listings rarely change on demand, so the "new information" step injects a clearly labelled simulated availability event for the real recommended listing.

### The Scenario (what `adaptive_run.py` demonstrates)

1. The user asks for *"a minimalist three-seater sofa under ₹30,000"*.
2. Gemini returns structured requirements (budget 30000, 3 seats, minimalist). They are validated again by the deterministic rules, recorded as knowledge, and applied to the BDI agent as beliefs. Room limits (210 cm wall, 110 cm depth) come from the user's earlier home-project knowledge, not from the model.
3. Product discovery retrieves listings; incomplete or unverifiable ones are rejected with reasons.
4. Listings are saved as knowledge (source URL, retrieval time, confidence). The optimizer ranks the feasible ones; the BDI agent adopts the best and an explanation is produced from the optimizer's own figures.
5. New information arrives (a listing goes out of stock): the belief updates, the intention is found invalid, the agent replans, and the recommendation changes.
6. The user says *"I'd prefer contemporary instead"*: Gemini proposes the change, validation applies it, the market is searched again, and the agent replans.

### Packages

| Package | Responsibility |
|---|---|
| `knowledge/` | `KnowledgeRecord` (id, kind, entity, attributes, relationships, provenance, confidence, version, history, invalidation), per-kind validated attributes, and `KnowledgeStore` over an injected repository. No database driver, LLM or agent code. |
| `integrations/gemini/` | `GeminiSettings`, `GenaiClient` (the only code touching the `google-genai` SDK), `GeminiService`, Pydantic schemas, error types with secret redaction, `ScriptedLLMClient` for tests. Independent of BDI, CNP and the optimizer. |
| `integrations/products/` | `GeminiSearchProvider`, listing normalizer, URL validation and provenance, `LiveCandidateSource` / `CompositeCandidateSource` (the BDI `CandidateSource` interface), change detection. |
| `adaptive/` | `AdaptiveSession`: the only module that combines Gemini, discovery, knowledge and the BDI agent. Nothing imports it. |

#### Knowledge

Seven kinds: `requirements`, `home_project`, `room`, `budget`, `product_spec`, `observation`, `agent_knowledge`. Records are validated on creation and on every update (a failed update leaves the record untouched). An update that changes nothing is a no-op; a real change bumps the version and keeps the replaced values in `history`. Records are never silently deleted: they are **invalidated** with a reason and then excluded from queries unless asked for. Timestamps are strictly increasing within a process, so "most recently modified" is unambiguous even on Windows' coarse clock. Records live in the `knowledge` collection (`KNOWLEDGE_COLLECTION`), one document per record keyed by `knowledge_id`.

#### Gemini (`GeminiService`)

| Capability | Method |
|---|---|
| Requirement extraction (budgets, size limits, seats, style, colour) | `extract_requirements` |
| Preference and constraint extraction (liked/disliked, must-haves, hard/soft constraints with quotes) | `extract_preferences` |
| Product information extraction from supplied text | `extract_products` |
| Recommendation explanation | `explain_recommendation` |
| Conversational plan changes | `interpret_change_request` |

Every answer is requested with a Pydantic schema and then **validated**. An invalid answer is re-asked once with the specific problem, then rejected (`GeminiResponseError`); it is never patched up. Transient errors (429, 5xx, timeouts) are retried with exponential backoff. User messages and web text are wrapped in tags and the system prompt says they are data, not instructions. Explanations are checked against the optimizer's facts: if the text contains a number that is not in the facts, or does not name the recommended product, a deterministic template is used instead.

The client uses the SDK's `models.generate_content` call, which the current documentation describes as *legacy but fully supported* (the newer Interactions API can be added behind the same `LLMClient` protocol). Tools and a response schema are **never combined in one call**: grounded search returns text plus source metadata, and structured extraction is a separate, tool-free call. That works on every model regardless of tool-combination support.

#### Real product discovery (`GeminiSearchProvider`)

1. **Search**: one call with Google Search grounding (and the URL-context tool so product pages are read).
2. **Extract**: a tool-free structured call turns that text into listings; the model may only use URLs from the sources the search reported.
3. **Normalize**: every listing is validated or rejected with reasons.

Rules that keep invented data out:
- **A link must be a reported source.** Google reports each source as a `vertexaisearch.cloud.google.com/grounding-api-redirect/...` link; `RedirectResolver` reads its `Location` header (the only host ever contacted, and the destination page is never requested) to get the real page URL. A listing whose URL is not among the reported sources is rejected, as are non-`https` URLs, URLs with credentials, IP addresses and non-public hosts. Tracking parameters are stripped.
- **No search sources, no listings.** The provider never falls back to the model's own knowledge.
- **The price must appear in the retrieved text**, be positive and in INR.
- **Nothing is defaulted.** Missing rating, review count, seats, material, colour, style, or dimension unit rejects the listing (the optimizer needs them). Dimensions are converted from mm/in and checked for plausibility. A style the model had to infer is recorded in `provenance.inferred_fields`. An unstated availability stays `unknown`, which the optimizer then refuses to treat as available.
- **Images** are kept only if a `HEAD` request shows they really serve an image; otherwise dropped (never a reason to reject the product).
- **Freshness:** each record carries `verified_at`; `LiveCandidateSource` withholds listings older than `max_age` (default 6 h) or with no usable timestamp.

Honest limits: the prose a grounded model writes still comes from a model. These checks bound what can slip through; they do not make a listing guaranteed correct. Every record keeps its source URL so a person can verify it. Real listings often lack a depth, rating or review count, so a live run may reject many listings; the demo and `gemini_check.py --search` print every rejection with its reason.

#### Safeguards: Gemini cannot override hard constraints

- It returns *proposals*; every value is re-validated by `normalise_requirement` and then judged by the optimizer's constraint validator. Model claims like `utility: 100` or `fits_room: true` are not in any schema and are ignored.
- **A model-extracted value can never relax a limit already on file** (a larger budget, width or depth, or fewer seats). It is held as *pending* and reported; it applies only after `session.confirm_pending()`. Tightening a limit, stating one for the first time, and changing style or colour apply immediately.
- Utility is computed only by the `UtilityOptimizer`. Gemini only turns its output into prose.

#### Live information and the BDI agent

`LiveCandidateSource` satisfies the BDI `CandidateSource` interface and **never touches the network from `get_candidates()`**: it serves a snapshot filled by an explicit `refresh()`. `detect_changes()` turns the difference between snapshots into ordinary `EnvironmentalEvent`s (price changes, availability changes, delisted products), which the existing belief-update and replanning machinery handles unchanged. An empty candidate pool makes the sofa goal *wait*, not become *blocked*. The existing `CatalogCandidateSource` still works, alone or combined with live listings through `CompositeCandidateSource`.

### Retailer APIs (inspected, none integrated)

The spec prefers official retailer APIs "when credentials and access are available". They are not, so none is bundled, and `ProductProvider` is the plug-in point for one. What the current documentation says:

- **Amazon:** the Product Advertising API 5.0 is **retired** (migration cut-off 30 April 2026, retirement around 15 May 2026) and replaced by the **Creators API**, which needs new credentials and uses OAuth 2.0. See https://affiliate-program.amazon.com/creatorsapi/docs/en-us/paapiv5-deprecation and https://affiliate-program.amazon.in/creatorsapi/docs/en-us/migrating-to-creatorsapi-from-paapi. The old API must not be used.
- **Flipkart:** the Affiliate API needs a registered affiliate account and a private token sent in `Fk-Affiliate-Id` / `Fk-Affiliate-Token` headers. See https://affiliate.flipkart.com/api-docs/af_faq.html.
- **Scraping retailer sites** is not done.

### Tests

386 new tests (1351 in total, 23 skipped), none needing credentials or network:
- `test_knowledge.py`: creation, validation, updates, history, invalidation, serialisation, the store, failures
- `test_gemini_integration.py`: configuration; the SDK client against real SDK response and error objects (grounding metadata, URL-context status, error translation, key redaction); the service (invalid, truncated and hostile outputs, re-asks, retries, explanation grounding)
- `test_product_discovery.py`: URL checks and provenance, redirect resolution, normalization rejections, the three-stage provider, live-source staleness, `CandidateSource` compatibility with the BDI planner, change detection
- `test_adaptive_session.py`: the whole scenario, new information, conversation, the no-silent-loosening safeguard, and failure handling at every stage
- `test_integration_architecture.py`: dependency direction (Gemini imports no BDI/CNP/optimizer; the SDK is not even loaded on import), secret scans, the live-check script, the demo

Optional live tests (skipped unless enabled; the discovery tests skip with the reason if the key has no grounding quota):

```bash
SMARTNEST_RUN_LIVE_TESTS=1 python -m pytest -m live -s -rs # real Gemini calls (needs GEMINI_API_KEY)
SMARTNEST_RUN_ATLAS_TESTS=1 python -m pytest -m atlas      # real Atlas (includes knowledge persistence)
```

## SerpApi Google Shopping Product Discovery

SerpApi's `google_shopping` engine is now SmartNest's **primary live product provider**. Gemini keeps the conversational and reasoning role only (understanding requests, extracting requirements, follow-ups, explanations); **Google Search grounding is no longer used for product discovery**. `GeminiSearchProvider` remains in the code but nothing constructs it by default.

```
User -> Gemini -> structured requirements -> SerpApiCandidateSource -> parsing/validation
     -> UtilityOptimizer -> BDI agent -> MongoDB
```

> **Verification status (1 Nov 2026).**
> - **Verified live:** one real SerpApi search (`3 seater minimalist sofa`, `gl=in`) with 3 product-detail lookups (4 requests of the 10 allowed for this project). It returned 40 shopping results (10 considered) from real Indian retailers, with prices in INR, merchants, direct retailer URLs (Home Centre, Amazon.in, Wakefit, Flipkart) and "In stock online" offer text.
> - **The live run found and I fixed two mapping errors** that the mocked tests could not: product details are nested under `product_results` (the parser had read them at the top level, so every offer and spec was ignored), and Google's `product_link` contains raw spaces (it was classed invalid). A trimmed copy of the real response is kept in `tests/fixtures/serpapi_real_sample.json` and replayed by `TestRealResponseShape`.
> - **Since then:** with optional attributes (see "Optional Product Attributes and Live Recommendations" below) real listings now reach the optimizer; `python src/serpapi_check.py --atlas` shows them end to end, from the cached real response at 0 extra requests.
> - **Not run live:** the full `adaptive_run.py --live` flow (Gemini + SerpApi + BDI), to protect the 10-request project quota; its output code was only exercised on the saved real response. Note that its scenario pre-loads a 210x110 cm room, which real listings cannot verify, so it would list products as "fit not verified" instead of recommending one.

### Setup and running

Add to `backend/.env` (git-ignored; never commit it): `SERPAPI_API_KEY=...` (optional: `SERPAPI_TIMEOUT_SECONDS`, `SERPAPI_MAX_RESULTS`, `SERPAPI_MAX_DETAIL_CALLS`, `SERPAPI_CACHE_TTL_HOURS`; see `.env.example`).

```bash
python src/serpapi_check.py [--atlas] [--dump raw.json]   # ONE real search; shows names, prices, merchants, URLs, optimizer result
python src/adaptive_run.py --live [--atlas]               # Gemini -> SerpApi -> optimizer -> BDI, all real (needs both keys)
SMARTNEST_RUN_SERPAPI_TESTS=1 python -m pytest -m serpapi_live -s   # optional live test
```

A fresh search costs 1 SerpApi search plus up to `SERPAPI_MAX_DETAIL_CALLS` (default 3) product-detail lookups. Results are cached in the `search_cache` collection (MongoDB when `--atlas`, memory otherwise) for `SERPAPI_CACHE_TTL_HOURS` (default 6), so repeating a search costs nothing.

### Package (`smartnest/integrations/products/serpapi/`)

| File | Purpose |
|------|---------|
| `config.py` | `SerpApiSettings`, `load_serpapi_settings()` (key hidden from `repr`, placeholders refused) |
| `client.py` | `httpx` client for `https://serpapi.com/search.json` (`gl=in`, `hl=en`); 401/403, 429, timeout, 5xx, bad-JSON handling; no automatic retries (a retry could cost another search); redacts the key from errors **and from httpx's own request logging** |
| `parsing.py` | the missing-data policy, price/URL/spec parsing, records for the optimizer |
| `cache.py` | `SearchCache` over the repository interface: fresh/stale flags, key hash excludes the API key |
| `provider.py` | `SerpApiProvider` (`ProductProvider`) and `SerpApiCandidateSource` (a `LiveCandidateSource`, so it satisfies `CandidateSource`; `get_candidates()` never calls the network) |

### Missing-data policy (explicit, no defaults)

- **Mandatory for a listing:** a product name, a valid positive INR price, and a **usable listing URL** (a direct retailer URL or, failing that, the Google Shopping product page, labelled as such). Without these the listing is rejected with the reason.
- **Optional, stays `null` when not stated:** rating, review count, style, colour, material, seating capacity, width, depth, height. Nothing is defaulted, estimated or inferred by a model. Each product reports which optional fields were unstated (`unstated`).
- Values are read only from structured fields or explicit wording: `about_the_product.features` (`Width`, `Depth`, `Material`, `Color`, `Style`), and "N seater" in the title/snippet. Dimensions need an explicit unit. `Length` is never assumed to be width; unlabelled `210 x 90 x 80` is not guessed; conflicting statements count as missing; a rating outside 0-5 is treated as not stated.
- Currency comes from the price text (`₹`, `Rs.`, `INR`). Other currencies are rejected, never converted; an unstated currency is never assumed to be INR.
- Availability is `in_stock` / `out_of_stock` only when offer or listing text says so; otherwise `unknown` (see the next section).

### URLs

`url_type` is one of `retailer` (an https link on a non-Google domain, from a product-details offer), `google_shopping` (any Google domain, including ad redirects), `invalid` or `missing`. The record's `url` holds **only** a retailer URL; a Google page goes in `product_page_url` and is never presented as a retailer listing. The cheapest INR offer with a direct retailer URL is preferred, via the product-details endpoint for shortlisted products (within budget, capped by `SERPAPI_MAX_DETAIL_CALLS`). Each record also carries `listing_url` / `listing_url_type`: the best usable link and its honest label (`retailer`, or `google_shopping`, or `missing` / `invalid` which are rejected). Recommendations print a Google page as "Google Shopping page, NOT a retailer link".

### Known limitations (confirmed by the live run)

- Google Shopping states little: in the real response no product stated width, depth, style or colour, and rating/review count appeared for only a few. These now stay `null` and the product is still scored on what is stated (see "Optional Product Attributes" below), so such a recommendation can rest on **price alone** and says so.
- Most results carry only a Google Shopping page and **unknown stock**; only products that received a product-details lookup get a direct retailer URL and a stock statement. The number of lookups is capped (`SERPAPI_MAX_DETAIL_CALLS`) to protect a small free quota.
- Some offers have only a relative `/goto?...` Google link; those are not used as URLs.
- The integration for other retailers is a new `ProductProvider`; the BDI and optimizer need no change.

## Optional Dimensions and Dynamic Utility Weights

Recommendations no longer need room dimensions. Room limits are optional user inputs, a product's width, depth and height may be null, and nothing is ever invented. Dimensional constraints are enforced **only** for the limits the user actually gave.

```bash
python src/dimensions_run.py        # four sofas under three situations (no limits / width+depth / width only)
```

### Rules

| Situation | Spatial status | What happens |
|---|---|---|
| The user gave no room limit | `not_required` ("No room limits specified") | Fit is not a question; every product is ranked together. |
| Every given limit is checked against a stated size and holds | `verified` ("Fit verified") | Eligible; ranked; may be recommended. |
| A stated size exceeds a given limit | `incompatible` ("Dimensionally incompatible") | Rejected with a reason such as `width 235 cm exceeds maximum width 210 cm by 25 cm`. |
| A size needed for a given limit is not stated (and no stated size breaks a limit) | `unknown` ("Fit not verified") | **Not rejected, not claimed to fit.** Status `unverified`: scored, but kept in `OptimizationResult.unverified` and never `selected`. |

Only limits the user gave are checked: with a width limit alone, a missing depth does not matter. One exceeded limit makes a product incompatible even if another size is unknown. Boundaries stay inclusive. Malformed values that *are* present (`0`, negative, `NaN`, text) are still invalid records.

### Utility formula

```
Fit verified against a width and/or depth limit (SpaceFit counted):
    U = 100 × (0.30·P + 0.25·Q + 0.20·F + 0.15·S + 0.10·C)

SpaceFit excluded (no limits given, size not stated, or only a height limit):
    U = 100 × (0.375·P + 0.3125·Q + 0.1875·S + 0.125·C)
```

When F is excluded the remaining weights are scaled up in proportion (`w / (1 − w_spatial)`; ratios are preserved and the effective weights always sum to 1, checked on every breakdown). F is **excluded, never scored as 0**: the `spatial` key is absent from `components`, `weights` and `contributions`, and `UtilityBreakdown.spatial_included` is `False`. With one limit only, F uses that axis alone (its sub-weight rescales to 1); with both it is the unchanged `0.5·(w/W) + 0.5·(d/D)`. `ScoringConfig.effective_weights(spatial)` holds the redistribution, and it refuses if no other component has any weight.

### Changes by module

- **`optimization/constraint_validator.py`**: `UserRequirements.max_width_cm / max_depth_cm / max_height_cm` optional (+ `has_spatial_limits`); `ProductCandidate.width_cm / depth_cm / height_cm` optional; `assess_space()` → `SpatialAssessment`; new `unverified` status; `FIT_LABELS`.
- **`optimization/utility_model.py`**: `spatial_fit_score()`, dynamic weights in `score()`. `spatial_score()` is unchanged.
- **`optimization/utility_optimizer.py`**: `OptimizationResult.unverified`, `unverified_count`, `incompatible`, `by_fit()`; `CandidateEvaluation.spatial`, `.fit_label`, `.unverified`; the explanation mentions unverified candidates.
- **BDI** (`bdi/planner.py`, `bdi_agent.py`): room limits are no longer required before planning; fit preconditions are created only for limits that were given; `max_height_cm` is accepted; a limit that was given and then invalidated is still *unknown* (the agent waits) rather than silently dropped.
- **Recommendations** (`adaptive/session.py`, Gemini facts/prompt/template): the selected product's `fit_status` is part of the facts; the explanation prompt forbids implying a fit unless it is "Fit verified"; the fallback template states it.
- **SerpApi parsing**: width/depth no longer required for a product to reach the optimizer; `unstated_dimensions` is reported.
- **CNP is untouched.** `UserRequirements.from_cnp()` still requires width and depth by default (`require_room_limits=True`). Without that, a bidder scoring without SpaceFit and a coordinator scoring with it would disagree and honest bids would be flagged as discrepancies.

### Backward compatibility

- All complete-record behaviour is identical: with both limits given and sizes stated, results, weights and scores are unchanged (the existing 3.2, CNP, BDI and demo outputs match apart from random IDs and ports).
- Behaviour that **intentionally changed** (these existing tests were updated): a record missing `width_cm` or `depth_cm` was `invalid` and is now valid (`unverified` if a limit needs it, `eligible` otherwise); the BDI planner no longer lists `user.max_width_cm` / `user.max_depth_cm` as missing; a session without room information now plans instead of asking for it; the 3.2 demo's "missing depth" probe now prints `UNVERIFIED`.
- API additions with defaults only: new optional fields on `UserRequirements`, `ProductCandidate`, `EligibilityResult`, `CandidateEvaluation`, `OptimizationResult`, `UtilityBreakdown` and `RecommendationFacts`. Code that indexed `breakdown.components["spatial"]` must now handle its absence when SpaceFit is excluded (no such code exists in this repository).
- `OptimizationResult.rejected_count` no longer includes unverified candidates (`eligible + unverified + rejected == total_inspected`).
- (Superseded by the next section: rating, review count, style, colour, material, seating and availability are now optional too.)

## Optional Product Attributes and Live Recommendations

Real Google Shopping data is sparse, so SmartNest now recommends from what a listing actually states, says what it did not, and never fills a gap.

```bash
python src/serpapi_check.py --atlas [--explain]     # real SerpApi listings -> optimizer -> recommendation (cached 6 h in Atlas)
```

### Field requirements

| | Fields |
|---|---|
| **Mandatory** | product name, a valid positive price (and, for a live listing, a usable listing URL) |
| **Optional (null if not stated)** | rating, review count, style, colour, material, seating capacity, dimensions |

`validate_record(record, require_complete=False)`: only `product_id`, `name`, `price` and `source_agent_id` are mandatory. A value that *is* present must still be valid (a rating of 9 or a width of 0 is an error, not "missing"). `require_complete=True` restores the strict legacy rules and is what CNP negotiations use, so negotiated bids still have to be complete.

### Dynamic utility scoring

Only components supported by actual data are scored, and their configured weights are scaled up in proportion (`ScoringConfig.weights_for`) so the effective weights always sum to 1.

| Component | Scored when | Otherwise |
|---|---|---|
| Price P | always | (mandatory) |
| Quality Q | the product has a rating (the review count refines it if stated; a missing count is not scored as zero reviews) | excluded, "rating not stated" |
| Style S / Colour C | the user stated that preference **and** the product states it | excluded: "not stated" if the product lacks it, "no preference given" (not applicable) if the user stated none |
| SpaceFit F | a width/depth limit was given and the fit is verified | excluded (see previous section) |

A missing attribute is excluded, **never scored as 0**. Every `UtilityBreakdown` now also carries `completeness` (the configured weight of the scored components over that of the *applicable* ones; 1.0 = everything that applies was scored), `excluded` (component → reason) and `unavailable` (the attributes that were needed but not stated), plus a one-sentence `completeness_note`, e.g. *"Scored on 43% of the applicable criteria (P=price). Not available, so not scored: rating, review count, style."*

Ranking is still utility first; exact ties go to the more complete product, then higher quality, lower price, then id. A known consequence: a product scored on price alone can outrank a well-documented one, which is why completeness and the unavailable attributes are always shown next to the score.

### Availability: IN_STOCK, OUT_OF_STOCK, UNKNOWN

| State | Meaning | Outcome |
|---|---|---|
| `in_stock` | a configured available state (`in_stock`, `low_stock`) | eligible |
| `out_of_stock` | confirmed (`out_of_stock`, `sold_out`, `unavailable`, `discontinued`) | **rejected**: `confirmed out of stock (availability: out_of_stock)` |
| `unknown` | not stated, or any unrecognised value | **unverified**: scored and listed in `OptimizationResult.unverified`, never `selected`; "Availability unknown - check stock on the retailer's listing" |

An unstated seating capacity (when the user needs more than one seat) is unverified in the same way. Each unverified candidate lists exactly what is unconfirmed in `gaps` (`fit`, `seating`, `availability`) and why. `AdaptiveSession` exposes the best ones as `SessionTurn.unverified_options`, separate from the confirmed recommendation.

### Live result (real SerpApi data)

`serpapi_check.py` for "a minimalist three-seater sofa in India under ₹30,000" (no room limits) on a real Google Shopping search retrieved on 31 Oct 2026 (4 SerpApi requests: 1 search, 3 product-details lookups; later runs reused the 6-hour Atlas cache at 0 requests). Of the 10 listings, 7 were over the ₹30,000 budget and rejected for that reason; the optimizer's result for the rest:

| | Product | Price | Merchant | Listing | Stock | U | Completeness |
|---|---|---|---|---|---|---|---|
| **Recommended** | AMATA Eagle Solid Wood 3 Seater Sofa Cum Bed (Cream, Medium) | ₹15,009 | Amazon.in | direct retailer URL | in stock | 49.97 | 43% (price only) |
| Confirmed #2 | Helios by Home Centre Helios Scott Nxt 3-Seater Sofa | ₹22,999 | Home Centre | direct retailer URL | in stock | 23.34 | 43% (price only) |
| **Unverified** | Wakefit Miami Seater Sofa | ₹18,628 | Wakefit | direct retailer URL | in stock | 56.68 | 79%; seating capacity not stated |

The recommendation rests on price alone: rating, review count and style were not stated by Google Shopping, so they were not scored (the minimalist preference could not be assessed). The AMATA listing is a sofa-cum-bed; nothing in the retrieved data says whether it is minimalist.

### Changes and compatibility

- `optimization/constraint_validator.py`: optional attributes, three availability states, `verification_gaps`, `require_complete`. `optimization/utility_model.py`: dynamic components, `completeness`, `excluded`, `unavailable`. `optimization/scoring_config.py`: `weights_for`. `optimization/utility_optimizer.py`: gaps, availability, completeness tie-break, richer explanation, `best_unverified`.
- **SerpApi parsing:** only name, price and a usable URL are mandatory; `listing_url` / `listing_url_type` added.
- **Gemini facts/prompt/template** and **session**: availability, URL type and completeness are part of the facts; the prompt forbids praising unavailable attributes or calling a Google link a retailer link.
- **CNP**: two call sites pass `require_complete=True`; negotiation output is byte-identical to before.
- **Changed behaviour** (existing tests updated): a record missing rating / review count / style / colour / material / seating / availability is no longer `invalid`; `out_of_stock` is still rejected but with the reason `confirmed out of stock (availability: ...)`; `unknown` or unrecognised availability used to be rejected as "availability not confirmed" and is now `unverified`; with no stated style or colour preference those components are no longer scored as a constant 1.0 (rankings among products with the same data are unchanged, absolute U values shift upward-normalised).

## Assignment 4.2 — Live Workflow Verification (Gemini → SerpApi → Optimizer → BDI → MongoDB)

```bash
python src/live_run.py --atlas                      # 0 SerpApi requests: uses the 6-hour search cache only
python src/live_run.py --atlas --allow-requests 4   # permit a fresh search (1 search + 3 detail lookups)
```

`live_run.py` is the live counterpart of `adaptive_run.py`, built for *real* listings. Every language task is a real Gemini call; every product is real SerpApi Google Shopping data. A per-run request allowance (default **0**) sits on top of the lifetime ledger: if the cache is cold the run stops and says so instead of spending quota.

| Step | What happens | Verified |
|---|---|---|
| 1. Request, no room limits | Gemini extracts ₹30,000 / 3 seats / minimalist; 10 real listings; recommendation from what the data states, with completeness and the unavailable attributes | direct Amazon.in retailer URL, in stock, ₹15,009; matches an independent optimizer re-run |
| 2. New information (a **simulated** event on a real listing, labelled) | the recommended listing goes out of stock; belief updated, intention invalidated, replanned | next confirmed product (Home Centre, ₹22,999) chosen = optimizer's best remaining |
| 3. User adds "210 cm wide, 110 cm deep" | Gemini extracts the limits; no real listing states its size | confirmed recommendation **withdrawn**; candidates shown as "Fit not verified". Constraints are not weakened |
| 4. Persistence | knowledge records (SerpApi provenance), agent state saved to Atlas, agent restarted and restored | beliefs, history and the room limit survive the restart |

27 checks passed, 0 new SerpApi requests (data retrieved 31 Oct 2026 17:45 UTC; prices and stock must be refreshed before buying). The first live run found and fixed a real bug that the simulated demo could not reveal: the product-knowledge schema rejected the URL-type fields, so real listings were not being saved (now covered by a regression test).

### What Google Shopping actually states (measured on the real response: 40 results, 3 product-detail lookups)

| Attribute | Stated | Notes |
|---|---|---|
| name, price, merchant, image, Google product page | 40/40 | |
| "N seater" in the title | 27/40 | used (explicit text); otherwise seating is unverified |
| rating / review count | 6/40 (15%) | |
| direct retailer URL, stock text | only for products that got a detail lookup (3/3 did: "In stock online") | capped by `SERPAPI_MAX_DETAIL_CALLS` |
| material | sometimes, as a detail "feature" | stored, not scored |
| dimensions, style, colour | never as structured fields | |

Evidence for the open design questions:
- **Store ratings in the details are seller ratings, not product ratings**: one merchant shows an identical 4.1 (1,520 reviews) on two different sofas. They are deliberately not used as product quality.
- **Style and colour words do appear in some titles** (16/40 and 15/40), but the search itself contained "minimalist", so a title match is circular evidence. They are not used. Using them would need to be an explicit, labelled "inferred" tier, which is a policy decision.

### Open decisions for the project owner

1. Whether to add a labelled *inferred* tier (title-derived style/colour) or a second evidence source for ratings; today unstated attributes are simply not scored.
2. The price-only ranking policy: with 15% rating coverage, a recommendation will usually rest on price (completeness is always shown). Options: a minimum completeness for "recommended", or showing price-only results in a separate "limited evidence" tier.
3. How strict CNP product records stay in the final application (currently complete records are required there).

## Assignment 5.1: Agent Authentication and Security (Ed25519)

Inter-agent messages are now authenticated with Ed25519 digital signatures. **Signatures give authenticity and integrity, not secrecy**: anyone on the wire can still read a message; encrypting traffic (for example TLS) is a separate concern.

```bash
python src/security_run.py      # nine scenarios over real TCP, with the audit trail and a leak scan of its own output
```

### Package (`smartnest/security/`)

| Module | Purpose |
|---|---|
| `keys.py` | `AgentKeyPair`: Ed25519 generation from the OS secure random source; `save()` writes PKCS#8 PEM (optionally passphrase-encrypted, mode 0600 where supported, never overwrites silently); `load()`; public key and 16-hex fingerprint are safe to share and log; the private key never appears in `repr`/`str`/logs and cannot be pickled |
| `signing.py` | `sign_envelope` / `verify_envelope`. Signs a domain prefix plus canonical JSON of **every** field except the signature (`message_id, sender, receiver, message_type, task_id, payload, timestamp, schema_version`); refuses to sign as another agent; the sender's key comes only from the registry, never from the message |
| `registry.py` | `TrustedAgentRegistry`: agent id → public key + roles (public data only; JSON round-trip); one key per id, no aliasing, explicit rotation, revocation |
| `policy.py` | `AuthorizationPolicy`: roles (`coordinator`, `worker`, `observer`) → permitted message types, with optional `task_type` restrictions. Authenticated is not authorized: a trusted worker cannot issue commands (`task_request`, `state_sync`, `cfp`, `award`, `accept_proposal`, `cancel`, `request`) |
| `replay.py` | `ReplayGuard`: freshness window (default 120 s, 30 s skew) plus one-time message ids; bounded memory; consulted only after the signature verifies, so forged traffic cannot fill it |
| `audit.py` | `AuditLog`: every accepted/rejected message with code and reason (never the payload or key material), also written to the `smartnest.security.audit` logger |
| `secure_router.py` | `SecureMessageRouter` wraps a `MessageRouter` and has the same `route()` method, so the existing `RemoteAgentServer` and transport are used **unchanged**. Checks, in order: addressed to me → signed → trusted and not revoked → signature valid → fresh and unseen → role allows this type → only then the handler runs. Replies are signed with the server's own key; a rejected peer is told only a coarse reason (`authentication_failed`, `message_rejected`, `not_authorized`) |
| `channel.py` | `SecureChannel`: signs requests and verifies replies (valid signature, expected server, addressed to me, fresh) |

### Demonstration (`security_run.py`, actual output summary)

| # | Scenario | Result |
|---|---|---|
| 1 | trusted coordinator sends `state_sync` and `task_request` | accepted; handlers ran; Agent B's signed replies verified |
| 2 | the same command modified in transit (sofa width 230 → 150 cm) | rejected `bad_signature` |
| 3 | unsigned command | rejected `unsigned` |
| 4 | attacker claims to be the coordinator, signs with her own key | rejected `bad_signature` |
| 5 | unknown agent, validly signed | rejected `unknown_sender` |
| 6 | trusted **worker** issues a command | rejected `not_authorized` (its `INFORM` report is accepted) |
| 7 | captured valid command re-sent | rejected `replay` |
| 8 | command addressed to another agent forwarded to Agent B | rejected `wrong_receiver` |
| 9 | worker key reported stolen and revoked | rejected `revoked_sender` |

Handlers ran only for messages from trusted senders; nothing from the attacker reached one; the run scans its own output and logs for every private key (hex, base64, PEM) and finds none.

### Key handling and honest limits

- Keep private keys in the git-ignored `backend/keys/` (`*.pem` is ignored everywhere); never commit them. Use a passphrase (for example from an environment variable) for stored keys.
- On Windows `chmod` cannot restrict a file to one user, so an encrypted key is the real protection there.
- The registry is trust-on-configuration: SmartNest authenticates agents whose public keys an operator has registered; it does not solve first-contact key distribution.
- Not applied to the CNP negotiation traffic or the 2.2 demos yet (they are untouched, byte-identical output): the secure router can wrap any of their routers when that is wanted.
- Replay protection assumes clocks agree within the skew; the seen-id window is per process (restarting the receiver forgets it, but the freshness window still bounds replay).


---

## SmartNest Application: Whole-Home Furnishing (Furnishing Layer, API, Interface)

The earlier assignments built one decision at a time (a sofa). This layer turns them into the product: *"I just moved
into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture. Help me furnish the entire
apartment."* The existing optimizer, validators, security and persistence are reused unchanged; the BDI single-item agent
remains for single-item flows. The architecture diagram is [docs/architecture.svg](../docs/architecture.svg).

### Layers

| Layer | Where | What it does |
|---|---|---|
| Interface | `../frontend` (React, Tailwind, Recharts) | Chat, room-wise plan, budget dashboard, compare + shortlist, agent monitor |
| API | `smartnest/api` (FastAPI) | Validation, sessions, chat, events, views; no business logic |
| Chat | `furnishing/intent.py`, `assistant.py` | Rule-based parsing (₹, lakh, BHK, rooms, commands); Gemini only as fallback and rewording |
| Agents | `furnishing/agents.py`, `coordinator.py` | Space Planning, Product Research, Comparison, Budget, Coordinator |
| Messaging | `furnishing/activity.py` | Every agent request/reply is an Ed25519-signed `MessageEnvelope` through `SecureMessageRouter` |
| Data | `furnishing/catalog.py` | Real SerpApi snapshots; labelled simulated samples only as a fallback |

### The agents
* **Space Planning** expands "2BHK" into living room, bedrooms and kitchen and the items each needs; room dimensions are
  optional (none are invented) and its assumptions are listed in the plan.
* **Product Research** serves a category's listings and drops accessories (a ₹499 "refrigerator" is a magnet) with the
  reason recorded.
* **Comparison** runs the existing `UtilityOptimizer` per item: hard limits, dynamic weights, confirmed vs unverified.
* **Budget** splits the budget into **hard per-item ceilings** (they sum to the budget), re-shares leftovers, and
  produces the analytics. If an even split leaves every open item just short, the whole remainder is offered to open
  items one at a time, so money is not wasted.
* **Coordinator** holds the plan-level BDI state (beliefs in the same `BeliefBase`, desire "furnish everything inside the
  budget", one intention per item) and re-plans on events. Each item's intention is `retained`, `replaced` or
  `invalidated`, with an explanation (`transitions`).

Events: out of stock, price change, budget change, preference change, remove item, choose product (the user's own pick:
accepted only if the optimizer still accepts it and the total stays inside the budget, otherwise refused with a reason).

### Shopping cart
A buy list the user fills from **any listing SmartNest has found** (a plan pick, an alternative, a shortlisted product, or
any other listing in a category) with a quantity of 1 to 10 per product. SmartNest sells nothing and has no checkout:
each line links out to the retailer (or the labelled Google Shopping page) and the cart keeps a running total.
* Rules (`furnishing/cart.py`): only found products; no product known to be out of stock; **the cart total can never
  exceed the total budget** (an addition that would is refused with the exact shortfall and the cart is left untouched);
  stock stays "unverified" unless the listing states it; sample data has no link.
* Later events never silently edit the cart: after a budget cut, a price change or an out-of-stock event the cart view
  reports what no longer holds (over budget, now out of stock) and the user decides. Reducing a cart is always allowed.
* Events: `cart_add`, `cart_set`, `cart_remove`, `cart_clear`, `cart_add_plan` (sets each plan pick to its count, so it is
  idempotent; both bedrooms sharing a bed gives quantity 2). Cart events do not re-plan. They are stored in the session's
  event list, so the cart survives a restart like the plan.
* Chat: "add the sofa to my cart", "add the bed in bedroom 1 to cart", "add everything to my cart", "show my cart",
  "remove the sofa from my cart", "clear my cart". An unclear item is asked about, never guessed.
* Interface: **Add to cart** on every plan item, alternative and shortlist row; a **Cart** tab with quantity steppers,
  totals against the budget, warnings, "Add all plan picks", "Clear cart" and "Copy buy list" (plain text with links).

### Versatile requests: any room, who it is for, themes
The assistant no longer assumes a "BHK" home. A request is a **brief** (`furnishing/brief.py`, `intent.py`):

| Piece | Examples understood |
|---|---|
| Scope | a home type (`2BHK`) or rooms: "my son's bedroom", "living room and kitchen", "my study", "nursery" |
| Budget | `₹2 lakh`, `80k`, `under 50000`, `within ₹45,000`, `budget of 1.5 lakh`, `60000 to spend`, or a bare number when the assistant just asked ("40000"); "1200 sq ft" or "under 3 seats" are never read as money |
| Who | age ("10 year old"), son / daughter / kid / teen, which turns a bedroom into a **kid's bedroom** |
| Themes | "he loves cartoons and monsters", "into dinosaurs", "space themed" (a preference, never a requirement) |

It asks only for what is missing ("Which room or home...? your total budget?") and says back what it understood, so it can
be corrected. Gemini reads free-form briefs the rules cannot (`rooms`, `budget`, `audience`, `age`, `themes`), but every
value is validated (known rooms only, positive budget up to 1e9, age 0-99, themes as short plain words) and the rules always
win over Gemini.

New categories: kid's bed, study table, study chair, bookshelf, wall decor, table lamp, rug, bean bag, toy storage (rooms:
kid's bedroom, bedroom, study, plus the living room and kitchen). Room templates live in `rooms.py` (`SCOPE_ROOMS`).

**Themes** rank by the listing's own title: among the options for an item, those whose title mentions a theme word come
first (stock is still labelled; an unverified themed listing is shown as unverified). Nothing is inferred; if no title
mentions the theme the optimizer's order stands and the reply says "No listing I have mentions ...".

**Categories with no listings yet** (the four above have no captured data) are shown as "No listings yet" and get **no
share of the budget** (their money goes to items that can be bought); they are never filled with invented products.

**Live search (opt-in, quota-limited).** A "Search live (1 request)" button on those items runs ONE Google Shopping search
for that category (theme words are added to wall decor queries), saves the result as a snapshot and re-plans. It is off by
default and does nothing unless `SMARTNEST_ALLOW_LIVE_SEARCH=1`, a SerpApi key is set and the lifetime allowance
(`SERPAPI_LIFETIME_LIMIT`, counted in `.serpapi_usage.json`) is not used up; only categories in the current plan can be
searched; the browser asks for confirmation; the planner never searches by itself. `POST /api/sessions/{id}/search`
returns 409 (switched off / allowance used up), 502 (the search failed) or 404 (nothing usable found).

**Tight budgets.** Item budgets start as shares by category weight, but a share can be smaller than the cheapest listing even
when everything would fit. If a buyable item is left unfilled, SmartNest asks the Comparison agent for the cheapest valid price
of every item. If those minimums fit the budget, each item gets its own minimum plus a share of the slack and the plan is
re-made, so **every item is filled whenever filling them all is possible** (tested as a property over many budgets, on real and
simulated data). If they do not fit, nothing is relaxed: the unfilled item says what the cheapest listing costs and how much is
left, the plan reports `minimum_total`, `shortfall` and `budget_needed`, and the interface offers **Raise budget to ₹X** and
**Remove item** on the card (for example ₹25,000 for a 10-year-old's room: the cheapest wardrobe alone is ₹12,319, the cheapest
valid choice for all 11 items totals ₹29,581, so the wardrobe is left out and ₹30,000 is suggested).

### Dark mode
A sun/moon button in the header switches between a light and a dark theme (`frontend/src/theme.ts`, tokens in `index.css`).
* **Follows the system on a first visit**, then remembers an explicit choice (`localStorage`, with blocked storage handled);
  while no choice has been made it keeps following the system setting if that changes.
* **No flash:** a tiny script in `index.html` sets `data-theme` before the first paint, and colour transitions are enabled
  only afterwards, so a dark-mode user never sees white or a fade-in on load.
* **Token based:** every colour is a CSS variable (`paper`, `surface`, `card`, `subtle`, `ink`, `muted`, `line`, `brand`,
  `fill`, `ok`, `warn`, `bad` and their soft tints). Dark is a warm charcoal with a restrained teal accent; a separate `fill`
  token keeps white-on-teal buttons readable. Product photos sit on a light tile so white-background images do not glare.
* **Charts** (Recharts takes literal colours) use a theme-aware palette, light axis labels, a dark tooltip and neutral legends.
* **Checked, not eyeballed:** `node e2e/dark-mode.mjs` runs a WCAG contrast audit over every visible text element (including
  chart labels) on all five tabs in both themes. It found a real light-mode defect (the green "ok" badge was 4.47:1) which is
  fixed; every tab now passes 4.5:1 (3:1 for large text) in both themes.

### Categories from names (any item, not a fixed list)
The 16 curated categories carry knowledge (search phrase, price range, accessory words, budget weight) and saved real data.
They are no longer the only things SmartNest can plan for. When a request names things to buy ("a telescope and a bed, budget
40000"), **Gemini is asked for category NAMES only** (`bed`, `telescope`...), never prices, links, ranges or brands. Everything
else comes from the shopping search (SerpApi) or, where there is no data yet, an honest "No listings yet".
* `naming.py` is the gate between names and the planner. A name maps to a **curated** category only if its extra words are plain
  modifiers ("queen size bed" is a `bed`, "3 door wardrobe" a `wardrobe`, "couch" a `sofa`); anything else ("standing desk",
  "gaming chair", "telescope") becomes its **own** category, so it is searched for by name instead of being squeezed into the
  nearest curated one. "bed and wardrobe" is split.
* **Model output is screened like any input:** short noun phrases only (up to 4 words, 40 characters), instruction-like words
  blocked (`ignore`, `reveal`, `prompt`, `password`, URLs, scripts...), validated as a whole before splitting (`../../etc/passwd`
  is not two items), at most 12 items, and the valid ones are kept next to the refused ones.
* A new category's id is a strict slug (`x_<=4 words>`, only `[a-z0-9_]`) because it becomes a snapshot file name and a search
  phrase sent to a paid API; anything else is refused (`categories.category()` raises).
* **No curated knowledge, so no made-up knowledge:** a custom category has no price range of its own. Its price floor is taken
  from the search results themselves (a listing under 25% of the median, with at least 5 results, is dropped as a likely
  accessory, with the reason recorded). A word that is part of the name ("toy" in "toys") is not treated as an accessory marker.
* When rules can already read the request (a home or room with a budget) Gemini is not called at all. For a list of items it
  always is, because rules only know the curated words; Gemini's list then replaces the rules' word-spotting, and the rules' list
  is the fallback if Gemini is down or offers nothing.
* A custom item with no data is a plan line like any other: no budget share, "Search live" (quota-gated, one request, the search
  phrase is the name). An item list can also be added **on top of** a room or home ("2BHK" plus "a telescope" goes in an
  "Extras" room; items the template already has are not duplicated). The list is stored with the session, so it replays exactly.
* Chat understands custom items already in the plan ("why this telescope?", "I don't need a telescope").
* Gemini rewording of replies must now keep **every** number of the rule-based text (it already could not add any), so a budget
  or shortfall figure cannot silently vanish.

### Saved listings must answer THIS request (themes)
Wall decor and rugs depend on the theme, so a saved search is only reused when it covers the requested theme
(`SnapshotCatalog.listings_for`):
* every themed live search is saved in its **own file** (`wall_decor__<words>.json`), so a new theme never overwrites an old one;
* for a request, the best saved search wins: an exact match, else one that mentions a requested theme (more is better), else a
  generic search with no theme words (a plain "kids rug" is valid for any theme);
* if nothing covers the theme (a "monster cartoon wall poster" snapshot for a princess / Barbie request) the item gets **no
  listings** ("No listings yet ... the saved listings were searched for 'monster cartoon wall poster', which does not cover your
  theme") and its share of the budget goes to the other items; it is never filled with the wrong thing;
* a generic pick that mentions none of the requested themes is flagged **"Not themed"** (`theme_unmatched`) with the themed
  search it would take; changing the themes in chat re-reads these listings.

### Using live requests
Live search never runs on its own. With `SMARTNEST_ALLOW_LIVE_SEARCH=1` the interface offers: **Search live** on an item with no
listings, **Search live for the themed query** on a "Not themed" pick, **Refresh listings (1 request)** in any item's "Why this?"
panel (to re-fetch fresh data for any category), and one banner, **Search live for all N (N requests)**, that runs them one at a
time (a failure stops the run) after a single confirmation that states the cost. Everything else is served from saved snapshots, by design,
so a normal plan costs no request.

### Honesty rules (tested)
* A pick is **confirmed** only if its listing states it is in stock; otherwise it is **unverified** (still counted in the
  budget). Missing ratings, sizes, styles and colours stay missing and are listed as "not stated".
* Item budgets are hard ceilings; the plan can never exceed the total. An item that cannot be filled is explained, and no
  constraint is relaxed to force it.
* Simulated listings are labelled "Sample data", have no purchase link and no image. A Google Shopping page is never called
  a retailer link.
* Gemini never selects a product, price or budget. Its rewording is discarded if it adds a number or drops a caution.

### Real data and the SerpApi quota
`python src/capture_catalog.py --plan` shows what would be spent. **Sixteen categories** hold real listings:
* the first seven (sofa, TV unit, bed, mattress, wardrobe, dining table, refrigerator) were captured with the first 10
  requests (the sofa from the earlier cache, 6 fresh one-request searches, details only for the sofa);
* nine more for a child's room (kid's bed, study table, study chair, bookshelf, wall decor, table lamp, rug, bean bag, toy
  storage) with the second allowance of 10: 9 one-request searches (wall decor searched as "monster cartoon wall poster"
  for the cartoon/monster request), 1 request kept spare. **19 of 20 requests are used.**

Raw responses are cached in Atlas and parsed listings saved to `backend/data/catalog/*.json` (git-ignored: third-party
data). **The running application makes no SerpApi request** unless the opt-in live search is enabled (below); the local
ledger refuses anything beyond `SERPAPI_LIFETIME_LIMIT` when one is set. The cap is currently **removed on purpose**
(`SERPAPI_LIFETIME_LIMIT=unlimited`; an absent value is different: the capture script refuses to run, because the `.env` may not
have loaded). Requests are still counted, each capture run is still limited by `--spend`, and SerpApi's own plan limit (the free
plan: 250 searches a month, 50 an hour) still applies; a 429 from SerpApi is reported as a quota error. Listings captured without a details lookup link to the
Google Shopping product page (labelled); only the three sofas with details have direct retailer links and stock evidence.

**Child-safety of theme matches.** For a child's room, listings whose title has an obviously mature word (a small list in
`categories.MATURE_WORDS`: thriller, horror, adult, 18+...) are dropped with the reason recorded, and among theme matches the
title matching more theme words, then the one worded for children, wins. A title cannot prove a product is suitable (for
example an adult manga poster titled "Monster"), so every theme match is shown as "matched by title only" and the user is
asked to check it.

### Run it
```
pip install -e .                               # backend (adds fastapi, uvicorn)
python src/serve.py                            # API on http://127.0.0.1:8000   (--atlas persists sessions in MongoDB)
cd ../frontend && npm install && npm run dev   # UI on http://localhost:5173 (proxies /api to the backend)
python src/furnishing_run.py                   # terminal demo of the agents, the plan and replanning
```
Settings (read from `backend/.env`; a real environment variable wins): `SMARTNEST_ALLOW_SIMULATED=auto|0|1` (auto: only when no real snapshot exists),
`SMARTNEST_USE_GEMINI=auto|0|1`, `SMARTNEST_USE_ATLAS=0|1`, `SMARTNEST_ALLOW_LIVE_SEARCH=0|1`, `SMARTNEST_CATALOG_DIR`, `SMARTNEST_CORS_ORIGINS`.
Sessions are stored as `{request, events}` and rebuilt by replay (planning is deterministic), so a restart loses nothing.

### Testing
* `pytest` (backend): furnishing layer, chat, API (validation, persistence, rate limit, CORS, no secrets) and
  `test_e2e_real_catalog.py` (the demo scenario on the real snapshots; every recommended product must exist in the
  snapshot with unaltered name, price, seller and link; skipped if no snapshots).
* `npm test` (frontend, vitest): formatting, item card states (no link for simulated data, unverified never "confirmed"),
  chat panel, budget view, API client errors.
* `node e2e/princess.mjs` (a theme with no saved listings; nothing is spent), `node e2e/items.mjs` (a request for things incl. an unknown one, needs the Gemini key), `node e2e/dark-mode.mjs` (theme toggle, persistence, contrast audit), `node e2e/tight-budget.mjs` (a budget too small
  for everything) and `node e2e/kids-room.mjs` (a child's bedroom with themes, a vague first message, a bare-number answer) and
  `node e2e/demo.mjs` (frontend, Playwright against the running stack): 13 and 23 checks from the demo message to the agent
  monitor, a page reload and the mobile layout.
* `node e2e/record-demo.mjs` records the ~3 minute captioned demo to `docs/demo/smartnest-demo.webm`;
  the narration is in [docs/DEMO_SCRIPT.md](../docs/DEMO_SCRIPT.md).

### Known limits
* Real Google Shopping data states little: ratings on about 15% of results; no sizes, styles or colours. Ranking therefore
  rests mostly on price (and rating where present), so the cheapest valid option usually wins and a large budget is
  mostly left unspent; the user can upgrade any item from its alternatives. A style preference only matters where a
  listing states a style.
* Both bedrooms get the same top product (matching furniture, counted twice).
* The agents run in one process over the signed router; moving one to another host needs only a `RemoteAgentServer`
  and a `SecureChannel`. Signatures give authenticity and integrity, not encryption.
* CNP (Assignment 3.1) traffic is still unsigned.
