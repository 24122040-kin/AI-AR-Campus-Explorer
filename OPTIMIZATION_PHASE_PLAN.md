# LocalNavBot Optimization Phase Plan

## Goal

Prepare a focused optimization phase for this workspace that improves:

- web quality and UX consistency
- module boundaries and maintainability
- runtime performance and failure transparency
- testability and observability

This plan is repo-specific. It is based on the current structure and hotspot modules in this folder, not a generic template.

## Current Architecture Snapshot

Core runtime layers:

- `web/app.py`: app bootstrap plus many legacy endpoints
- `web/routes/*`: newer route split by concern
- `routing/router.py`: main route engine and fallback logic
- `bot/nav_bot.py`: bot orchestration, intent parsing, route presentation
- `core/vpr_engine.py`: VPR pipeline and fallback indexing
- `core/realtime_manager.py`: realtime navigation/session coordination
- `core/database.py`: SQLite access layer
- `web/static/js/*`: client logic split into route/chat/gps/floor/speech/camera/ar

## Critical Modules To Examine First

These are the modules that stand out and should be examined transparently before broad optimization work.

### 1. `routing/router.py` — Critical

Why it stands out:

- largest and most coupled backend module
- mixes graph loading, routing strategy, Valhalla integration, fallback logic, indoor switching, geocoding, and route parsing
- performance-sensitive and correctness-sensitive
- hard to reason about regressions due to broad responsibility

Primary risks:

- hidden routing regressions
- fallback path drift between Valhalla and OSMNX flows
- expensive graph operations repeated more than necessary
- limited observability around why a given route was selected

Optimization objectives:

- split by responsibility
- make route selection and fallback decisions observable
- reduce repeated graph or heuristic work
- standardize route analysis output across providers

### 2. `web/app.py` — Critical

Why it stands out:

- still contains many `_legacy/*` endpoints
- mixes startup, state wiring, route registration, and old controller logic
- creates duplicate surface area with `web/routes/*`
- increases maintenance cost and web inconsistency

Primary risks:

- duplicated behavior between legacy and new APIs
- route drift and inconsistent validation
- startup becoming harder to test and optimize

Optimization objectives:

- shrink `web/app.py` to bootstrap only
- move or remove legacy endpoints behind a deprecation plan
- centralize shared request/response schemas

### 3. `bot/nav_bot.py` — Critical

Why it stands out:

- large orchestration layer
- blends LLM interaction, route formatting, VPR context, OCR context, and fallback intent logic
- user-facing quality depends on it directly

Primary risks:

- inconsistent response quality
- hidden prompt/data coupling
- weak error classification around LLM/VPR/OCR failures

Optimization objectives:

- separate intent parsing, context building, and response formatting
- make bot failures degrade predictably
- improve test coverage for route formatting and fallback intent rules

### 4. `core/vpr_engine.py` — High

Why it stands out:

- ML-heavy path with several fallbacks
- critical for image recognition quality and indexing performance
- likely sensitive to environment drift

Primary risks:

- slow indexing
- inconsistent extractor fallback behavior
- poor visibility into recall/precision and index health

Optimization objectives:

- define benchmark path for indexing/query latency
- isolate model backend selection
- add structured index-health diagnostics

### 5. `core/realtime_manager.py` + `bot/session_manager.py` — High

Why they stand out:

- session/state logic tends to accumulate hidden bugs
- likely source of race conditions and stale route/session state
- central to live navigation quality

Primary risks:

- state desync between websocket, GPS updates, and route state
- hard-to-reproduce timing bugs

Optimization objectives:

- formalize state transitions
- make session events traceable
- add tests for reroute and active-session flows

## Web Quality Improvement Plan

### Web priorities

1. simplify backend API surface
2. improve frontend consistency
3. remove duplicated flow between old and new endpoints
4. make loading, error, and empty states explicit
5. reduce large JS globals and hidden cross-module coupling

### Web issues visible now

- `web/app.py` still carries extensive legacy API logic
- frontend uses many global functions across `web/static/js/*`
- route, speech, camera, GPS, AR, and floor features are loosely coupled through shared globals
- likely inconsistent UX across tabs because behavior is distributed rather than composed

### Web optimization targets

- one primary API path per feature
- one frontend state model for:
  - current route
  - current GPS fix
  - current session
  - current upload/job state
  - current floor/AR mode
- explicit error rendering instead of silent failure
- minimal blocking work on page load

### Web refactor slices

#### Slice A: Backend web cleanup

- reduce `web/app.py` to app startup, middleware, static serving, router inclusion
- migrate or remove `_legacy/*` routes
- unify request validation strategy across route modules

#### Slice B: Frontend state cleanup

- define a single app state object in `web/static/js/globals.js` or a replacement module
- remove ad hoc cross-file mutation patterns
- standardize fetch wrapper and error handling

#### Slice C: UX polish

- loading states for route search, VPR query, OCR, upload, and traffic refresh
- stable toasts/error banners
- empty-state copy for search, routes, map, and data views
- performance pass for initial render and route/map interactions

## Module-by-Module Optimization Draft

### `routing/`

#### `routing/router.py`

Planned split:

- `routing/osm_graph.py`
- `routing/valhalla_client.py`
- `routing/route_analysis.py`
- `routing/route_merging.py`
- `routing/location_resolver.py`
- keep `NavRouter` as orchestrator only

Quality gates:

- same route output schema for Valhalla, OSMNX, chained, and indoor paths
- route provider decision included in structured analysis/logs
- benchmark route latency before/after refactor

#### `routing/route_renderer.py`

Improve:

- avoid broad silent exceptions
- separate data formatting from HTML generation
- define map rendering fallback behavior explicitly

### `bot/`

#### `bot/nav_bot.py`

Planned split:

- `bot/intent_parser.py`
- `bot/context_builder.py`
- `bot/response_formatter.py`
- `bot/media_context.py`

Quality gates:

- deterministic tests for:
  - fallback intent detection
  - route formatting
  - image/VPR/OCR context assembly

#### `bot/session_manager.py`

Improve:

- make state transitions explicit
- isolate reroute policy
- define replayable session event log for debugging

### `core/`

#### `core/vpr_engine.py`

Improve:

- separate backend selection from indexing/query orchestration
- benchmark ORB vs DINOv2 fallback path
- add structured metadata for index readiness and backend reason

#### `core/database.py`

Improve:

- review query hotspots
- standardize row-to-domain mapping
- define repository-style access for high-use flows

#### `core/realtime_manager.py`

Improve:

- isolate event ingestion from navigation decision logic
- measure update frequency, queueing behavior, and stale-session cleanup

#### `core/image_manager.py`

Improve:

- reduce broad exception swallowing
- define ingest pipeline stages explicitly:
  - validate
  - read EXIF
  - quality filter
  - dedupe
  - caption
  - persist

### `web/routes/`

Priority order:

1. `chat.py`
2. `navigation.py`
3. `realtime.py`
4. `data.py`
5. `speech.py`
6. `indoor.py`
7. `traffic.py`
8. `system.py`
9. `experimental.py`

Refactor rule:

- keep route modules thin
- move business logic into service/core layers
- avoid per-route custom error behavior unless necessary

### `web/static/js/`

Priority order:

1. `globals.js`
2. `route.js`
3. `chat.js`
4. `websocket.js`
5. `floor.js`
6. `speech.js`
7. `camera.js`
8. `ar.js`

Targets:

- eliminate hidden global coupling
- centralize API calls and error handling
- make route and session state the backbone for UI updates

## Transparent Optimization Workflow

This phase should be run transparently, with measurable checkpoints.

### Phase 0: Baseline

Before changing architecture:

- capture route latency for representative scenarios
- capture startup time
- capture VPR query latency
- capture upload/import latency
- list legacy endpoints still actively used
- note current test coverage and gaps

Deliverables:

- baseline metrics file
- dependency/runtime map
- risk register

### Phase 1: Observability First

Add visibility before deep refactors:

- structured logs for route provider selection
- timing for route, VPR, OCR, upload, and realtime update paths
- failure reasons instead of broad silent fallback where possible

Deliverables:

- instrumentation patch
- simple diagnostics endpoint or internal debug report

### Phase 2: Critical Backend Refactor

Order:

1. `routing/router.py`
2. `web/app.py`
3. `bot/nav_bot.py`
4. `core/realtime_manager.py`
5. `core/vpr_engine.py`

Rule:

- each critical module gets a dedicated refactor PR or branch slice
- no simultaneous deep refactor across multiple critical modules

### Phase 3: Web Quality Pass

Order:

1. unify API usage
2. simplify frontend state
3. polish UX states
4. remove dead legacy paths from UI

### Phase 4: Performance and Regression Pass

- benchmark route generation
- benchmark VPR indexing/query
- benchmark page load and route interaction
- add missing tests for regressions introduced by refactor

## Proposed KPIs

### Backend KPIs

- route generation latency:
  - median
  - p95
- startup time
- VPR query latency
- image batch import throughput
- reroute decision latency
- error classification coverage

### Web KPIs

- first usable render time
- route request to first visible feedback
- number of silent frontend failures
- number of duplicated API paths still exposed

### Code quality KPIs

- line count reduction in:
  - `routing/router.py`
  - `web/app.py`
  - `bot/nav_bot.py`
- broad `except Exception` count reduced
- test coverage added on critical paths

## Recommended Execution Order

### Sprint 1: Diagnostic and safety

- baseline metrics
- observability
- identify active vs dead legacy web flows
- define route contract tests

### Sprint 2: Routing core

- split `routing/router.py`
- stabilize route analysis and provider selection
- regression tests for route output

### Sprint 3: Web backend

- shrink `web/app.py`
- move or retire legacy endpoints
- normalize route module boundaries

### Sprint 4: Bot and realtime

- split `bot/nav_bot.py`
- harden session and realtime state transitions

### Sprint 5: Frontend quality

- unify state management
- reduce globals
- polish UX and failure states

## Immediate Action List

If starting now, these are the first concrete tasks:

1. audit `routing/router.py` and extract responsibility map
2. inventory every `_legacy/*` endpoint in `web/app.py` and mark:
   - keep
   - migrate
   - remove
3. add timing/logging around route generation and VPR query
4. write regression tests for route schema and bot route formatting
5. define frontend shared state contract for route/session/GPS

## Decision Rule For The Next Optimization Pass

Do not optimize everything at once.

Prioritize by:

1. correctness risk
2. architectural coupling
3. user-facing impact
4. measurable runtime cost

By that rule, the first module to inspect deeply is:

- [routing/router.py](/C:/Users/Admin/Downloads/local_nav_bot/routing/router.py)

The second is:

- [web/app.py](/C:/Users/Admin/Downloads/local_nav_bot/web/app.py)

The third is:

- [bot/nav_bot.py](/C:/Users/Admin/Downloads/local_nav_bot/bot/nav_bot.py)

