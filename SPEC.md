# Implementation Specification — External Truck Gate Scheduling

**Status:** Implementable spec (v1) — updated to match the implemented codebase  
**Source problem:** `project-5-trucks-scheduling.md` (the corrected/polished formulation)

The full specification is split under [`spec/`](spec/) so behavior changes usually
edit **one section file**. This root file is only the index.

**Editing rule:** change the smallest relevant file under `spec/` (e.g. Auto policy →
[`spec/08-dispatcher.md`](spec/08-dispatcher.md) and/or [`spec/07-tier-switch.md`](spec/07-tier-switch.md);
ALNS params → [`spec/05-alns.md`](spec/05-alns.md)). Do not rewrite unrelated sections.

## Contents

- [0. Summary of the decision](spec/00-summary.md)
- [1. Problem model](spec/01-problem-model.md)
- [2. Data formats & I/O](spec/02-data-io.md)
- [3. Module / repo architecture](spec/03-architecture.md)
- [4. Tier 3 — Greedy ERD/SPT](spec/04-greedy.md)
- [4A. Tier 1 — CP-SAT](spec/04a-cpsat.md)
- [4B. Metaheuristics (Tabu, GA, hybrids)](spec/04b-metaheuristics.md)
- [5. Tier 2 — ALNS](spec/05-alns.md)
- [6. Instance generator](spec/06-instance-gen.md)
- [6H. ALNS hyperparameter tuning](spec/06h-alns-tuning.md)
- [7. Tier-switch determination](spec/07-tier-switch.md)
- [8. Runtime dispatcher](spec/08-dispatcher.md)
- [9. Experiments & reporting](spec/09-experiments.md)
- [10. UI (Streamlit)](spec/10-ui.md)
- [11. Testing](spec/11-testing.md)
- [12. Deliverables checklist](spec/12-checklist.md)
- [13. Defaults if you skip tuning/profiling](spec/13-defaults.md)

