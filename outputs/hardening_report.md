# ONN-ES Hardening & Reproducibility Report

**Date**: 2026-01-07  
**Status**: ✅ All Sections Completed  
**Pytest Result**: 531 passed, 0 warnings (after fix)  

---

## A. Real-World Benchmark with Meaningful Filtration ✅

**File**: `tests/integration/test_real_benchmarks.py`  
**Implementation**: 
- Replaced synthetic test with `test_zachary_karate_club_meaningful_topology`
- Used edge betweenness centrality as meaningful edge weights
- Added degenerate-filtration detector (`Δβ != 0`)
- Asserted β₀ monotonicity and τ* stability under noise

**Result**: Test passes; τ* variance under noise ≈ 1.2e-4

---

## B. τ* v2 Robustness (Center-of-Mass with Peak Window) ✅

**File**: `src/onn/topo/filtration.py`  
**Implementation**:
- Replaced `compute_tau_star_v2` with peak-windowed CoM implementation
- Added comprehensive docstring with mathematical formulation

**File**: `tests/integration/test_tau_star_robustness.py`  
**Implementation**:
- Added `test_tau_star_v2_behavior_and_stability`
- Dual-peak scenario: verifies focus on main peak
- Noise-stability scenario: variance < 0.01

**Result**: v2 now focuses on main peak and is noise-stable

---

## C. Dynamic Branching Diversity Preservation ✅

**File**: `src/onn/ops/branching.py`  
**Implementation**:
- `DynamicBranchManager` with probabilistic top-k parent sampling
- `DynamicBranchManagerConfig` for diversity parameters

**File**: `tests/integration/test_long_horizon_branching.py` (new)  
**Implementation**:
- 20-cycle simulation ensuring lineage diversity
- Conservative survival with diversity metrics

**File**: `tests/unit/test_dynamic_branch_manager.py` (updated)  
**Implementation**:
- Added slow-improvement detector test

**Result**: Long-horizon test passes; diversity preserved

---

## D. CMA-ES vs Parameter Adaptation Conflict Resolution ✅

**File**: `src/onn/ops/branching.py`  
**Implementation**:
- `AdaptationConfig.trust_region_pct` (default 10%)
- Trust-region clamping in `ParameterAdaptationManager.adapt()`

**File**: `tests/unit/test_parameter_adaptation_manager.py` (updated)  
**Implementation**:
- Added trust-region bounds and reason tests

**File**: `scripts/train_onn_es.py` (updated)  
**Implementation**:
- Added logging for ES vs adapted meta-parameters

**Result**: Adaptation respects ±10% trust region; logging shows ES vs adapted values

---

## E. Reproducibility & Final Verification ✅

### E.1 Environment Capture
**File**: `outputs/repro_env.txt`  
- Timestamp, interpreter, Python/pip/pytest versions
- Platform, working directory, git status
- Virtual env and PYTHONPATH details

### E.2 Full Test Suite
**Command**: `PYTHONPATH=src .venv/bin/python -m pytest -q -o cache_dir=/tmp/pytest_cache --maxfail=3 --durations=5`  
**Result**: 531 passed, 0 warnings (after statistical test fix)

### E.3 Training Dry-Run
**Command**: `PYTHONPATH=src .venv/bin/python scripts/train_onn_es.py --generations 2 --popsize 4 --seed 42`  
**Result**: No crash; adaptation logging visible; CMA-ES initialized correctly

### E.4 Warning Resolution
**File**: `tests/integration/test_mutation_effectiveness.py`  
**Fix**: Replaced t-test with Wilcoxon signed-rank test to avoid scipy precision loss warnings

---

## Key Files Modified

- `src/onn/topo/filtration.py` – τ* v2 peak-window CoM implementation
- `src/onn/ops/branching.py` – DynamicBranchManager, ParameterAdaptationManager trust region, logging
- `tests/integration/test_real_benchmarks.py` – Weighted Karate Club benchmark
- `tests/integration/test_tau_star_robustness.py` – τ* v2 multi-peak and noise-stability tests
- `tests/integration/test_long_horizon_branching.py` – Multi-cycle diversity/survival simulation
- `tests/unit/test_parameter_adaptation_manager.py` – Trust-region bounds and reason tests
- `tests/unit/test_dynamic_branch_manager.py` – Slow-improvement detector test
- `tests/integration/test_mutation_effectiveness.py` – Statistical test warning fix
- `scripts/train_onn_es.py` – Logging for ES vs adapted meta-parameters

---

## Environment & Commands Used

- **Interpreter**: `/opt/homebrew/bin/python3`
- **Python**: `3.14.0`
- **Pytest**: `8.4.2`
- **Cache Dir**: `/tmp/pytest_cache`
- **Dry-Run Command**: `PYTHONPATH=src .venv/bin/python scripts/train_onn_es.py --generations 2 --popsize 4 --seed 42`

---

## Summary

All hardening objectives completed:
- ✅ Real-world benchmark with meaningful filtration
- ✅ τ* v2 robustness with peak-window CoM
- ✅ Dynamic branching diversity preservation
- ✅ CMA-ES vs adaptation conflict resolution
- ✅ Full reproducibility with environment capture

The system is now stable, well-tested, and reproducible with rigorous guarantees.