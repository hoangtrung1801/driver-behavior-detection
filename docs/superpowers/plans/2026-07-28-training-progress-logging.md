# Training Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make C-DIRA and baseline training visibly report batch and epoch progress in the terminal.

**Architecture:** Extend the existing `Trainer` loop with a small progress-output helper. The helper receives the phase, epoch, batch index, batch total, and elapsed time, and prints flushed status lines without changing optimization, metrics, or checkpoint behavior. Tests capture stdout around a deterministic one-epoch fit and assert both periodic batch and epoch messages.

**Tech Stack:** Python 3.12, PyTorch, pytest.

## Global Constraints

- Preserve the existing training calculations and public `Trainer` API.
- Keep output terminal-friendly and flush every line immediately.
- Use no new runtime dependencies.
- Maintain Apple Silicon MPS compatibility.

---

### Task 1: Add failing progress-output coverage

**Files:**
- Modify: `tests/test_training_engine.py`
- Modify: `src/cdira/training/engine.py`

**Interfaces:**
- Consumes: existing `Trainer.fit(train_loader, validation_loader)`.
- Produces: flushed output containing periodic `train`/`validation` batch messages and one epoch summary.

- [x] **Step 1: Write the failing test**

Add a deterministic test that builds tiny tensor loaders, calls `Trainer.fit` for one epoch, captures stdout, and asserts the output contains `epoch 1/1`, `train`, `validation`, `train_loss=`, and `val_loss=`.

- [x] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training_engine.py -q`

Expected: FAIL because `Trainer.fit` currently emits no progress lines.

- [x] **Step 3: Implement minimal progress output**

Update `Trainer._run_epoch` to print progress every 50 batches and on the final batch, including phase, completed/total batches, and loss. Update `Trainer.fit` to print one flushed epoch summary with elapsed seconds after validation.

- [x] **Step 4: Run the focused test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training.py -q`

Expected: PASS.

- [x] **Step 5: Run full verification**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src/cdira
```

Expected: all tests pass, Ruff is clean, and mypy reports no issues.

- [ ] **Step 6: Commit**

Commit is pending because this environment denies writes to `.git/index.lock`.

```bash
git add docs/superpowers/plans/2026-07-28-training-progress-logging.md tests/test_training_engine.py src/cdira/training/engine.py
git commit -m "feat: log training progress"
```
