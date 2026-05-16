"""Compare old vs new prompt-write paths under simulated Hub latency.

OLD: ``Dataset.from_pandas(df).push_to_hub(...)`` per save, no cache, no
concurrency control (last writer wins → clobbering under contention).
NEW: ``commit_prompts_with_cas`` with 30 s read cache + slot reservations.

Hub I/O is stubbed with ``time.sleep`` based on realistic round-trip
times measured against real HF datasets. The point isn't precision —
it's the ratio between the two paths, which holds across plausible
real-world latencies.

Run from the repo root::

    python -m tests.bench_compare    # recommended
    python tests/bench_compare.py    # also works (sys.path tweak below)

Not collected by ``python -m unittest discover tests`` because it has no
``TestCase`` subclasses — it's a dev tool, not a unit test.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

# Make the repo root importable when this file is run as
# ``python tests/bench_compare.py``: the script's own dir (tests/) is on
# sys.path by default, but ``data`` / ``app`` live one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("hackathon").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Simulated Hub latencies (ms) — order-of-magnitude estimates of real HF.
# ---------------------------------------------------------------------------

HUB_READ_MS = 400          # load_dataset for a few-thousand-row table
HUB_PUSH_TO_HUB_MS = 1800  # Dataset.push_to_hub: parquet + README + commit
HUB_CAS_COMMIT_MS = 600    # api.create_commit with one parquet upload
HUB_DATASET_INFO_MS = 80   # api.dataset_info call (just to get HEAD sha)

# ---------------------------------------------------------------------------
# Shared in-memory "Hub" used by both implementations under test.
# Mutable through a lock so concurrent writers don't race during the bench.
# ---------------------------------------------------------------------------

STATE_LOCK = threading.Lock()
STATE = {
    "prompts": None,   # pd.DataFrame
    "sha": "sha-0",    # incrementing pretend sha
    "sha_counter": 0,
}


def _bump_sha() -> str:
    STATE["sha_counter"] += 1
    STATE["sha"] = f"sha-{STATE['sha_counter']}"
    return STATE["sha"]


def _empty_val():
    return {"choice": "", "username": ""}


def _empty_vote():
    return {"choice": "", "username": ""}


def seed_state(n_prompts: int = 2000, country: str = "es") -> None:
    """Seed STATE with ``n_prompts`` empty-slot rows authored by 'v0'."""
    rows = []
    for i in range(n_prompts):
        rows.append({
            "id": i + 1,
            "username": "v0",
            "language": "es",
            "country": country,
            "system_prompt": "",
            "prompt": f"prompt #{i + 1}",
            "prompt_validation_1": _empty_val(),
            "prompt_validation_2": _empty_val(),
            "prompt_validation_3": _empty_val(),
            "answer_a": "ans A",
            "model_a": "m_a",
            "answer_b": "ans B",
            "model_b": "m_b",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        })
    STATE["prompts"] = pd.DataFrame(rows)
    STATE["sha_counter"] = 0
    STATE["sha"] = "sha-0"


def reset_state(n_prompts: int = 2000, country: str = "es") -> None:
    """Reset STATE and the (process-global) reservation table + cache."""
    seed_state(n_prompts, country)
    import data
    with data._reservations_lock:
        data._reservations.clear()
    with data._cache_lock:
        data._cache.clear()


# ---------------------------------------------------------------------------
# Stub HF I/O — installed on the ``data`` module before any handler runs.
# ---------------------------------------------------------------------------

def stub_load_dataset(*_args, **_kwargs):
    time.sleep(HUB_READ_MS / 1000)
    with STATE_LOCK:
        df = STATE["prompts"].copy()
    return SimpleNamespace(to_pandas=lambda: df)


def stub_dataset_info(*_args, **_kwargs):
    time.sleep(HUB_DATASET_INFO_MS / 1000)
    with STATE_LOCK:
        sha = STATE["sha"]
    return SimpleNamespace(sha=sha)


def stub_push_to_hub_via_dataset(df, _features, *_args, **_kwargs):
    """Emulates ``Dataset.from_pandas(df).push_to_hub(...)`` — the OLD path."""
    time.sleep(HUB_PUSH_TO_HUB_MS / 1000)
    with STATE_LOCK:
        STATE["prompts"] = df.copy()
        _bump_sha()


# CAS stub: emulates api.create_commit. Respects parent_commit (raises 412
# if STATE's sha has moved on).
class FakeHfHubHTTPError(Exception):
    def __init__(self, status: int):
        super().__init__(f"{status} fake error")
        self.response = SimpleNamespace(status_code=status)


def stub_create_commit(repo_id, operations, parent_commit, **_kwargs):
    time.sleep(HUB_CAS_COMMIT_MS / 1000)
    with STATE_LOCK:
        if parent_commit != STATE["sha"]:
            raise FakeHfHubHTTPError(412)
        # Read parquet back from the in-flight operation.
        op = operations[0]
        buf = op.path_or_fileobj
        buf.seek(0)
        import io
        if isinstance(buf, io.BytesIO):
            import pyarrow.parquet as pq
            new_df = pq.read_table(buf).to_pandas()
            STATE["prompts"] = new_df
        _bump_sha()


def stub_list_repo_files(*_args, **_kwargs):
    return ["data/train-00000-of-00001.parquet"]


def install_stubs() -> None:
    """Monkey-patch ``data`` module's Hub calls. Called once per bench run."""
    import data
    from huggingface_hub.utils import HfHubHTTPError as RealHfHubHTTPError

    data.load_dataset = stub_load_dataset
    # Patch the api object's methods via a fake HfApi factory.
    fake_api = SimpleNamespace(
        dataset_info=stub_dataset_info,
        create_commit=stub_create_commit,
        list_repo_files=stub_list_repo_files,
    )
    data._hf_api = lambda: fake_api
    # Translate FakeHfHubHTTPError into the real exception type the
    # CAS layer catches.
    real_create_commit = stub_create_commit

    def translating_create_commit(*args, **kwargs):
        try:
            real_create_commit(*args, **kwargs)
        except FakeHfHubHTTPError as e:
            real_exc = RealHfHubHTTPError("412 conflict")
            real_exc.response = e.response
            raise real_exc

    fake_api.create_commit = translating_create_commit


# ---------------------------------------------------------------------------
# OLD implementation — inlined from HEAD~2 to bench against current code.
# ---------------------------------------------------------------------------

def old_load_prompts_df() -> pd.DataFrame:
    """OLD: full Hub read per call, no cache."""
    return stub_load_dataset().to_pandas()


def old_push_prompts_df(df: pd.DataFrame) -> None:
    """OLD: Dataset.from_pandas(df).push_to_hub() — full ceremony per save."""
    stub_push_to_hub_via_dataset(df, None)


def old_save_validation(username: str, country: str) -> bool:
    """Pick a slot and write it the OLD way. Returns True if it 'landed'.

    Mirrors HEAD~2 save_validation: load → mutate → push. No CAS, no
    reservations. Under concurrency the last writer wins — earlier
    writers' mutations are silently clobbered (the "True" we return is
    misleading because by the time we check, someone else's push may
    have overwritten ours)."""
    df = old_load_prompts_df()
    for idx in df.index:
        row = df.loc[idx]
        if row["country"] != country or row["username"] == username:
            continue
        if any(row[f"prompt_validation_{i}"]["username"] == username
               for i in (1, 2, 3)):
            continue
        for i in (1, 2, 3):
            if not row[f"prompt_validation_{i}"]["username"]:
                df.at[idx, f"prompt_validation_{i}"] = {
                    "choice": "knowledge", "username": username,
                }
                old_push_prompts_df(df)
                # OLD fetch-next would do another load_prompts_df here.
                _ = old_load_prompts_df()
                return True
    return False


def old_save_prompt(username: str, country: str) -> int:
    """Append a new prompt the OLD way: load → concat → push. Returns new id."""
    df = old_load_prompts_df()
    next_id = int(df["id"].max()) + 1 if len(df) else 1
    new_row = {
        "id": next_id,
        "username": username, "language": "es", "country": country,
        "system_prompt": "", "prompt": f"new prompt from {username}",
        "prompt_validation_1": _empty_val(),
        "prompt_validation_2": _empty_val(),
        "prompt_validation_3": _empty_val(),
        "answer_a": "", "model_a": "", "answer_b": "", "model_b": "",
        "answer_chosen_1": _empty_vote(),
        "answer_chosen_2": _empty_vote(),
        "answer_chosen_3": _empty_vote(),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    old_push_prompts_df(df)
    return next_id


# ---------------------------------------------------------------------------
# NEW implementation — current app.py handlers, against the stubbed Hub.
# ---------------------------------------------------------------------------

def new_save_validation(username: str, country: str) -> bool:
    """Call current app.save_validation via picker → save → auto-advance.

    Catches the RuntimeError that ``commit_with_cas`` raises after
    exhausting its retry budget, so a stress-test scenario where many
    users hit the Hub at the same instant returns False (failed save)
    rather than crashing the whole worker."""
    import app
    profile = SimpleNamespace(username=username)
    try:
        idx, slot, _prompt, _status = app.fetch_next_validation("es", profile)
        if idx < 0:
            return False
        out = app.save_validation(idx, slot, None, "knowledge", "es", profile)
        return out[6] == app._t("es")["validation_saved"]
    except RuntimeError:
        return False


def new_save_prompt(username: str, country: str) -> bool:
    import app
    profile = SimpleNamespace(username=username)
    try:
        out = app.save_prompt("", f"new prompt from {username}", "es", profile)
    except RuntimeError:
        return False
    return "guardado" in out[0].lower() or "saved" in out[0].lower()


# Patch participant_info so new handlers think every bench user is registered.
def install_app_stubs() -> None:
    import app
    import data
    def fake_info(username, df=None):
        return {"username": username, "language": "es",
                "country": "es", "gmail": ""}
    app.participant_info = fake_info
    data.participant_info = fake_info


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.samples: list[float] = []

    def record(self, dt: float):
        self.samples.append(dt)

    def report(self, label: str) -> str:
        if not self.samples:
            return f"{label}: no samples"
        n = len(self.samples)
        s = sorted(self.samples)
        mean = sum(s) / n
        p50 = s[n // 2]
        p95 = s[min(n - 1, int(n * 0.95))]
        return (f"{label:35s} n={n:3d}  "
                f"mean={mean*1000:6.0f}ms  "
                f"p50={p50*1000:6.0f}ms  "
                f"p95={p95*1000:6.0f}ms  "
                f"total={sum(s):.2f}s")


def time_call(fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    return result, time.monotonic() - t0


def scenario_serial(name: str, save_fn, n: int = 10):
    print(f"\n--- {name}: 1 user × {n} sequential saves ---")
    reset_state(n_prompts=500)
    install_stubs()
    install_app_stubs()
    timer = Timer()
    landed = 0
    t_total = time.monotonic()
    for _ in range(n):
        ok, dt = time_call(save_fn, "alice", "es")
        if ok:
            landed += 1
        timer.record(dt)
    wall = time.monotonic() - t_total
    print(timer.report("per-save latency"))
    print(f"  total wall: {wall:.2f}s,  landed: {landed}/{n}")
    return wall, landed


def scenario_concurrent(name: str, save_fn, users: int = 10, per_user: int = 5,
                        stagger_ms: int = 100):
    """``stagger_ms`` between consecutive worker starts approximates the
    real-world fact that even simultaneous "Save" clicks land at the
    server at least one network-round-trip apart — without it, the bench
    creates a perfect thundering herd that's harder to reach in practice."""
    import random as _random
    print(f"\n--- {name}: {users} concurrent users × {per_user} saves each"
          f" (stagger={stagger_ms}ms) ---")
    reset_state(n_prompts=500)
    install_stubs()
    install_app_stubs()
    timer = Timer()

    def worker(user: str, start_delay: float):
        time.sleep(start_delay)
        local = []
        for _ in range(per_user):
            ok, dt = time_call(save_fn, user, "es")
            local.append((ok, dt))
            # Tiny gap between saves from the same user — a human
            # validating wouldn't click Save in a tight loop.
            time.sleep(_random.uniform(0.05, 0.2))
        return local

    t_total = time.monotonic()
    landed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=users) as ex:
        futures = [
            ex.submit(worker, f"u{i:02d}", i * stagger_ms / 1000)
            for i in range(users)
        ]
        for f in as_completed(futures):
            for ok, dt in f.result():
                timer.record(dt)
                if ok:
                    landed += 1
                else:
                    failed += 1
    wall = time.monotonic() - t_total
    actual_writes = 0
    df = STATE["prompts"]
    for _, row in df.iterrows():
        for i in (1, 2, 3):
            if str(row[f"prompt_validation_{i}"]["username"]).startswith("u"):
                actual_writes += 1
    print(timer.report("per-save latency"))
    print(f"  total wall: {wall:.2f}s")
    print(f"  reported landed: {landed},  failed: {failed},  "
          f"actual writes in dataset: {actual_writes}")
    if landed > actual_writes:
        print(f"  → {landed - actual_writes} writes CLOBBERED (silent data loss)")
    return wall, landed, actual_writes, failed


def scenario_writing_concurrent(name: str, save_fn, users: int = 10, per_user: int = 3):
    print(f"\n--- {name}: {users} concurrent users writing × {per_user} prompts each (Writing tab) ---")
    reset_state(n_prompts=500)
    install_stubs()
    install_app_stubs()
    timer = Timer()

    def worker(user: str):
        local = []
        for _ in range(per_user):
            ok, dt = time_call(save_fn, user, "es")
            local.append((ok, dt))
        return local

    t_total = time.monotonic()
    with ThreadPoolExecutor(max_workers=users) as ex:
        futures = [ex.submit(worker, f"w{i:02d}") for i in range(users)]
        for f in as_completed(futures):
            for ok, dt in f.result():
                timer.record(dt)
    wall = time.monotonic() - t_total
    df = STATE["prompts"]
    seeded = 500
    actual_new = len(df) - seeded
    ids = df["id"].tolist()
    unique_ids = len(set(ids))
    print(timer.report("per-save latency"))
    print(f"  total wall: {wall:.2f}s")
    print(f"  expected new rows: {users * per_user},  actual new rows: {actual_new}")
    print(f"  unique ids: {unique_ids}/{len(df)} "
          f"({'OK' if unique_ids == len(df) else 'COLLISION'})")
    return wall, actual_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Simulated Hub latencies: read={HUB_READ_MS}ms  "
          f"push_to_hub={HUB_PUSH_TO_HUB_MS}ms  "
          f"create_commit={HUB_CAS_COMMIT_MS}ms  "
          f"dataset_info={HUB_DATASET_INFO_MS}ms")

    print("\n" + "=" * 72)
    print("  VALIDATION SAVES")
    print("=" * 72)
    old_w_serial, old_l_serial = scenario_serial("OLD", old_save_validation, n=10)
    new_w_serial, new_l_serial = scenario_serial("NEW", new_save_validation, n=10)

    old_w_conc, old_l_conc, old_a_conc, _ = scenario_concurrent(
        "OLD", old_save_validation, users=10, per_user=5,
    )
    new_w_conc, new_l_conc, new_a_conc, new_f_conc = scenario_concurrent(
        "NEW", new_save_validation, users=10, per_user=5,
    )

    print("\n" + "=" * 72)
    print("  PROMPT WRITES (Writing tab)")
    print("=" * 72)
    old_w_write, _ = scenario_writing_concurrent(
        "OLD", old_save_prompt, users=10, per_user=3,
    )
    new_w_write, _ = scenario_writing_concurrent(
        "NEW", new_save_prompt, users=10, per_user=3,
    )

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    def pct(old, new):
        if old == 0:
            return "n/a"
        return f"{(1 - new / old) * 100:+.0f}%"

    print(f"  Serial validation (10 saves):       "
          f"OLD {old_w_serial:5.1f}s → NEW {new_w_serial:5.1f}s  "
          f"({pct(old_w_serial, new_w_serial)})")
    print(f"  Concurrent validation (50 saves):   "
          f"OLD {old_w_conc:5.1f}s → NEW {new_w_conc:5.1f}s  "
          f"({pct(old_w_conc, new_w_conc)})")
    print(f"  Concurrent writes (30 prompts):     "
          f"OLD {old_w_write:5.1f}s → NEW {new_w_write:5.1f}s  "
          f"({pct(old_w_write, new_w_write)})")
    print(f"  Writes lost to clobbering (OLD):    "
          f"{old_l_conc - old_a_conc} of {old_l_conc} "
          f"({(old_l_conc - old_a_conc) / max(1, old_l_conc) * 100:.0f}%)")
    print(f"  Writes lost to clobbering (NEW):    "
          f"{new_l_conc - new_a_conc} of {new_l_conc}")
    print(f"  Saves that 412-exhausted retries (NEW): {new_f_conc}")
