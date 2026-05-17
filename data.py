"""Hugging Face dataset I/O for the hackathon Space.

Both the participants and the cultural-preferences datasets are private; the
calling environment must expose ``HF_TOKEN``. On a Hugging Face Space this is
configured as a secret. Locally, drop a ``.env`` file at the repo root with
``HF_TOKEN=hf_...`` and ``python-dotenv`` will load it on import.

Writes to both datasets use optimistic concurrency (compare-and-swap): each
commit carries the parent SHA the writer was working from. The Hub rejects a
commit whose parent doesn't match the current head with HTTP 412, which we
translate to ``CommitConflictError``. ``commit_with_cas`` turns that into a
refresh-and-retry loop and re-runs the caller's mutator against the refreshed
state — so a save handler can re-pick a slot if its original target was
claimed by someone else in the meantime.

Reads go through a short-TTL in-process cache (``_CACHE_TTL_SECONDS``). A
successful CAS commit updates the cache with the new ``(sha, df)`` so the
next read is instant. Cache misses pin the read to ``dataset_info().sha``
so the SHA we use for ``parent_commit`` matches the rows we just read.
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import threading
import time
from typing import Callable, Optional, Tuple

import pandas as pd
from datasets import Dataset, Features, Value, load_dataset
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

PARTICIPANTS_REPO = "mariagrandury/hackathon_participants"
PROMPTS_REPO = "mariagrandury/cultural_preferences"

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

log = logging.getLogger("hackathon")
# Make ``log.info`` messages actually appear in the terminal — without
# this, the cache-miss / commit-conflict / latency logs go to the void
# because Python's root logger defaults to WARNING. Override via
# ``HACKATHON_LOG_LEVEL=WARNING python app.py`` if it gets too chatty.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("HACKATHON_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

# In-process cache for HF dataset reads. Each entry is
# ``(loaded_at_monotonic, sha, df)``; the SHA is what CAS writers use as
# ``parent_commit``. A successful commit overwrites the entry with the new
# (sha, df) so the next reader is instant; on a 412 conflict the entry is
# invalidated so the retry loop re-fetches.
#
# 5 min is long enough to absorb the typical "read, think, save" cycle
# without re-paying the Hub round-trip (and CAS detects any actual staleness
# at commit time via parent_commit). Shorter TTLs (e.g. 30 s) were tripping
# on every save_prompt because participant lookups during the click happen
# >30 s after init_ui's first read.
_CACHE_TTL_SECONDS = 300.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Optional[str], pd.DataFrame]] = {}


class CommitConflictError(Exception):
    """The push's ``parent_commit`` no longer matches the server's head.

    Raised by the inner ``_commit_*`` wrappers when the Hub returns HTTP
    412, and caught by ``commit_with_cas`` to drive its retry loop. The
    original ``HfHubHTTPError`` is kept as ``__cause__`` so callers can
    chain it through if they want to surface the underlying status."""


# Slot reservation table. Optimization on top of CAS: the picker reserves
# the (row, slot) it hands out so the next concurrent picker walks past it,
# avoiding the round-trip + 412 + retry that the CAS layer would otherwise
# absorb. Reservations expire by TTL so a user who navigates away without
# saving doesn't permanently block the slot.
#
# Local to this process — on a multi-replica deployment this would need to
# move to Redis or similar. HF Spaces are single-process by default, so it's
# fine here.
_RESERVATION_TTL_SECONDS = 300.0
_reservations_lock = threading.Lock()
# Key is ``(row_idx, slot, kind)`` where kind is ``"validation"`` or
# ``"vote"``; value is ``(username, expiry_monotonic)``.
_reservations: dict[tuple[int, int, str], tuple[str, float]] = {}


def reserve_slot(idx: int, slot: int, user: str, kind: str) -> bool:
    """Atomically claim ``(idx, slot, kind)`` for ``user``.

    Returns True if the slot is now reserved by ``user`` — either it was
    free, expired, or already owned by ``user``. Returns False if someone
    else still holds a non-expired reservation."""
    now = time.monotonic()
    key = (idx, slot, kind)
    with _reservations_lock:
        existing = _reservations.get(key)
        if existing is not None:
            holder, expiry = existing
            if holder != user and now < expiry:
                return False
        _reservations[key] = (user, now + _RESERVATION_TTL_SECONDS)
        return True


def release_slot(idx: int, slot: int, user: str, kind: str) -> None:
    """Release the reservation iff it's currently owned by ``user``."""
    key = (idx, slot, kind)
    with _reservations_lock:
        existing = _reservations.get(key)
        if existing is not None and existing[0] == user:
            _reservations.pop(key, None)


def is_slot_reserved_by_other(idx: int, slot: int, user: str, kind: str) -> bool:
    """Picker helper: True if another user holds a fresh reservation."""
    now = time.monotonic()
    key = (idx, slot, kind)
    with _reservations_lock:
        existing = _reservations.get(key)
        if existing is None:
            return False
        holder, expiry = existing
        return holder != user and now < expiry


def _hf_api() -> HfApi:
    return HfApi(token=HF_TOKEN)


def _cache_get(key: str) -> Optional[tuple[Optional[str], pd.DataFrame]]:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        loaded_at, sha, df = hit
        if (time.monotonic() - loaded_at) >= _CACHE_TTL_SECONDS:
            return None
        return sha, df


def _cache_put(key: str, sha: Optional[str], df: pd.DataFrame) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), sha, df)


def _cache_invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)


EMPTY_VALIDATION = {"choice": "", "username": ""}
EMPTY_VOTE = {"choice": "", "username": ""}
# JSON-encoded ``{attempt_number_str: score_float}``. String column because
# ``datasets.Features`` has no open-ended dict type, and the dict is small
# enough that JSON encoding is fine.
EMPTY_TEST_SCORE = "{}"
# JSON-encoded ``{attempt_number_str: {question_id: chosen_answer}}``.
# Keyed by the same attempt numbers as ``test_score`` so the two columns
# can be cross-indexed for per-attempt analysis (mirroring the 2025
# response analysis under data/analysis_test_2025.md).
EMPTY_TEST_RESPONSES = "{}"

# Pass mark for the entry test, in RAW POINTS. The current grading scheme
# (see ``test_data.grade``) tops out at 14 classification points + 2 MCQ
# points = 16; ``12`` means "need at least 12 raw points to unlock the
# action tabs". Stored test scores are raw points (the same float the user
# sees in the "X / 16" status message), so this is directly comparable
# with ``best_test_score``.
TEST_PASS_THRESHOLD = 12.0

# Canonical validation bucket ordering, shared across the app and the entry
# test. Lives here (not in app.py) so non-Gradio modules like ``test_data``
# can import it without dragging the UI layer in.
#   - REJECT_CHOICES: three reject buckets.
#   - ACCEPT_CHOICES: the four AlKhamissi et al. (2025) cultural dimensions.
#   - VALIDATION_CHOICES: REJECT + ACCEPT, the full radio order.
#   - ACCEPT_VALIDATION_CHOICES: set form of ACCEPT_CHOICES, used by row
#     predicates (``is_fully_validated`` etc.) where set membership matters.
REJECT_CHOICES = ("trivial", "stereotype", "unrelated")
ACCEPT_CHOICES = ("knowledge", "preference", "dynamics", "bias_probe")
VALIDATION_CHOICES = REJECT_CHOICES + ACCEPT_CHOICES
ACCEPT_VALIDATION_CHOICES = frozenset(ACCEPT_CHOICES)

# Usernames that aren't real participants and must be hidden from user-facing
# aggregations (leaderboard, known-users list). ``"v0"`` is the author
# sentinel used by ``import_dpo_pairs.py`` for the ~2k prompts migrated from
# the 2025 dataset — without this filter it would sit at the top of the
# ranking with thousands of "prompts sent".
EXCLUDED_USERNAMES = frozenset({"v0"})

VALIDATION_STRUCT = {"choice": Value("string"), "username": Value("string")}
VOTE_STRUCT = {"choice": Value("string"), "username": Value("string")}

PARTICIPANTS_FEATURES = Features(
    {
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        # Email is intentionally NOT pushed to the Hub. The importer
        # collects it locally (so the missing-HF report can name attendees)
        # and inspect_hf_dataset joins demographics on ``username`` instead.
        "test_score": Value("string"),
        "test_responses": Value("string"),
    }
)

PROMPTS_FEATURES = Features(
    {
        # Stable, human-readable identifier (1-indexed, sequential). Used in
        # the commit message of every save so the HF dataset's commit
        # history doubles as an audit log.
        "id": Value("int64"),
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        # Optional LLM-steering preamble shown to annotators alongside the
        # prompt. Empty string when the prompt is self-contained.
        "system_prompt": Value("string"),
        "prompt": Value("string"),
        "prompt_validation_1": VALIDATION_STRUCT,
        "prompt_validation_2": VALIDATION_STRUCT,
        "prompt_validation_3": VALIDATION_STRUCT,
        "answer_a": Value("string"),
        "model_a": Value("string"),
        "answer_b": Value("string"),
        "model_b": Value("string"),
        "answer_chosen_1": VOTE_STRUCT,
        "answer_chosen_2": VOTE_STRUCT,
        "answer_chosen_3": VOTE_STRUCT,
    }
)

# 2-letter ISO → display name for the country shown to annotators (in the
# in-progress status messages). Falls back to the uppercase code if a
# country isn't in the map.
COUNTRY_DISPLAY_NAMES: dict[str, str] = {
    "es": "España",
    "cu": "Cuba",
    "co": "Colombia",
    "py": "Paraguay",
    "ec": "Ecuador",
    "cl": "Chile",
    "pe": "Perú",
    "mx": "México",
    "ni": "Nicaragua",
    "br": "Brasil",
    "pt": "Portugal",
    "us": "USA",
    "ar": "Argentina",
    "uy": "Uruguay",
    "ve": "Venezuela",
    "bo": "Bolivia",
    "cr": "Costa Rica",
    "do": "República Dominicana",
    "gt": "Guatemala",
    "hn": "Honduras",
    "pa": "Panamá",
    "pr": "Puerto Rico",
    "sv": "El Salvador",
}


def country_display(code: str | None) -> str:
    if not code:
        return "?"
    return COUNTRY_DISPLAY_NAMES.get(code, code.upper())


def _fetch_dataset(repo_id: str, sha: str | None = None) -> pd.DataFrame:
    """Read ``repo_id`` from the Hub, optionally pinned to ``sha``.

    ``verification_mode="no_checks"`` skips the split-size verification
    against the dataset card's metadata: CAS commits rewrite only the
    parquet shard, not ``README.md``, so the card's recorded
    ``num_examples`` goes stale after the first save. The parquet at the
    pinned ``revision=sha`` is authoritative; file-level integrity is
    still checked by ``hf_hub_download`` and schema by pyarrow."""
    kwargs: dict = {
        "split": "train",
        "token": HF_TOKEN,
        "verification_mode": "no_checks",
    }
    if sha is not None:
        kwargs["revision"] = sha
    return load_dataset(repo_id, **kwargs).to_pandas()


def _fetch_with_sha(repo_id: str) -> tuple[str, pd.DataFrame]:
    """Read ``repo_id`` pinned to its current HEAD SHA.

    The SHA is what CAS writers use as ``parent_commit``. Pinning the read
    to the SHA we just looked up closes a tiny window where ``load_dataset``
    might otherwise resolve to a different revision than ``dataset_info``
    reported a moment later."""
    info = _hf_api().dataset_info(repo_id=repo_id, token=HF_TOKEN)
    sha = info.sha
    df = _fetch_dataset(repo_id, sha=sha)
    return sha, df


def _cached_load(
    repo_id: str, *, fresh: bool = False
) -> tuple[Optional[str], pd.DataFrame]:
    """Return ``(sha, df)`` for ``repo_id`` from the in-process cache, or
    refetch on miss / when ``fresh=True``. ``sha`` is the revision the df
    was read at — pass it back as ``parent_commit`` for CAS writes."""
    if not fresh:
        hit = _cache_get(repo_id)
        if hit is not None:
            return hit
    t0 = time.monotonic()
    sha, df = _fetch_with_sha(repo_id)
    log.info(
        "cache miss: loaded %s in %.2fs (rows=%d, sha=%s)",
        repo_id,
        time.monotonic() - t0,
        len(df),
        (sha[:8] if sha else "n/a"),
    )
    _cache_put(repo_id, sha, df)
    return sha, df


def load_participants_df() -> pd.DataFrame:
    _, df = _cached_load(PARTICIPANTS_REPO)
    return df


def load_prompts_df() -> pd.DataFrame:
    _, df = _cached_load(PROMPTS_REPO)
    return df


def load_prompts_with_sha(fresh: bool = False) -> tuple[str, pd.DataFrame]:
    """Read path for CAS write handlers — returns ``(sha, df)`` so the
    caller can hand ``sha`` back as ``parent_commit`` when it pushes."""
    sha, df = _cached_load(PROMPTS_REPO, fresh=fresh)
    assert sha is not None, "prompts cache entry must carry a SHA"
    return sha, df


def load_participants_with_sha(fresh: bool = False) -> tuple[str, pd.DataFrame]:
    sha, df = _cached_load(PARTICIPANTS_REPO, fresh=fresh)
    assert sha is not None, "participants cache entry must carry a SHA"
    return sha, df


def participant_info(username: str, df: pd.DataFrame | None = None) -> Optional[dict]:
    if df is None:
        df = load_participants_df()
    matches = df[df["username"] == username]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def parse_test_score(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) else {}


def parse_test_responses(raw: str | None) -> dict[str, dict[str, str]]:
    """Mirror of ``parse_test_score`` for the ``test_responses`` column.

    Returns ``{attempt_number_str: {question_id: chosen_answer}}``. An
    unparseable / missing cell decays to an empty dict — same defensive
    contract as ``parse_test_score``."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(attempt): {str(qid): str(ans) for qid, ans in answers.items()}
        for attempt, answers in data.items()
        if isinstance(answers, dict)
    }


def best_test_score(username: str | None, df: pd.DataFrame | None = None) -> float:
    """Return the user's best attempt score in raw points (matches the
    ``X / 16`` numbers shown in the UI), or 0.0 if no attempts."""
    if not username:
        return 0.0
    if df is None:
        df = load_participants_df()
    matches = df[df["username"] == username]
    if matches.empty:
        return 0.0
    scores = parse_test_score(matches.iloc[0].get("test_score"))
    return max(scores.values()) if scores else 0.0


# How many times to retry a CAS commit when another writer landed in
# between. Hackathon contention is bounded by Hub commit throughput
# (~hundreds of ms each), so a handful of retries with jittered exponential
# backoff is enough in practice.
_COMMIT_MAX_RETRIES = 8


def _single_parquet_path(api: HfApi, repo_id: str) -> str:
    """Locate the single Parquet shard of ``repo_id`` on the Hub.

    ``Dataset.push_to_hub`` writes one ``data/train-XXXXX-of-XXXXX.parquet``
    for a small table; we replace that same path on CAS updates so we don't
    leave orphan shards behind. Raises if the dataset has grown past a
    single shard — at that point this helper needs a real shard strategy."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
    parquets = [f for f in files if f.endswith(".parquet")]
    if len(parquets) != 1:
        raise RuntimeError(
            f"expected exactly one parquet shard in {repo_id}, "
            f"got {len(parquets)}: {parquets!r}"
        )
    return parquets[0]


def _participants_parquet_path(api: HfApi) -> str:
    """Back-compat shim — kept for the existing test mocks that patch this
    by name."""
    return _single_parquet_path(api, PARTICIPANTS_REPO)


def _prompts_parquet_path(api: HfApi) -> str:
    return _single_parquet_path(api, PROMPTS_REPO)


def _commit_dataset(
    df: pd.DataFrame,
    *,
    api: HfApi,
    repo_id: str,
    path_in_repo: str,
    features: Features,
    parent_commit: str,
    commit_message: str,
) -> str:
    """Replace the dataset's single Parquet shard at ``path_in_repo`` with
    the encoded ``df``, requiring the repo to still be at ``parent_commit``.

    Returns the new commit SHA so callers (e.g. ``_commit_with_cas``) can
    warm the read cache with the just-committed ``(sha, df)`` pair, sparing
    the next reader a Hub round-trip.

    Writes the parquet to a temp file (not a ``BytesIO``) so HF's Xet
    storage backend can pick it up — Xet does chunk-based uploads that
    skip blocks already present on the server, which is meaningfully
    faster for ~MB parquet shards where most rows are unchanged between
    saves. With ``BytesIO`` we'd silently fall back to a plain HTTP POST
    of the entire 2 MB on every commit (see the warning
    "Uploading files as a binary IO buffer is not supported by Xet
    Storage. Falling back to HTTP upload.").

    Raises ``HfHubHTTPError`` with status 412 if another writer landed first."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Dataset.from_pandas(df, preserve_index=False, features=features).to_parquet(tmp_path)
        info = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=tmp_path)
            ],
            commit_message=commit_message,
            parent_commit=parent_commit,
            token=HF_TOKEN,
        )
        return info.oid
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _commit_participants(
    df: pd.DataFrame,
    *,
    api: HfApi,
    path_in_repo: str,
    parent_commit: str,
    commit_message: str,
) -> str:
    return _commit_dataset(
        df,
        api=api,
        repo_id=PARTICIPANTS_REPO,
        path_in_repo=path_in_repo,
        features=PARTICIPANTS_FEATURES,
        parent_commit=parent_commit,
        commit_message=commit_message,
    )


def _commit_prompts(
    df: pd.DataFrame,
    *,
    api: HfApi,
    path_in_repo: str,
    parent_commit: str,
    commit_message: str,
) -> str:
    return _commit_dataset(
        df,
        api=api,
        repo_id=PROMPTS_REPO,
        path_in_repo=path_in_repo,
        features=PROMPTS_FEATURES,
        parent_commit=parent_commit,
        commit_message=commit_message,
    )


def _commit_with_cas(
    *,
    repo_id: str,
    fetch: Callable[[bool], tuple[str, pd.DataFrame]],
    mutator: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
    commit: Callable[[pd.DataFrame, str], str],
    max_retries: int = _COMMIT_MAX_RETRIES,
) -> Tuple[Optional[bool], int]:
    """Apply ``mutator`` and push via ``commit`` under compare-and-swap.

    Steps each attempt:

      1. ``fetch(fresh)`` returns ``(parent_sha, df)``. ``fresh=True`` on
         retries to bypass the cache and read the new HEAD.
      2. ``mutator(df.copy())`` decides what to do given the current state.
         It returns either:
           * a mutated df → commit it
           * ``None``     → abort (e.g. precondition no longer holds)
      3. ``commit(df, parent_sha)`` uploads and returns the new commit
         SHA. On HTTP 412 (another writer landed since we read), the cache
         is invalidated and the loop re-runs — the mutator gets to
         re-evaluate against the new state.

    On success, the cache is updated to ``(new_sha, mutated_df)``. Because
    the parent_sha was still HEAD when our commit landed, no other writer
    raced in between, so our mutated df IS the new state — the next reader
    can serve it without a Hub round-trip. This is the single biggest
    latency win for the save+load cycle (eliminates one Hub read per click).

    Returns ``(status, attempts_used)``:
      * ``(True, n)``    committed after ``n + 1`` attempts
      * ``(False, n)``   ``mutator`` returned ``None`` on attempt ``n + 1``
      * ``(None, ...)``  raises ``RuntimeError`` after ``max_retries``
        consecutive 412s (chained from the last ``HfHubHTTPError``).
    """
    last_conflict: HfHubHTTPError | None = None
    for retry in range(max_retries):
        parent_sha, df = fetch(retry > 0)
        result = mutator(df.copy())
        if result is None:
            return False, retry
        try:
            new_sha = commit(result, parent_sha)
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 412:
                raise
            log.info(
                "commit conflict on %s: parent=%s no longer matches HEAD",
                repo_id,
                parent_sha[:8],
            )
            _cache_invalidate(repo_id)
            last_conflict = exc
            # Jittered exponential backoff: 50ms · 2^retry · [0.5, 1.5).
            time.sleep(0.05 * (2**retry) * (0.5 + random.random()))
            continue
        # Success — warm the cache with our committed state so the next
        # reader is instant. parent_sha was still HEAD when we committed,
        # so ``result`` IS the canonical new state.
        if new_sha:
            _cache_put(repo_id, new_sha, result)
        else:
            _cache_invalidate(repo_id)
        return True, retry
    raise RuntimeError(
        f"could not commit to {repo_id} after {max_retries} retries"
    ) from last_conflict


def commit_prompts_with_cas(
    mutator: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
    *,
    commit_message: str | Callable[[pd.DataFrame], str] = "update prompts",
    max_retries: int = _COMMIT_MAX_RETRIES,
) -> Tuple[Optional[bool], int]:
    """CAS wrapper for the prompts dataset. See ``_commit_with_cas`` for the
    return-value contract.

    ``commit_message`` can be a callable that takes the mutated df and
    returns the message — useful when the message depends on per-mutation
    state (e.g. the freshly-assigned row id) and would be stale if computed
    outside the retry loop."""
    api = _hf_api()
    path_in_repo = _prompts_parquet_path(api)

    def fetch(fresh: bool) -> tuple[str, pd.DataFrame]:
        return load_prompts_with_sha(fresh=fresh)

    def commit(df: pd.DataFrame, parent_sha: str) -> str:
        msg = commit_message(df) if callable(commit_message) else commit_message
        return _commit_prompts(
            df,
            api=api,
            path_in_repo=path_in_repo,
            parent_commit=parent_sha,
            commit_message=msg,
        )

    return _commit_with_cas(
        repo_id=PROMPTS_REPO,
        fetch=fetch,
        mutator=mutator,
        commit=commit,
        max_retries=max_retries,
    )


def record_test_attempt(
    username: str,
    score: float,
    responses: dict[str, str] | None = None,
) -> int:
    """Append ``score`` (and optionally per-question ``responses``) as the
    next attempt for ``username`` via CAS.

    Writes both ``test_score`` and ``test_responses`` in the same commit
    under the same attempt-number key, so the two columns stay aligned for
    later cross-indexed analysis. ``responses`` is a ``{question_id:
    chosen_answer}`` map; ``None`` (back-compat for older callers) is
    treated as an empty dict.

    Returns the new attempt number (1-indexed). Raises ``LookupError`` if
    the user isn't a registered participant; ``RuntimeError`` if all
    retries kept losing the race (shouldn't happen for hackathon-scale
    contention)."""
    api = _hf_api()
    path_in_repo = _participants_parquet_path(api)
    captured_attempt: list[int] = []
    responses = responses or {}

    def fetch(_fresh: bool) -> tuple[str, pd.DataFrame]:
        # ``record_test_attempt`` writes are rare; bypass the cache entirely
        # so we never operate on a stale snapshot. (Each call reads HEAD
        # directly via ``api.dataset_info`` + ``load_dataset(revision=…)``,
        # matching the pre-refactor behaviour the tests expect.)
        info = api.dataset_info(repo_id=PARTICIPANTS_REPO, token=HF_TOKEN)
        sha = info.sha
        df = load_dataset(
            PARTICIPANTS_REPO,
            split="train",
            token=HF_TOKEN,
            revision=sha,
        ).to_pandas()
        return sha, df

    def mutator(df: pd.DataFrame) -> pd.DataFrame:
        matches = df.index[df["username"] == username].tolist()
        if not matches:
            raise LookupError(f"user {username!r} is not in the participants dataset")
        idx = matches[0]
        scores = parse_test_score(df.at[idx, "test_score"])
        attempt = max((int(k) for k in scores), default=0) + 1
        scores[str(attempt)] = float(score)
        df.at[idx, "test_score"] = json.dumps(scores)
        # Mirror the same write into test_responses. The column may be
        # missing on older datasets that pre-date the schema migration;
        # in that case we initialise it from EMPTY_TEST_RESPONSES so the
        # commit always lands a complete row.
        existing_responses_cell = (
            df.at[idx, "test_responses"]
            if "test_responses" in df.columns
            else EMPTY_TEST_RESPONSES
        )
        all_responses = parse_test_responses(existing_responses_cell)
        all_responses[str(attempt)] = dict(responses)
        if "test_responses" not in df.columns:
            df["test_responses"] = EMPTY_TEST_RESPONSES
        df.at[idx, "test_responses"] = json.dumps(all_responses)
        captured_attempt.append(attempt)
        return df

    def commit(df: pd.DataFrame, parent_sha: str) -> str:
        return _commit_participants(
            df,
            api=api,
            path_in_repo=path_in_repo,
            parent_commit=parent_sha,
            commit_message=(
                f"{username} took the test (attempt {captured_attempt[-1]}): "
                f"{score:.2f}"
            ),
        )

    _commit_with_cas(
        repo_id=PARTICIPANTS_REPO,
        fetch=fetch,
        mutator=mutator,
        commit=commit,
    )
    return captured_attempt[-1]


def is_fully_validated(row) -> bool:
    return all(
        row[f"prompt_validation_{i}"]["choice"] in ACCEPT_VALIDATION_CHOICES
        for i in (1, 2, 3)
    )


def has_answers(row) -> bool:
    return bool(str(row.get("answer_a", "")).strip()) and bool(
        str(row.get("answer_b", "")).strip()
    )


VALIDATION_COLS = (
    "prompt_validation_1",
    "prompt_validation_2",
    "prompt_validation_3",
)
VOTE_COLS = ("answer_chosen_1", "answer_chosen_2", "answer_chosen_3")


def _validator_usernames(df: pd.DataFrame) -> pd.Series:
    parts = [df[col].str["username"] for col in VALIDATION_COLS]
    s = pd.concat(parts, ignore_index=True)
    return s[s != ""]


def _voter_usernames(df: pd.DataFrame) -> pd.Series:
    parts = [df[col].str["username"] for col in VOTE_COLS]
    s = pd.concat(parts, ignore_index=True)
    return s[s != ""]


def _fully_validated_mask(df: pd.DataFrame) -> pd.Series:
    """Vectorised equivalent of ``df.apply(is_fully_validated, axis=1)``."""
    return (
        df["prompt_validation_1"].str["choice"].isin(ACCEPT_VALIDATION_CHOICES)
        & df["prompt_validation_2"].str["choice"].isin(ACCEPT_VALIDATION_CHOICES)
        & df["prompt_validation_3"].str["choice"].isin(ACCEPT_VALIDATION_CHOICES)
    )


def user_stats(username: str, df: pd.DataFrame) -> dict:
    """Counts of prompts written, validations recorded, and votes recorded
    by ``username``."""
    if not username or df.empty:
        return {"sent": 0, "validated": 0, "voted": 0}
    sent = int((df["username"] == username).sum())
    validated = int(
        sum((df[col].str["username"] == username).sum() for col in VALIDATION_COLS)
    )
    voted = int(sum((df[col].str["username"] == username).sum() for col in VOTE_COLS))
    return {"sent": sent, "validated": validated, "voted": voted}


def all_known_usernames(df: pd.DataFrame) -> list[str]:
    """Every username that has authored, validated, or voted on a prompt."""
    if df.empty:
        return []
    names: set[str] = set(df["username"].dropna().astype(str))
    names.update(_validator_usernames(df))
    names.update(_voter_usernames(df))
    return sorted(n for n in names if n and n not in EXCLUDED_USERNAMES)


def country_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country: ``fully_validated`` and ``pending`` (sent but not yet
    fully validated). Sums to total prompts sent."""
    if df.empty:
        return pd.DataFrame(columns=["country", "fully_validated", "pending"])
    grouped = (
        df.assign(_fully=_fully_validated_mask(df))
        .dropna(subset=["country"])
        .groupby("country", sort=True)
        .agg(total=("country", "size"), fully_validated=("_fully", "sum"))
        .reset_index()
    )
    grouped["pending"] = grouped["total"] - grouped["fully_validated"]
    return grouped[["country", "fully_validated", "pending"]].astype(
        {"fully_validated": int, "pending": int}
    )


def ranking_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per known user with their three counts, sorted by prompts sent.
    ``EXCLUDED_USERNAMES`` (the ``"v0"`` import sentinel etc.) are filtered
    out so they don't pollute the leaderboard."""
    columns = ["username", "prompts sent", "prompts validated", "answers voted"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    sent = df["username"].value_counts()
    validated = _validator_usernames(df).value_counts()
    voted = _voter_usernames(df).value_counts()
    users = sorted(
        (set(sent.index) | set(validated.index) | set(voted.index)) - EXCLUDED_USERNAMES
    )
    out = pd.DataFrame(
        {
            "username": users,
            "prompts sent": [int(sent.get(u, 0)) for u in users],
            "prompts validated": [int(validated.get(u, 0)) for u in users],
            "answers voted": [int(voted.get(u, 0)) for u in users],
        },
        columns=columns,
    )
    return out.sort_values(
        ["prompts sent", "prompts validated", "answers voted"],
        ascending=False,
    ).reset_index(drop=True)
