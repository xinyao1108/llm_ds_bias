"""
Full experiment: run all 1500 counterfactual pairs through OpenAI API.

Usage:
  python run_experiment.py --model gpt-4o-mini
  python run_experiment.py --model gpt-4o --workers 10
  python run_experiment.py --model gpt-3.5-turbo --resume

Features:
  - Processes all 1500 pairs × 3 race variants × 5 aspects = 22,500 API calls
  - Concurrent requests via ThreadPoolExecutor
  - Automatic retry with exponential backoff on rate-limit / transient errors
  - Incremental checkpointing (saves every N pairs)
  - Resume from last checkpoint
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# ── Constants ──────────────────────────────────────────────────────────────────
ASPECTS = ["A", "B", "C", "D", "E"]
CHECKPOINT_EVERY = 50  # save after every N pairs
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2.0
DEFAULT_WORKERS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_labels(response_text: str) -> list[str]:
    """Extract the 4 comma-separated labels from LLM response."""
    for line in reversed(response_text.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            return parts
    return [response_text.strip()]


def call_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SEC,
) -> str:
    """Call the OpenAI API with exponential-backoff retry on transient errors."""
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            return response.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIError) as exc:
            if attempt == max_retries:
                raise
            log.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt, max_retries, type(exc).__name__, backoff,
            )
            time.sleep(backoff)
            backoff *= 2
    # Should not reach here, but just in case
    raise RuntimeError("Exhausted retries")


def process_pair(
    client: OpenAI,
    model: str,
    pair: dict,
) -> dict:
    """Process a single counterfactual pair (3 variants × 5 aspects = 15 calls)."""
    result = {
        "pair_id": pair["pair_id"],
        "model": model,
        "source_race": pair["source_race"],
        "chief_complaint": pair["chief_complaint"],
        "ed_diagnosis": pair["ed_diagnosis"],
        "variants": [],
    }

    for variant in pair["variants"]:
        aspect_results = {}
        for aspect_key in ASPECTS:
            prompt = variant["aspect_prompts"][aspect_key]
            raw = call_with_retry(client, model, prompt)
            labels = parse_labels(raw)
            aspect_results[aspect_key] = {
                "labels": labels,
                "raw_response": raw,
            }

        result["variants"].append({
            "race": variant["race"],
            "race_key": variant["race_key"],
            "is_original": variant["is_original"],
            "aspects": aspect_results,
        })

    return result


def output_path(model: str) -> Path:
    """Return the output JSON path for a given model."""
    safe_name = model.replace("/", "_")
    return Path(f"experiment_results_{safe_name}.json")


def load_checkpoint(path: Path) -> dict:
    """Load existing results from checkpoint file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, all_results: dict) -> None:
    """Atomically save results to checkpoint file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2)
    tmp.replace(path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full counterfactual bias experiment via OpenAI API"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--pairs-file", type=str, default="counterfactual_pairs.json",
        help="Path to counterfactual pairs JSON",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of concurrent workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint (skip completed pairs)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index (0-based) in the pairs list",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End index (exclusive) in the pairs list (default: all)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
        help=f"Save checkpoint every N pairs (default: {CHECKPOINT_EVERY})",
    )
    args = parser.parse_args()

    # ── Load pairs ─────────────────────────────────────────────────────────
    log.info("Loading pairs from %s", args.pairs_file)
    with open(args.pairs_file) as f:
        data = json.load(f)

    pairs = data["pairs"]
    end_idx = args.end if args.end is not None else len(pairs)
    pairs_slice = pairs[args.start:end_idx]
    total = len(pairs_slice)

    log.info(
        "Model: %s | Pairs: %d (index %d–%d) | Workers: %d",
        args.model, total, args.start, end_idx - 1, args.workers,
    )

    # ── Load checkpoint / resume ───────────────────────────────────────────
    out_path = output_path(args.model)
    all_results: dict = load_checkpoint(out_path) if args.resume else {}
    completed_ids = set(all_results.keys())

    if args.resume and completed_ids:
        log.info("Resuming: %d pairs already completed", len(completed_ids))

    # ── Filter out already-done pairs ──────────────────────────────────────
    pending_pairs = [
        p for p in pairs_slice
        if f"{p['pair_id']}_{args.model}" not in completed_ids
    ]
    log.info("Pending pairs: %d / %d", len(pending_pairs), total)

    if not pending_pairs:
        log.info("Nothing to do — all pairs already completed.")
        return

    # ── Process ────────────────────────────────────────────────────────────
    client = OpenAI()  # uses OPENAI_API_KEY env var
    done_count = 0
    errors = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_pair = {
            pool.submit(process_pair, client, args.model, pair): pair
            for pair in pending_pairs
        }

        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            pair_id = pair["pair_id"]
            key = f"{pair_id}_{args.model}"

            try:
                result = future.result()
                all_results[key] = result
                done_count += 1

                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = (len(pending_pairs) - done_count) / rate if rate > 0 else 0

                log.info(
                    "[%d/%d] %s done  (%.1f pairs/min, ~%.0f min left)",
                    done_count, len(pending_pairs), pair_id,
                    rate * 60, remaining / 60,
                )

                # Periodic checkpoint
                if done_count % args.checkpoint_every == 0:
                    save_checkpoint(out_path, all_results)
                    log.info("Checkpoint saved (%d total results)", len(all_results))

            except Exception as exc:
                log.error("FAILED %s: %s", pair_id, exc)
                errors.append({"pair_id": pair_id, "error": str(exc)})

    # ── Final save ─────────────────────────────────────────────────────────
    save_checkpoint(out_path, all_results)
    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info("Experiment complete for model: %s", args.model)
    log.info("Total results saved: %d", len(all_results))
    log.info("Errors: %d", len(errors))
    log.info("Time: %.1f min", elapsed / 60)
    log.info("Output: %s", out_path)

    if errors:
        err_path = Path(f"experiment_errors_{args.model}.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        log.warning("Error log saved → %s", err_path)

    # ── File size report ───────────────────────────────────────────────────
    size_bytes = out_path.stat().st_size
    log.info("Output file size: %.2f MB", size_bytes / 1024 / 1024)


if __name__ == "__main__":
    main()
