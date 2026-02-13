"""
Test experiment: run 1 counterfactual case through OpenAI API.
Usage:
  python test_experiment.py --model gpt-4o
  python test_experiment.py --model gpt-4o-mini
  python test_experiment.py --model gpt-3.5-turbo --pair-index 5
"""

import argparse
import json
import os
from openai import OpenAI

OUTPUT_FILE = "test_experiment_results.json"


def parse_labels(response_text: str) -> list[str]:
    """Extract the 4 comma-separated labels from LLM response."""
    # Take last line that looks like comma-separated labels
    for line in reversed(response_text.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            return parts
    # Fallback: return raw text
    return [response_text.strip()]


def main():
    parser = argparse.ArgumentParser(description="Test 1 counterfactual case via OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--pair-index", type=int, default=0,
                        help="Index of pair to test (default: 0)")
    parser.add_argument("--pairs-file", type=str, default="counterfactual_pairs.json",
                        help="Path to counterfactual pairs JSON")
    args = parser.parse_args()

    # ── Load pairs ────────────────────────────────────────────────────────────
    with open(args.pairs_file) as f:
        data = json.load(f)

    pair = data["pairs"][args.pair_index]
    print(f"Pair: {pair['pair_id']}")
    print(f"Source race: {pair['source_race']}")
    print(f"Chief complaint: {pair['chief_complaint']}")
    print(f"ED diagnosis: {pair['ed_diagnosis']}")
    print(f"Model: {args.model}")
    print("=" * 60)

    client = OpenAI()  # uses OPENAI_API_KEY env var

    # ── Run each variant × aspect ─────────────────────────────────────────────
    results = {
        "pair_id": pair["pair_id"],
        "model": args.model,
        "source_race": pair["source_race"],
        "chief_complaint": pair["chief_complaint"],
        "ed_diagnosis": pair["ed_diagnosis"],
        "variants": [],
    }

    for variant in pair["variants"]:
        race = variant["race"]
        print(f"\n--- Race: {race} {'(original)' if variant['is_original'] else '(counterfactual)'} ---")

        aspect_results = {}
        for aspect_key, prompt in variant["aspect_prompts"].items():
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content
            labels = parse_labels(raw)
            aspect_results[aspect_key] = {
                "labels": labels,
                "raw_response": raw,
            }
            print(f"  Aspect {aspect_key}: {labels}")

        results["variants"].append({
            "race": race,
            "race_key": variant["race_key"],
            "is_original": variant["is_original"],
            "aspects": aspect_results,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Aspect':<10}", end="")
    for v in results["variants"]:
        print(f"{v['race']:<30}", end="")
    print()
    print("-" * 100)

    for asp in ["A", "B", "C", "D", "E"]:
        print(f"{asp:<10}", end="")
        for v in results["variants"]:
            labels = v["aspects"][asp]["labels"]
            print(f"{', '.join(labels):<30}", end="")
        print()

    # ── Save (append to existing results, keyed by model) ──────────────────
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"{pair['pair_id']}_{args.model}"
    all_results[key] = results

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {OUTPUT_FILE} (key: {key})")


if __name__ == "__main__":
    main()
