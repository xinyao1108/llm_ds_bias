"""
Prompt Stability Experiment for ER-Reason Discharge Summary Evaluation
======================================================================
Three tests:
  Test 1 – N-Repetition Reliability  (same input, 10 runs, T=0.5)
  Test 2 – Ordinal Variance           (SD of numeric labels from Test 1)
  Test 3 – Structural Perturbation    (Original / Reversed / No-Lettering)

Requirements:
  export OPENAI_API_KEY="sk-..."
  pip install openai pandas scikit-learn
"""

import os, sys, json, time, re, random, csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL = "gpt-4o"
TEMPERATURE = 0.5
N_REPETITIONS = 10
N_SAMPLES = 10
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CSV_PATH = PROJECT_DIR / "er_reason.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Label definitions ──────────────────────────────────────────────────────────
VALID_LABELS = {"Fully", "Partial", "Unacceptance", "Missing"}
LABEL_TO_NUM = {"Fully": 3, "Partial": 2, "Unacceptance": 1, "Missing": 0}
ASPECTS = ["A", "B", "C", "D", "E"]
QUESTIONS = ["Q1", "Q2", "Q3", "Q4"]

# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are the attending physician on record, auditing an ICU discharge "
    "summary to ensure complete documentation."
)

# --- Original prompt (A-E ordering with letters) ---
PROMPT_ORIGINAL = """\
Task: Assess the discharge summary below for five main aspects, with four \
sub-items under each aspect. Assign one of the following labels to each \
sub-item: Fully, Partial, Unacceptance, or Missing.

Sub-items for every aspect:
Q1. Sufficient information (enough information for its purpose; includes pertinent details)
Q2. Concise (no duplicated information across sections; no irrelevant details unrelated to the aspect; appropriate length relative to clinical complexity)
Q3. Clear (understandable to providers and others)
Q4. Organized (properly grouped, chronological, can find important information)

Aspects:
A. Reason for Hospitalization: Chief complaint and/or history of present illness.
B. Significant Findings: Primary diagnoses.
C. Procedures and Treatment Provided: Hospital course and/or hospital consults or procedures.
D. Patient's Discharge Condition: Documentation conveying the patient's status at discharge.
E. Patient and Family Instructions: Instructions provided to the patient and/or family members.

Labels:
Fully – All expected elements are present with specific, granular detail (e.g., exact dates, dosages, named procedures, specific measurements).
Partial – The main topic is addressed but lacks one or more specific details (e.g., says "antibiotics given" without naming the drug or duration).
Unacceptance – Information is present but is vague, contradictory, or disorganized enough to risk misinterpretation (e.g., "patient improved" with no objective markers).
Missing – The aspect is entirely absent from the discharge summary.

Instructions: For each aspect, first briefly identify the relevant excerpt(s) \
from the discharge summary (or note "not found"), then assign your four labels \
based on that evidence.

Discharge Summary:
{discharge_summary}

Output Format: After your reasoning, provide a final answer block with each \
aspect on its own line, the aspect letter followed by four labels separated \
by commas. Example:
A: Fully, Partial, Fully, Fully
B: Partial, Fully, Fully, Partial
C: Fully, Fully, Partial, Missing
D: Missing, Missing, Missing, Missing
E: Fully, Partial, Fully, Fully
"""

# --- Reversed prompt (E-A ordering) ---
PROMPT_REVERSED = """\
Task: Assess the discharge summary below for five main aspects, with four \
sub-items under each aspect. Assign one of the following labels to each \
sub-item: Fully, Partial, Unacceptance, or Missing.

Sub-items for every aspect:
Q1. Sufficient information (enough information for its purpose; includes pertinent details)
Q2. Concise (no duplicated information across sections; no irrelevant details unrelated to the aspect; appropriate length relative to clinical complexity)
Q3. Clear (understandable to providers and others)
Q4. Organized (properly grouped, chronological, can find important information)

Aspects:
E. Patient and Family Instructions: Instructions provided to the patient and/or family members.
D. Patient's Discharge Condition: Documentation conveying the patient's status at discharge.
C. Procedures and Treatment Provided: Hospital course and/or hospital consults or procedures.
B. Significant Findings: Primary diagnoses.
A. Reason for Hospitalization: Chief complaint and/or history of present illness.

Labels:
Fully – All expected elements are present with specific, granular detail (e.g., exact dates, dosages, named procedures, specific measurements).
Partial – The main topic is addressed but lacks one or more specific details (e.g., says "antibiotics given" without naming the drug or duration).
Unacceptance – Information is present but is vague, contradictory, or disorganized enough to risk misinterpretation (e.g., "patient improved" with no objective markers).
Missing – The aspect is entirely absent from the discharge summary.

Instructions: For each aspect, first briefly identify the relevant excerpt(s) \
from the discharge summary (or note "not found"), then assign your four labels \
based on that evidence.

Discharge Summary:
{discharge_summary}

Output Format: After your reasoning, provide a final answer block with each \
aspect on its own line, the aspect letter followed by four labels separated \
by commas. Example:
E: Fully, Partial, Fully, Fully
D: Partial, Fully, Fully, Partial
C: Fully, Fully, Partial, Missing
B: Missing, Missing, Missing, Missing
A: Fully, Partial, Fully, Fully
"""

# --- No-lettering prompt (bullet points only) ---
PROMPT_NO_LETTER = """\
Task: Assess the discharge summary below for five main aspects, with four \
sub-items under each aspect. Assign one of the following labels to each \
sub-item: Fully, Partial, Unacceptance, or Missing.

Sub-items for every aspect:
- Sufficient information (enough information for its purpose; includes pertinent details)
- Concise (no duplicated information across sections; no irrelevant details unrelated to the aspect; appropriate length relative to clinical complexity)
- Clear (understandable to providers and others)
- Organized (properly grouped, chronological, can find important information)

Aspects:
- Reason for Hospitalization: Chief complaint and/or history of present illness.
- Significant Findings: Primary diagnoses.
- Procedures and Treatment Provided: Hospital course and/or hospital consults or procedures.
- Patient's Discharge Condition: Documentation conveying the patient's status at discharge.
- Patient and Family Instructions: Instructions provided to the patient and/or family members.

Labels:
Fully – All expected elements are present with specific, granular detail (e.g., exact dates, dosages, named procedures, specific measurements).
Partial – The main topic is addressed but lacks one or more specific details (e.g., says "antibiotics given" without naming the drug or duration).
Unacceptance – Information is present but is vague, contradictory, or disorganized enough to risk misinterpretation (e.g., "patient improved" with no objective markers).
Missing – The aspect is entirely absent from the discharge summary.

Instructions: For each aspect, first briefly identify the relevant excerpt(s) \
from the discharge summary (or note "not found"), then assign your four labels \
based on that evidence.

Discharge Summary:
{discharge_summary}

Output Format: After your reasoning, provide a final answer block with each \
aspect on its own line, the aspect name followed by four labels separated \
by commas. Example:
Reason for Hospitalization: Fully, Partial, Fully, Fully
Significant Findings: Partial, Fully, Fully, Partial
Procedures and Treatment Provided: Fully, Fully, Partial, Missing
Patient's Discharge Condition: Missing, Missing, Missing, Missing
Patient and Family Instructions: Fully, Partial, Fully, Fully
"""

# Mapping for no-letter output parsing
ASPECT_NAME_TO_LETTER = {
    "reason for hospitalization": "A",
    "significant findings": "B",
    "procedures and treatment provided": "C",
    "patient's discharge condition": "D",
    "patient and family instructions": "E",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_samples(csv_path: Path, n: int, seed: int) -> pd.DataFrame:
    """Load CSV and randomly sample n rows with non-null discharge summaries."""
    print(f"Loading CSV from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df[df["Discharge_Summary_Text"].notna()].reset_index(drop=True)
    print(f"  Total rows with discharge summaries: {len(df)}")
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  Sampled {n} discharge summaries.")
    return sampled


def call_llm(client: OpenAI, prompt: str, temperature: float = TEMPERATURE) -> str:
    """Call OpenAI chat completion and return the content string."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return ""


def parse_response(text: str, prompt_type: str = "original") -> dict:
    """
    Parse model output into {aspect: [label, label, label, label]}.
    Returns dict keyed by aspect letter A-E.
    """
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Try letter prefix: "A: ..." or "A. ..."
        m = re.match(r"^([A-E])[.:]\s*(.+)", line, re.IGNORECASE)
        if m:
            aspect = m.group(1).upper()
            labels_str = m.group(2)
        else:
            # Try name prefix for no-letter variant
            matched = False
            for name, letter in ASPECT_NAME_TO_LETTER.items():
                if line.lower().startswith(name):
                    aspect = letter
                    labels_str = line[len(name):].lstrip(":").strip()
                    matched = True
                    break
            if not matched:
                continue

        labels = [l.strip().capitalize() for l in labels_str.split(",")]
        # Normalize common variations
        normalized = []
        for l in labels:
            if l in VALID_LABELS:
                normalized.append(l)
            elif l.lower().startswith("full"):
                normalized.append("Fully")
            elif l.lower().startswith("partial"):
                normalized.append("Partial")
            elif l.lower().startswith("unaccept"):
                normalized.append("Unacceptance")
            elif l.lower().startswith("miss"):
                normalized.append("Missing")
            else:
                normalized.append(l)  # keep raw for debugging

        if len(normalized) == 4:
            result[aspect] = normalized

    return result


def flatten_labels(parsed: dict) -> list:
    """Flatten parsed dict into a list of 20 labels in A-Q1..A-Q4..E-Q4 order."""
    flat = []
    for asp in ASPECTS:
        if asp in parsed:
            flat.extend(parsed[asp])
        else:
            flat.extend(["PARSE_ERROR"] * 4)
    return flat


# ── Test 1: N-Repetition Reliability ──────────────────────────────────────────

def run_test1(client, samples, n_reps=N_REPETITIONS):
    """
    Run each discharge summary n_reps times with the Original prompt.
    Returns: all_results[sample_idx][rep_idx] = list of 20 labels
    """
    print("\n" + "="*70)
    print("TEST 1: N-Repetition Reliability")
    print("="*70)
    all_results = []

    for idx in range(len(samples)):
        ds_text = samples.iloc[idx]["Discharge_Summary_Text"]
        encounter = samples.iloc[idx]["encounterkey"]
        ds_truncated = ds_text[:8000] if len(ds_text) > 8000 else ds_text
        prompt = PROMPT_ORIGINAL.format(discharge_summary=ds_truncated)

        print(f"\nSample {idx+1}/{len(samples)} (encounter: {encounter})")
        reps = []
        for rep in range(n_reps):
            print(f"  Rep {rep+1}/{n_reps} ...", end=" ", flush=True)
            raw = call_llm(client, prompt)
            parsed = parse_response(raw, "original")
            flat = flatten_labels(parsed)
            reps.append(flat)
            print("done" if "PARSE_ERROR" not in flat else "PARSE WARNING")
            time.sleep(0.5)  # rate limiting
        all_results.append(reps)

    return all_results


def compute_agreement_rate(all_results):
    """
    For each sub-item position, compute agreement rate = freq(most_common) / n_reps.
    Returns per-item rates and overall mean.
    """
    n_samples = len(all_results)
    n_items = 20  # 5 aspects x 4 questions
    item_names = [f"{asp}_{q}" for asp in ASPECTS for q in QUESTIONS]

    per_item_rates = []  # shape: (n_samples, n_items)
    for sample_reps in all_results:
        sample_rates = []
        for item_idx in range(n_items):
            labels_at_pos = [rep[item_idx] for rep in sample_reps]
            counter = Counter(labels_at_pos)
            most_common_count = counter.most_common(1)[0][1]
            rate = most_common_count / len(sample_reps)
            sample_rates.append(rate)
        per_item_rates.append(sample_rates)

    per_item_rates = np.array(per_item_rates)
    overall_mean = per_item_rates.mean()
    per_item_mean = per_item_rates.mean(axis=0)

    return per_item_rates, per_item_mean, overall_mean, item_names


# ── Test 2: Ordinal Variance ──────────────────────────────────────────────────

def compute_ordinal_variance(all_results):
    """
    Convert labels to numeric and compute SD per sub-item per sample.
    Returns per-item SDs and overall mean SD.
    """
    n_items = 20
    item_names = [f"{asp}_{q}" for asp in ASPECTS for q in QUESTIONS]
    per_item_sds = []

    for sample_reps in all_results:
        sample_sds = []
        for item_idx in range(n_items):
            labels_at_pos = [rep[item_idx] for rep in sample_reps]
            nums = [LABEL_TO_NUM.get(l, np.nan) for l in labels_at_pos]
            nums = [n for n in nums if not np.isnan(n)]
            sd = np.std(nums) if len(nums) > 1 else 0.0
            sample_sds.append(sd)
        per_item_sds.append(sample_sds)

    per_item_sds = np.array(per_item_sds)
    overall_mean_sd = per_item_sds.mean()
    per_item_mean_sd = per_item_sds.mean(axis=0)

    return per_item_sds, per_item_mean_sd, overall_mean_sd, item_names


# ── Test 3: Structural Perturbation ───────────────────────────────────────────

def run_test3(client, samples):
    """
    Run each sample once with Original, Reversed, and No-Lettering prompts.
    Compute Cohen's Kappa between Original vs Reversed, and Original vs No-Lettering.
    """
    print("\n" + "="*70)
    print("TEST 3: Structural Perturbation")
    print("="*70)

    prompts = {
        "original": PROMPT_ORIGINAL,
        "reversed": PROMPT_REVERSED,
        "no_letter": PROMPT_NO_LETTER,
    }

    all_labels = {variant: [] for variant in prompts}

    for idx in range(len(samples)):
        ds_text = samples.iloc[idx]["Discharge_Summary_Text"]
        encounter = samples.iloc[idx]["encounterkey"]
        ds_truncated = ds_text[:8000] if len(ds_text) > 8000 else ds_text
        print(f"\nSample {idx+1}/{len(samples)} (encounter: {encounter})")

        for variant, template in prompts.items():
            print(f"  Variant: {variant} ...", end=" ", flush=True)
            prompt = template.format(discharge_summary=ds_truncated)
            raw = call_llm(client, prompt, temperature=0.0)  # T=0 for deterministic comparison
            parsed = parse_response(raw, variant)
            flat = flatten_labels(parsed)
            all_labels[variant].append(flat)
            print("done" if "PARSE_ERROR" not in flat else "PARSE WARNING")
            time.sleep(0.5)

    # Flatten all samples into single label vectors for kappa
    def flatten_all(variant_labels):
        return [l for sample in variant_labels for l in sample]

    orig_flat = flatten_all(all_labels["original"])
    rev_flat = flatten_all(all_labels["reversed"])
    nolet_flat = flatten_all(all_labels["no_letter"])

    # Filter out PARSE_ERROR pairs
    def filter_pairs(a, b):
        a_f, b_f = [], []
        for x, y in zip(a, b):
            if x != "PARSE_ERROR" and y != "PARSE_ERROR":
                a_f.append(x)
                b_f.append(y)
        return a_f, b_f

    orig_f, rev_f = filter_pairs(orig_flat, rev_flat)
    orig_f2, nolet_f = filter_pairs(orig_flat, nolet_flat)

    kappa_orig_rev = cohen_kappa_score(orig_f, rev_f) if len(orig_f) > 0 else float("nan")
    kappa_orig_nolet = cohen_kappa_score(orig_f2, nolet_f) if len(orig_f2) > 0 else float("nan")

    return all_labels, kappa_orig_rev, kappa_orig_nolet


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    samples, all_results,
    per_item_rates, per_item_mean_rate, overall_agreement,
    per_item_sds, per_item_mean_sd, overall_sd,
    test3_labels, kappa_orig_rev, kappa_orig_nolet,
    item_names
):
    """Generate and save a comprehensive report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PROMPT STABILITY EXPERIMENT REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Model: {MODEL}")
    report_lines.append(f"Temperature (Test 1&2): {TEMPERATURE}")
    report_lines.append(f"N Repetitions: {N_REPETITIONS}")
    report_lines.append(f"N Samples: {N_SAMPLES}")
    report_lines.append(f"Random Seed: {SEED}")
    report_lines.append("")

    # Test 1
    report_lines.append("-" * 70)
    report_lines.append("TEST 1: N-Repetition Reliability (Agreement Rate)")
    report_lines.append("-" * 70)
    report_lines.append(f"Overall Mean Agreement Rate: {overall_agreement:.3f}")
    report_lines.append("")
    report_lines.append("Per-item mean agreement rates (averaged across samples):")
    for i, name in enumerate(item_names):
        report_lines.append(f"  {name}: {per_item_mean_rate[i]:.3f}")

    report_lines.append("")
    report_lines.append("Per-sample agreement rates (averaged across items):")
    for idx in range(len(samples)):
        enc = samples.iloc[idx]["encounterkey"]
        mean_rate = per_item_rates[idx].mean()
        report_lines.append(f"  Sample {idx+1} ({enc}): {mean_rate:.3f}")

    # Test 2
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("TEST 2: Ordinal Variance (Standard Deviation)")
    report_lines.append("-" * 70)
    report_lines.append(f"Overall Mean SD: {overall_sd:.4f}")
    report_lines.append(f"  Interpretation: {'Highly stable' if overall_sd < 0.2 else 'Moderate' if overall_sd < 1.0 else 'Unstable (high jitter)'}")
    report_lines.append("")
    report_lines.append("Per-item mean SD (averaged across samples):")
    for i, name in enumerate(item_names):
        report_lines.append(f"  {name}: {per_item_mean_sd[i]:.4f}")

    report_lines.append("")
    report_lines.append("Per-sample mean SD (averaged across items):")
    for idx in range(len(samples)):
        enc = samples.iloc[idx]["encounterkey"]
        mean_sd = per_item_sds[idx].mean()
        report_lines.append(f"  Sample {idx+1} ({enc}): {mean_sd:.4f}")

    # Test 3
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("TEST 3: Structural Perturbation (Cohen's Kappa)")
    report_lines.append("-" * 70)
    report_lines.append(f"Cohen's Kappa (Original vs Reversed):      {kappa_orig_rev:.4f}")
    report_lines.append(f"Cohen's Kappa (Original vs No-Lettering):  {kappa_orig_nolet:.4f}")
    report_lines.append("")
    bias_status_rev = "No position bias" if kappa_orig_rev >= 0.8 else "POSITION BIAS DETECTED"
    bias_status_nolet = "No format bias" if kappa_orig_nolet >= 0.8 else "FORMAT BIAS DETECTED"
    report_lines.append(f"  Original vs Reversed:     {bias_status_rev} (threshold: kappa >= 0.8)")
    report_lines.append(f"  Original vs No-Lettering: {bias_status_nolet} (threshold: kappa >= 0.8)")

    report_lines.append("")
    report_lines.append("Per-sample label comparison (Original vs Reversed):")
    for idx in range(len(samples)):
        enc = samples.iloc[idx]["encounterkey"]
        orig = test3_labels["original"][idx]
        rev = test3_labels["reversed"][idx]
        matches = sum(1 for a, b in zip(orig, rev) if a == b and a != "PARSE_ERROR")
        total = sum(1 for a, b in zip(orig, rev) if a != "PARSE_ERROR" and b != "PARSE_ERROR")
        pct = matches / total * 100 if total > 0 else 0
        report_lines.append(f"  Sample {idx+1} ({enc}): {matches}/{total} match ({pct:.1f}%)")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    return report_text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load samples
    samples = load_samples(CSV_PATH, N_SAMPLES, SEED)

    # Save selected sample metadata
    sample_meta = samples[["encounterkey", "primarychiefcomplaintname",
                           "primaryeddiagnosisname"]].copy()
    sample_meta.to_csv(OUTPUT_DIR / "selected_samples.csv", index=False)
    print(f"Saved sample metadata to {OUTPUT_DIR / 'selected_samples.csv'}")

    # ── Test 1: N-Repetition Reliability ──
    all_results = run_test1(client, samples)

    # Save raw results
    raw_data = []
    item_names = [f"{asp}_{q}" for asp in ASPECTS for q in QUESTIONS]
    for s_idx, reps in enumerate(all_results):
        for r_idx, labels in enumerate(reps):
            row = {
                "sample_idx": s_idx,
                "encounterkey": samples.iloc[s_idx]["encounterkey"],
                "rep": r_idx,
            }
            for i, name in enumerate(item_names):
                row[name] = labels[i]
            raw_data.append(row)
    pd.DataFrame(raw_data).to_csv(OUTPUT_DIR / "test1_raw_results.csv", index=False)
    print(f"\nSaved Test 1 raw results to {OUTPUT_DIR / 'test1_raw_results.csv'}")

    # Compute Test 1 metrics
    per_item_rates, per_item_mean_rate, overall_agreement, item_names = compute_agreement_rate(all_results)

    # ── Test 2: Ordinal Variance ──
    print("\n" + "="*70)
    print("TEST 2: Ordinal Variance (computed from Test 1 data)")
    print("="*70)
    per_item_sds, per_item_mean_sd, overall_sd, _ = compute_ordinal_variance(all_results)
    print(f"Overall mean SD: {overall_sd:.4f}")

    # ── Test 3: Structural Perturbation ──
    test3_labels, kappa_orig_rev, kappa_orig_nolet = run_test3(client, samples)

    # Save Test 3 raw results
    test3_data = []
    for variant, label_lists in test3_labels.items():
        for s_idx, labels in enumerate(label_lists):
            row = {
                "sample_idx": s_idx,
                "encounterkey": samples.iloc[s_idx]["encounterkey"],
                "variant": variant,
            }
            for i, name in enumerate(item_names):
                row[name] = labels[i]
            test3_data.append(row)
    pd.DataFrame(test3_data).to_csv(OUTPUT_DIR / "test3_raw_results.csv", index=False)
    print(f"\nSaved Test 3 raw results to {OUTPUT_DIR / 'test3_raw_results.csv'}")

    # ── Generate report ──
    report = generate_report(
        samples, all_results,
        per_item_rates, per_item_mean_rate, overall_agreement,
        per_item_sds, per_item_mean_sd, overall_sd,
        test3_labels, kappa_orig_rev, kappa_orig_nolet,
        item_names,
    )

    report_path = OUTPUT_DIR / "stability_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nReport saved to {report_path}")

    # Save summary metrics as JSON
    summary = {
        "model": MODEL,
        "temperature_test1_2": TEMPERATURE,
        "n_repetitions": N_REPETITIONS,
        "n_samples": N_SAMPLES,
        "test1_overall_agreement_rate": round(overall_agreement, 4),
        "test2_overall_mean_sd": round(overall_sd, 4),
        "test2_interpretation": "Highly stable" if overall_sd < 0.2 else "Moderate" if overall_sd < 1.0 else "Unstable",
        "test3_kappa_original_vs_reversed": round(kappa_orig_rev, 4),
        "test3_kappa_original_vs_no_lettering": round(kappa_orig_nolet, 4),
        "test3_position_bias": bool(kappa_orig_rev < 0.8),
        "test3_format_bias": bool(kappa_orig_nolet < 0.8),
    }
    with open(OUTPUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics saved to {OUTPUT_DIR / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
