import os
import sys
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


REQUESTY_API_KEY = os.getenv("REQUESTY_API_KEY")
BASE_URL         = "https://router.requesty.ai/v1"
MODEL            = "openai/gpt-4o"       
MAX_ITERATIONS   = 3

client = OpenAI(
    api_key=REQUESTY_API_KEY,
    base_url=BASE_URL,
    default_headers={
        "HTTP-Referer": "https://yourapp.com",
        "X-Title":      "Log Analyzer",
    },
)


def chat(system: str, user: str, temperature: float = 0.3) -> str:
    """Single-turn chat completion."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


DRAFT_SYSTEM = """
You are a UX researcher and cognitive analyst specializing in mobile app behavior analysis.

Given raw timestamped logs of a user session in a food-ordering app (Zomato / Swiggy),
produce a **Cognitive Task Report (CTR)** that breaks down:

1. **Session Overview** – brief summary (user, app, date, outcome).
2. **Step-by-Step Breakdown** – for every meaningful action, describe:
   - Timestamp
   - Action taken
   - Inferred thought process / intent behind the action
   - Any friction, hesitation, or decision-making moment observed
3. **Key Decision Points** – moments where the user made a notable choice.
4. **UX Observations** – patterns, pain-points, or positive flows noticed.
5. **Session Outcome** – result of the session.

Format the report in clean Markdown with headers and numbered lists.
Be thorough but concise. Base everything strictly on the logs provided.
""".strip()

VALIDATOR_SYSTEM = """
You are a meticulous QA analyst reviewing a Cognitive Task Report (CTR) against original app session logs.

Your job is to identify **every inconsistency** between the CTR and the logs, including:
- Missing events that appear in the logs but not in the CTR
- Incorrect timestamps cited in the CTR
- Wrong prices, item names, order IDs, ratings, or other factual details
- Misinterpreted user intent that contradicts observable log evidence
- Extra events in the CTR that do not appear in the logs

Output your findings as a **numbered list of issues** in this exact format:

ISSUE 1: <brief title>
  - Location in CTR: <section / step reference>
  - Problem: <what is wrong>
  - Evidence in logs: <exact log line or detail>

If there are NO issues, output exactly:
  NO_ISSUES_FOUND

Be exhaustive. Do not skip minor discrepancies.
""".strip()

CORRECTION_SYSTEM = """
You are a precise technical writer. You will be given:
1. The original raw logs
2. A Cognitive Task Report (CTR) that may contain errors
3. A validation report listing specific issues

Your task is to produce a **fully corrected CTR** that:
- Fixes every issue listed in the validation report
- Retains all correct content from the original CTR
- Adds any missing log events with correct timestamps and analysis
- Does NOT introduce new information not present in the logs

Output the complete corrected CTR in clean Markdown. Do not include any preamble
like "Here is the corrected CTR" — output only the report itself.
""".strip()


def initial_draft_agent(logs: str) -> str:
    print("\n[Agent 1] Initial Draft Agent running...")
    prompt = f"Here are the session logs:\n\n{logs}"
    ctr = chat(DRAFT_SYSTEM, prompt)
    print("  → Draft CTR produced.")
    return ctr


def validator_agent(logs: str, ctr: str) -> str:
    print("\n[Agent 2] Validator Agent running...")
    prompt = (
        "## Original Logs\n\n"
        f"{logs}\n\n"
        "## Current CTR\n\n"
        f"{ctr}"
    )
    report = chat(VALIDATOR_SYSTEM, prompt, temperature=0.1)
    print("  → Validation report produced.")
    return report


def correction_agent(logs: str, ctr: str, issues: str) -> str:
    print("\n[Agent 3] Correction Agent running...")
    prompt = (
        "## Original Logs\n\n"
        f"{logs}\n\n"
        "## Current CTR (may have errors)\n\n"
        f"{ctr}\n\n"
        "## Validation Issues to Fix\n\n"
        f"{issues}"
    )
    corrected = chat(CORRECTION_SYSTEM, prompt)
    print("  → Corrected CTR produced.")
    return corrected


def run_pipeline(logs: str, output_dir: str = "output") -> str:
    ctr = initial_draft_agent(logs)
    save_file(f"{output_dir}/ctr_draft.md", ctr)
    print(f"  → Saved: {output_dir}/ctr_draft.md")

    # Step 2+3 – Iterative validation & correction
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")
        print(f"{'='*60}")

        issues = validator_agent(logs, ctr)
        save_file(f"{output_dir}/validation_iter_{iteration}.md", issues)
        print(f"  → Saved: {output_dir}/validation_iter_{iteration}.md")

        if "NO_ISSUES_FOUND" in issues:
            print(f"\n No issues found in iteration {iteration}. Pipeline complete.")
            break

        ctr = correction_agent(logs, ctr, issues)
        save_file(f"{output_dir}/ctr_iter_{iteration}.md", ctr)
        print(f"  → Saved: {output_dir}/ctr_iter_{iteration}.md")

        if iteration == MAX_ITERATIONS:
            print(f"\n  Reached maximum iterations ({MAX_ITERATIONS}). Saving final CTR.")

    # Save final output
    save_file(f"{output_dir}/ctr_final.md", ctr)
    print(f"\n Final CTR saved to: {output_dir}/ctr_final.md")
    return ctr


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "example-log.txt"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "output"

    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        sys.exit(1)

    print(f"   Starting Log Analyzer Pipeline")
    print(f"   Log file : {log_file}")
    print(f"   Output   : {out_dir}/")
    print(f"   Model    : {MODEL}")

    logs = load_file(log_file)
    run_pipeline(logs, out_dir)