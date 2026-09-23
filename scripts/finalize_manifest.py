# -*- coding: utf-8 -*-
"""Finalize: update run_manifest with A25-A40 verification results, rebuild agent roster log,
refresh artifact manifest hash for changed files."""
import json, hashlib, os
from pathlib import Path
import pandas as pd

OUT = Path("/home/z/my-project/trac_phish_revision8_FULL_RUN")

# --- 1. assemble the 40-agent roster with final statuses ---
AGENTS = {
    "A01": ("Archive & Integrity Auditor", "pre-execution LLM subagent", "PASS"),
    "A02": ("Schema & Label Auditor", "pre-execution LLM subagent", "PASS"),
    "A03": ("Temporal Integrity Auditor", "pre-execution LLM subagent", "PASS"),
    "A04": ("Class Balance & Duplicate Auditor", "pre-execution LLM subagent", "PASS"),
    "A05": ("Cross-Dataset Overlap Pre-Scanner", "pre-execution LLM subagent", "PASS"),
    "A06": ("Execution Plan & Environment Reviewer", "pre-execution LLM subagent", "GO-WITH-CHANGES (adopted)"),
}
for aid, name in [("A07","Dataset audit"),("A08","Cleaning & dedup ledger"),("A09","Leakage/overlap analysis"),
                  ("A10","Partitioning"),("A11","Lexical feature extraction"),("A12","Char n-gram TF-IDF"),
                  ("A13","Representation analysis"),("A14","Baselines"),("A15","N-gram model training"),
                  ("A16","Lexical models"),("A17","Model selection"),("A18","Main test evaluation"),
                  ("A19","Transfer zero-shot"),("A20","Robustness suite"),("A21","Calibration"),
                  ("A22","SHAP/XAI"),("A23","Explanation stability / ERS"),("A24","DTS")]:
    AGENTS[aid] = (name, "in-pipeline stage agent", "COMPLETED")
POST = {"A25": ("Main Metrics Verifier", "PASS"),
        "A26": ("Leakage Audit Verifier", "PASS"),
        "A27": ("Transfer Results Verifier", "PASS"),
        "A28": ("Calibration Verifier", "PASS"),
        "A29": ("Robustness Verifier", "PASS"),
        "A30": ("SHAP/ERS Verifier", "PASS"),
        "A31": ("DTS Verifier", "PASS"),
        "A32": ("Statistical Tests Verifier", "PASS_WITH_DEFECTS -> MCC int64-overflow bug FIXED, notebook re-executed (run 15); fix verified [0.9482, 0.9512]"),
        "A33": ("Gates & Negative-Results Auditor", "PASS (honest reporting confirmed)"),
        "A34": ("Case Studies QA", "PASS (12/12)"),
        "A35": ("Sanity Checks Auditor", "PASS (determinism independently reproduced)"),
        "A36": ("Reproducibility Auditor", "PASS"),
        "A37": ("Tables/LaTeX QA", "CONDITIONAL FAIL -> LaTeX escaping FIXED, notebook re-executed (run 16); escaping verified"),
        "A38": ("Figures QA", "PASS (9/9)"),
        "A39": ("Artifact & Manifest QA", "PASS (128 files, hashes verified)"),
        "A40": ("Notebook Validation", "PASS (31/31 cells, FULL banner, SELF-VALIDATION PASS)")}
for aid, (name, status) in POST.items():
    AGENTS[aid] = (name, "post-execution LLM verification subagent", status)

roster = [{"agent": aid, "role": r[0], "mode": r[1], "status": r[2]} for aid, r in sorted(AGENTS.items())]
pd.DataFrame(roster).to_csv(OUT / "manifests/agent_roster.csv", index=False)

# --- 2. update run_manifest.json ---
rm_path = OUT / "manifests/run_manifest.json"
rm = json.loads(rm_path.read_text())
rm["agents"] = {"pre_execution_A01_A06": 6, "in_pipeline_A07_A24": 18, "post_execution_A25_A40": 16,
                "total_specialists": 40, "coordinator": 1,
                "roster_file": "manifests/agent_roster.csv",
                "verification_defects_found_and_fixed": 2,
                "defects": ["MCC bootstrap int64 overflow (A32) — fixed run 15, verified",
                            "LaTeX escaping (A37) — fixed run 16, verified"]}
rm["verification"] = {"post_execution_verifiers": "16/16 dispatched, all completed",
                      "defects_found": 2, "defects_fixed_and_rerun": 2}
rm["final_notebook_run"] = {"cells_executed": 31, "error_outputs": 0,
                            "notebook": "trac-phish-revision8_FULL_EXECUTED.ipynb"}
rm["finished_utc"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S UTC", __import__("time").gmtime())
rm_path.write_text(json.dumps(rm, indent=2, default=str))

# --- 3. refresh the artifact manifest (agent logs + roster + manifest changed) ---
def sha(p, chunk=1 << 20):
    h = hashlib.sha256(); fd = os.open(str(p), os.O_RDONLY)
    try:
        while True:
            b = os.read(fd, chunk)
            if not b: break
            h.update(b)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return h.hexdigest()

inv = []
for root, dirs, files in os.walk(OUT):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fn in sorted(files):
        p = Path(root) / fn
        inv.append({"file": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p)})
inv_df = pd.DataFrame(inv).sort_values("file").reset_index(drop=True)
inv_df.to_csv(OUT / "manifests/artifact_manifest.csv", index=False)
# re-hash the manifest file itself into run manifest stats
rm["artifacts"] = {"count": len(inv_df), "total_bytes": int(inv_df.bytes.sum()),
                   "manifest": "manifests/artifact_manifest.csv"}
rm_path.write_text(json.dumps(rm, indent=2, default=str))

print(f"agent roster: {len(roster)} specialists")
print(f"artifacts: {len(inv_df)} files, {inv_df.bytes.sum()/1e6:.1f} MB")
print("run_manifest.json updated with verification results")
