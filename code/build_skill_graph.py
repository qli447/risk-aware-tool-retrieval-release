# -*- coding: utf-8 -*-
"""
build_skill_graph.py
--------------------
Build the offline ToolGraph used for post-hoc score smoothing.

4-type ToolGraph:
  Type 1: co-occurrence   (tools appearing together in queries, count >= T)
  Type 2: semantic        (cosine(desc_i, desc_j) > SEM_THR)
  Type 3: permission overlap  (tools sharing >= 1 high-risk permission category)
  Type 4: risk propagation    (both risk >= 3 AND co-occur >= 1)

Also stores perm_arr [N] (permission level 0-5) for optional rule/filter utilities.

Usage:
  python code/build_skill_graph.py [--threshold 2] [--sem-threshold 0.65]
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

DEV_PATH    = ROOT / "data/ultratool/data/English-dataset/dev.json"
RISK_PATH   = ROOT / "data/ultratool/ultratool_risk_labels_resolved.json"
TOOLS_PATH  = ROOT / "data/ultratool/ultratool_tools_with_desc.json"
OUT_PATH    = ROOT / "data/ultratool/skill_graph.pt"
OUT_PATH_4T = ROOT / "data/ultratool/risk_graph_4type.pt"

# Permission keyword categories used for permission-overlap edges.
PERM_KEYWORDS = {
    "shell":   ["execute", "shell", "command", "bash", "terminal", "subprocess", "run script"],
    "write":   ["write file", "create file", "modify file", "delete", "save file",
                "overwrite", "append to file", "file write"],
    "network": ["http", "https", "request", "api call", "download", "upload",
                "fetch url", "web request", "url", "endpoint"],
    "secret":  ["password", "token", "credential", "secret", "api key",
                "authentication", "login", "oauth"],
    "code":    ["python", "javascript", "script", "compile", "eval", "exec",
                "code execution", "run code", "interpreter"],
}


def extract_perm_level(desc: str) -> int:
    """Count how many of the 5 high-risk permission categories appear in the description."""
    desc_lower = desc.lower()
    return sum(
        1 for kws in PERM_KEYWORDS.values()
        if any(kw in desc_lower for kw in kws)
    )


def build_symmetric_edges(edge_dict):
    if not edge_dict:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
    pairs = sorted(edge_dict.keys())
    src_f = [i for i, j in pairs]
    dst_f = [j for i, j in pairs]
    w_f   = [edge_dict[k] for k in pairs]
    src = src_f + dst_f
    dst = dst_f + src_f
    w   = w_f + w_f
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ultratool", choices=["ultratool", "sealtools"])
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--sem-threshold", type=float, default=0.65)
    parser.add_argument("--encoder", type=str,
                        default="mangopy/ToolRet-trained-bge-large-en-v1.5")
    args = parser.parse_args()
    T, SEM_THR = args.threshold, args.sem_threshold

    # ── Dataset-specific paths ───────────────────────────────────────────────
    if args.dataset == "sealtools":
        tools_path = ROOT / "data/sealtools/sealtools_tools.json"
        risk_path  = ROOT / "data/sealtools/sealtools_risk_labels_resolved.json"
        train_path = ROOT / "data/sealtools/sealtools_train.jsonl"
        out_path   = ROOT / "data/sealtools/sealtools_skill_graph.pt"
        out_path_4t= ROOT / "data/sealtools/sealtools_risk_graph_4type.pt"
    else:
        tools_path = TOOLS_PATH
        risk_path  = RISK_PATH
        train_path = DEV_PATH
        out_path   = OUT_PATH
        out_path_4t= OUT_PATH_4T

    # ── Tool vocabulary ──────────────────────────────────────────────────────
    with open(tools_path, encoding="utf-8") as f:
        tools_raw = json.load(f)
    tool_names = [t["name"] for t in tools_raw]
    tool_descs = [t["description"] for t in tools_raw]
    tool_to_idx = {n: i for i, n in enumerate(tool_names)}
    N = len(tool_names)
    print(f"Dataset: {args.dataset}  Tool vocabulary: {N} tools")

    # ── Permission levels [N] ────────────────────────────────────────────────
    perm_levels = [extract_perm_level(d) for d in tool_descs]
    perm_arr = torch.tensor(perm_levels, dtype=torch.float32)
    n_high_perm = sum(1 for p in perm_levels if p >= 2)
    print(f"Tools with perm_level >= 2: {n_high_perm}/{N}")

    # ── Risk labels ──────────────────────────────────────────────────────────
    with open(risk_path, encoding="utf-8") as f:
        risk_raw = json.load(f)
    risk_level = {n: (info["risk_level"] if info["risk_level"] is not None else 2)
                  for n, info in risk_raw.items()}
    risk_arr = torch.tensor(
        [float(risk_level.get(n, 1)) for n in tool_names], dtype=torch.float32
    )

    # ── Co-occurrences ────────────────────────────────────────────────────────
    with open(train_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Training queries: {len(records)}")

    cooccur: dict = {}
    domain_to_id: dict = {}
    query_domains: dict = {}

    for rec in records:
        if args.dataset == "sealtools":
            domain = "All"
            q = rec.get("query", "").strip()
            names = [t for t in rec.get("correct_tools", []) if t in tool_to_idx]
        else:
            domain = rec.get("domain", "Other")
            q = rec.get("question", "").strip()
            names = [t["name"] for t in rec.get("tools", [])
                     if isinstance(t, dict) and t.get("name") in tool_to_idx]

        if domain not in domain_to_id:
            domain_to_id[domain] = len(domain_to_id)
        if q:
            query_domains[q] = domain_to_id[domain]

        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                i, j = tool_to_idx[names[a]], tool_to_idx[names[b]]
                if i > j: i, j = j, i
                cooccur[(i, j)] = cooccur.get((i, j), 0) + 1

    cooccur_thresh = {(i, j): cnt for (i, j), cnt in cooccur.items() if cnt >= T}
    print(f"Co-occur edges (>= {T}): {len(cooccur_thresh)}")

    # ── Backward-compat skill_graph.pt (co-occur only) ───────────────────────
    max_cnt = max(cooccur_thresh.values()) if cooccur_thresh else 1
    cooccur_w = {(i, j): cnt / max_cnt for (i, j), cnt in cooccur_thresh.items()}
    edge_index, edge_weight = build_symmetric_edges(cooccur_w)
    torch.save({
        "tool_names": tool_names, "tool_to_idx": tool_to_idx,
        "edge_index": edge_index, "edge_weight": edge_weight,
        "num_nodes": N, "domain_to_id": domain_to_id, "query_domains": query_domains,
    }, out_path)
    print(f"Saved co-occurrence graph -> {out_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Build 4-type ToolGraph
    # ═══════════════════════════════════════════════════════════════════════════
    print("\nBuilding 4-type ToolGraph (Type1:co-occur, Type2:sem, Type3:perm_overlap, Type4:risk_prop)...")
    combined: dict = {}

    # Type 1: Co-occurrence (weight 0.4)
    for (i, j), w in cooccur_w.items():
        combined[(i, j)] = combined.get((i, j), 0.0) + 0.4 * w
    print(f"  Type 1 (co-occurrence): {len(cooccur_w)} edges")

    # Type 2: Semantic similarity (weight 0.3)
    encoder_path = args.encoder

    n_sem = 0
    try:
        from sentence_transformers import SentenceTransformer
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Encoding {N} descs with {encoder_path} on {_dev}...")
        enc = SentenceTransformer(encoder_path, device=_dev)
        embs = enc.encode(tool_descs, batch_size=256, normalize_embeddings=True,
                          convert_to_tensor=True, show_progress_bar=True)
        sim_matrix = (embs @ embs.T).cpu()
        for i in range(N):
            for j in range(i + 1, N):
                s = float(sim_matrix[i, j])
                if s > SEM_THR:
                    combined[(i, j)] = combined.get((i, j), 0.0) + 0.3 * s
                    n_sem += 1
        print(f"  Type 2 (semantic sim > {SEM_THR}): {n_sem} edges")
        del embs, sim_matrix
    except Exception as e:
        print(f"  WARNING: Semantic edges skipped ({e})")

    # Type 3: Permission overlap (weight 0.15 × shared/5)
    n_po = 0
    perm_categories = list(PERM_KEYWORDS.keys())
    # Precompute per-tool permission bitmask
    perm_bits = []
    for desc in tool_descs:
        dl = desc.lower()
        perm_bits.append([
            1 if any(kw in dl for kw in PERM_KEYWORDS[cat]) else 0
            for cat in perm_categories
        ])
    for i in range(N):
        for j in range(i + 1, N):
            shared = sum(perm_bits[i][k] & perm_bits[j][k] for k in range(5))
            if shared >= 1:
                w_perm = 0.15 * shared / 5.0
                combined[(i, j)] = combined.get((i, j), 0.0) + w_perm
                n_po += 1
    print(f"  Type 3 (permission overlap >= 1 shared): {n_po} edges")

    # Type 4: Risk propagation (both risk>=3 AND co-occur>=1, weight 0.15)
    n_rprop = 0
    for (i, j) in cooccur:
        if float(risk_arr[i]) >= 3 and float(risk_arr[j]) >= 3:
            combined[(i, j)] = combined.get((i, j), 0.0) + 0.15
            n_rprop += 1
    print(f"  Type 4 (risk propagation, both risk>=3 & co-occur>=1): {n_rprop} edges")

    # Normalize to [0,1]
    if combined:
        max_w = max(combined.values())
        combined = {k: v / max_w for k, v in combined.items()}

    edge_index_4t, edge_weight_4t = build_symmetric_edges(combined)

    print(f"\n4-type ToolGraph: {N} nodes, "
          f"{edge_index_4t.shape[1]//2} undirected ({edge_index_4t.shape[1]} directed) edges")

    torch.save({
        "tool_names":   tool_names,
        "tool_to_idx":  tool_to_idx,
        "edge_index":   edge_index_4t,
        "edge_weight":  edge_weight_4t,
        "risk_arr":     risk_arr,
        "perm_arr":     perm_arr,
        "num_nodes":    N,
        "domain_to_id": domain_to_id,
        "query_domains": query_domains,
        "edge_type_counts": {
            "cooccurrence":     len(cooccur_w),
            "semantic":         n_sem,
            "perm_overlap":     n_po,
            "risk_propagation": n_rprop,
        },
    }, out_path_4t)
    print(f"Saved 4-type ToolGraph -> {out_path_4t}")
    print(f"Total domains: {len(domain_to_id)}")


if __name__ == "__main__":
    main()
