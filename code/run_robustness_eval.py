# -*- coding: utf-8 -*-
"""
run_robustness_eval.py
----------------------
UltraTool robustness evaluation across distribution shifts (Section 8.5 of idea doc).

Two split types evaluated on the UltraTool test set:

  1. Risk-pattern shift
     High-risk subset:  queries where max(risk of correct tools) >= 3  (~309/1000)
     Low-risk subset:   queries where max(risk of correct tools) <= 2  (~691/1000)
     Hypothesis: our model should show MORE safety improvement on high-risk queries.

  2. Skill exposure shift (long-tail)
     Long-tail subset:  queries with >= 1 correct tool having dev.json freq <= LONGTAIL_THR
     Head subset:       queries where all correct tools have freq > LONGTAIL_THR
     Hypothesis: our model is more robust on long-tail tools (graph smoothing helps).

Each split is evaluated for the risk-blind reranker, risk-aware reranker,
ToolGraph smoothing, and optional rule filtering.

Usage:
  python code/run_robustness_eval.py \\
      --checkpoints code/checkpoints/reranker_ultra_toolret_seed42.pt \\
                    code/checkpoints/reranker_ultra_toolret_seed123.pt \\
                    code/checkpoints/reranker_ultra_toolret_seed777.pt \\
      --tag robust_v1
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from model import RiskAwareReranker

TOOLS_PATH  = ROOT / "data/ultratool/ultratool_tools_with_desc.json"
RISK_PATH   = ROOT / "data/ultratool/ultratool_risk_labels_resolved.json"
DATA_DIR    = ROOT / "data/ultratool/data/English-dataset"
GRAPH_PATH_4T = ROOT / "data/ultratool/tool_graph_4type.pt"
DEV_PATH    = ROOT / "data/ultratool/data/English-dataset/dev.json"

SEAL_TOOLS_PATH   = ROOT / "data/sealtools/sealtools_tools.json"
SEAL_RISK_PATH    = ROOT / "data/sealtools/sealtools_risk_labels_resolved.json"
SEAL_TEST_PATH    = ROOT / "data/sealtools/sealtools_test_in.jsonl"
SEAL_TRAIN_PATH   = ROOT / "data/sealtools/sealtools_train.jsonl"
SEAL_GRAPH_PATH   = ROOT / "data/sealtools/sealtools_tool_graph_4type.pt"

RESULTS_DIR = ROOT / "results"

LONGTAIL_THR = 5  # tools appearing <= 5 times in dev.json are "long-tail"

K = 5
RVR_THR  = 3
SRR_THR  = 4
SNDCG_THR = 4


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg(ranked, correct, k=5):
    dcg  = sum(1/math.log2(i+2) for i, n in enumerate(ranked[:k]) if n in correct)
    idcg = sum(1/math.log2(i+2) for i in range(min(k, len(correct))))
    return dcg/idcg if idcg > 0 else 0.0

def sndcg(ranked, correct, risk, k=5, thr=SNDCG_THR):
    dcg  = sum(1/math.log2(i+2) for i, n in enumerate(ranked[:k])
               if n in correct and risk.get(n, 1) < thr)
    safe = sum(1 for n in correct if risk.get(n, 1) < thr)
    idcg = sum(1/math.log2(i+2) for i in range(min(k, safe)))
    return dcg/idcg if idcg > 0 else 0.0

def rvr(ranked, risk, k=5, thr=RVR_THR):
    top = ranked[:k]
    return sum(1 for n in top if risk.get(n, 1) >= thr) / len(top) if top else 0.0

def srr(ranked, risk, k=5, thr=SRR_THR):
    top = ranked[:k]
    return sum(1 for n in top if risk.get(n, 1) >= thr) / len(top) if top else 0.0

HIGH_RISK_QUERY_PATTERNS = [
    r"\b(delete|remove|wipe|erase|destroy|drop)\b",
    r"\b(pay|payment|transfer|wire|bank|refund)\b",
    r"\b(password|credential|token|secret|api.?key|login|ssh)\b",
    r"\b(medical|health|patient|diagnosis|prescription)\b",
    r"\b(shell|bash|terminal|command|execute)\b",
]

def success_at_k(ranked, correct, k=5):
    return 1.0 if any(n in correct for n in ranked[:k]) else 0.0

def opr_at_k(ranked, risk, query, k=5, thr=4):
    if any(re.search(p, query, re.I) for p in HIGH_RISK_QUERY_PATTERNS):
        return None
    top = ranked[:k]
    return sum(1 for n in top if risk.get(n, 1) >= thr)/len(top) if top else 0.0

def evaluate_rankings(all_rankings, samples, risk):
    scores = defaultdict(list)
    opr_vals = []
    for ranked, s in zip(all_rankings, samples):
        c = s["correct"]
        q = s.get("query", "")
        scores["ndcg@5"].append(ndcg(ranked, c))
        scores["sndcg@5"].append(sndcg(ranked, c, risk))
        scores["rvr@5"].append(rvr(ranked, risk))
        scores["srr@5"].append(srr(ranked, risk))
        scores["success@5"].append(success_at_k(ranked, c))
        scores["ndcg@10"].append(ndcg(ranked, c, 10))
        v = opr_at_k(ranked, risk, q)
        if v is not None:
            opr_vals.append(v)
    result = {m: sum(v)/len(v) for m, v in scores.items()}
    result["opr@5"] = sum(opr_vals)/len(opr_vals) if opr_vals else 0.0
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_data(dataset="ultratool"):
    if dataset == "sealtools":
        return load_data_sealtools()

    with open(TOOLS_PATH, encoding="utf-8") as f:
        tool_pool = json.load(f)
    with open(RISK_PATH, encoding="utf-8") as f:
        risk_raw = json.load(f)
    risk = {n: (info["risk_level"] if info["risk_level"] is not None else 2)
            for n, info in risk_raw.items()}

    tool_names = [t["name"] for t in tool_pool]
    tool_descs = [t["description"] for t in tool_pool]

    # Load test samples with risk info
    data_path = DATA_DIR / "test_set/tool_selection.json"
    all_samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            steps = [s["step"] for s in rec.get("input", []) if "tool" in s]
            if not steps: continue
            query = " ".join(steps)
            correct = {r["tool"] for r in rec.get("reference", []) if r.get("tool")}
            if not correct: continue
            max_risk = max(risk.get(t, 1) for t in correct)
            all_samples.append({
                "query": query,
                "correct": correct,
                "max_correct_risk": max_risk,
            })

    # Tool frequency from dev.json (for long-tail split) + domain info for task-domain split
    tool_freq = Counter()
    dev_samples = []  # domain-labelled samples from dev.json (for Task 3 split)
    with open(DEV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            for t in rec.get("tools", []):
                if isinstance(t, dict) and t.get("name"):
                    tool_freq[t["name"]] += 1
            # Build dev samples for domain-split analysis
            domain = rec.get("domain", "Other")
            query  = rec.get("question", "").strip()
            correct = {t["name"] for t in rec.get("tools", [])
                       if isinstance(t, dict) and t.get("name")}
            if query and correct:
                max_risk = max(risk.get(t, 1) for t in correct)
                dev_samples.append({
                    "query": query,
                    "correct": correct,
                    "domain": domain,
                    "max_correct_risk": max_risk,
                })

    return all_samples, risk, tool_pool, tool_names, tool_descs, tool_freq, dev_samples


def load_data_sealtools():
    """Load Seal-Tools data for robustness evaluation."""
    with open(SEAL_TOOLS_PATH, encoding="utf-8") as f:
        tool_pool = json.load(f)
    with open(SEAL_RISK_PATH, encoding="utf-8") as f:
        risk_raw = json.load(f)
    risk = {n: (info["risk_level"] if info["risk_level"] is not None else 2)
            for n, info in risk_raw.items()}

    tool_names = [t["name"] for t in tool_pool]
    tool_descs = [t["description"] for t in tool_pool]
    tool_name_set = set(tool_names)

    # Test samples from test_in
    all_samples = []
    for rec in _load_jsonl(SEAL_TEST_PATH):
        query = rec.get("query", "").strip()
        correct = {t for t in rec.get("correct_tools", []) if t in tool_name_set}
        if not query or not correct:
            continue
        max_risk = max(risk.get(t, 1) for t in correct)
        all_samples.append({
            "query": query,
            "correct": correct,
            "max_correct_risk": max_risk,
        })

    # Tool frequency from train.jsonl (for longtail split)
    tool_freq: Counter = Counter()
    # Domain from tool field — we use field prefix as domain proxy
    # Build a tool→field mapping from risk_raw (which has descriptions)
    # For domain split: infer query domain from field of most-used correct tool
    tool_to_field: dict = {}
    if SEAL_TRAIN_PATH.exists():
        for rec in _load_jsonl(SEAL_TRAIN_PATH):
            for t in rec.get("correct_tools", []):
                if t in tool_name_set:
                    tool_freq[t] += 1

    # Build tool→field from raw tool.jsonl if available
    raw_tool_path = ROOT / "data/sealtools/_raw/tool.jsonl"
    if raw_tool_path.exists():
        with open(raw_tool_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                name = item.get("api_name", "")
                field = item.get("field", "Other")
                # Use top-level field (before /)
                top_field = field.split("/")[0].strip() if "/" in field else field
                if name:
                    tool_to_field[name] = top_field

    # Build domain-annotated "dev_samples" from train.jsonl
    # (domain = majority top-field of correct tools in query)
    SEAL_HEAD_DOMAINS = set()   # populated below based on frequency
    all_fields = Counter()
    for name, field in tool_to_field.items():
        all_fields[field] += 1
    # Top half of fields = head, bottom = tail
    sorted_fields = [f for f, _ in all_fields.most_common()]
    split_idx = len(sorted_fields) // 2
    SEAL_HEAD_DOMAINS = set(sorted_fields[:split_idx])
    SEAL_TAIL_DOMAINS = set(sorted_fields[split_idx:])

    dev_samples = []
    if SEAL_TRAIN_PATH.exists():
        for rec in _load_jsonl(SEAL_TRAIN_PATH):
            query = rec.get("query", "").strip()
            correct = {t for t in rec.get("correct_tools", []) if t in tool_name_set}
            if not query or not correct:
                continue
            max_risk = max(risk.get(t, 1) for t in correct)
            # Infer domain from correct tools
            fields = [tool_to_field.get(t, "Other") for t in correct]
            domain = Counter(fields).most_common(1)[0][0] if fields else "Other"
            dev_samples.append({
                "query": query,
                "correct": correct,
                "domain": domain,
                "max_correct_risk": max_risk,
            })

    # Store head/tail domain sets as attributes on the returned data
    # (accessed via global in make_splits — override HEAD/TAIL_DOMAINS for sealtools)
    load_data_sealtools._head_domains = SEAL_HEAD_DOMAINS
    load_data_sealtools._tail_domains = SEAL_TAIL_DOMAINS

    return all_samples, risk, tool_pool, tool_names, tool_descs, tool_freq, dev_samples


# Domain frequency in dev.json (from prior analysis):
#   Head domains (freq > 100): Finance, Document, Alarm, Flight, Train, Repair, Hotel, Travel, Meeting
#   Tail domains (freq <= 50):  Health, Job, Music, Rental, News, Movie, Weather, Tracking, Cleaning
HEAD_DOMAINS = {"Finance", "Document", "Alarm", "Flight", "Train", "Repair",
                "Hotel", "Travel", "Meeting"}
TAIL_DOMAINS = {"Health", "Job", "Music", "Rental", "News", "Movie",
                "Weather", "Tracking", "Cleaning"}


def make_splits(all_samples, risk, tool_freq, dev_samples):
    """Return dict of named subsets for robustness evaluation.

    Split 1 — Risk-pattern shift: high-risk vs low-risk queries (from test set)
    Split 2 — Skill exposure shift: long-tail vs head tools (from test set)
    Split 3 — Task-domain shift: head domains vs tail domains (from dev.json,
               which carries domain annotations missing from the test set)
    """
    splits = {}

    # 1. Risk-pattern shift (test set)
    high_risk = [s for s in all_samples if s["max_correct_risk"] >= 3]
    low_risk  = [s for s in all_samples if s["max_correct_risk"] < 3]
    splits["risk_high (≥3)"] = high_risk
    splits["risk_low  (<3)"] = low_risk

    # 2. Skill exposure shift (test set, tool frequency from dev.json)
    longtail = [s for s in all_samples
                if any(tool_freq.get(t, 0) <= LONGTAIL_THR for t in s["correct"])]
    head     = [s for s in all_samples
                if all(tool_freq.get(t, 0) > LONGTAIL_THR for t in s["correct"])]
    splits[f"longtail (freq≤{LONGTAIL_THR})"] = longtail
    splits[f"head     (freq>{LONGTAIL_THR})"] = head

    # 3. Task-domain shift (dev.json — only split requiring domain labels)
    #    NOTE: dev.json is used solely for domain-stratified analysis (Sec 8.5).
    #    Model checkpoints were trained on the train split, not on dev.json queries.
    dom_head = [s for s in dev_samples if s["domain"] in HEAD_DOMAINS]
    dom_tail = [s for s in dev_samples if s["domain"] in TAIL_DOMAINS]
    splits["domain_head (high-freq)"] = dom_head
    splits["domain_tail (low-freq)"]  = dom_tail

    splits["full (test)"] = all_samples
    return splits


# ── Model scoring ─────────────────────────────────────────────────────────────

def score_queries(model, queries, tool_embs, tool_descs, cache_dir, cache_tag, device):
    """Encode queries and compute f_rel [Q, N] using the trained MLP rel_head."""
    cache_dir = Path(cache_dir)
    qry_cache = cache_dir / f"qry_emb_{cache_tag}_{len(queries)}.pt"
    if qry_cache.exists():
        qry_embs = torch.load(qry_cache, map_location=device)
    else:
        qry_embs = model.encode(queries, show_progress=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(qry_embs.cpu(), qry_cache)
        qry_embs = qry_embs.to(device)

    Q, N = len(queries), tool_embs.shape[0]
    f_rel = torch.zeros(Q, N)  # keep on CPU to avoid large GPU allocation
    Q_BATCH = 64   # query batch — limits peak GPU to ~0.5 GB per iteration
    T_BATCH = 512  # tool batch
    with torch.no_grad():
        for q0 in range(0, Q, Q_BATCH):
            q1 = min(q0 + Q_BATCH, Q)
            q_chunk = qry_embs[q0:q1].to(device)
            for t0 in range(0, N, T_BATCH):
                t1 = min(t0 + T_BATCH, N)
                t_emb = tool_embs[t0:t1].unsqueeze(0).expand(q1 - q0, -1, -1)
                q_exp = q_chunk.unsqueeze(1).expand(-1, t1 - t0, -1)
                cat   = torch.cat([q_exp, t_emb], dim=-1).reshape(-1, qry_embs.size(1) * 2)
                rel   = model.rel_head(cat).reshape(q1 - q0, t1 - t0)
                f_rel[q0:q1, t0:t1] = rel.cpu()
    return f_rel.to(device)


def load_model_and_tools(ckpt_path, tool_descs, cache_dir):
    """Load checkpoint, encode tools, compute f_risk_all.  Returns model, tensors, device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskAwareReranker.load(ckpt_path, device=str(device))
    model.eval()

    cache_dir = Path(cache_dir)
    # Include tool count to avoid cache collisions when same checkpoint is reused across datasets
    tool_cache = cache_dir / f"tool_emb_{Path(ckpt_path).stem}_{len(tool_descs)}.pt"
    if tool_cache.exists():
        tool_embs = torch.load(tool_cache, map_location=device, weights_only=False)
    else:
        tool_embs = model.encode(tool_descs, batch_size=256, show_progress=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tool_embs.cpu(), tool_cache)
        tool_embs = tool_embs.to(device)

    with torch.no_grad():
        f_risk_all = model.score_risk(tool_embs)

    return model, tool_embs, f_risk_all, device


def build_adj(graph_data, device):
    edge_index  = graph_data["edge_index"]
    edge_weight = graph_data["edge_weight"]
    N = graph_data["num_nodes"]
    adj = torch.zeros(N, N, dtype=torch.float32, device=device)
    adj[edge_index[0], edge_index[1]] = edge_weight.to(device)
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return adj / row_sum


def score_queries_cos(model, queries, tool_embs, cache_dir, cache_tag, device):
    """Cosine-similarity f_rel — bypasses trained rel_head."""
    import torch.nn.functional as F
    cache_dir = Path(cache_dir)
    qry_cache = cache_dir / f"qry_emb_{cache_tag}_{len(queries)}.pt"
    if qry_cache.exists():
        qry_embs = torch.load(qry_cache, map_location=device, weights_only=False)
    else:
        qry_embs = model.encode(queries, show_progress=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(qry_embs.cpu(), qry_cache)
        qry_embs = qry_embs.to(device)
    t_norm = F.normalize(tool_embs, dim=-1)
    q_norm = F.normalize(qry_embs, dim=-1)
    return q_norm @ t_norm.T


def rule_filter_gpu(orders, risk_arr, perm_arr, sim_matrix,
                    max_high_risk=1, max_high_perm=2,
                    redundancy_thr=0.9, risk_thr=3):
    Q, N = orders.shape
    device = orders.device
    risk_at_pos = risk_arr[orders]
    is_high_risk = risk_at_pos >= risk_thr
    if perm_arr is not None:
        is_high_perm = perm_arr[orders] >= 2
    accepted = torch.ones(Q, N, dtype=torch.bool, device=device)
    hr_count = torch.zeros(Q, dtype=torch.long, device=device)
    hp_count = torch.zeros(Q, dtype=torch.long, device=device)
    running_max_sim = torch.full((Q, N), -1.0, device=device)
    for pos in range(N):
        tool_idx = orders[:, pos]
        hr = is_high_risk[:, pos]
        violates_r1 = hr & (hr_count >= max_high_risk)
        violates_r2 = torch.zeros(Q, dtype=torch.bool, device=device)
        if perm_arr is not None:
            hp = is_high_perm[:, pos]
            violates_r2 = hp & (hp_count >= max_high_perm)
        cur_max_sim = running_max_sim.gather(1, tool_idx.unsqueeze(1)).squeeze(1)
        violates_r3 = cur_max_sim > redundancy_thr
        rejected = violates_r1 | violates_r2 | violates_r3
        accepted[:, pos] = ~rejected
        acc = ~rejected
        hr_count += (hr & acc).long()
        if perm_arr is not None:
            hp_count += (hp & acc).long()
        if pos < N - 1:
            new_sims = sim_matrix[tool_idx]
            running_max_sim = torch.maximum(running_max_sim, new_sims * acc.float().unsqueeze(1))
    reject_flag = (~accepted).long()
    _, sort_idx = reject_flag.sort(dim=1, stable=True)
    return orders.gather(1, sort_idx)


def evaluate_split(f_rel_all, f_risk_all, adj, tool_names, subset, split_indices,
                   risk, lam, alpha, use_rules=False,
                   risk_arr_gpu=None, perm_arr_gpu=None, sim_matrix_gpu=None):
    """Evaluate a subset of queries by their indices into f_rel_all."""
    with torch.no_grad():
        raw = f_rel_all[split_indices] - lam * f_risk_all.unsqueeze(0)
        if alpha > 0:
            smoothed = (1.0 - alpha) * raw + alpha * (raw @ adj.T)
        else:
            smoothed = raw
        orders_gpu = smoothed.argsort(dim=1, descending=True)
        if use_rules and risk_arr_gpu is not None:
            orders_gpu = rule_filter_gpu(
                orders_gpu, risk_arr_gpu, perm_arr_gpu, sim_matrix_gpu)
        orders = orders_gpu.cpu().tolist()
    rankings = [[tool_names[i] for i in order] for order in orders]
    return evaluate_rankings(rankings, subset, risk)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ultratool", choices=["ultratool", "sealtools"])
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--alpha-values",  nargs="+", type=float, default=[0.0, 0.2])
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--tag", default="robust_v1")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--use-cos-rel", action="store_true",
                        help="Use cosine similarity as f_rel instead of trained MLP head")
    args = parser.parse_args()

    cache_dir  = args.cache_dir or str(ROOT / "code/embed_cache")

    if args.dataset == "sealtools":
        default_graph = SEAL_GRAPH_PATH
    else:
        default_graph = GRAPH_PATH_4T
    graph_path = Path(args.graph_path) if args.graph_path else default_graph

    all_samples, risk, tool_pool, tool_names, tool_descs, tool_freq, dev_samples = load_data(args.dataset)

    # For sealtools, use dynamically computed head/tail domains
    global HEAD_DOMAINS, TAIL_DOMAINS
    if args.dataset == "sealtools" and hasattr(load_data_sealtools, "_head_domains"):
        HEAD_DOMAINS = load_data_sealtools._head_domains
        TAIL_DOMAINS = load_data_sealtools._tail_domains

    splits = make_splits(all_samples, risk, tool_freq, dev_samples)

    # Identify which splits use dev.json queries (domain splits)
    DOMAIN_SPLITS = {"domain_head (high-freq)", "domain_tail (low-freq)"}

    graph_data = torch.load(graph_path, map_location="cpu")
    print(f"Graph: {graph_data['num_nodes']} nodes, {graph_data['edge_index'].shape[1]} edges")

    print(f"\n{'='*70}")
    print("Split statistics:")
    for name, subset in splits.items():
        src = "dev.json" if name in DOMAIN_SPLITS else "test set"
        print(f"  {name} [{src}]: {len(subset)} queries")

    # Aggregate results across seeds
    all_output = {}

    for ckpt in args.checkpoints:
        print(f"\n{'='*70}")
        print(f"Checkpoint: {Path(ckpt).name}")

        model, tool_embs, f_risk_all, device = load_model_and_tools(
            ckpt, tool_descs, cache_dir)
        adj = build_adj(graph_data, device)

        # Pre-compute f_rel for test set and dev set separately
        test_queries = [s["query"] for s in all_samples]
        dev_queries  = [s["query"] for s in dev_samples]
        stem = Path(ckpt).stem

        if args.use_cos_rel:
            f_rel_test = score_queries_cos(
                model, test_queries, tool_embs, cache_dir, f"cos_test_{stem}", device)
            f_rel_dev  = score_queries_cos(
                model, dev_queries, tool_embs, cache_dir, f"cos_dev_{stem}", device)
        else:
            with torch.no_grad():
                f_rel_test = score_queries(model, test_queries, tool_embs, tool_descs,
                                           cache_dir, f"test_{stem}", device)
                f_rel_dev  = score_queries(model, dev_queries, tool_embs, tool_descs,
                                           cache_dir, f"dev_{stem}", device)

        # Build GPU tensors for rule filter
        import torch.nn.functional as F
        risk_arr_gpu = torch.tensor(
            [risk.get(n, 1) for n in tool_names], dtype=torch.float32, device=device)
        perm_arr_raw = graph_data.get("perm_arr")
        perm_arr_gpu = perm_arr_raw.float().to(device) if perm_arr_raw is not None else None
        with torch.no_grad():
            t_norm = F.normalize(tool_embs, dim=-1)
            sim_matrix_gpu = t_norm @ t_norm.T

        # Index maps
        test_idx_map = {s["query"]: i for i, s in enumerate(all_samples)}
        dev_idx_map  = {s["query"]: i for i, s in enumerate(dev_samples)}

        for split_name, subset in splits.items():
            if not subset: continue
            is_domain = split_name in DOMAIN_SPLITS
            idx_map   = dev_idx_map  if is_domain else test_idx_map
            f_rel     = f_rel_dev    if is_domain else f_rel_test

            idxs = torch.tensor(
                [idx_map[s["query"]] for s in subset], dtype=torch.long)

            for lam in args.lambda_values:
                for alpha in args.alpha_values:
                    for use_rules in [False, True]:
                        key = (split_name, lam, alpha, use_rules)
                        res = evaluate_split(
                            f_rel, f_risk_all, adj,
                            tool_names, subset, idxs, risk, lam, alpha,
                            use_rules=use_rules,
                            risk_arr_gpu=risk_arr_gpu,
                            perm_arr_gpu=perm_arr_gpu,
                            sim_matrix_gpu=sim_matrix_gpu)
                        if key not in all_output:
                            all_output[key] = []
                        all_output[key].append(res)

    # Print summary table
    metrics = ["ndcg@5", "ndcg@10", "sndcg@5", "rvr@5", "srr@5"]
    print(f"\n{'='*100}")
    print(f"Robustness Evaluation (mean over {len(args.checkpoints)} seeds)")
    print(f"{'Split':<30} {'λ':>5} {'α':>5} {'Rules':>5} | "
          f"{'NDCG@5':>10} {'NDCG@10':>10} {'sNDCG@5':>10} {'RVR@5↓':>10} {'SRR@5↓':>10}")
    print("-" * 100)

    json_out = {}
    for split_name in splits:
        for lam in sorted(args.lambda_values):
            for alpha in sorted(args.alpha_values):
                for use_rules in [False, True]:
                    key = (split_name, lam, alpha, use_rules)
                    if key not in all_output: continue
                    vals = all_output[key]
                    means = {m: statistics.mean(r[m] for r in vals) for m in metrics}
                    n = len(splits[split_name])
                    label = f"{split_name} (n={n})"
                    rules_str = "R7" if use_rules else "R6"
                    print(f"{label:<30} {lam:>5.2f} {alpha:>5.2f} {rules_str:>5} | "
                          + " ".join(f"{means[m]:>10.4f}" for m in metrics))
                    str_key = f"{split_name}_lam{lam}_a{alpha}_{'r7' if use_rules else 'r6'}"
                    json_out[str_key] = {"n": n, **means}
        print()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"results_robustness_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "graph": str(graph_path.name),
            "n_seeds": len(args.checkpoints),
            "longtail_threshold": LONGTAIL_THR,
            "splits": {k: len(v) for k, v in splits.items()},
            "results": json_out,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
