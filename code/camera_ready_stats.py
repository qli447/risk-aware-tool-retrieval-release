# -*- coding: utf-8 -*-
"""
camera_ready_stats.py
---------------------
Camera-ready supplementary statistics. Read-only w.r.t. existing artifacts;
writes results/camera_ready_stats_<tag>.{json,txt}.

For the paper operating point (lam=0.1, K=1; alpha=0.2 for UltraTool and
alpha=0.02 for Seal-Tools) this computes:
  1. Paired significance tests (Wilcoxon signed-rank + sign-flip permutation +
     paired t) of Core / +Graph / +Rule vs the first-stage ToolRet-BGE ranking,
     on per-query metrics averaged over 3 seeds.
  2. Rule-filter activation stats: % queries whose top-5 changes, mean tools
     deferred from top-5, fallback rate (accepted pool < K), per-rule attribution.
  3. Wall-clock latency of graph smoothing and rule filtering.

No scipy dependency (tests implemented with stdlib + torch).
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_graph_smooth import (ROOT, RVR_THR, load_samples, build_adj_matrix,
                              smooth_scores_batch, ndcg, sndcg, mrr_fn, rvr,
                              srr, success_at_k)

RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "code/embed_cache"

# Default checkpoints for the released hidden-size-64 learned-head setup.
DEFAULT_CKPTS = {
    "ultratool": [f"code/checkpoints/reranker_ultra_toolret_seed{s}.pt"
                  for s in (123, 42, 777)],
    "sealtools": [f"code/checkpoints/reranker_seal_toolret_seed{s}.pt"
                  for s in (42, 123, 777)],
}
USE_COS_REL = {"ultratool": False, "sealtools": False}

PQ_METRICS = ["ndcg@5", "sndcg@5", "mrr", "rvr@5", "srr@5", "success@5"]


# ── head-only loading + cached embeddings ─────────────────────────────────────
# The encoder is frozen (model.py sets requires_grad=False), so only the two MLP
# heads carry trained weights. We rebuild the heads directly from the checkpoint
# and reuse embedding caches written by the training and evaluation scripts.
# This avoids reloading the frozen encoder during the statistics pass.

def load_heads(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, emb_dim, hidden = ckpt["state_dict"], ckpt["emb_dim"], ckpt["hidden"]
    rel = nn.Sequential(nn.Linear(emb_dim * 2, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1), nn.Sigmoid())
    risk = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1), nn.Sigmoid())
    rel.load_state_dict({k[len("rel_head."):]: v for k, v in sd.items()
                         if k.startswith("rel_head.")})
    risk.load_state_dict({k[len("risk_head."):]: v for k, v in sd.items()
                          if k.startswith("risk_head.")})
    return rel.to(device).eval(), risk.to(device).eval(), emb_dim


def resolve_cache(kind, stem, dataset, n_tools, n_queries=None):
    """Canonical cache names used by run_graph_smooth.py. Older caches under
    different names are deliberately NOT reused: they were written by earlier
    script versions with different input text, which silently corrupts the MLP
    relevance head's input distribution."""
    if kind == "tool":
        return CACHE_DIR / f"tool_emb_{stem}_{dataset}_{n_tools}.pt"
    return CACHE_DIR / f"qry_emb_{stem}_{dataset}_{n_tools}_{n_queries}.pt"


def build_encoder(ckpt_path, device):
    """Rebuild the frozen encoder and load its weights straight from the
    checkpoint, so the encoder is bit-identical to the published run regardless
    of the sentence-transformers version installed today."""
    from sentence_transformers import SentenceTransformer
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = SentenceTransformer(ckpt["encoder_name"], device=str(device))
    enc_sd = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("_encoder."):
            kk = k[len("_encoder."):].replace("0.model.", "0.auto_model.")
            enc_sd[kk] = v
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    print(f"    encoder weights loaded from checkpoint "
          f"(matched={len(enc_sd) - len(unexpected)}, missing={len(missing)}, "
          f"unexpected={len(unexpected)})")
    if len(enc_sd) - len(unexpected) < 300:
        print("ERROR: encoder weight remap failed — refusing to run with a "
              "partially-initialised encoder.")
        sys.exit(3)
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval()


def encode_texts(enc, texts, batch_size=256):
    with torch.no_grad():
        embs = enc.encode(texts, batch_size=batch_size, show_progress_bar=False,
                          convert_to_tensor=True, normalize_embeddings=True)
    return embs.float()


# ── statistics helpers ────────────────────────────────────────────────────────

def wilcoxon_signed_rank_p(diffs):
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie
    correction (zero differences dropped)."""
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n < 10:
        return float("nan"), n
    ranked = sorted((abs(x), x > 0) for x in d)
    # average ranks over ties
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        t = j - i
        if t > 1:
            tie_term += t ** 3 - t
        i = j
    w_pos = sum(r for r, (_, pos) in zip(ranks, ranked) if pos)
    mu = n * (n + 1) / 4.0
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if sigma2 <= 0:
        return float("nan"), n
    z = (w_pos - mu) / math.sqrt(sigma2)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return p, n


def permutation_p(diffs, n_iter=20000, device="cpu", seed=0):
    """Two-sided sign-flip permutation test on the mean difference."""
    d = torch.tensor(diffs, dtype=torch.float32, device=device)
    if float(d.abs().sum()) == 0:
        return 1.0
    obs = d.mean().abs()
    g = torch.Generator(device=device).manual_seed(seed)
    count, chunk = 0, 2000
    for start in range(0, n_iter, chunk):
        m = min(chunk, n_iter - start)
        signs = torch.randint(0, 2, (m, d.numel()), generator=g,
                              device=device, dtype=torch.float32) * 2 - 1
        perm_means = (signs * d.unsqueeze(0)).mean(dim=1).abs()
        count += int((perm_means >= obs - 1e-12).sum())
    return (count + 1) / (n_iter + 1)


def paired_t_p(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    if var == 0:
        return float("nan"), float("nan")
    t = mean / math.sqrt(var / n)
    # normal approximation (n >= 1000 in all our splits)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return p, t


# ── per-query metric computation ─────────────────────────────────────────────

def per_query_metrics(all_rankings, samples, risk):
    out = {m: [] for m in PQ_METRICS}
    for ranked, s in zip(all_rankings, samples):
        c = s["correct"]
        out["ndcg@5"].append(ndcg(ranked, c))
        out["sndcg@5"].append(sndcg(ranked, c, risk))
        out["mrr"].append(mrr_fn(ranked, c))
        out["rvr@5"].append(rvr(ranked, risk))
        out["srr@5"].append(srr(ranked, risk))
        out["success@5"].append(success_at_k(ranked, c))
    return out


# ── rule filter with stats (same logic as run_graph_smooth.rule_filter_gpu) ──

def rule_filter_gpu_stats(orders, risk_arr, perm_arr, sim_matrix,
                          max_high_risk=1, max_high_perm=2,
                          redundancy_thr=0.9, risk_thr=RVR_THR):
    Q, N = orders.shape
    device = orders.device
    risk_at_pos = risk_arr[orders]
    is_high_risk = risk_at_pos >= risk_thr
    if perm_arr is not None:
        is_high_perm = perm_arr[orders] >= 2

    accepted = torch.ones(Q, N, dtype=torch.bool, device=device)
    viol1 = torch.zeros(Q, N, dtype=torch.bool, device=device)
    viol2 = torch.zeros(Q, N, dtype=torch.bool, device=device)
    viol3 = torch.zeros(Q, N, dtype=torch.bool, device=device)
    hr_count = torch.zeros(Q, dtype=torch.long, device=device)
    hp_count = torch.zeros(Q, dtype=torch.long, device=device)
    running_max_sim = torch.full((Q, N), -1.0, device=device)

    for pos in range(N):
        tool_idx = orders[:, pos]
        hr = is_high_risk[:, pos]
        v1 = hr & (hr_count >= max_high_risk)
        v2 = torch.zeros(Q, dtype=torch.bool, device=device)
        if perm_arr is not None:
            hp = is_high_perm[:, pos]
            v2 = hp & (hp_count >= max_high_perm)
        cur_max_sim = running_max_sim.gather(1, tool_idx.unsqueeze(1)).squeeze(1)
        v3 = cur_max_sim > redundancy_thr
        rejected = v1 | v2 | v3
        accepted[:, pos] = ~rejected
        viol1[:, pos], viol2[:, pos], viol3[:, pos] = v1, v2, v3
        acc = ~rejected
        hr_count += (hr & acc).long()
        if perm_arr is not None:
            hp_count += (hp & acc).long()
        if pos < N - 1:
            new_sims = sim_matrix[tool_idx]
            acc_mask = acc.float().unsqueeze(1)
            running_max_sim = torch.maximum(running_max_sim, new_sims * acc_mask)

    reject_flag = (~accepted).long()
    _, sort_idx = reject_flag.sort(dim=1, stable=True)
    return orders.gather(1, sort_idx), accepted, viol1, viol2, viol3


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, risk, tool_pool, tool_names, tool_descs = load_samples(
        args.dataset, split=args.split)
    Q_full = len(samples)          # caches are named by the full query count
    all_queries_full = [s["query"] for s in samples]
    if args.limit:
        samples = samples[:args.limit]
    Q, N = len(samples), len(tool_names)
    print(f"Dataset={args.dataset} split={args.split}: {Q} queries, {N} tools")

    graph_path = (ROOT / "data/sealtools/sealtools_risk_graph_4type.pt"
                  if args.dataset == "sealtools"
                  else ROOT / "data/ultratool/risk_graph_4type.pt")
    graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
    adj, _ = build_adj_matrix(graph_data, device=device, dr_mode=False)

    risk_arr_gpu = torch.tensor([risk.get(n, 1) for n in tool_names],
                                dtype=torch.float32, device=device)
    perm_arr_raw = graph_data.get("perm_arr")
    perm_arr_gpu = perm_arr_raw.float().to(device) if perm_arr_raw is not None else None

    use_cos = USE_COS_REL[args.dataset]
    ckpts = args.checkpoints or [str(ROOT / c) for c in DEFAULT_CKPTS[args.dataset]]

    # per-method, per-seed, per-query metric lists
    seeds_pq = {meth: [] for meth in ("toolret", "core", "graph", "rule")}
    rule_stats_seeds = []
    timing_seeds = []

    for ckpt in ckpts:
        stem = Path(ckpt).stem
        print(f"  Loading {Path(ckpt).name}...")
        rel_head, risk_head, emb_dim = load_heads(ckpt, device)

        tool_cache = resolve_cache("tool", stem, args.dataset, N)
        qry_cache = resolve_cache("qry", stem, args.dataset, N, Q_full)
        if tool_cache.exists() and qry_cache.exists():
            print(f"    reusing caches {tool_cache.name} / {qry_cache.name}")
            tool_embs = torch.load(tool_cache, map_location=device, weights_only=False).float()
            qry_embs = torch.load(qry_cache, map_location=device, weights_only=False).float()
        else:
            enc = build_encoder(ckpt, device)
            print(f"    encoding {N} tools and {Q_full} queries...")
            tool_embs = encode_texts(enc, tool_descs)
            qry_embs = encode_texts(enc, all_queries_full)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(tool_embs.cpu(), tool_cache)
            torch.save(qry_embs.cpu(), qry_cache)
            tool_embs, qry_embs = tool_embs.to(device), qry_embs.to(device)
            del enc
            torch.cuda.empty_cache()
        if args.limit:
            qry_embs = qry_embs[:args.limit]
        assert tool_embs.shape[0] == N, (tool_embs.shape, N)
        assert qry_embs.shape[0] == Q, (qry_embs.shape, Q)

        with torch.no_grad():
            f_risk_all = risk_head(tool_embs).squeeze(-1)
            t_norm = F.normalize(tool_embs, dim=-1)
            q_norm = F.normalize(qry_embs, dim=-1)
            cos_scores = q_norm @ t_norm.T                       # ToolRet-BGE 1st stage
            if use_cos:
                f_rel_all = cos_scores
            else:
                f_rel_all = torch.zeros(Q, N, device=device)
                BATCH = 64
                for start in range(0, N, BATCH):
                    end = min(start + BATCH, N)
                    t_emb = tool_embs[start:end].unsqueeze(0).expand(Q, -1, -1)
                    q_exp = qry_embs.unsqueeze(1).expand(-1, end - start, -1)
                    cat = torch.cat([q_exp, t_emb], dim=-1).reshape(-1, qry_embs.size(1) * 2)
                    f_rel_all[:, start:end] = rel_head(cat).reshape(Q, end - start)
            sim_matrix_gpu = t_norm @ t_norm.T

            raw = f_rel_all - args.lam * f_risk_all.unsqueeze(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            smoothed = smooth_scores_batch(raw, adj, args.alpha, K=args.K)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_smooth = time.perf_counter() - t0

            orders_toolret = cos_scores.argsort(dim=1, descending=True)
            orders_core = raw.argsort(dim=1, descending=True)
            orders_graph = smoothed.argsort(dim=1, descending=True)

            t0 = time.perf_counter()
            orders_rule, accepted, v1, v2, v3 = rule_filter_gpu_stats(
                orders_graph, risk_arr_gpu, perm_arr_gpu, sim_matrix_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_rule = time.perf_counter() - t0

        for meth, orders in (("toolret", orders_toolret), ("core", orders_core),
                             ("graph", orders_graph), ("rule", orders_rule)):
            rankings = [[tool_names[i] for i in row] for row in orders.cpu().tolist()]
            seeds_pq[meth].append(per_query_metrics(rankings, samples, risk))

        k = 5
        acc_count = accepted.sum(dim=1)
        deferred_top5 = (~accepted[:, :k]).sum(dim=1).float()
        rule_stats_seeds.append({
            "pct_queries_top5_changed": float((deferred_top5 > 0).float().mean()),
            "mean_deferred_in_top5": float(deferred_top5.mean()),
            "max_deferred_in_top5": int(deferred_top5.max()),
            "fallback_rate_pool_lt_k": float((acc_count < k).float().mean()),
            "mean_accepted_pool": float(acc_count.float().mean()),
            "top5_rejections_by_rule": {
                "risk_cap": int(v1[:, :k].sum()),
                "perm_cap": int(v2[:, :k].sum()),
                "redundancy": int(v3[:, :k].sum()),
            },
            "all_rejections_by_rule": {
                "risk_cap": int(v1.sum()),
                "perm_cap": int(v2.sum()),
                "redundancy": int(v3.sum()),
            },
        })
        timing_seeds.append({
            "graph_smooth_total_ms": t_smooth * 1000,
            "graph_smooth_per_query_ms": t_smooth * 1000 / Q,
            "rule_filter_total_ms": t_rule * 1000,
            "rule_filter_per_query_ms": t_rule * 1000 / Q,
        })
        del rel_head, risk_head, tool_embs, qry_embs, f_rel_all, sim_matrix_gpu, smoothed
        torch.cuda.empty_cache()

    # average per-query metrics across seeds
    def seed_avg(meth, metric):
        per_seed = [s[metric] for s in seeds_pq[meth]]
        return [sum(vals) / len(vals) for vals in zip(*per_seed)]

    sig = {}
    print(f"\n=== Paired tests vs ToolRet-BGE (lam={args.lam}, alpha={args.alpha}, "
          f"{len(ckpts)} seeds, Q={Q}) ===")
    for meth in ("core", "graph", "rule"):
        sig[meth] = {}
        for metric in PQ_METRICS:
            base = seed_avg("toolret", metric)
            ours = seed_avg(meth, metric)
            diffs = [o - b for o, b in zip(ours, base)]
            p_w, n_eff = wilcoxon_signed_rank_p(diffs)
            p_perm = permutation_p(diffs, device=str(device))
            p_t, t_stat = paired_t_p(diffs)
            sig[meth][metric] = {
                "mean_base": sum(base) / len(base),
                "mean_ours": sum(ours) / len(ours),
                "mean_diff": sum(diffs) / len(diffs),
                "wilcoxon_p": p_w, "wilcoxon_n_nonzero": n_eff,
                "perm_p": p_perm, "t_p": p_t, "t_stat": t_stat,
            }
            print(f"  {meth:6s} {metric:10s} base={sig[meth][metric]['mean_base']:.4f} "
                  f"ours={sig[meth][metric]['mean_ours']:.4f} "
                  f"diff={sig[meth][metric]['mean_diff']:+.4f} "
                  f"wilcoxon_p={p_w:.2e} perm_p={p_perm:.2e}")

    def avg_dicts(ds):
        out = {}
        for key in ds[0]:
            if isinstance(ds[0][key], dict):
                out[key] = {k: sum(d[key][k] for d in ds) / len(ds) for k in ds[0][key]}
            else:
                out[key] = sum(d[key] for d in ds) / len(ds)
        return out

    rule_stats = avg_dicts(rule_stats_seeds)
    timing = avg_dicts(timing_seeds)

    # ── reproduction check against the published results file ────────────────
    repro = {}
    ref_file = {
        "ultratool": "results_graph_smooth_ultra_toolret_final_3seed.json",
        "sealtools": "results_graph_smooth_sealtools_main.json",
    }[args.dataset]
    ref_path = RESULTS_DIR / ref_file
    if ref_path.exists() and not args.limit:
        ref = json.load(open(ref_path, encoding="utf-8"))
        key = f"a{args.alpha}_l{args.lam}"
        print(f"\n=== Reproduction check vs {ref_file} [{key}] ===")
        for block, meth in (("R6b", "graph"), ("R7b", "rule")):
            if key not in ref.get(block, {}):
                print(f"  [{block}] key {key} absent; available: "
                      f"{sorted(ref.get(block, {}))[:12]}")
                continue
            for metric in ("ndcg@5", "sndcg@5", "mrr", "rvr@5", "srr@5"):
                if metric not in ref[block][key]:
                    continue
                pub = ref[block][key][metric]
                now = sum(seed_avg(meth, metric)) / Q
                repro[f"{block}_{metric}"] = {"reference": pub, "recomputed": now,
                                              "abs_diff": abs(pub - now)}
                flag = "OK " if abs(pub - now) < 5e-3 else "DIFF"
                print(f"  [{flag}] {block} {metric:10s} reference={pub:.4f} "
                      f"recomputed={now:.4f} diff={abs(pub-now):.5f}")

    print(f"\n=== Rule-filter stats (mean over seeds) ===")
    print(json.dumps(rule_stats, indent=2))
    print(f"\n=== Timing (mean over seeds) ===")
    print(json.dumps(timing, indent=2))

    tag = f"{args.dataset}_{args.split}" if args.dataset == "sealtools" else args.dataset
    if args.limit:
        tag += f"_limit{args.limit}"
    out = {
        "config": {"dataset": args.dataset, "split": args.split, "lam": args.lam,
                   "alpha": args.alpha, "K": args.K, "checkpoints": ckpts,
                   "n_queries": Q, "n_tools": N, "use_cos_rel": use_cos},
        "significance_vs_toolret": sig,
        "rule_filter_stats": rule_stats,
        "rule_filter_stats_per_seed": rule_stats_seeds,
        "timing_ms": timing,
        "reproduction_check": repro,
    }
    out_path = RESULTS_DIR / f"camera_ready_stats_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ultratool", choices=["ultratool", "sealtools"])
    ap.add_argument("--split", default="test_in", choices=["test_in", "test_out"])
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--checkpoints", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit #queries")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
