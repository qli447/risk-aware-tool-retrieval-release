# -*- coding: utf-8 -*-
"""
topk_probe.py — camera-ready control experiments. Writes new files only.

(1) Candidate-set control: rank our method under
      A  full tool pool                       (main-table configuration)
      B  ToolRet-BGE top-100, full-graph smoothing then mask
      C  ToolRet-BGE top-100, candidate-induced subgraph smoothing (appendix control)
(2) Rule-filter threshold sensitivity: risk cap x redundancy threshold.
(3) Impact of the 32 corrected Seal risk labels, re-scored on identical rankings.
"""
import argparse, json, sys, math
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_graph_smooth import (ROOT, RVR_THR, load_samples, build_adj_matrix,
                              smooth_scores_batch, ndcg, sndcg, mrr_fn, rvr,
                              srr, success_at_k)
from camera_ready_stats import (load_heads, resolve_cache, build_encoder,
                                encode_texts, DEFAULT_CKPTS, USE_COS_REL)


def rule_filter_stats(orders, risk_arr, perm_arr, sim_matrix,
                      max_high_risk=1, max_high_perm=2,
                      redundancy_thr=0.9, risk_thr=RVR_THR):
    """Same rules as run_graph_smooth.rule_filter_gpu, but accepts a candidate
    list of any length M <= N. running_max_sim stays indexed by GLOBAL tool id,
    which is what breaks if the upstream [Q,N] version is fed M<N columns."""
    Q, M = orders.shape
    N = sim_matrix.shape[0]
    device = orders.device
    is_high_risk = risk_arr[orders] >= risk_thr
    is_high_perm = (perm_arr[orders] >= 2) if perm_arr is not None else None

    accepted = torch.ones(Q, M, dtype=torch.bool, device=device)
    viol = [torch.zeros(Q, M, dtype=torch.bool, device=device) for _ in range(3)]
    hr_count = torch.zeros(Q, dtype=torch.long, device=device)
    hp_count = torch.zeros(Q, dtype=torch.long, device=device)
    running_max_sim = torch.full((Q, N), -1.0, device=device)   # global-indexed

    for pos in range(M):
        tool_idx = orders[:, pos]
        hr = is_high_risk[:, pos]
        v1 = hr & (hr_count >= max_high_risk)
        v2 = torch.zeros(Q, dtype=torch.bool, device=device)
        if is_high_perm is not None:
            hp = is_high_perm[:, pos]
            v2 = hp & (hp_count >= max_high_perm)
        v3 = running_max_sim.gather(1, tool_idx.unsqueeze(1)).squeeze(1) > redundancy_thr
        rejected = v1 | v2 | v3
        accepted[:, pos] = ~rejected
        viol[0][:, pos], viol[1][:, pos], viol[2][:, pos] = v1, v2, v3
        acc = ~rejected
        hr_count += (hr & acc).long()
        if is_high_perm is not None:
            hp_count += (hp & acc).long()
        if pos < M - 1:
            running_max_sim = torch.maximum(
                running_max_sim, sim_matrix[tool_idx] * acc.float().unsqueeze(1))

    _, sort_idx = (~accepted).long().sort(dim=1, stable=True)
    return orders.gather(1, sort_idx), accepted, viol[0], viol[1], viol[2]

RESULTS_DIR = ROOT / "results"
TOP_K_FIRST = 100
ALPHA = {"ultratool": 0.2, "sealtools": 0.02}
METRICS = ["ndcg@5", "sndcg@5", "mrr", "rvr@5", "srr@5", "success@5"]


def eval_rankings(rankings, samples, risk):
    acc = {m: 0.0 for m in METRICS}
    for ranked, s in zip(rankings, samples):
        c = s["correct"]
        acc["ndcg@5"] += ndcg(ranked, c)
        acc["sndcg@5"] += sndcg(ranked, c, risk)
        acc["mrr"] += mrr_fn(ranked, c)
        acc["rvr@5"] += rvr(ranked, risk)
        acc["srr@5"] += srr(ranked, risk)
        acc["success@5"] += success_at_k(ranked, c)
    n = len(samples)
    return {m: v / n for m, v in acc.items()}


def subgraph_smooth(raw_c, adj, cand, alpha, K):
    """LightGCN-style smoothing restricted to the candidate-induced subgraph,
    renormalised on that subgraph (paper: 'the corresponding normalized adjacency')."""
    sub = adj[cand.unsqueeze(2), cand.unsqueeze(1)]        # [Q,M,M]
    deg = sub.sum(-1)
    dinv = deg.pow(-0.5)
    dinv[deg == 0] = 0.0
    sub = sub * dinv.unsqueeze(2) * dinv.unsqueeze(1)
    if alpha == 0.0 or K == 0:
        return raw_c
    hops, h = [raw_c], raw_c
    subT = sub.transpose(1, 2)
    for _ in range(K):
        h = (1.0 - alpha) * raw_c + alpha * torch.bmm(h.unsqueeze(1), subT).squeeze(1)
        hops.append(h)
    return torch.stack(hops, dim=0).mean(dim=0)


def mean_std(dicts):
    out = {}
    for k in dicts[0]:
        v = [d[k] for d in dicts]
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        out[k] = {"mean": mu, "std": sd}
    return out


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, risk, tool_pool, tool_names, tool_descs = load_samples(args.dataset, split=args.split)
    Q_full = len(samples)
    all_queries = [s["query"] for s in samples]
    Q, N = len(samples), len(tool_names)
    alpha = args.alpha if args.alpha is not None else ALPHA[args.dataset]
    print(f"{args.dataset}/{args.split}: Q={Q} N={N} lam={args.lam} alpha={alpha} K={args.K}")

    # corrected Seal labels (32 previously un-adjudicated tools)
    risk_fixed = None
    if args.dataset == "sealtools" and args.fixed_labels:
        fix = json.load(open(args.fixed_labels, encoding="utf-8"))
        risk_fixed = dict(risk)
        n_ch = 0
        for k, v in fix.items():
            if k in risk_fixed and risk_fixed[k] != v:
                risk_fixed[k] = v
                n_ch += 1
        hi_before = sum(1 for v in risk.values() if v >= 3)
        hi_after = sum(1 for v in risk_fixed.values() if v >= 3)
        print(f"  corrected labels: {n_ch} tools changed; high-risk pool {hi_before} -> {hi_after}")

    graph_path = (ROOT / "data/sealtools/sealtools_risk_graph_4type.pt" if args.dataset == "sealtools"
                  else ROOT / "data/ultratool/risk_graph_4type.pt")
    graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
    adj, _ = build_adj_matrix(graph_data, device=device, dr_mode=False)
    risk_arr = torch.tensor([risk.get(n, 1) for n in tool_names], dtype=torch.float32, device=device)
    perm_raw = graph_data.get("perm_arr")
    perm_arr = perm_raw.float().to(device) if perm_raw is not None else None

    use_cos = USE_COS_REL[args.dataset]
    ckpts = args.checkpoints or [str(ROOT / c) for c in DEFAULT_CKPTS[args.dataset]]

    variants = ["A_full", "B_top100_fullgraph", "C_top100_subgraph"]
    per_seed = {f"{v}_{s}": [] for v in variants for s in ("core", "graph", "rule")}
    per_seed_fixed = {f"{v}_{s}": [] for v in variants for s in ("core", "graph", "rule")}
    sens_seed, fallback_seed = [], []

    for ckpt in ckpts:
        stem = Path(ckpt).stem
        print(f"  {Path(ckpt).name}")
        rel_head, risk_head, _ = load_heads(ckpt, device)
        tc, qc = resolve_cache("tool", stem, args.dataset, N), resolve_cache("qry", stem, args.dataset, N, Q_full)
        if tc.exists() and qc.exists():
            tool_embs = torch.load(tc, map_location=device, weights_only=False).float()
            qry_embs = torch.load(qc, map_location=device, weights_only=False).float()
        else:
            enc = build_encoder(ckpt, device)
            tool_embs, qry_embs = encode_texts(enc, tool_descs), encode_texts(enc, all_queries)
            torch.save(tool_embs.cpu(), tc); torch.save(qry_embs.cpu(), qc)
            tool_embs, qry_embs = tool_embs.to(device), qry_embs.to(device)
            del enc; torch.cuda.empty_cache()

        with torch.no_grad():
            f_risk = risk_head(tool_embs).squeeze(-1)
            t_norm, q_norm = F.normalize(tool_embs, dim=-1), F.normalize(qry_embs, dim=-1)
            cos_scores = q_norm @ t_norm.T
            if use_cos:
                f_rel = cos_scores
            else:
                f_rel = torch.zeros(Q, N, device=device)
                for s0 in range(0, N, 64):
                    s1 = min(s0 + 64, N)
                    te = tool_embs[s0:s1].unsqueeze(0).expand(Q, -1, -1)
                    qe = qry_embs.unsqueeze(1).expand(-1, s1 - s0, -1)
                    f_rel[:, s0:s1] = rel_head(torch.cat([qe, te], -1).reshape(-1, tool_embs.shape[1] * 2)).reshape(Q, s1 - s0)
            sim = t_norm @ t_norm.T
            raw = f_rel - args.lam * f_risk.unsqueeze(0)
            cand = cos_scores.topk(TOP_K_FIRST, dim=1).indices          # [Q,100]

            smoothed_full = smooth_scores_batch(raw, adj, alpha, K=args.K)

            orders_by_variant = {}
            # A: full pool
            orders_by_variant["A_full"] = (raw.argsort(1, descending=True),
                                           smoothed_full.argsort(1, descending=True))
            # B: top-100 mask over full-graph smoothing
            NEG = torch.finfo(raw.dtype).min
            mask_raw = torch.full_like(raw, NEG).scatter_(1, cand, raw.gather(1, cand))
            mask_sm = torch.full_like(smoothed_full, NEG).scatter_(1, cand, smoothed_full.gather(1, cand))
            orders_by_variant["B_top100_fullgraph"] = (
                mask_raw.argsort(1, descending=True)[:, :TOP_K_FIRST],
                mask_sm.argsort(1, descending=True)[:, :TOP_K_FIRST])
            # C: candidate-induced subgraph
            raw_c = raw.gather(1, cand)
            sm_c = subgraph_smooth(raw_c, adj, cand, alpha, args.K)
            orders_by_variant["C_top100_subgraph"] = (
                cand.gather(1, raw_c.argsort(1, descending=True)),
                cand.gather(1, sm_c.argsort(1, descending=True)))

            for vname, (o_core, o_graph) in orders_by_variant.items():
                o_rule, accepted, v1, v2, v3 = rule_filter_stats(o_graph, risk_arr, perm_arr, sim)
                for stage, o in (("core", o_core), ("graph", o_graph), ("rule", o_rule)):
                    rk = [[tool_names[i] for i in row] for row in o.cpu().tolist()]
                    per_seed[f"{vname}_{stage}"].append(eval_rankings(rk, samples, risk))
                    if risk_fixed is not None:
                        per_seed_fixed[f"{vname}_{stage}"].append(eval_rankings(rk, samples, risk_fixed))
                if vname == args.sens_variant:
                    k5 = 5
                    fallback_seed.append({
                        "fallback_rate": float((accepted.sum(1) < k5).float().mean()),
                        "pct_top5_changed": float(((~accepted[:, :k5]).sum(1) > 0).float().mean()),
                        "mean_deferred_top5": float((~accepted[:, :k5]).sum(1).float().mean()),
                        "rej_risk_cap": int(v1[:, :k5].sum()), "rej_perm_cap": int(v2[:, :k5].sum()),
                        "rej_redundancy": int(v3[:, :k5].sum())})
                    sens = {}
                    for cap in (1, 2):
                        for thr in (0.85, 0.90, 0.95):
                            o_s, acc_s, _, _, _ = rule_filter_stats(
                                o_graph, risk_arr, perm_arr, sim,
                                max_high_risk=cap, redundancy_thr=thr)
                            rk = [[tool_names[i] for i in row] for row in o_s.cpu().tolist()]
                            m = eval_rankings(rk, samples, risk)
                            m["fallback_rate"] = float((acc_s.sum(1) < 5).float().mean())
                            sens[f"cap{cap}_sim{thr}"] = m
                    sens_seed.append(sens)

        del tool_embs, qry_embs, f_rel, sim, raw, smoothed_full
        torch.cuda.empty_cache()

    out = {"config": {"dataset": args.dataset, "split": args.split, "lam": args.lam,
                      "alpha": alpha, "K": args.K, "top_k_first": TOP_K_FIRST,
                      "n_queries": Q, "n_tools": N, "seeds": len(ckpts),
                      "sens_variant": args.sens_variant},
           "variants": {k: mean_std(v) for k, v in per_seed.items() if v},
           "fallback": mean_std(fallback_seed) if fallback_seed else {},
           "threshold_sensitivity": {k: mean_std([s[k] for s in sens_seed])
                                     for k in sens_seed[0]} if sens_seed else {}}
    if risk_fixed is not None:
        out["variants_corrected_labels"] = {k: mean_std(v) for k, v in per_seed_fixed.items() if v}

    print(f"\n{'variant':<26}{'NDCG@5':>9}{'sNDCG@5':>10}{'MRR':>9}{'RVR@5':>9}{'SRR@5':>9}")
    for k, v in out["variants"].items():
        print(f"{k:<26}{v['ndcg@5']['mean']:>9.4f}{v['sndcg@5']['mean']:>10.4f}"
              f"{v['mrr']['mean']:>9.4f}{v['rvr@5']['mean']:>9.4f}{v['srr@5']['mean']:>9.4f}")
    if out.get("variants_corrected_labels"):
        print("\n-- same rankings, corrected Seal labels --")
        for k, v in out["variants_corrected_labels"].items():
            print(f"{k:<26}{v['ndcg@5']['mean']:>9.4f}{v['sndcg@5']['mean']:>10.4f}"
                  f"{v['mrr']['mean']:>9.4f}{v['rvr@5']['mean']:>9.4f}{v['srr@5']['mean']:>9.4f}")
    if out["threshold_sensitivity"]:
        print(f"\n-- rule-filter sensitivity on {args.sens_variant} --")
        print(f"{'setting':<20}{'NDCG@5':>9}{'RVR@5':>9}{'SRR@5':>9}{'fallback':>10}")
        for k, v in out["threshold_sensitivity"].items():
            print(f"{k:<20}{v['ndcg@5']['mean']:>9.4f}{v['rvr@5']['mean']:>9.4f}"
                  f"{v['srr@5']['mean']:>9.4f}{v['fallback_rate']['mean']:>10.4f}")

    tag = f"{args.dataset}_{args.split}" if args.dataset == "sealtools" else args.dataset
    p = RESULTS_DIR / f"topk_probe_{tag}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ultratool", choices=["ultratool", "sealtools"])
    ap.add_argument("--split", default="test_in", choices=["test_in", "test_out"])
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--checkpoints", nargs="+", default=None)
    ap.add_argument("--fixed-labels", default=None)
    ap.add_argument("--sens-variant", default="A_full")
    run(ap.parse_args())
