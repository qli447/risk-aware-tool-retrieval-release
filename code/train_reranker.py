"""
train_reranker.py
-----------------
Train the RiskAwareReranker (frozen encoder + two MLP heads).

Loss:
  L = L_rel + mu * L_risk
  L_rel  = MarginRankingLoss(f_rel(q,pos), f_rel(q,neg), margin=0.1)
  L_risk = MSELoss(f_risk(t), (risk_level - 1) / 4)

Usage:
  python code/train_reranker.py --dataset ultratool --epochs 10 --mu 0.5
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import RiskAwareReranker

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def build_embed_cache_path(cache_dir, dataset, encoder_name, n_triplets):
    enc_safe = encoder_name.replace("/", "__").replace("\\", "__")
    fname = f"{dataset}__{enc_safe}__{n_triplets}.pt"
    return Path(cache_dir) / fname


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_risk_labels(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {name: info["risk_level"] for name, info in raw.items()}


def risk_norm(level):
    """Normalize risk_level 1-5 to 0-1."""
    return (level - 1) / 4.0


# ---------------------------------------------------------------------------
# UltraTool sample construction
# ---------------------------------------------------------------------------

def build_ultratool_triplets(negs_per_query=5, seed=42):
    """
    Load dev.json (4824 queries, completely disjoint from test_set) and build triplets.
    dev.json format: {question, plan, tools, domain}
      query    = question field
      pos_tools = all tool names in the tools list (tools actually used for this task)
      neg_tools = random sample from full 2032-tool corpus not in pos_tools
    """
    random.seed(seed)

    risk_path  = ROOT / "data/ultratool/ultratool_risk_labels_resolved.json"
    tools_path = ROOT / "data/ultratool/ultratool_tools_with_desc.json"
    data_path  = ROOT / "data/ultratool/data/English-dataset/dev.json"

    risk_labels = load_risk_labels(risk_path)

    with open(tools_path, encoding="utf-8") as f:
        tool_pool = json.load(f)
    tool_map = {t["name"]: t["description"] for t in tool_pool}

    records = load_jsonl(data_path)

    all_tool_names = list(tool_map.keys())

    triplets = []
    skipped = 0
    for rec in records:
        # dev.json: query is the question field
        query = rec.get("question", "").strip()
        if not query:
            skipped += 1
            continue

        # dev.json: tools field is the list of tool dicts used in this task
        tools_field = rec.get("tools", [])
        if isinstance(tools_field, list):
            pos_names = [t["name"] for t in tools_field if isinstance(t, dict) and t.get("name")]
        else:
            pos_names = []
        pos_names = [n for n in pos_names if n in tool_map]
        if not pos_names:
            skipped += 1
            continue

        neg_pool = [n for n in all_tool_names if n not in set(pos_names)]
        if not neg_pool:
            skipped += 1
            continue

        neg_names = random.sample(neg_pool, min(negs_per_query * len(pos_names), len(neg_pool)))

        for pos in pos_names:
            chosen_negs = random.sample(neg_names, min(negs_per_query, len(neg_names)))
            for neg in chosen_negs:
                triplets.append({
                    "query": query,
                    "pos_desc": tool_map[pos],
                    "neg_desc": tool_map[neg],
                    "pos_risk": risk_norm(risk_labels.get(pos, 1)),
                    "neg_risk": risk_norm(risk_labels.get(neg, 1)),
                })

    print(f"UltraTool 训练三元组: {len(triplets)} 条 (跳过 {skipped} 条无效样本)")
    return triplets


# ---------------------------------------------------------------------------
# ATBench sample construction
# ---------------------------------------------------------------------------

def build_atbench_triplets(negs_per_query=5, seed=42):
    """
    Load ATBench from HuggingFace cache and build triplets.
    Positive tools from tool_used; negatives sampled from corpus.
    """
    random.seed(seed)

    try:
        from datasets import load_dataset
    except ImportError:
        print("请安装: pip install datasets")
        sys.exit(1)

    risk_path = ROOT / "data/atbench/atbench_risk_labels_resolved.json"
    cache_dir  = ROOT / "data/atbench"

    risk_labels = load_risk_labels(risk_path)

    print("加载 ATBench 数据集...")
    ds = load_dataset("AI45Research/ATBench", "ATBench", split="test", cache_dir=str(cache_dir))

    # Build full tool corpus from dataset
    tool_map = {}
    for row in ds:
        for tool in row["tool_used"]:
            name = tool.get("name", "").strip()
            desc = tool.get("description", "").strip()
            if name and name not in tool_map:
                tool_map[name] = desc

    all_tool_names = list(tool_map.keys())
    print(f"ATBench 工具库: {len(all_tool_names)} 个")

    triplets = []
    skipped = 0
    for row in ds:
        contents = row["contents"]
        if not contents:
            skipped += 1
            continue
        first = contents[0]
        if isinstance(first, dict):
            query = first.get("content", "")
        elif isinstance(first, list):
            user_turns = [m for m in first if isinstance(m, dict) and m.get("role") == "user"]
            query = user_turns[0]["content"] if user_turns else ""
        else:
            query = str(first)
        if not query:
            skipped += 1
            continue

        pos_names = [t["name"] for t in row["tool_used"] if t.get("name") and t["name"] in tool_map]
        if not pos_names:
            skipped += 1
            continue

        neg_pool = [n for n in all_tool_names if n not in set(pos_names)]
        if not neg_pool:
            skipped += 1
            continue

        neg_names = random.sample(neg_pool, min(negs_per_query * len(pos_names), len(neg_pool)))

        for pos in pos_names:
            chosen_negs = random.sample(neg_names, min(negs_per_query, len(neg_names)))
            for neg in chosen_negs:
                triplets.append({
                    "query": query,
                    "pos_desc": tool_map[pos],
                    "neg_desc": tool_map[neg],
                    "pos_risk": risk_norm(risk_labels.get(pos, 1)),
                    "neg_risk": risk_norm(risk_labels.get(neg, 1)),
                })

    print(f"ATBench 训练三元组: {len(triplets)} 条 (跳过 {skipped} 条无效样本)")
    return triplets


# ---------------------------------------------------------------------------
# Seal-Tools sample construction
# ---------------------------------------------------------------------------

def build_sealtools_triplets(negs_per_query=5, seed=42):
    """
    Load Seal-Tools train.jsonl and build triplets.
    query    = query field
    pos_tool = correct_tools list (tools called in this instance)
    neg_tool = random sample from full 4076-tool corpus not in correct_tools
    """
    random.seed(seed)

    risk_path  = ROOT / "data/sealtools/sealtools_risk_labels_resolved.json"
    tools_path = ROOT / "data/sealtools/sealtools_tools.json"
    data_path  = ROOT / "data/sealtools/sealtools_train.jsonl"

    risk_labels = load_risk_labels(risk_path)
    # handle None risk_levels (needs_adjudication → default 2)
    risk_labels = {k: (v if v is not None else 2) for k, v in risk_labels.items()}

    with open(tools_path, encoding="utf-8") as f:
        tool_pool = json.load(f)
    tool_map = {t["name"]: t["description"] for t in tool_pool}
    all_tool_names = list(tool_map.keys())

    records = load_jsonl(data_path)

    triplets = []
    skipped = 0
    for rec in records:
        query = rec.get("query", "").strip()
        if not query:
            skipped += 1
            continue

        pos_names = [t for t in rec.get("correct_tools", []) if t in tool_map]
        if not pos_names:
            skipped += 1
            continue

        neg_pool = [n for n in all_tool_names if n not in set(pos_names)]
        if not neg_pool:
            skipped += 1
            continue

        neg_names = random.sample(neg_pool, min(negs_per_query * len(pos_names), len(neg_pool)))

        for pos in pos_names:
            chosen_negs = random.sample(neg_names, min(negs_per_query, len(neg_names)))
            for neg in chosen_negs:
                triplets.append({
                    "query": query,
                    "pos_desc": tool_map[pos],
                    "neg_desc": tool_map[neg],
                    "pos_risk": risk_norm(risk_labels.get(pos, 1)),
                    "neg_risk": risk_norm(risk_labels.get(neg, 1)),
                })

    print(f"Seal-Tools 训练三元组: {len(triplets)} 条 (跳过 {skipped} 条无效样本)")
    return triplets


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, triplets, model, cache_path=None):
        self.triplets = triplets
        n = len(triplets)

        if cache_path and Path(cache_path).exists():
            print(f"  从缓存加载 embedding: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.q_embs   = cached["q_embs"]
            self.pos_embs = cached["pos_embs"]
            self.neg_embs = cached["neg_embs"]
        else:
            texts = (
                [t["query"]    for t in triplets]
                + [t["pos_desc"] for t in triplets]
                + [t["neg_desc"] for t in triplets]
            )
            print(f"  编码 {3 * n} 条文本 (query + pos + neg)...")
            all_embs = model.encode(texts, show_progress=True)
            # Store on CPU: saves GPU VRAM; batch transfer to device happens in the training loop.
            self.q_embs   = all_embs[:n].cpu()
            self.pos_embs = all_embs[n : 2 * n].cpu()
            self.neg_embs = all_embs[2 * n :].cpu()
            if cache_path:
                p = Path(cache_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"q_embs": self.q_embs, "pos_embs": self.pos_embs, "neg_embs": self.neg_embs},
                    p,
                )
                print(f"  Embedding 缓存已保存 → {p}")

        self.pos_risks = torch.tensor([t["pos_risk"] for t in triplets], dtype=torch.float32)
        self.neg_risks = torch.tensor([t["neg_risk"] for t in triplets], dtype=torch.float32)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return (
            self.q_embs[idx],
            self.pos_embs[idx],
            self.neg_embs[idx],
            self.pos_risks[idx],
            self.neg_risks[idx],
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def val_loss(model, val_dataset, alpha, batch_size=256):
    """在 val 集上计算 loss（不更新参数），用于 best-checkpoint 选择。"""
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    margin_loss = nn.MarginRankingLoss(margin=0.1)
    mse_loss    = nn.MSELoss()
    device = model.device
    model.eval()
    total = 0.0
    with torch.no_grad():
        for q_emb, pos_emb, neg_emb, pos_risk, neg_risk in loader:
            q_emb    = q_emb.to(device)
            pos_emb  = pos_emb.to(device)
            neg_emb  = neg_emb.to(device)
            pos_risk = pos_risk.to(device)
            neg_risk = neg_risk.to(device)
            pos_rel = model.score_rel(q_emb, pos_emb)
            neg_rel = model.score_rel(q_emb, neg_emb)
            l_rel   = margin_loss(pos_rel, neg_rel, torch.ones(len(pos_rel), device=device))
            l_risk  = mse_loss(
                torch.cat([model.score_risk(pos_emb), model.score_risk(neg_emb)]),
                torch.cat([pos_risk, neg_risk]),
            )
            total += (l_rel + alpha * l_risk).item()
    return total / len(loader)


def train(model, triplets, epochs, alpha, lr, batch_size, save_path, cache_path=None, val_ratio=0.1):
    print(f"\n开始训练: epochs={epochs}, alpha={alpha}, lr={lr}, batch_size={batch_size}")
    print(f"三元组总数: {len(triplets)}")

    # 先编码全量（可复用 cache，对所有 seed 都有效），再按 index 切 train/val
    print("预计算 embedding (冻结 encoder 只做一次)...")
    full_dataset = TripletDataset(triplets, model, cache_path=cache_path)

    n = len(triplets)
    n_val   = max(1, int(n * val_ratio))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]
    print(f"训练集: {len(trn_idx)}  验证集: {n_val}  (val_ratio={val_ratio})")

    class IndexDataset(Dataset):
        def __init__(self, base, idx): self.base = base; self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i): return self.base[self.idx[i]]

    trn_dataset = IndexDataset(full_dataset, trn_idx)
    val_dataset = IndexDataset(full_dataset, val_idx)
    loader      = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.rel_head.parameters()) + list(model.risk_head.parameters()), lr=lr
    )
    margin_loss = nn.MarginRankingLoss(margin=0.1)
    mse_loss    = nn.MSELoss()
    device = model.device

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_rel = total_risk = total_margin = 0.0
        n_batches = len(loader)

        bar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs}", unit="batch", leave=False) if use_tqdm else loader
        for q_emb, pos_emb, neg_emb, pos_risk, neg_risk in bar:
            q_emb    = q_emb.to(device)
            pos_emb  = pos_emb.to(device)
            neg_emb  = neg_emb.to(device)
            pos_risk = pos_risk.to(device)
            neg_risk = neg_risk.to(device)
            pos_rel = model.score_rel(q_emb, pos_emb)
            neg_rel = model.score_rel(q_emb, neg_emb)
            l_rel   = margin_loss(pos_rel, neg_rel, torch.ones(len(pos_rel), device=device))

            pos_risk_pred = model.score_risk(pos_emb)
            neg_risk_pred = model.score_risk(neg_emb)
            l_risk = mse_loss(
                torch.cat([pos_risk_pred, neg_risk_pred]),
                torch.cat([pos_risk, neg_risk]),
            )

            loss = l_rel + alpha * l_risk
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss   += loss.item()
            total_rel    += l_rel.item()
            total_risk   += l_risk.item()
            total_margin += (pos_rel - neg_rel).mean().item()
            if use_tqdm:
                bar.set_postfix(loss=f"{loss.item():.4f}")

        v_loss = val_loss(model, val_dataset, alpha)
        is_best = v_loss < best_val
        if is_best:
            best_val   = v_loss
            best_epoch = epoch
            model.save(save_path)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss={total_loss/n_batches:.4f}  "
            f"L_rel={total_rel/n_batches:.4f}  "
            f"L_risk={total_risk/n_batches:.4f}  "
            f"margin={total_margin/n_batches:+.4f}  "
            f"val={v_loss:.4f}" + (" ← best" if is_best else "")
        )

    print(f"\nBest checkpoint: epoch {best_epoch}  val_loss={best_val:.4f}")
    print(f"Checkpoint 保存 → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ultratool", "atbench", "sealtools"])
    parser.add_argument("--encoder", default="mangopy/ToolRet-trained-bge-large-en-v1.5")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--mu", "--alpha", dest="mu", type=float, default=0.5,
        help="Risk-loss weight in L_rel + mu * L_risk. --alpha is retained as a legacy alias.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--negs-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save",
        default=None,
        help="checkpoint 保存路径 (默认: code/checkpoints/reranker_<dataset>.pt)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Embedding 缓存目录 (默认: code/embed_cache)。相同 dataset+encoder+n_triplets 时直接复用，跳过重新编码。",
    )
    args = parser.parse_args()

    save_path = args.save or str(ROOT / f"code/checkpoints/reranker_{args.dataset}.pt")

    print(f"初始化模型: encoder={args.encoder}, hidden={args.hidden}")
    model = RiskAwareReranker(encoder_name=args.encoder, hidden=args.hidden)

    if args.dataset == "ultratool":
        triplets = build_ultratool_triplets(negs_per_query=args.negs_per_query, seed=args.seed)
    elif args.dataset == "sealtools":
        triplets = build_sealtools_triplets(negs_per_query=args.negs_per_query, seed=args.seed)
    else:
        triplets = build_atbench_triplets(negs_per_query=args.negs_per_query, seed=args.seed)

    if not triplets:
        print("没有训练数据，退出。")
        sys.exit(1)

    cache_dir = args.cache_dir or str(ROOT / "code/embed_cache")
    cache_path = build_embed_cache_path(cache_dir, args.dataset, args.encoder, len(triplets))

    train(
        model=model,
        triplets=triplets,
        epochs=args.epochs,
        alpha=args.mu,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=save_path,
        cache_path=cache_path,
    )


if __name__ == "__main__":
    main()
