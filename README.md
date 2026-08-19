# Risk-Aware Reranking for Agentic Tool Retrieval

This repository contains the code and resolved tool-risk labels used in our
CIKM 2026 paper, *Risk-Aware Reranking for Agentic Tool Retrieval*.

## Method components

The learned model has exactly two heads:

- `f_rel(q, t)`: a query-conditioned relevance head.
- `f_risk(t)`: a tool-level exposure-risk head.

ToolGraph smoothing and the optional rule filter are parameter-free inference
steps. The rule filter implements the three constraints in Appendix C: the
risk cap, permission cap, and similarity-based redundancy constraint.

## Candidate-pool settings

The release contains both evaluation settings:

- `code/run_graph_smooth.py` scores the complete tool registry. This is the
  end-to-end deployment configuration reported for our method in the main
  table.
- `code/topk_probe.py` first selects the ToolRet-BGE top-100 candidates and then
  applies the same trained heads, candidate-induced ToolGraph smoothing, and
  optional rule filter. This is the candidate-matched comparison reported in
  the appendix.

The two settings use the same checkpoints and do not change the architecture.

### UltraTool top-100 control

The table below reports NDCG@5 over three random seeds when our method uses the
same ToolRet-BGE top-100 candidates as the reranking baselines.

| Variant | NDCG@5 |
|---|---:|
| Core | 0.609 +/- 0.029 |
| Core + Graph | 0.621 +/- 0.024 |
| Core + Graph + Rule | 0.573 +/- 0.032 |

### Standard deviations

The following values are standard deviations over three random seeds. The
corresponding means are reported in the paper.

Main results:

| Dataset | Variant | NDCG@5 | MRR | sNDCG@5 | RVR@5 | SRR@5 |
|---|---|---:|---:|---:|---:|---:|
| UltraTool | Core + Graph | 0.039 | 0.036 | 0.046 | 0.028 | 0.027 |
| UltraTool | Core + Graph + Rule | 0.047 | 0.042 | 0.060 | 0.011 | 0.012 |
| Seal-Tools test-in | Core + Graph | 0.005 | 0.001 | 0.004 | 0.004 | 0.001 |
| Seal-Tools test-in | Core + Graph + Rule | 0.003 | 0.001 | 0.002 | 0.003 | 0.001 |

UltraTool ablation results:

| Variant | NDCG@5 | MRR | sNDCG@5 | RVR@5 | SRR@5 |
|---|---:|---:|---:|---:|---:|
| Risk-label penalty | 0.003 | 0.002 | 0.002 | 0.039 | 0.024 |
| Trained relevance head | 0.048 | 0.047 | 0.052 | 0.017 | 0.022 |
| Trained risk head | 0.046 | 0.041 | 0.052 | 0.029 | 0.028 |
| Graph smoothing | 0.039 | 0.036 | 0.046 | 0.028 | 0.027 |
| Rule filter | 0.047 | 0.042 | 0.060 | 0.011 | 0.012 |

UltraTool stress-test results:

| Variant | NDCG@5 | RVR@5 | SRR@5 |
|---|---:|---:|---:|
| ToolRet-BGE | 0.000 | 0.000 | 0.000 |
| Graph smoothing | 0.008 | 0.097 | 0.081 |
| Rule filter | 0.009 | 0.027 | 0.019 |

## Contents

- `code/model.py`: frozen-encoder dual-head reranker.
- `code/train_reranker.py`: training for the relevance and risk heads.
- `code/build_skill_graph.py`: construction of the four-type ToolGraph.
- `code/run_graph_smooth.py`: full-pool deployment evaluation.
- `code/topk_probe.py`: controlled full-pool/top-100 comparison and rule-filter
  sensitivity.
- `code/camera_ready_stats.py`: per-query uncertainty, significance, filter
  activation, and timing statistics.
- `code/run_robustness_eval.py`: subset and robustness evaluation utilities.
- `code/requirements.txt`: Python dependencies.
- `results/topk_probe_*.json`: the exact three-seed candidate-pool control
  outputs, including the rule-filter sensitivity runs.
- `data/ultratool/ultratool_risk_labels_resolved.json`: UltraTool risk labels.
- `data/sealtools/sealtools_risk_labels_resolved.json`: Seal-Tools risk labels.

## Data layout

The benchmark query and tool files are public upstream data and are not
redistributed here. Download the benchmark resources from the
[ToolRet repository](https://github.com/mangopy/tool-retrieval-benchmark).
The original dataset repositories are
[UltraTool](https://github.com/JoeYing1019/UltraTool) and
[Seal-Tools](https://github.com/fairyshine/Seal-Tools). Place the prepared
files under the following paths:

```text
data/
  ultratool/
    ultratool_risk_labels_resolved.json
    ultratool_tools_with_desc.json
    risk_graph_4type.pt
    data/English-dataset/dev.json
    data/English-dataset/test_set/tool_selection.json
  sealtools/
    sealtools_risk_labels_resolved.json
    sealtools_tools.json
    sealtools_train.jsonl
    sealtools_test_in.jsonl
    sealtools_risk_graph_4type.pt
```

The graph files are generated locally with `build_skill_graph.py`.

## Reproduction

Install dependencies and train the lightweight heads:

```bash
pip install -r code/requirements.txt
python code/train_reranker.py --dataset ultratool --epochs 10 --mu 0.5
python code/build_skill_graph.py --dataset ultratool --threshold 2 --sem-threshold 0.65
```

Evaluate the main full-pool configuration:

```bash
python code/run_graph_smooth.py \
  --dataset ultratool \
  --checkpoints code/checkpoints/reranker_ultratool_seed42.pt \
                code/checkpoints/reranker_ultratool_seed123.pt \
                code/checkpoints/reranker_ultratool_seed777.pt \
  --lambda-values 0.1 \
  --alpha-values 0.2 \
  --K 1 \
  --tag ultratool_main
```

Run the candidate-pool control with the same checkpoints:

```bash
python code/topk_probe.py \
  --dataset ultratool \
  --checkpoints code/checkpoints/reranker_ultratool_seed42.pt \
                code/checkpoints/reranker_ultratool_seed123.pt \
                code/checkpoints/reranker_ultratool_seed777.pt \
  --lam 0.1 \
  --alpha 0.2 \
  --K 1
```

For Seal-Tools, use `--dataset sealtools` and the paper's graph-smoothing value
`--alpha 0.02` (or `--alpha-values 0.02` for `run_graph_smooth.py`). Multiple
checkpoint paths reproduce the reported three-seed means and standard
deviations.

By default, the scripts use
`mangopy/ToolRet-trained-bge-large-en-v1.5`, the frozen ToolRet-BGE encoder in
the paper.

## Risk labels

The released labels use five ordinal levels. The distribution is:

| Dataset | Tools | L1 | L2 | L3 | L4 | L5 | L3-L5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltraTool | 2,032 | 958 | 799 | 155 | 115 | 5 | 275 |
| Seal-Tools | 4,076 | 2,807 | 1,197 | 51 | 20 | 1 | 72 |

The annotation protocol, LLM agreement statistics, and independent human
audit are reported in the paper.
