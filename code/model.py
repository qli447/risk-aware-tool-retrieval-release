"""
model.py
--------
RiskAwareReranker: frozen encoder + two trained MLP heads.

  score(q, t) = f_rel(q, t) - lambda * f_risk(t)

  f_rel = MLP_rel(cat(E(q), E(t)))   -- relevance head
  f_risk = MLP_risk(E(t))             -- risk head
"""

import torch
import torch.nn as nn


class RiskAwareReranker(nn.Module):
    def __init__(self, encoder_name="mangopy/ToolRet-trained-bge-large-en-v1.5", hidden=64, emb_dim=None):
        super().__init__()
        self.encoder_name = encoder_name

        # Freeze encoder at load time — only MLP heads are trained
        from sentence_transformers import SentenceTransformer
        # Prefer local cache to avoid httpx closed-client errors across sequential subprocesses
        # (huggingface_hub >= 0.23 uses httpx which can fail on re-use after close)
        try:
            self._encoder = SentenceTransformer(encoder_name, local_files_only=True)
        except Exception:
            self._encoder = SentenceTransformer(encoder_name)
        for p in self._encoder.parameters():
            p.requires_grad = False

        # Auto-detect embedding dim from encoder (handles MiniLM=384, BGE-large=1024, etc.)
        if emb_dim is None:
            emb_dim = self._encoder.get_sentence_embedding_dimension()
        self.emb_dim = emb_dim

        self.rel_head = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # Move MLP heads to the same device as the encoder so that
        # encode() output (CUDA on GPU servers) can flow directly into them.
        try:
            enc_device = next(self._encoder.parameters()).device
        except StopIteration:
            enc_device = torch.device("cpu")
        self.rel_head  = self.rel_head.to(enc_device)
        self.risk_head = self.risk_head.to(enc_device)

    @property
    def device(self):
        return next(self.rel_head.parameters()).device

    def encode(self, texts, batch_size=128, show_progress=False):
        """Encode texts to embeddings using the frozen encoder."""
        with torch.no_grad():
            embs = self._encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        # Cast to float32 — some encoders (e.g. bge-large) output float16
        return embs.float()

    def score_rel(self, q_emb, t_emb):
        """q_emb and t_emb: (N, emb_dim) tensors. Returns (N,)."""
        q_emb = q_emb.float()
        t_emb = t_emb.float()
        return self.rel_head(torch.cat([q_emb, t_emb], dim=-1)).squeeze(-1)

    def score_risk(self, t_emb):
        """t_emb: (N, emb_dim). Returns (N,)."""
        return self.risk_head(t_emb.float()).squeeze(-1)

    def score(self, q_emb, t_emb, lam):
        """Combined score for a single query against N tools.

        q_emb: (N, emb_dim) — query emb repeated N times
        t_emb: (N, emb_dim) — one embedding per tool
        Returns: (N,) final scores
        """
        return self.score_rel(q_emb, t_emb) - lam * self.score_risk(t_emb)

    def save(self, path):
        torch.save(
            {
                "encoder_name": self.encoder_name,
                "hidden": self.rel_head[0].out_features,
                "emb_dim": self.emb_dim,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(
            encoder_name=ckpt["encoder_name"],
            hidden=ckpt["hidden"],
            emb_dim=ckpt["emb_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
