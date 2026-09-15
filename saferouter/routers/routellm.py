"""RouteLLM: a per-model latent vector scored against the projected query."""
import torch.nn as nn
import torch.nn.functional as F

from cost import N_MODELS


class BilinearMF(nn.Module):
    def __init__(self, n_models=N_MODELS, dim=128, text_dim=1024):
        super().__init__()
        self.n_models = n_models
        self.P = nn.Embedding(n_models, dim)
        self.text_proj = nn.Linear(text_dim, dim, bias=False)
        self.classifier = nn.Linear(dim, 1, bias=False)

    def project_text(self, q_emb):
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        return self.text_proj(q_emb)

    def forward(self, model_win, model_loss, q_proj):
        v_win = F.normalize(self.P(model_win), p=2, dim=-1)
        v_loss = F.normalize(self.P(model_loss), p=2, dim=-1)
        h = v_win - v_loss
        if q_proj.dim() == 1:
            q_proj = q_proj.unsqueeze(0)
        interaction = h * q_proj
        logit = self.classifier(interaction).squeeze(-1)
        return logit

    def score_all(self, q_proj):
        """Scores for all models, from a pre-projected embedding."""
        P_all = F.normalize(self.P.weight, p=2, dim=-1)  # (n_models, dim)
        if q_proj.dim() == 1:
            interaction = P_all * q_proj.unsqueeze(0)  # (n_models, dim)
        else:
            interaction = P_all.unsqueeze(0) * q_proj.unsqueeze(1)  # (batch, n_models, dim)
        logits = self.classifier(interaction).squeeze(-1)
        return logits
