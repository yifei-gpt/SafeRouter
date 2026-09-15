"""IRT-Router (MIRT): P(correct) = sigmoid(a_q . theta_m - b_q), model ability
theta from the model-profile embedding, a_q/b_q from the query."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cost import MODELS, N_MODELS
from data_io import CKPT_DIR, EMB_LLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MIRTNet(nn.Module):
    def __init__(self, llm_dim=1024, query_dim=1024, latent=25):
        super().__init__()
        self.theta_proj = nn.Linear(llm_dim, latent, bias=False)
        self.a_proj = nn.Linear(query_dim, latent, bias=False)
        self.b_proj = nn.Linear(query_dim, 1, bias=False)

    def forward(self, llm_emb, query_emb):
        theta = self.theta_proj(llm_emb)
        a = F.softplus(self.a_proj(query_emb))
        b = self.b_proj(query_emb).squeeze(-1)
        return torch.sigmoid((a * theta).sum(-1) - b)

def load_mirt(device=DEVICE):
    """The trained net plus the model-profile matrix it scores against."""
    net = MIRTNet().to(device).eval()
    net.load_state_dict(torch.load(CKPT_DIR / "mirt.pt", map_location=device,
                                   weights_only=False))
    prof = torch.load(EMB_LLM, map_location="cpu", weights_only=False)
    return net, torch.stack([prof[m] for m in MODELS]).float().to(device)

def irt_quality(net, llm, x):
    """P(correct) for every model -> (batch, n_models)."""
    return torch.stack([net(llm[i].expand(len(x), -1), x) for i in range(N_MODELS)], 1)
