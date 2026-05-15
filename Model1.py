import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Feature Extractors (The "Readers")
# ==========================================
class DrugTokenizer(nn.Module):
    def __init__(self, in_dim=384, out_dim=256, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim * num_tokens),
            nn.LayerNorm(out_dim * num_tokens),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        out = self.fc(x)
        return out.view(-1, self.num_tokens, self.out_dim)


class ProtTokenizer(nn.Module):
    def __init__(self, in_dim=1280, out_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(x)


# ==========================================
# 2. Co-Attention (The "Handshake")
# ==========================================
class AdvancedCoAttentionBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.channel_gate = nn.Parameter(torch.ones(1, 1, dim))
        self.tau = nn.Parameter(torch.tensor([5.0]))

        self.post_norm_d = nn.LayerNorm(dim)
        self.post_norm_p = nn.LayerNorm(dim)
        self.refine_proj_d = nn.Linear(dim, dim)
        self.refine_proj_p = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.ffn_d = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ffn_p = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_ffn_d = nn.LayerNorm(dim)
        self.norm_ffn_p = nn.LayerNorm(dim)

    def forward(self, F_A, F_B):
        F_A_norm = F.normalize(F_A, dim=-1)
        F_B_norm = F.normalize(F_B, dim=-1)

        S2_cosine = torch.matmul(F_A_norm, F_B_norm.transpose(-1, -2)) * self.tau

        attn_A = F.softmax(torch.tanh(S2_cosine), dim=-1)
        attn_B = F.softmax(torch.tanh(S2_cosine.transpose(-1, -2)), dim=-1)

        Z_A = torch.matmul(attn_A, F_B)
        Z_B = torch.matmul(attn_B, F_A)

        gate = torch.sigmoid(self.channel_gate)
        out_A = F_A + self.alpha * self.refine_proj_d(self.post_norm_d(Z_A * gate))
        out_B = F_B + self.alpha * self.refine_proj_p(self.post_norm_p(Z_B * gate))

        out_A = self.norm_ffn_d(out_A + self.ffn_d(out_A))
        out_B = self.norm_ffn_p(out_B + self.ffn_p(out_B))
        return out_A, out_B

class CoAffDTI(nn.Module):
    def __init__(self, drug_dim=384, prot_dim=1280, hidden_dim=256, num_drug_tokens=8):   # 256
        super().__init__()

        self.d_proj = DrugTokenizer(drug_dim, hidden_dim, num_drug_tokens)
        self.p_proj = ProtTokenizer(prot_dim, hidden_dim)
        self.co_attn = AdvancedCoAttentionBlock(hidden_dim)

        fuse_dim = hidden_dim * 4

        # The MoE Layer replaces the standard MLP classification head
        self.mlp = nn.Sequential(
            nn.Linear(fuse_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # <--- 改成 LayerNorm
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def _fuse_with_affinity_gating(self, d_ref, p_ref):
        sim_raw = torch.matmul(F.normalize(d_ref, dim=-1), F.normalize(p_ref, dim=-1).transpose(-1, -2))
        aff_drug = sim_raw.max(dim=-1, keepdim=True)[0]
        aff_prot = sim_raw.max(dim=-2, keepdim=True)[0].transpose(-1, -2)

        d_gated = d_ref * (1 + torch.tanh(aff_drug))
        p_gated = p_ref * (1 + torch.tanh(aff_prot))

        d_pool = d_gated.mean(dim=1)
        p_pool = p_gated.mean(dim=1)
        return torch.cat([d_pool, p_pool, d_pool * p_pool, torch.abs(d_pool - p_pool)], dim=1)


    def forward(self, drug, prot):
        # Feature Extraction
        d_feat = self.d_proj(drug)
        p_feat = self.p_proj(prot)

        # Interaction
        d_att, p_att = self.co_attn(d_feat, p_feat)
        z_shared = self._fuse_with_affinity_gating(d_att, p_att)
        # 💡 逐层通过 MLP，拦截倒数第一层之前的特征
        hidden_feat = z_shared
        for layer in self.mlp[:-1]:  # 遍历除了最后一层 Linear 之外的所有层
            hidden_feat = layer(hidden_feat)
        # MoE Prediction
        # pred_logits, routing_logits = self.moe_predictor(z_shared)
        pred_logits = self.mlp(z_shared)
        return pred_logits, hidden_feat

