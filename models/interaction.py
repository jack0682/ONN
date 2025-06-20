import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCausalAttention(nn.Module):
    def __init__(self, d_model, d_relation=32, d_hidden=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_relation = d_relation

        # Linear projections for attention
        self.W_q = nn.Linear(d_model, d_hidden)
        self.W_k = nn.Linear(d_model, d_hidden)
        self.W_r = nn.Linear(d_relation, d_hidden)

        # Interaction fusion
        self.fuse = nn.Linear(d_model * 2 + d_relation + d_model, d_model)  # 더 많은 정보를 fuse
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H, R, delta_H):
        """
        H: (B, N, T, D) - N objects의 의미 흐름
        R: (B, N, N, d_relation) - 객체 간 관계 텐서
        delta_H: (B, N, T, D) - 의미 텐서 변화율
        return Z: (B, N, T, D)
        """
        B, N, T, D = H.shape
        device = H.device

        H_q = self.W_q(H)  # (B, N, T, H)
        H_k = self.W_k(H)  # (B, N, T, H)

        Z_out = []

        for i in range(N):
            q_i = H_q[:, i, :, :]            # (B, T, H)
            h_i = H[:, i, :, :]              # (B, T, D)

            z_t_all = []

            for j in range(N):
                if i == j:
                    continue

                k_j = H_k[:, j, :, :]        # (B, T, H)
                h_j = H[:, j, :, :]          # (B, T, D)
                r_ij = R[:, i, j, :]         # (B, d_relation)
                delta_h_j = delta_H[:, j, :, :]  # (B, T, D)

                r_expand = self.W_r(r_ij).unsqueeze(1)  # (B, 1, H)

                # attention score (B, T)
                attn_score = torch.sum(q_i * (k_j + r_expand), dim=-1) / (self.d_model ** 0.5)
                attn_weight = F.softmax(attn_score, dim=1).unsqueeze(-1)  # (B, T, 1)

                # concat(H_i, H_j, R_ij, delta_H_j) for change rate influence
                r_broadcast = r_ij.unsqueeze(1).repeat(1, T, 1)  # (B, T, d_relation)
                delta_broadcast = delta_h_j.sum(dim=1).unsqueeze(1)  # (B, 1, T, D)
                z_in = torch.cat([h_i, h_j, r_broadcast, delta_broadcast], dim=-1)  # (B, T, 2D + d_r + D)

                z_fused = self.fuse(z_in)  # (B, T, D)
                z_attended = attn_weight * z_fused  # (B, T, D)
                z_t_all.append(z_attended)

            z_sum = torch.sum(torch.stack(z_t_all, dim=0), dim=0)  # sum over j≠i → (B, T, D)
            z_out = self.norm(h_i + self.dropout(z_sum))  # residual + norm
            Z_out.append(z_out)

        Z_out = torch.stack(Z_out, dim=1)  # (B, N, T, D)
        return Z_out
