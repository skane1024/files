import torch

torch.manual_seed(456)

# ========== 基本配置 ==========
N      = 16        # Q 的 token 长度
N_kv   = 2 * N     # KV cache 长度是 Q 的 2 倍（含历史）
h_q    = 8         # Q 的头数
h_kv   = 4         # KV 的头数
d_head = 8         # 每个头的维度

assert h_q % h_kv == 0, "Q 头数必须能被 KV 头数整除"
group_size = h_q // h_kv

# ========== 准备 Q、K、V ==========
Q_mat = torch.rand((N,    h_q,  d_head))
K_mat = torch.rand((N_kv, h_kv, d_head))
V_mat = torch.rand((N_kv, h_kv, d_head))

# ========== 标准答案（带 causal mask）==========
expected_attention = torch.zeros((N, h_q, d_head))

# 构造完整的 (N, N_kv) mask：True 表示可见，False 表示屏蔽
# Q 在完整 KV 序列里的"绝对位置" = i + (N_kv - N)
q_pos_full = torch.arange(N)[:, None] + (N_kv - N)        # (N, 1)，值是 16~31
k_pos_full = torch.arange(N_kv)[None, :]                  # (1, N_kv)，值是 0~31
full_mask  = k_pos_full <= q_pos_full                     # (N, N_kv) 布尔矩阵

for hq in range(h_q):
    hkv = hq // group_size
    Q_h = Q_mat[:, hq,  :]
    K_h = K_mat[:, hkv, :]
    V_h = V_mat[:, hkv, :]

    scores = Q_h @ K_h.T                                  # (N, N_kv)
    scores = scores.masked_fill(~full_mask, float('-inf'))  # 屏蔽未来位置
    attn = torch.softmax(scores, dim=1) @ V_h
    expected_attention[:, hq, :] = attn

# ========== Flash Attention V2 分块实现（带 mask）==========
Br = 4
Bc = 8

O = torch.zeros((N, h_q, d_head))

for hq in range(h_q):
    hkv = hq // group_size

    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br

        Qi = Q_mat[block_start_Br:block_end_Br, hq, :]    # (Br, d_head)

        Oi = torch.zeros((Br, d_head))
        li = torch.zeros((Br, 1))
        mi = torch.full((Br, 1), -torch.inf)

        for block_start_Bc in range(0, N_kv, Bc):
            block_end_Bc = block_start_Bc + Bc

            Kj = K_mat[block_start_Bc:block_end_Bc, hkv, :]   # (Bc, d_head)
            Vj = V_mat[block_start_Bc:block_end_Bc, hkv, :]   # (Bc, d_head)

            Sij = Qi @ Kj.T                                   # (Br, Bc)

            # ===== 关键改动：构造当前 block 的 causal mask =====
            # Q 在完整 KV 序列里的绝对位置
            q_pos = torch.arange(block_start_Br, block_end_Br)[:, None] + (N_kv - N)  # (Br, 1)
            # KV 在完整 KV 序列里的绝对位置
            k_pos = torch.arange(block_start_Bc, block_end_Bc)[None, :]               # (1, Bc)
            # 因果规则：k_pos <= q_pos 才可见
            block_mask = k_pos <= q_pos                                               # (Br, Bc)
            # 把不可见位置设成 -inf（softmax 后变成 0）
            Sij = Sij.masked_fill(~block_mask, float('-inf'))
            # =====================================================

            # ===== 在线 softmax 更新（逻辑完全不变）=====
            mi_new = torch.max(
                torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]),
                dim=1
            ).values[:, None]

            Pij_hat = torch.exp(Sij - mi_new)
            li = torch.exp(mi - mi_new) * li + torch.sum(Pij_hat, dim=1)[:, None]
            Oi = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj
            mi = mi_new

        Oi = Oi / li
        O[block_start_Br:block_end_Br, hq, :] = Oi

# ========== 验证 ==========
assert torch.allclose(O, expected_attention, atol=1e-5), "结果不一致！"
print("✓ 验证通过！带 Causal Mask 的 GQA + Flash Attention V2 + KV Cache 2倍长 实现正确")
print(f"  Q 形状: {tuple(Q_mat.shape)}")
print(f"  K 形状: {tuple(K_mat.shape)}")
print(f"  V 形状: {tuple(V_mat.shape)}")
print(f"  输出形状: {tuple(O.shape)}")
print(f"  Mask 规则: 第 i 个 Q 可见 KV 位置 0 ~ {N_kv - N} + i")
