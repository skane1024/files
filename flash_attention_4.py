import torch
import numpy as np

torch.manual_seed(456)

# ========== 基本配置 ==========
seq_len_q      = 512        # Q 的 token 长度（不切块，整个 512 一起算）
seq_len_kv   = 1 * seq_len_q     # KV cache 长度是 Q 的 2 倍（含历史）
q_head    = 8         # Q 的头数
kv_head   = 2         # KV 的头数
head_dims = 128         # 每个头的维度

assert q_head % kv_head == 0, "Q 头数必须能被 KV 头数整除"
group_size = q_head // kv_head    # 每个 KV 头对应 4 个 Q 头

seq_len_kv_part = 128   # KV 的 block 大小：保证每次 matmul 都是 (4,512,128) @ (1,128,128)

# ========== 准备 Q、K、V ==========
Q_mat = torch.rand((seq_len_q,  q_head,  head_dims))
K_mat = torch.rand((seq_len_kv, kv_head, head_dims))
V_mat = torch.rand((seq_len_kv, kv_head, head_dims))

# ========== 标准答案（带 causal mask）==========
expected_attention = torch.zeros((seq_len_q, q_head, head_dims))

# 构造完整的 (seq_len_q, seq_len_kv) mask：True 表示可见，False 表示屏蔽
# Q 在完整 KV 序列里的"绝对位置" = i + (seq_len_kv - seq_len_q)
q_pos_full = torch.arange(seq_len_q)[:, None] + (seq_len_kv - seq_len_q)        # (seq_len_q, 1)，值是 16~31
k_pos_full = torch.arange(seq_len_kv)[None, :]                  # (1, seq_len_kv)，值是 0~31
full_mask  = k_pos_full <= q_pos_full                     # (seq_len_q, seq_len_kv) 布尔矩阵

#一个头一个头的进行计算
for hq in range(q_head):
    hkv = hq // group_size
    Q_h = Q_mat[:, hq,  :]  # (seq_len_q, head_dims)
    K_h = K_mat[:, hkv, :]  # (seq_len_kv, head_dims)
    V_h = V_mat[:, hkv, :]  # (seq_len_kv, head_dims)
    scores = Q_h @ K_h.T                                  # (seq_len_q, seq_len_kv)
    scores = scores.masked_fill(~full_mask, float('-inf'))  # 屏蔽未来位置
    attn = torch.softmax(scores, dim=1) @ V_h
    expected_attention[:, hq, :] = attn



# ========== Flash Attention V2 实现（Q 不切块 + GQA 分组批量 + KV 分块）==========
# Q 的 512 整个作为一个 block，共享同一 KV 头的 4 个 Q 头放到 batch 维，
# 每次 matmul 形状固定为 (4,512,128) @ (1,128,128)
O = torch.zeros((seq_len_q, q_head, head_dims))

for hkv in range(kv_head):
    # (4, 512, 128)：batch 维是同一个 KV 头对应的 4 个 Q 头
    Qg = Q_mat[:, hkv * group_size:(hkv + 1) * group_size, :].permute(1, 0, 2)

    shuchu_add = torch.zeros((group_size, seq_len_q, head_dims))       # 累积输出和
    softmax_fenmu = torch.zeros((group_size, seq_len_q, 1))            # 累积分母
    max_value = torch.full((group_size, seq_len_q, 1), -torch.inf)     # 见过的最大值

    for block_start_Bc in range(0, seq_len_kv, seq_len_kv_part):
        block_end_Bc = block_start_Bc + seq_len_kv_part

        # K^T: (1, head_dims, seq_len_kv_part) = (1, 128, 128)
        Kj_T = K_mat[block_start_Bc:block_end_Bc, hkv, :].T.unsqueeze(0)
        # V:   (1, seq_len_kv_part, head_dims) = (1, 128, 128)
        Vj = V_mat[block_start_Bc:block_end_Bc, hkv, :].unsqueeze(0)

        Sij = Qg @ Kj_T       # (4,512,128) @ (1,128,128) -> (4,512,128)，batch 维自动广播

        # ===== 构造当前 block 的 causal mask =====
        # Q 在完整 KV 序列里的绝对位置
        q_pos = torch.arange(seq_len_q)[:, None] + (seq_len_kv - seq_len_q)   # (512, 1)
        # 当前 KV block 在完整序列里的绝对位置
        k_pos = torch.arange(block_start_Bc, block_end_Bc)[None, :]           # (1, 128)
        # 因果规则：k_pos <= q_pos 才可见；(512,128) 对 4 个 Q 头共享，广播即可
        block_mask = k_pos <= q_pos
        Sij = Sij.masked_fill(~block_mask, float('-inf'))
        # =====================================================

        # ===== 在线 softmax 更新（在最后一维 KV 上累加）=====
        mi_new = torch.maximum(max_value, torch.max(Sij, dim=-1, keepdim=True).values)

        Pij_hat = torch.exp(Sij - mi_new)
        softmax_fenmu = torch.exp(max_value - mi_new) * softmax_fenmu + Pij_hat.sum(dim=-1, keepdim=True)
        shuchu_add = shuchu_add * torch.exp(max_value - mi_new) + Pij_hat @ Vj   # (4,512,128) @ (1,128,128)
        max_value = mi_new

    shuchu_add = shuchu_add / softmax_fenmu               # (4, 512, 128)
    # 还原回 (512, 4, 128) 写回对应的 4 个 Q 头
    O[:, hkv * group_size:(hkv + 1) * group_size, :] = shuchu_add.permute(1, 0, 2)

# ========== 验证 ==========
assert torch.allclose(O, expected_attention, atol=1e-5), "结果不一致！"
print("✓ 验证通过！Q 不切块 + GQA 分组批量 + Flash Attention V2 实现正确")
print(f"  Q 形状: {tuple(Q_mat.shape)}")
print(f"  K 形状: {tuple(K_mat.shape)}")
print(f"  V 形状: {tuple(V_mat.shape)}")
print(f"  输出形状: {tuple(O.shape)}")
print(f"  matmul 形状: (4,512,128) @ (1,128,128)，KV block = {seq_len_kv_part}")
print(f"  Mask 规则: 第 i 个 Q 可见 KV 位置 0 ~ {seq_len_kv - seq_len_q} + i")
