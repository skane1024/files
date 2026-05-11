import torch

torch.manual_seed(456)

N, d = 16, 8           # N: 当前 Q 的 token 长度； d: 每个 token 的维度
N_kv = 2 * N           # KV cache 的长度是 Q 的 2 倍（模拟有历史缓存的场景）

Q_mat = torch.rand((N, d))        # Q 仍然是 N 行
K_mat = torch.rand((N_kv, d))     # K 变成 2N 行（包含历史 K + 当前 K）
V_mat = torch.rand((N_kv, d))     # V 变成 2N 行（包含历史 V + 当前 V）

# 标准做法的"标准答案"，用来对比验证
expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)   # 形状 (N, N_kv)
expected_attention = expected_softmax @ V_mat              # 形状 (N, d)

Br = 4      # Q 方向的 block 大小（外循环每次处理 4 个 Q token）
Bc = 8      # K、V 方向的 block 大小（内循环每次处理 d 个 KV token）

O = torch.zeros((N, d))   # 最终输出，形状和 Q 一致


# 外循环：把 Q 切成多个 block，每次处理 Br 个 Q token
for block_start_Br in range(0, N, Br):
    block_end_Br = block_start_Br + Br

    # 从 HBM(大显存) 加载一块 Q 到 SRAM(小快内存)
    Qi = Q_mat[block_start_Br:block_end_Br, :]      # 形状 (Br, d)

    # 初始化这一块 Q 对应的中间结果
    Oi = torch.zeros((Br, d))                       # 累积输出
    li = torch.zeros((Br, 1))                       # softmax 分母累加
    mi = torch.full((Br, 1), -torch.inf)            # 当前见过的最大值

    # 内循环：把 K、V 切成多个 block
    # 关键改动：循环范围从 range(0, N, Bc) 改成 range(0, N_kv, Bc)
    # 因为现在 K、V 有 2N 行，需要多扫一倍的 block
    for block_start_Bc in range(0, N_kv, Bc):
        block_end_Bc = block_start_Bc + Bc

        # 加载一块 K、V 到 SRAM
        Kj = K_mat[block_start_Bc:block_end_Bc, :]  # 形状 (Bc, d)
        Vj = V_mat[block_start_Bc:block_end_Bc, :]  # 形状 (Bc, d)

        # 算这一小块的注意力分数 S = Q @ K^T
        Sij = Qi @ Kj.T                             # 形状 (Br, Bc)

        # === 在线 softmax 更新 ===
        # 1) 更新当前见过的最大值
        mi_new = torch.max(
            torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]),
            dim=1
        ).values[:, None]

        # 2) 用新的最大值算 exp（数值稳定）
        Pij_hat = torch.exp(Sij - mi_new)

        # 3) 把旧的分母 li 用 exp(mi - mi_new) 缩放，再加上新 block 的贡献
        li = torch.exp(mi - mi_new) * li + torch.sum(Pij_hat, dim=1)[:, None]

        # 4) 同样地，把旧的累积输出 Oi 缩放，再加上新 block 的贡献
        Oi = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj

        # 5) 更新最大值，为下一轮做准备
        mi = mi_new

    # 内循环结束后，用累计的分母 li 做最终归一化（完成 softmax）
    Oi = Oi / li

    # 把这一块的结果写回最终输出
    O[block_start_Br:block_end_Br, :] = Oi

# 验证：分块算法的结果应该和标准做法完全一致
assert torch.allclose(O, expected_attention)
print("通过验证：Flash Attention V2 在 KV cache 长度为 Q 的 2 倍场景下结果正确！")
print("Q 形状:", Q_mat.shape, " K 形状:", K_mat.shape, " 输出 O 形状:", O.shape)
