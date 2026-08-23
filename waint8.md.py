import numpy as np

# ----------------------------------------------------------
# 1. 仿射量化 / 反量化基础函数
# ----------------------------------------------------------

def quantize(x_fp, scale, zero_point, qmin=-128, qmax=127):
    """把float数组量化成int8: q = clip(round(x/scale) + z, qmin, qmax)"""
    q = np.round(x_fp / scale) + zero_point
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int32)  # 用int32容器承载，避免中间运算溢出

def dequantize(q, scale, zero_point):
    """反量化: r = scale * (q - z)，仅用于生成参考真值，不参与整数计算路径"""
    return scale * (q.astype(np.float32) - zero_point)


# ----------------------------------------------------------
# 2. 浮点缩放系数 -> 定点乘子(Q31) + shift 的分解
#    对应文中: M = M0 * 2^-(31-shift)，M0落在[2^30, 2^31)区间
# ----------------------------------------------------------

def quantize_multiplier(M, bits=31):
    """
    把正的浮点数M分解成 (M0, shift)，使得 M ≈ M0 * 2^(shift-bits)
    M0是Q(bits)定点整数（int32），shift是右移位数
    """
    if M == 0:
        return 0, 0
    # 把M规约到[0.5, 1)区间，同时记录用了多少次乘2
    shift = 0
    while M < 0.5:
        M *= 2.0
        shift -= 1
    while M >= 1.0:
        M /= 2.0
        shift += 1
    # 此时 0.5 <= M < 1.0，转成Q31定点整数
    M0 = int(round(M * (1 << bits)))
    if M0 == (1 << bits):  # 四舍五入溢出的边界修正
        M0 //= 2
        shift += 1
    return M0, shift  # 运行时: round(x * M0 / 2^bits) 再右移 shift 位


def round_shift(x, shift, bits=31):
    """
    模拟VDSP上的 定点高位乘 + 带舍入右移，全程整数运算：
      1) 先做 x * M0，得到扩展精度的整数积
      2) 右移 (bits) 位取高位（等价于定点乘法取高位部分）
      3) 再按shift做带舍入的右移
    这里为了让复合右移一次做完，把两步移位合并处理
    """
    total_shift = bits - shift if shift >= 0 else bits + (-shift)
    x = x.astype(np.int64)
    if total_shift <= 0:
        return (x << (-total_shift)).astype(np.int64)
    rounding = 1 << (total_shift - 1)
    return (x + rounding) >> total_shift


# ----------------------------------------------------------
# 3. 量化MAC算子: output = input_1 + input_2 * weight
#    weight为浮点标量，融合进M2；两路输入为int8量化
# ----------------------------------------------------------

def quant_madd_int8(q1, q2, z1, z2, zo, M1_0, shift1, M2_0, shift2):
    """按照公式 q_o = clip(round(M1*(q1-z1) + M2*(q2-z2)) + z_o, -128, 127) 计算"""
    d1 = (q1 - z1).astype(np.int64)
    d2 = (q2 - z2).astype(np.int64)

    r1 = round_shift(d1 * M1_0, shift1)   # M1*(q1-z1) 的定点近似
    r2 = round_shift(d2 * M2_0, shift2)   # M2*(q2-z2) 的定点近似

    q_o = r1 + r2 + zo
    q_o = np.clip(q_o, -128, 127)
    return q_o.astype(np.int32)


# ----------------------------------------------------------
# 4. 构造模拟数据，分别跑浮点路径与量化路径，统计误差
# ----------------------------------------------------------

np.random.seed(42)
N = 200000  # 模拟样本数量

# 4.1 随机生成"真实"浮点输入和浮点标量权重
input_1_fp = np.random.uniform(-3.0, 3.0, N)
input_2_fp = np.random.uniform(-2.0, 2.0, N)
weight_fp  = 0.37   # 浮点标量权重（举例）

# 4.2 浮点参考结果（Baseline）
output_ref_fp = input_1_fp + input_2_fp * weight_fp

# 4.3 分别为input_1、input_2、output确定量化参数(scale, zero_point)
#     这里用对称量化(zero_point=0)简化演示，也可换成非对称
def calc_scale(x, qmax=127):
    return np.max(np.abs(x)) / qmax

s1, z1 = calc_scale(input_1_fp), 0
s2, z2 = calc_scale(input_2_fp), 0
s_o, z_o = calc_scale(output_ref_fp), 0


# 4.4 量化输入
q1 = quantize(input_1_fp, s1, z1)
q2 = quantize(input_2_fp, s2, z2)

# 4.5 计算缩放系数 M1、M2，并分解成定点乘子
M1 = s1 / s_o
M2 = (weight_fp * s2) / s_o

M1_0, shift1 = quantize_multiplier(M1)
M2_0, shift2 = quantize_multiplier(M2)

print(f"M1 = {M1:.8f}  -> M1_0={M1_0}, shift1={shift1}")
print(f"M2 = {M2:.8f}  -> M2_0={M2_0}, shift2={shift2}")

# 4.6 用整数域的量化算子计算输出
q_o = quant_madd_int8(q1, q2, z1, z2, z_o, M1_0, shift1, M2_0, shift2)

# 4.7 把量化输出反量化回浮点，与浮点参考结果对比
output_quant_fp = dequantize(q_o, s_o, z_o)

abs_err = np.abs(output_quant_fp - output_ref_fp)
rel_err = abs_err / (np.abs(output_ref_fp) + 1e-8)

# 余弦相似度: cos = <a,b> / (|a|*|b|)，衡量量化结果与浮点参考的整体方向一致性
cos_sim = float(np.dot(output_quant_fp, output_ref_fp) /
                (np.linalg.norm(output_quant_fp) * np.linalg.norm(output_ref_fp)))

print("\n---- 精度验证结果 ----")
print(f"样本数量:            {N}")
print(f"输出scale(s_o):       {s_o:.6f}  (即int8量化的最小分辨单位)")
print(f"平均绝对误差(MAE):     {abs_err.mean():.6f}")
print(f"最大绝对误差:         {abs_err.max():.6f}")
print(f"平均相对误差:         {rel_err.mean()*100:.4f}%")
print(f"余弦相似度:           {cos_sim:.6f}")

