import numpy as np

NUM_TOKENS   = 1             # token 总数
N_DIMS       = 3072             # 每 token 特征维度
NUM_EXPERTS  = 116              #专家的个数
TOP_K        = 8                # 每个 token 被路由到的专家数

rng = np.random.default_rng(55)

expert_path = "expert/model_1core_opt_w1_int8.onnx"
# expert_path = ""


import numpy as np

def int8_quantize(data,scale = 0.0, zero_point = 0.0):
    x = np.asarray(data, dtype=np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    
    if scale == 0.0:
        scale = max(np.abs(x_min), np.abs(x_max)) / 127.0
        zero_point = 0
    q = np.round(x / scale + zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    if scale == 0.0:
        return q, scale, zero_point
    return q, scale, zero_point

def to_np128(data):
    data = data.reshape((NUM_TOKENS, N_DIMS))
    data = data.transpose((1,0))
    data = data.reshape(N_DIMS // 128,128, NUM_TOKENS)
    data = data.transpose(0,2,1)
    return data




# 从 router 矩阵扫描当前专家的 (token_id, weight)
def scan_expert_tokens(router_matrix, expert_id):
    expert_row = router_matrix[expert_id]
    token_ids  = np.nonzero(expert_row)[0]
    weights    = expert_row[token_ids]
    return token_ids, weights


# 专家计算
def expert_forward(input_data, expert_onnx_model=""):
    print(input_data.shape)
    if expert_onnx_model != "":
        import onnxruntime as ort
        input_data_1 = input_data.transpose((1,0))
        input_data_1 = input_data_1.reshape((1,3072,1,1))
        input_data_1 = input_data_1.astype(np.float32)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(expert_onnx_model, so)
        input_name_0 = session.get_inputs()[0].name
        input_data_map = {input_name_0: input_data_1 }
        outputs = session.run(None, input_data_map)[0]
        outputs = outputs.transpose(0,3,2,1)
        outputs = outputs.reshape(1,3072)
        print("expert result: ", np.min(outputs), np.max(outputs))
        return outputs

    return  input_data


# ===================== moe_ffn.c 加权求和过程的完整模拟 =====================
# 输入 token: int8 (量化参数 in_scale/in_zeropoint)
# 路由权重:   float16
# 输出:       int8 (量化参数 out_scale/out_zeropoint)
# 专家前向(NPU): 用 QDQ ONNX 模拟, NPU 输出量化参数 npu_out_scale/npu_out_zeropoint

from moe_ffn_ref import expert_onnx_forward


def round_away_from_zero(x):
    # 四舍五入, 恰为 .5 时远离零, 与 golden 的 round_away_from_zero 一致
    x = np.asarray(x)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def firound_f16(x):
    # IVP_FIROUNDNXF16: f16 值就近舍入到整数, 恰为 0.5 时远离零 [moe_ffn.c:1221/248]
    return round_away_from_zero(x.astype(np.float32))


def _token_input(token_data, in_scale, in_zeropoint):
    # int8 token(C 的 in1 输入): (q - in_zp) * in_scale 直接反量化(f32);
    # float token: 先按 in_scale/in_zeropoint 做一次 QDQ
    token = np.asarray(token_data)
    if np.issubdtype(token.dtype, np.integer):
        return (token.astype(np.float32) - in_zeropoint) * np.float32(in_scale)
    x = token.astype(np.float32)
    return (round_away_from_zero(x / np.float32(in_scale) + in_zeropoint)
            - in_zeropoint) * np.float32(in_scale)


def _npu_out_int8(expert_out, npu_out_scale, npu_out_zeropoint):
    # NPU 的 int8 输出: 专家 QDQ ONNX 输出本就在 npu_out 网格上, 量化还原 e0
    return np.clip(round_away_from_zero(expert_out / np.float32(npu_out_scale)
                                        + npu_out_zeropoint), -128, 127)


def moe_ffn_decoder_int8(token_data, router_weights, expert_model_path,
                         in_scale, in_zeropoint, out_scale, out_zeropoint,
                         npu_out_scale, npu_out_zeropoint):
    """完全模拟 golden/moe_ffn.c moe_ffn_decoder_int8 [moe_ffn.c:1055-1231] (单 token).

    token_data:        (1, n_dims) int8 量化 token 或 float 原始 token
    router_weights:    (num_experts,) float16 各专家对该 token 的路由权重
    expert_model_path: w1 QDQ ONNX 路径, "" 表示跳过专家计算(直通)
    返回 (1, n_dims) int8 (NP128 布局只影响存放, 不影响数值)
    """
    assert token_data.shape[0] == 1
    n_dims = token_data.shape[1]

    # C 中的常量: 都经 f16 参与运算
    sn = np.float16(npu_out_scale)      # npu_out_scale_v  [moe_ffn.c:1154]
    zn = int(npu_out_zeropoint)
    so_recip = np.float16(np.float32(1.0) / np.float32(out_scale))  # [moe_ffn.c:1137]
    zp = int(out_zeropoint)

    input_data = _token_input(token_data, in_scale, in_zeropoint)

    # expert_token_search [moe_ffn.c:473-488]: 权重 >= 1e-5 的专家, 按专家 id 升序
    # (vWeight 用原始专家 id 索引权重数组 [moe_ffn.c:1156], 数组不做压缩)
    router_weights = np.asarray(router_weights, dtype=np.float16)
    expert_ids = [e for e in range(router_weights.size)
                  if np.float32(router_weights[e]) >= 1e-5]

    # NPU int8 输出 e0(所有选中专家同模型同输入, e0 相同)
    expert_out = expert_onnx_forward(input_data, 1, n_dims, expert_model_path)
    e0 = _npu_out_int8(expert_out, npu_out_scale, npu_out_zeropoint)

    # f16 累加器, memset 0 [moe_ffn.c:1149]
    acc = np.zeros((1, n_dims), dtype=np.float16)
    for e in expert_ids:
        # 反量化: (e0 - zn) 精确转 f16, * f16(sn) 后 f16 舍入 [moe_ffn.c:1186]
        deq = (e0 - zn).astype(np.float16) * sn
        # 路由权重: expert_token_weight[原始专家 id] [moe_ffn.c:1156]
        w = np.float32(router_weights[e])
        # MOE_FFN_WSUM_F16F32 [moe_ffn.c:223-237, 1200-1205]:
        #   acc = f16(f32(acc) + f32(w) * f32(deq)), 每专家累加后回写 f16
        acc = (acc.astype(np.float32) + w * deq.astype(np.float32)).astype(np.float16)

    # 最终重量化(全部专家累加完后只做一次) [moe_ffn.c:1221]:
    #   sat(firound(f16(acc * f16(1/out_scale))) + out_zp)
    prod = acc * so_recip                # f16 乘法, 结果 f16 舍入
    code = np.clip(firound_f16(prod) + zp, -128, 127).astype(np.int8)
    return code


def moe_ffn_encoder_int8(token_data, router_matrix, expert_model_path,
                         in_scale, in_zeropoint, out_scale, out_zeropoint,
                         npu_out_scale, npu_out_zeropoint, expert_compute_count=None):
    """完全模拟 golden/moe_ffn.c moe_ffn_encoder_int8 [moe_ffn.c:537-754] (多 token prefill).

    token_data:    (num_tokens, n_dims) int8 量化 token 或 float 原始 token
    router_matrix: (num_experts, num_tokens) float16 路由权重矩阵
    expert_compute_count: None=处理全部专家(默认); 对应 C 的 EXPERT_COMPUTE_COUNT
    返回 (num_tokens, n_dims) int8

    与 decoder 的区别: 累加器就是 int8 输出 buffer(初值 out_zeropoint),
    每个专家处理完立即 requant 回 int8, 下一专家再 dequant 出来累加
    (MOE_FFN_WSUM_INT8_F32 融合宏 [moe_ffn.c:253-272, 728-735])。
    """
    num_tokens, n_dims = token_data.shape

    sn = np.float16(npu_out_scale)      # npu_out_scale_v  [moe_ffn.c:643]
    zn = int(npu_out_zeropoint)
    so = np.float16(out_scale)          # out_scale_v     [moe_ffn.c:614]
    so_recip = np.float16(np.float32(1.0) / np.float32(out_scale))  # [moe_ffn.c:615]
    zp = int(out_zeropoint)
    router_matrix = np.asarray(router_matrix, dtype=np.float16)

    input_data = _token_input(token_data, in_scale, in_zeropoint)

    # 输出 buffer memset 为 out_zeropoint [moe_ffn.c:596]
    acc8 = np.full((num_tokens, n_dims), zp, dtype=np.int8)

    expert_count = router_matrix.shape[0] if expert_compute_count is None \
        else expert_compute_count
    for expert_id in range(expert_count):
        # expert_token_search_opt [moe_ffn.c:491-533]: 权重 != 0 的 token, id 升序
        # (NPU 按 npu_max_tokens 分批只是性能问题, 逐 token 结果一致)
        row = router_matrix[expert_id]
        token_ids = np.nonzero(row)[0]
        if token_ids.size == 0:
            continue
        expert_out = expert_onnx_forward(input_data[token_ids], token_ids.size,
                                         n_dims, expert_model_path)
        e0 = _npu_out_int8(expert_out, npu_out_scale, npu_out_zeropoint)

        # 逐 token: 反量化 -> f32 乘加 -> requant 回 int8 (每专家一次)
        for i, tid in enumerate(token_ids):
            w = np.float32(row[tid])                                # f16 权重 -> f32 精确
            deq_in = (e0[i] - zn).astype(np.float16) * sn           # f16((e0-zn)*sn)
            deq_acc = (acc8[tid].astype(np.int16) - zp).astype(np.float16) * so
                                                                    # f16((acc-zp)*so)
            acc_f16 = (deq_acc.astype(np.float32)
                       + w * deq_in.astype(np.float32)).astype(np.float16)
            prod = acc_f16 * so_recip                               # f16(acc * 1/so)
            acc8[tid] = np.clip(firound_f16(prod) + zp, -128, 127)  # sat(...+zp)
    return acc8


#token 数据
token_data = rng.standard_normal(size=(NUM_TOKENS, N_DIMS)).astype(np.float16)
token_data_np128 = to_np128(token_data)
token_data_np128.tofile("test_data/token_data_np128_f16.bin")

# token_data_np128_int8,in_scale,in_zeropoint = int8_quantize(token_data_np128)
in_scale = 0.047759633511304855
in_zeropoint = 3
token_data_np128_int8,s,z = int8_quantize(token_data_np128, in_scale, in_zeropoint)
token_data_np128_int8.tofile("test_data/token_data_np128_int8.bin")
print("in scale_zeropoint:", in_scale,in_zeropoint, np.min(token_data), np.max(token_data))


# 2. router 矩阵 (NUM_EXPERTS, NUM_TOKENS), 非零值 = 权重
router_matrix = np.fromfile("test_data/expert_tokens.bin", dtype = np.float16).reshape((NUM_EXPERTS, NUM_TOKENS))


NPU_input = []

golden_result = np.zeros((NUM_TOKENS, N_DIMS), dtype=np.float32)
for expert_id in range(2):
    token_ids, token_weights = scan_expert_tokens(router_matrix, expert_id)
    print(token_ids)
    print(len(token_ids))
    input_data = token_data[token_ids].astype(np.float32)
    input_data_np128_int8 = token_data_np128_int8[:,token_ids,:]
    input_data_np128_int8.tofile(f"test_data/npu_input_{expert_id}.bin")
    npe_result =expert_forward(input_data,expert_path)
    npe_result_np128 = npe_result
    npe_result_np128 = npe_result_np128.reshape((72, N_DIMS))
    npe_result_np128 = npe_result_np128.transpose((1,0))
    npe_result_np128 = npe_result_np128.reshape(N_DIMS // 128,128, 72)
    npe_result_np128 = npe_result_np128.transpose(0,2,1)
    npe_result_np128,s,z = int8_quantize(npe_result_np128, scale = 0.032091207802295685, zero_point = 60)
    npe_result_np128.tofile(f"test_data/npu_out_{expert_id}.bin")
    golden_result[token_ids] += token_weights[:, None].astype(np.float32) * npe_result


golden_result = golden_result.astype(np.float16)
golden_result_np128 = to_np128(golden_result)
golden_result_np128.tofile("test_data/golden_result_np128_f16.bin")


golden_result_np128_int8,out_scale,out_zeropoint = int8_quantize(golden_result_np128)
# golden_result_np128_int8 = int8_quantize(golden_result_np128, out_scale, out_zeropoint)
golden_result_np128_int8.tofile("test_data/golden_result_np128_int8.bin")
print("out scale_zeropoint:", out_scale,out_zeropoint)




data = np.fromfile("test_data/out_img_cast1_r1_i0.out", dtype  = np.int8)

phone_result = data[:3072 * 1024]
phone_npu_io = data[3072 * 1024:3072 * 1024 + 3072 * 72 *4]
phone_npu_io = phone_npu_io.reshape(4,3072 * 72)
phone_npu_in_1 = phone_npu_io[0, :]
phone_npu_in_2 = phone_npu_io[2, :]
phone_npu_in_1.tofile("test_data/phone_npu_in_1.bin")
phone_npu_in_2.tofile("test_data/phone_npu_in_2.bin")

phone_npu_out_1 = phone_npu_io[1, :]
phone_npu_out_2 = phone_npu_io[3, :]
phone_npu_out_1.tofile("test_data/phone_npu_out_1.bin")
phone_npu_out_2.tofile("test_data/phone_npu_out_2.bin")

print(phone_npu_out_1)
print(phone_npu_out_2)
