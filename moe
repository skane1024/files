import numpy as np

# ============================================================
# 配置参数
# ============================================================
NUM_TOKENS   = 512          # token 总数
DIM          = 1024         # 每 token 特征维度
PACK         = 128          # NP128: 每物理块 128 维
NUM_PLANES   = DIM // PACK  # = 8 个 plane
NUM_EXPERTS  = 116
TOP_K        = 8           # 每个 token 被路由到的专家数
BATCH_TOKENS = 8            # VDSP 每批处理的 token 数

DT_STORE = np.float16       # 数据面 fp16, 每 token 2KB
DT_ACC   = np.float16       # 累加面 fp32

LOCAL_MEM_BYTES = 64 * 1024

# ============================================================
# DMA 模拟器: 只统计, 真正搬数由 numpy 切片完成
# ============================================================
class DmaSim:
    def __init__(self):
        self.stats = {}     # phase -> [次数, 字节数]

    def copy(self, nbytes, phase):
        entry = self.stats.setdefault(phase, [0, 0])
        entry[0] += 1
        entry[1] += nbytes

    def report(self):
        print("================ DMA 统计 ================")
        total_count, total_bytes = 0, 0
        for phase, (count, nbytes) in self.stats.items():
            print(f"  {phase:<28s} 次数 {count:>8d}   字节 {nbytes/1024/1024:8.2f} MB")
            total_count += count
            total_bytes += nbytes
        print(f"  {'TOTAL':<28s} 次数 {total_count:>8d}   字节 {total_bytes/1024/1024:8.2f} MB")

dma = DmaSim()

# ============================================================
# NCHW_NP128 容器
# 逻辑: 第 n 个 token 的 1024 维 = planes[g][n] (g=0..7) 拼接
# 物理: planes[g] 内所有 token 的第 g 组 128 维连续存放
# ============================================================
class NchwNp128:
    def __init__(self, max_tokens, dtype=DT_STORE):
        self.max_tokens = max_tokens
        self.dtype      = dtype
        self.planes     = np.zeros((NUM_PLANES, max_tokens, PACK), dtype=dtype)

    @classmethod
    def from_logical(cls, data):
        """(N, DIM) 逻辑数据 -> NCHW_NP128 物理布局"""
        num_tokens = data.shape[0]
        container = cls(num_tokens, data.dtype)
        container.planes = (data.reshape(num_tokens, NUM_PLANES, PACK)
                                .transpose(1, 0, 2).copy())
        return container

    def read_token(self, token_slot, phase):
        """单 token 读: 8 次跨 plane 的 DMA"""
        token_vector = np.empty(DIM, dtype=self.dtype)
        for plane_id in range(NUM_PLANES):
            dma.copy(PACK * np.dtype(self.dtype).itemsize, phase)
            token_vector[plane_id*PACK:(plane_id+1)*PACK] = \
                self.planes[plane_id][token_slot]
        return token_vector

    def write_token(self, token_slot, token_vector, phase):
        """单 token 写: 8 次跨 plane 的 DMA"""
        for plane_id in range(NUM_PLANES):
            dma.copy(PACK * np.dtype(self.dtype).itemsize, phase)
            self.planes[plane_id][token_slot] = \
                token_vector[plane_id*PACK:(plane_id+1)*PACK]

    def read_block(self, start_slot, token_count, phase):
        """连续块读: 每 plane 一次 DMA, 共 8 次"""
        block_data = np.empty((token_count, DIM), dtype=self.dtype)
        for plane_id in range(NUM_PLANES):
            dma.copy(token_count * PACK * np.dtype(self.dtype).itemsize, phase)
            block_data[:, plane_id*PACK:(plane_id+1)*PACK] = \
                self.planes[plane_id][start_slot:start_slot+token_count]
        return block_data

    def write_block(self, start_slot, block_data, phase):
        """连续块写: 每 plane 一次 DMA, 共 8 次"""
        token_count = block_data.shape[0]
        for plane_id in range(NUM_PLANES):
            dma.copy(token_count * PACK * np.dtype(self.dtype).itemsize, phase)
            self.planes[plane_id][start_slot:start_slot+token_count] = \
                block_data[:, plane_id*PACK:(plane_id+1)*PACK]

# ============================================================
# DDR 数据区
# ============================================================
rng = np.random.default_rng(42)

# 1. 原始 token 特征 (逻辑 -> NCHW_NP128 物理存放)
token_data_logical = rng.standard_normal((NUM_TOKENS, DIM)).astype(DT_STORE)
token_data = NchwNp128.from_logical(token_data_logical)

# 2. router 矩阵 (NUM_EXPERTS, NUM_TOKENS), 非零值 = 权重
router_matrix = np.zeros((NUM_EXPERTS, NUM_TOKENS), dtype=np.float32)
routing_scores = rng.random((NUM_EXPERTS, NUM_TOKENS), dtype=np.float32)
topk_experts = np.argpartition(-routing_scores, TOP_K, axis=0)[:TOP_K]
for token_id in range(NUM_TOKENS):
    selected_experts = topk_experts[:, token_id]
    raw_weights = np.exp(routing_scores[selected_experts, token_id])
    router_matrix[selected_experts, token_id] = raw_weights / raw_weights.sum()

# 3. NPU 输入/输出 buffer: NCHW_NP128 布局
MAX_TOKENS_PER_EXPERT = 64
npu_input  = NchwNp128(MAX_TOKENS_PER_EXPERT)
npu_output = NchwNp128(MAX_TOKENS_PER_EXPERT)

# 4. global_sum: 行主序布局(每 token 1024 维连续), fp32, MOE 开始前清零
global_sum = np.zeros((NUM_TOKENS, DIM), dtype=DT_ACC)

# ============================================================
# VDSP Local Memory: 两个 buffer, 校验 64KB 上限
# ============================================================
expert_out_buf  = np.zeros((BATCH_TOKENS, DIM), dtype=DT_ACC)  # npu_output 读入 + 原地加权
accumulate_buf  = np.zeros((BATCH_TOKENS, DIM), dtype=DT_ACC)  # global_sum 旧值 + 累加 + 写回

local_mem_used = expert_out_buf.nbytes + accumulate_buf.nbytes
print(f"Local Memory 占用: {local_mem_used/1024:.1f} KB / 64 KB "
      f"(expert_out_buf {expert_out_buf.nbytes//1024}KB + "
      f"accumulate_buf {accumulate_buf.nbytes//1024}KB)")
assert local_mem_used <= LOCAL_MEM_BYTES, "超出 local memory 容量!"

# ============================================================
# 专家子图(模拟 NPU 计算): 确定性的逐专家逐元素线性变换
# ============================================================
def expert_forward(expert_id, input_data):
    scale = 1.0 + 0.001 * expert_id
    shift = 0.0001 * expert_id
    return input_data * scale + shift

# ============================================================
# 从 router 矩阵扫描当前专家的 (token_id, weight)
# ============================================================
def scan_expert_tokens(expert_id):
    expert_row = router_matrix[expert_id]
    token_ids  = np.nonzero(expert_row)[0]
    weights    = expert_row[token_ids]
    return token_ids, weights

# ============================================================
# Gather: 按 token id 列表, 从 token_data 逐 token 收集到
# npu_input, 保持 token_ids 顺序, 使 npu_input 连续
# ============================================================
def gather_expert_tokens(token_ids):
    for slot, token_id in enumerate(token_ids):
        token_vector = token_data.read_token(token_id, phase="gather_read")
        npu_input.write_token(slot, token_vector, phase="gather_write")

# ============================================================
# Gather: 按 token id 列表, 逐个读回 global_sum 旧值到
# local buffer (行主序, 每 token 1 次 DMA)
# ============================================================
def gather_global_sum(token_ids, dst_buffer):
    for index, token_id in enumerate(token_ids):
        dma.copy(DIM * np.dtype(DT_ACC).itemsize, phase="vdsp_read_gsum")
        dst_buffer[index] = global_sum[token_id]

# ============================================================
# Scatter: 按 token id 列表, 把 local buffer 中的累加结果
# 逐个写回 global_sum (每 token 1 次 DMA)
# ============================================================
def scatter_global_sum(token_ids, src_buffer):
    for index, token_id in enumerate(token_ids):
        dma.copy(DIM * np.dtype(DT_ACC).itemsize, phase="vdsp_write_gsum")
        global_sum[token_id] = src_buffer[index]

# ============================================================
# 主流程: 专家间串行
# ============================================================
for expert_id in range(NUM_EXPERTS):
    token_ids, token_weights = scan_expert_tokens(expert_id)
    token_count = len(token_ids)

    # ---- Stage 1: Gather (token_data -> npu_input) ----
    gather_expert_tokens(token_ids)

    # ---- Stage 2: NPU 计算 (整块连续读入/写出) ----
    npu_input_data = npu_input.read_block(0, token_count,phase="npu_load").astype(DT_ACC)
    npu_result = expert_forward(expert_id, npu_input_data)
    npu_output.write_block(0, npu_result.astype(DT_STORE), phase="npu_store")

    # ---- Stage 3: VDSP 分批加权累加 ----
    for batch_start in range(0, token_count, BATCH_TOKENS):
        batch_size      = min(BATCH_TOKENS, token_count - batch_start)
        batch_token_ids = token_ids[batch_start : batch_start + batch_size]
        batch_weights   = token_weights[batch_start : batch_start + batch_size]

        # Step 1: npu_output 连续块 -> expert_out_buf (8 次 plane DMA)
        expert_out_buf[:batch_size] = \
            npu_output.read_block(batch_start, batch_size,
                                  phase="vdsp_read_out").astype(DT_ACC)

        # Step 2: gather 读回 global_sum 旧值 -> accumulate_buf
        gather_global_sum(batch_token_ids, accumulate_buf)

        # dma_wait_all()  同步点: 保证数据完整到达 local memory

        # Step 3: VDSP 计算, accumulate_buf 原地累加
        for index in range(batch_size):
            accumulate_buf[index] += expert_out_buf[index] * batch_weights[index]

        # Step 4: scatter 写回 global_sum
        scatter_global_sum(batch_token_ids, accumulate_buf)

        # dma_wait_all()  同步点: 写回完成后再处理下一个专家

# ============================================================
# 黄金参考: 纯 numpy 直接计算, 用于校验
# ============================================================
golden_result = np.zeros((NUM_TOKENS, DIM), dtype=np.float32)
for expert_id in range(NUM_EXPERTS):
    token_ids, token_weights = scan_expert_tokens(expert_id)
    input_data = token_data_logical[token_ids].astype(np.float32)
    golden_result[token_ids] += \
        token_weights[:, None].astype(np.float32) * \
        expert_forward(expert_id, input_data)

abs_error = np.abs(global_sum.astype(np.float32) - golden_result)
rel_error = abs_error.max() / (np.abs(golden_result).max() + 1e-9)
print(f"结果校验: max_abs_err = {abs_error.max():.6e}, "
      f"max_rel_err = {rel_error:.3e} -> "
      f"{'PASS' if rel_error < 1e-2 else 'FAIL'}")

dma.report()




在大模型MOE的推理中，token的数据是(512, 1024)，其中，512是token的个数，1024是每个 token的特征维度，会把它转换成NCHW_NP128的格式, 数据的shape就是(1, 8,512,128) ，把每个 token 的 1024 维按 128 一组，均匀切成了 8 ， 物理内存上存放的时候，并不是按 token 顺序存的，而是按“组”来存的。先把所有 512 个 token 的第一组特征（也就是每个 token 的前 128 维）连续地排好，然后再排所有 token 的第二组特征（128~255 维），就这样一组一组往后排，直到排完 8 组，这样如果想取出某一个完整 token 的数据，就没法一次性连续读完了。因为它的 1024 维被分散到了 8 个不同的数据块里，你必须分 8 次去取，每次从对应的那个块里取出 128 个数，这个 token 完整的特征向量。

在MOE expert的推理中，每个专家会被编译成一个固定shape的NPU子图，比如一次可以处理64个token, 输入输出的数据格式也是NCHW_NP128，专家是串行的，每次加载一个专家进行计算， 会先找到当前专家下的所有token, 按照NCHW_NP128的格式组织在一起，每次从多个位置取一个token的数据，再存在在多个位置上。需要做gather的操作


NPU 输入输出的数据都是在DDR上的，专家计算后，会在VDSP上token该专家上计算加权求和的结果，  加权求和的结果放在buffer A上  输入数据放在buffer B上  第一次先取buffer B上的数据，乘以权重做加权，将结果放到 buffer A上，  第二次就是再取buffer B上的新数据，乘以权重做加权， 再取buffer A上的数据和加权的结果做求和
VDSP上的计算需要把数据搬运到VDSP的local_memory上， local_memory  容量很小，只有64kb

// 总数据	512 tokens × 1024 维（存储在 DDR）
// 专家数 116
// 每专家 token 数	大约64（非连续，需 gather）gather到npu_input DDR上, 送入NPU计算一次，得到同样shape的结果，放在npu_output DDR上
// local_memory 容量有限 64kb, 	每次 10 个 token（10 × 1024）
// 计算目标	对每和个专家内 64个 token 特征做加权乘法，每个 token的权重不同，再对 116 个专家下每个token的结果求和
// 约束	专家间串行（处理完一个专家再处理下一个）
// Gather 的本质：通过索引表 expert_token_idx 将 DDR 上非连续的 token 地址逐一映射到 local_memory 的连续空间中。
// partial_sum  local memory复用：每个专家处理前清零，处理完后累加到 global_sum，然后清零给下一个专家复用


数据gather的问题这里先不考虑 , 就认为有一个函数处理数据的搬运
这个设计是否正确
是否有优化的空间
补充一些数据搬运的细节  这个伪代码不好理解，整理成更方便理解的伪代码


Local Memory 布局优化
当前使用了 data_buf、partial_sum、readback_buf 三个主要 buffer，各 16 KB，共 48 KB。实际上可以合并为一个 buffer 对：
• 将专家输出读到 buf_A（8×1024），乘上权重后原地更新为加权结果。
• 另用一个 buf_B 读回 global_sum 对应位置。
• 计算 buf_B += buf_A，然后将 buf_B 写回 global_sum。
这样只需两个 16 KB buffer，节省 16 KB，可用于增大 batch size（例如将每批提升到 10 token，进一步减少 DMA 次数）。


//这里NPU计算和VDSP计算是会流水线并行的，目前先串行，梳理清楚整个流程


对齐与 burst：每次 128 维 = 256B（fp16），恰好是常见的 DDR burst 友好粒度；DMA 描述符应按 128 维对齐组织，避免跨 burst 的残缺传输。
//硬件是否支持stride的访问搬运？
写回的必要性判断：如果某 token 在后续专家中还会出现，则写回 global_sum 是必须的；如果硬件支持 DDR 端的原子加（或者 VDSP 有 write-back-with-accumulate 模式），可以省掉 "读回旧值" 这一路，流量从 6KB/token 降到 4KB/token，值得查一下芯片手册。



一、先分析问题特征，决定优化方向
你的数据有三个决定性特征：

每行很小：512 个 fp16 = 1KB，恰好 16 个 512-bit 向量；
稀疏：每行约 66 个非零（512 token × TOP_K / 116 ≈ 13% 密度），意味着"读和比较"是全向量化的，"写出"只有 66 项；
非零总数固定：TOP_K 固定时，全矩阵 nnz = 512 × 15 = 7680，输出表可以静态分配，不需要动态扩容。

所以这件事的本质是经典的 stream compaction（流压缩）​ 问题：向量化地生成"哪里非零"的信息，再选择便宜的紧凑化方式。下面按指令级给出两种落地方案。
二、指令级方案 A：向量比较 + 位掩码 + 位扫描（推荐先用这个）​
每 32 个 lane 一块，流程为：

// ============ 伪指令 → Vision 230 指令类映射 ============
// vload_h(p)          : 512-bit 对齐向量加载 (32×fp16)
// vcmp_nez_h(v)       : fp16 向量比较 != 0.0h → 布尔/谓词向量
//                       (备选整数技巧见下文)
// vmovemask_h(pred)   : 布尔向量 → 32-bit 标量掩码
// popcnt32(m)         : 人口计数 (无硬件指令时用 256 项 byte LUT)
// ctz32(m)            : Xtensa 无直接 CTZ, 用 NEG+AND+NSAU 组合:
//                       t = m & (-m); ctz = 31 - NSAU(t)
// ============================================================

void scan_expert_row(const fp16 *row, uint16 *out_ids, fp16 *out_ws, int *out_cnt)
{
    int cnt = 0;
    for (int blk = 0; blk < NUM_TOKENS / 32; blk++) {      // 16 块
        vec512 v    = vload_h(row + blk * 32);
        uint32 mask = vmovemask_h(vcmp_nez_h(v));
        if (mask == 0) continue;                 // 整 32 全零直接跳过

        while (mask) {
            int b = ctz32(mask);                 // 最低非零 lane 的位置
            out_ws[cnt]  = row[blk * 32 + b];    // 标量 16-bit load/store
            out_ids[cnt] = blk * 32 + b;
            cnt++;
            mask &= mask - 1;                    // 清掉最低位, 取下一位
        }
    }
    *out_cnt = cnt;
}



这个方案的精髓是把分支不可预测的 512 次标量判断，变成 16 次向量比较 + 平均每块仅约 4 次的短位扫描。每行约 16×10 + 66×5 ≈ 500 cycles，116 个专家全部扫完约 6 万 cycles，几百 MHz 下就是几十微秒量级，相对每个专家的 NPU 计算时间完全可以忽略，还能和第一个专家的计算 overlap 掉。
关于零判定有一个实用细节：可以不依赖 fp16 比较指令，改用整数技巧——把向量当作 16-bit 整数，（bits & 0x7FFF) != 0。这有两个好处：一是整数向量的 AND + 比较在任何 SIMD 配置下都一定有；二是它把 -0.0（0x8000）也正确判为零，避免位级判等漏掉负零。你的权重来自 softmax 恒为正，两种写法都行，但整数版更稳。
