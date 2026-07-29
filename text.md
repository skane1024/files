xb_vecNxf16 *restrict pvecIn1;       
xb_vecNxf16 *restrict pvecIn2;  
xb_vecNxf16 *restrict pdvecOut; 
const int32_t vectorizationWidth2X = XCHAL_IVPN_SIMD_WIDTH; /* = 32 *
xb_vecNxf16 vaInData11, vaInData12; 
xb_vecNxf16 vaInData21, vaInData22; 
valign vaOutData = IVP_ZALIGN();    Mode Bit
xb_vecNxf16 dvecOut;    

pdvecOut = (xb_vecNxf16 *)((xb_f16 *)(&output_lm[0]));    //并不是向量寄存器的地址，而是指向一段可以一次加载 N 个 f16 的内存区，告诉编译器这个地址要向量访问
pvecIn1 = (xb_vecNxf16 *)((xb_f16 *)(&input1_lm[0])); 
pvecIn2 = (xb_vecNxf16 *)((xb_f16 *)(&input1_lm[0]));
valign vaInData1;                  
valign vaInData2;                 

xb_vecN_2xf32 fvecOut1, fvecOut2;  

vaInData1 = IVP_LANXF16_PP(pvecIn1);
vaInData2 = IVP_LANXF16_PP(pvecIn2);

IVP_LANXF16_XP(vaInData11, vaIndata1, pvecIn1,64);

xb_vecN_2xf32 fvecIn1DataEven =   IPV_CVTF32NXF16_0(vaInData11);





xb_vecNxf16 表示一个向量寄存器 ，里面装 N 个 f16 (half float)
xb_vecN_2xf32   一个 SIMD 向量寄存器，里面装 N/2 个 float32

Valign  对齐控制寄存器

restrict 关键字  编译器可以假设这个指针不会和别的指针重叠 允许进行更激进的优化
vaInData1 = IVP_LANXF16_PP(pvecIn1); 不加载数据，只计算对齐信息，为后续XP做准备
IVP_LANXF16_XP(vaInData11, vaIndata1, pvecIn1, 64); 才是真正的加载数据， 还会更新指针和对齐状态




Vision 230（IVP）有32个向量寄存器


我想要在VDSP上做MOE expert的计算，就是有两个输入，一个输入是 前面rms_norm的输出，输入的维度是(512,1024) 有512 个token 每个 token 的维度是1024 ， 另外一个输入是expert对应的token 信息， (116,512) 116个专家， 512 个token 每个token下的值代表权重，如果为0表示专家下没有这个token， 就不参与计算， 要遍历每一个专家， 找到对应的token， 然后将对应token的数据gather 组合在一起，调用一个函数计算，得到专家计算的结果， 然后要根据每个专家的权重，做加权求和


Step 1：预处理 routing
	先把 routing matrix 转换成：
	xpert_token_index[e] = {属于这个 expert 的 token 列表}
expert_token_weight[e] = {对应权重}
	比如：
	expert 0:
   tokens = [3, 9, 100, 220]
   weights = [0.8, 0.5, 0.6, 0.9]


Step 2：对每个 expert 做“连续 buffer 打包”
 就是将专家对应的token数据聚集在一起，调用NPU执行

加权 combine 
for i in expert_tokens:
    token_id = expert_token_index[e][i]
    w = weight[e][i]
    for d in 0..1024 step 32:
        load expert_out
        load Y[token_id]
        Y += w * expert_out
        store

xb_vecNxf16 vOut = ...
xb_vecNxf16 vY = ...
xb_vecNxf16 vW = IVP_MOVNXF16(w)
vY = IVP_MULANXF16(vOut, vW, vY)


Vision 有 multiply-accumulate 指令
xb_vecNxf16 vOut;     // expert 输出
xb_vecNxf16 vY;       // 原 Y
xb_vecNxf16 vWeight;  // 广播权重
vWeight = IVP_MOVNXF16(weight);
vY = IVP_MULANXF16(vOut, vWeight, vY);




这里必须是 expert-first” ， 因为权重常驻 LM，只能一次只能容纳加载一个 expert，计算算属于它的 token


preprocess_routing();   // 生成 index 列表
for (e = 0; e < 116; e++)
{
    load_expert_weight(e);
    for (k = 0; k < expert_token_count[e]; k++)
    {
        t = expert_token_index[e][k];
        w = expert_token_weight[e][k];
        // 1. 读 X[t]
        // 2. MLP compute
        // 3. 乘 w
        // 4. 累加回 Y[t]
    }
}


for each expert e:
    找到属于它的 token
    取出这些 token 的 1024 维向量
    做 expert 计算（通常是 MLP）
    得到输出
    乘以对应权重
    加回到输出 token 上






一个输入是expert对应的token 信息， (116,512) 116个专家 其实也可以是别的格式， 这里是初步的方法， 有什么别的形式推荐吗？ 原始的信息是 softmax的输出取 top_K 知道一个token下有那些专家以及权重，怎么样可以经过最少的转换得到我这里需要的最佳格式数据？

原始数据是：
• Softmax + Top-K 输出
• 对每个 token，你知道：expert_id[k]  weight[k]
计算方式是 expert-first，所以需要把它转换成 expert-first 的数据结构

我建议不要转换成 (116,512) 的矩阵
转换成一种压缩索引结构（类似 CSR，Compressed Sparse Row）


expert_offset[117]
expert_token_index[1024]    // 最多 512*K 个元素
expert_weight[1024]

expert_offset[117]
expert_token_index[512*K]
expert_weight[512*K]

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 


整个转换没有构造稠密矩阵，没有扫描 116×512，也没有额外的随机查找，是目前计算量和存储量都比较优的方案。

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 



你现在的情况是：
• ✅ 4 个 cluster
• ✅ 每个 cluster 处理 128 token
• ✅ 原始 routing 是 top‑K（token-first）
• ✅ 最终需要 expert-first 结构
• ✅ 最终要变成 512 token 的整体计算

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 

✅ 合并过程要简单
✅ 尽量不要反复重排
✅ gather 后格式对 expert-first 最友好

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 


每个 cluster 上你现在应该有：
expert_id[128][K]
weight[128][K]


✅ 合并 4 cluster 只需要一次 pass
✅ 没有矩阵
✅ 没有 CSR 合并
✅ 没有排序
✅ 没有二次重排
✅ 内存完全线性写

• 4 个 cluster
• ✅ 每个 cluster 只持有自己 128 个 token 的特征数据
• ✅ cluster 之间内存隔离
• ✅ 需要一个 gather 通信算子（cluster 间通信）把数据聚合
• ✅ gather 通信 必须连续访问，不能跳地址
• ✅ K = 8

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 



uint16_t expert_token_index[4][128*K];
float16  expert_weight[4][128*K];
作为 routing 表，这个格式没问题
✅ 每个 cluster 独立存自己的 routing，天然连续
✅ 后续可以合并成全局，也可以保留分块




• 每个 cluster 的 128*K 是一整块连续内存
• gather 时：

text
cluster0 block → dst[0]
cluster1 block → dst[1]
cluster2 block → dst[2]
cluster3 block → dst[3]
✅ 4 次连续拷贝
✅ 无跳跃
✅ 通信算子最喜欢这种
所以：
通信格式：[4][128*K] ✅ 完全正确

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 




✅ gather 完之后，数据就是 expert-first 格式
✅ 后续计算阶段完全不再做重排
✅ 避免再次 token→expert 转换

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 




cluster0:
    expert0: tokens...
    expert1: tokens...
cluster1:
    expert0: tokens...
    expert1: tokens...

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 


• 每个 cluster 内已经是 expert-first
• 每个 cluster 都知道：
• expert e 在本 cluster 有哪些 token
• 这些 token 的权重
你说：
✅ 当我有这个信息后
✅ 我已经知道 expert e 的所有 token
✅ 只需要 4 个 cluster 循环一遍
✅ 不需要再重排

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 





for (e = 0; e < 116; e++)
{
    load expert weight;
    total_count = 0;
    // 统计总 token 数
    for (c = 0; c < 4; c++)
        total_count += expert_offset[c][e+1] - expert_offset[c][e];
    if (total_count == 0)
        continue;
    // 分配一个 batch buffer
    // 或者直接流式喂给 GEMM
    for (c = 0; c < 4; c++)
    {
        begin = expert_offset[c][e];
        end   = expert_offset[c][e+1];
        if (begin == end)
            continue;
        // 把 cluster c 的 token 特征
        // 追加到当前 expert batch
    }
    batch_compute(all_tokens_of_expert_e);
}

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 




for (e)
{
    begin = expert_offset[e];
    end   = expert_offset[e+1];
    for (i = begin; i < end; i++)
    {
        token = expert_token_index[i];
        weight = expert_weight[i];
        Y[token] += weight * expert_out[i];
    }
}

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 








for (e = 0; e < 116; e++)
{
    for (c = 0; c < 4; c++)
    {
        begin = expert_offset[c][e];
        end   = expert_offset[c][e+1];
        for (i = begin; i < end; i++)
        {
            token  = expert_token_index[c][i];
            weight = expert_weight[c][i];
            Y[token] += weight * expert_out[c][i];
        }
    }
}

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 



✅ 多一层 cluster 循环
✅ 不会增加访存复杂度
✅ 不会增加带宽
✅ 不会增加 SIMD 复杂度

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 








for (e = 0; e < 116; e++)
{
    begin = expert_offset[e];
    end   = expert_offset[e+1];
    load expert weight;
    for (i = begin; i < end; i++)
    {
        token = expert_token_index[i];
        w     = expert_weight[i];
        compute(token);
        accumulate(token, w);
    }
}

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 


整个过程中没有任何 if 判断。
这是目前最推荐的格式。



for (e = 0; e < 116; e++)
{
    begin = expert_offset[e];
    end   = expert_offset[e+1];
    count = end - begin;
    if (count == 0)
        continue;
    load_expert_weight(e);
    // 1️⃣ dispatch
    for (i = 0; i < count; i++)
    {
        token = expert_token_index[begin + i];
        memcpy(expert_input_buffer[i],
               X[token],
               1024 * sizeof(f16));
    }
    // 2️⃣ batch compute
    expert_mlp(
        expert_input_buffer,
        expert_output_buffer,
        count
    );
    // 3️⃣ combine
    for (i = 0; i < count; i++)
    {
        token = expert_token_index[begin + i];
        w     = expert_weight[begin + i];
        Y[token] += w * expert_output_buffer[i];
    }
}

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 





加权求和
希望的逻辑是result[token] += weight * expert_output[token]


xb_vecNxf16 * restrict pOut;
xb_vecNxf16 * restrict pExp;
valign vaOut, vaExp;
xb_vecNxf16 vOut, vExp, vW;
pOut = (xb_vecNxf16 *)(result_token_ptr);
pExp = (xb_vecNxf16 *)(expert_output_ptr);
vaOut = IVP_LANXF16_PP(pOut);
vaExp = IVP_LANXF16_PP(pExp);
// 广播权重
vW = IVP_MOVNXF16(w);
for (int d = 0; d < 1024; d += 32)
{
    IVP_LANXF16_XP(vOut, vaOut, pOut, 64);
    IVP_LANXF16_XP(vExp, vaExp, pExp, 64);
    // vOut = vOut + vExp * vW
    //vOut = IVP_MULANXF16(vExp, vW, vOut);   //IVP_MULANXF16(a, b, c)  c + a * b

  if (first_expert_for_token)
      vOut = IVP_MULNXF16(vExp, vW);
  else
      vOut = IVP_MULANXF16(vExp, vW, vOut);
    IVP_SANXF16_XP(vOut, vaOut, pOut, 64);
}

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 





xb_vecNxf16 vOut16, vExp16;
xb_vecN_2xf32 vOut0, vOut1;
xb_vecN_2xf32 vExp0, vExp1;
xb_vecN_2xf32 vW32;
// 广播成 f32
vW32 = IVP_MOVN_2XF32((float)w);
IVP_LANXF16_XP(vOut16, vaOut, pOut, 64);
IVP_LANXF16_XP(vExp16, vaExp, pExp, 64);
// 拆成两半
vOut0 = IVP_CVTF32NXF16_0(vOut16);
vOut1 = IVP_CVTF32NXF16_1(vOut16);
vExp0 = IVP_CVTF32NXF16_0(vExp16);
vExp1 = IVP_CVTF32NXF16_1(vExp16);
// FMA
vOut0 = IVP_MULAN_2XF32(vExp0, vW32, vOut0);
vOut1 = IVP_MULAN_2XF32(vExp1, vW32, vOut1);
// 转回 f16
vOut16 = IVP_CVT16F32NXF16(vOut0, vOut1);
IVP_SANXF16_XP(vOut16, vaOut, pOut, 64);

来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 


来自 <https://arena.ai/c/019fae01-a143-7ea2-a77b-c5bb350e3ac6> 


