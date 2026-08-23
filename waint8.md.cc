
#include <xtensa/tie/xt_ivpn.h>   /* Cadence Vision IVP intrinsics头文件 */
#include <stdint.h>

/*
 * int8量化算子: output = input_1 + input_2 * weight
 * weight为浮点标量，已在离线阶段折算进 M1, M2 定点乘子
 *
 * 量化公式:
 *   q_o = clip( round( M1*(q1-z1) + M2*(q2-z2) ) + z_o, -128, 127 )
 *   Mi  = Mi_0 * 2^(-shift_i)，Mi_0为Q31定点乘子（int32）
 */

typedef struct {
    int8_t   z1, z2, zo;      /* 三路zero point */
    int32_t  M1_0, M2_0;      /* Q31定点乘子 */
    int8_t   shift1, shift2;  /* 对应右移位数(>=0) */
} vdsp_quant_param_t;

void vdsp_quant_madd_int8(
    const int8_t *in1,     /* input_1 量化数据 */
    const int8_t *in2,     /* input_2 量化数据 */
    int8_t       *out,     /* 输出量化数据 */
    int           len,     /* 元素个数 */
    const vdsp_quant_param_t *p)
{
    /* --- 循环外准备：把标量参数广播成向量寄存器，避免每次循环重复构造 --- */

    /* IVP_ADDN2X8(0, z1): 把标量z1复制到2N个int8通道里，
       等价于"0 + z1"广播，构造出一个全通道都是z1的向量寄存器 vz1 */
    xb_vec2Nx8   vz1  = IVP_ADDN2X8(0, p->z1);  //int8向量寄存器

    /* 同上，广播z2到向量寄存器 vz2 */
    xb_vec2Nx8   vz2  = IVP_ADDN2X8(0, p->z2);

    /* 同上，广播输出zero point zo到向量寄存器 vzo */
    xb_vec2Nx8   vzo  = IVP_ADDN2X8(0, p->zo);

    /* IVP_ADDN_2X32(0, M1_0): 把32位定点乘子M1_0广播到N个int32通道，
       构造出全通道都是M1_0的向量寄存器 vM1，供后续每个数据通道复用同一个乘子 */
    xb_vecNx32   vM1  = IVP_ADDN_2X32(0, p->M1_0);  //int32向量寄存器

    /* 同上，广播M2_0到向量寄存器 vM2 */
    xb_vecNx32   vM2  = IVP_ADDN_2X32(0, p->M2_0);  //int32向量寄存器

    /* XCHAL_IVPN_SIMD_WIDTH: 该型号向量寄存器每次能并行处理的int8通道数(硬件常量)，
       step即为每次循环实际推进的元素个数 */
    int step = 2 * XCHAL_IVPN_SIMD_WIDTH;

    for (int i = 0; i < len; i += step) {

        /* --- 步骤1: 向量加载 --- */

        /* IVP_LV2NX8_I: 从内存地址(in1+i)处连续读入一整个向量寄存器宽度的int8数据，
           "I"后缀表示这是立即数偏移的普通加载(non-aligned/aligned视具体型号而定) */
        xb_vec2Nx8 v1 = IVP_LV2NX8_I((xb_vec2Nx8 *)(in1 + i), 0); //加载input_1对应的一段int8数据

        /* 同上，加载input_2对应的一段int8数据 */
        xb_vec2Nx8 v2 = IVP_LV2NX8_I((xb_vec2Nx8 *)(in2 + i), 0); //加载input_2对应的一段int8数据

        /* --- 步骤2: 减零点 + 位宽扩展（widening subtract） --- */

        /* IVP_SUBWN2X8(v1_lo, v1_hi, v1, vz1):
           一次完成两件事——(a) 用v1中每个int8通道减去vz1对应通道(即减去zero point z1)，
           (b) 把差值从8bit有符号扩展(sign-extend)到16bit，
           并把结果拆成低半区v1_lo和高半区v1_hi两个int16向量(因为一个2N宽的int8向量
           扩展成int16后需要两个N宽的寄存器才能装下)，避免后续乘法时8bit溢出 */
        xb_vecNx16 v1_lo, v1_hi, v2_lo, v2_hi; //int16向量寄存器
        IVP_SUBWN2X8(v1_lo, v1_hi, v1, vz1); //input_1侧的差值
        IVP_SUBWN2X8(v2_lo, v2_hi, v2, vz2); //input_2侧的差值

        /* --- 步骤3: 定点高位乘（对应公式里的 Mi_0 * (qi - zi)） --- */

        /* IVP_MULNX16PACKL_0(v1_lo, vM1):
           把int16向量v1_lo与int32定点乘子vM1逐通道相乘，乘积本应是48bit左右的宽结果，
           该指令只截取乘积的高位有效部分(等价于ARM里"取乘法结果高32位"的定点乘语义)，
           这一步就是用整数乘法近似实现浮点缩放系数M1的乘法，不使用任何浮点单元 */
        xb_vecNx32 r1_lo = IVP_MULNX16PACKL_0(v1_lo, vM1);
        xb_vecNx32 r1_hi = IVP_MULNX16PACKL_0(v1_hi, vM1);

        /* 同上，用定点乘子vM2对input_2一侧的差值做定点高位乘 */
        xb_vecNx32 r2_lo = IVP_MULNX16PACKL_0(v2_lo, vM2);
        xb_vecNx32 r2_hi = IVP_MULNX16PACKL_0(v2_hi, vM2);

        /* --- 步骤4: 带舍入的算术右移（round-to-nearest shift），
                     对应公式里"除以2^shift"这一操作 --- */

        /* IVP_SRARIN_2X32(r1_lo, shift1):
           对int32向量r1_lo做算术右移shift1位，并在移位前自动加上0.5*2^shift1的舍入量，
           实现"四舍五入"而不是简单的截断右移，避免系统性的负偏差 */
        r1_lo = IVP_SRARIN_2X32(r1_lo, p->shift1);
        r1_hi = IVP_SRARIN_2X32(r1_hi, p->shift1);
        r2_lo = IVP_SRARIN_2X32(r2_lo, p->shift2);
        r2_hi = IVP_SRARIN_2X32(r2_hi, p->shift2);

        /* --- 步骤5: 两路缩放结果相加（对应公式里 M1*(...) + M2*(...)） --- */

        /* IVP_ADDN_2X32: 逐通道int32向量加法，把input_1路的缩放结果和
           input_2*weight路的缩放结果在int32精度上相加，避免提前截断丢精度 */
        xb_vecNx32 sum_lo = IVP_ADDN_2X32(r1_lo, r2_lo);
        xb_vecNx32 sum_hi = IVP_ADDN_2X32(r1_hi, r2_hi);

        /* --- 步骤6: 加输出zero point --- */

        /* 同样用IVP_ADDN_2X32做int32向量加法，把广播好的vzo(输出零点)
           加到sum上，对应公式里最外层的 "+ z_o" */
        sum_lo = IVP_ADDN_2X32(sum_lo, (xb_vecNx32)vzo);
        sum_hi = IVP_ADDN_2X32(sum_hi, (xb_vecNx32)vzo);

        /* --- 步骤7: 饱和窄化打包（对应公式里的clip到[-128,127]） --- */

        /* IVP_PACKVRNR2NX24_0(sum_hi, sum_lo):
           把两个int32向量(低/高半区)按饱和方式(saturating)窄化打包成一个
           2N宽的int8向量：任何超出[-128,127]范围的值会被夹紧(clamp)到边界值，
           "R"表示打包前带舍入，"NR"组合体现的是"round + narrow"，
           这一条指令同时完成了clip和int32->int8的位宽收窄 */
        xb_vec2Nx8 vout = IVP_PACKVRNR2NX24_0(sum_hi, sum_lo);

        /* --- 步骤8: 向量存储 --- */

        /* IVP_SV2NX8_I: 把打包好的int8向量结果一次性写回内存地址(out+i)处，
           完成一整个向量宽度的输出 */
        IVP_SV2NX8_I(vout, (xb_vec2Nx8 *)(out + i), 0);
    }
}