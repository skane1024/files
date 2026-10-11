### 在VDSP上处理int8量化数据时，应把浮点乘加转换为"整数MAC + 定点乘子(Multiplier)+移位+饱和"的定点运算流程，充分利用VDSP专有的宽位宽MAC、饱和、打包指令，从而避免浮点单元瓶颈并大幅提高并行度。

这个问题本质上是**量化神经网络（或DSP算子）推理中常见的"反量化-计算-重量化"（dequant-compute-requant）流程**，在VDSP（Vector DSP，向量数字信号处理器，通常具备针对int8/int16的宽位宽SIMD MAC单元）上实现时，核心思路不是先把int8还原成float再计算，而是全程留在整数域，用定点乘子近似浮点缩放系数。下面按照具体的推导和实现步骤展开说明。

#### 量化公式的建立

假设 input_1、input_2、weight 都是按照仿射量化（affine quantization）方式表示的，即：

$$
r = s \cdot (q - z)
$$

其中 $r$ 是真实值，$q$ 是int8整数值，$s$ 是scale，$z$ 是zero point（对称量化时 $z=0$）。分别记：

- input_1: $s_1, z_1, q_1$
- input_2: $s_2, z_2, q_2$
- weight: $s_w, z_w, q_w$（权重一般用对称量化，$z_w=0$）
- output: $s_o, z_o, q_o$

那么真实值域的运算：

$$
r_{out} = r_1 + r_2 \cdot r_w = s_1(q_1-z_1) + s_2 s_w (q_2-z_2)(q_w-z_w)
$$

要把它转成int8输出 $q_o$，需要：

$$
q_o = \mathrm{round}\left(\frac{r_{out}}{s_o}\right) + z_o
$$

直接算这个式子会引入浮点除法和浮点乘法，这正是要避免的部分。

#### 用定点乘子取代浮点缩放系数

标准做法（TFLite、CMSIS-NN、gemmlowp等量化推理库普遍采用）是把每一个浮点缩放因子（如 $s_1/s_o$、$s_2 s_w/s_o$）预先分解成一个**int32定点乘子 $M_0$（Q31格式，落在[0.5,1)区间）加一个右移位数 shift**，使得：

$$
M = s / s_o = M_0 \cdot 2^{-31+shift}
$$

这个分解是离线（编译期/量化calibration阶段）一次性算好的，运行时VDSP只需要做整数乘法和移位，完全不涉及浮点运算。这样一来，原来的浮点乘加：

$$
q_o - z_o \approx M_1 \cdot (q_1-z_1) + M_2 \cdot (q_2-z_2)(q_w-z_w)
$$

其中 $M_1 = s_1/s_o$、$M_2 = s_2 s_w/s_o$，都各自替换成"整数乘法+定点右移"的组合。

#### 在VDSP上的具体执行顺序

1. **零点减法与位宽扩展**：先用向量的widen减法指令，把int8的 $q_1, q_2, q_w$ 减去各自zero point，同时扩展到int16，避免后续乘法溢出。多数VDSP都有专门的"widening subtract"指令一次完成这两步。

2. **乘法累加（MAC）**：用int16×int16→int32的向量乘加指令，一次算出 $(q_2-z_2)(q_w-z_w)$，直接落入int32累加寄存器。这一步是VDSP相比CPU/GPU浮点管线最大的优势所在——很多VDSP的MAC单元本身就是为8/16bit定点乘加设计的，单周期可完成的lane数远高于float32，比如同样128bit宽度下int8可以塞下的通道数是float32的4倍。

3. **定点乘子缩放**：用类似ARM NEON里`VQRDMULH`（定点高位乘、带舍入）这类指令，把int32中间结果乘以预先算好的 $M_0$（Q31格式），取乘积高32位，再用带舍入的右移指令（如`VRSHR`）完成 shift，这一步等价于乘以浮点系数但全程整数运算。

4. **两路结果相加**：input_1项和input_2*weight项分别做完各自的定点缩放后，在int32精度上做向量加法，这样可以保证精度不因为提前截断而损失。

5. **加输出zero point并饱和截断**：加上 $z_o$ 后，用饱和截断指令（如`SSAT`/`VQMOVN`一类的int32→int8窄化打包指令）把结果clamp到int8合法范围[-128,127]，并一次性打包回int8向量寄存器输出。

#### 一些工程上的优化点



另外，如果发现 $s_1$ 和 $s_2 s_w$ 经过量化calibration后数值上比较接近，也可以尝试让二者共用同一个shift，这样两路结果在移位前就能先用int32做加法，只需要一次乘子缩放和一次饱和打包，减少一次向量操作的开销，这在算子融合（fused MAC）场景里比较常见。

总的来说，在VDSP上处理int8量化数据变快的关键不是"用int8去做浮点等价运算"，而是**把浮点缩放系数在离线阶段转换成整数乘子+移位的定点近似**，让整个计算链条（减零点、widen、乘加、定点缩放、饱和打包）全部落在VDSP最擅长的定点SIMD指令集上执行，这样既避免了浮点单元的性能瓶颈，也能利用更窄的位宽实现更高的向量并行度。

### **权重是浮点标量时，量化计算不需要对weight做量化和整数乘法，而是直接把weight折算进定点缩放乘子M2里，整条流水线简化为"两路各自反量化缩放（定点乘子实现）+ 整数域加法 + 重量化饱和输出"，比两个输入都量化时更省一次int×int乘法。**

这个变化看似很小，但对计算路径的影响是本质性的，因为一旦weight退化为浮点标量常数，它就不再参与"仿射量化-反量化"体系，而是直接作为缩放系数的一部分，在离线阶段就能和scale融合成一个数，这样运行时完全不需要对weight做整数乘法。下面把浮点计算和量化计算两条路径分别推导清楚，再做专业对比。

#### 浮点计算路径（Reference / Baseline）

在纯浮点场景下，input_1、input_2 都是float（或先反量化为float），weight本身就是float scalar，计算是最直接的FMA（fused multiply-add）：

$$
output_{fp} = input\_1_{fp} + input\_2_{fp} \times weight
$$

在VDSP上如果全程走浮点，这条路径需要：向量加载两路float32数据、执行一次向量FMA指令（大多数VDSP的浮点单元支持`VFMA`一类指令一次完成乘加）、再向量存储结果。它的瓶颈在于float32的位宽是int8的4倍，同样SIMD宽度下能并行处理的通道数只有int8方案的1/4，同时浮点MAC单元的吞吐通常也低于定点MAC单元，这是纯浮点路径在VDSP上效率低的根本原因。

#### 量化计算路径（int8输入 + float32标量权重）

设input_1、input_2为int8仿射量化，参数分别是 $(s_1,z_1,q_1)$、$(s_2,z_2,q_2)$，输出量化参数为 $(s_o,z_o,q_o)$，weight是已知的浮点标量 $w$（不量化，不带zero point）。真实值域的运算：

$$
r_{out} = s_1(q_1-z_1) + w \cdot s_2(q_2-z_2)
$$

重量化为int8输出：

$$
q_o = \mathrm{round}\!\left(\frac{r_{out}}{s_o}\right) + z_o = \mathrm{round}\Big(M_1\cdot(q_1-z_1) + M_2\cdot(q_2-z_2)\Big) + z_o
$$

这里两个缩放系数分别是：

$$
M_1 = \frac{s_1}{s_o}, \qquad M_2 = \frac{w \cdot s_2}{s_o}
$$

关键点在于 $M_2$ 已经把浮点权重 $w$ 直接吸收进去了，$w$ 只在离线阶段（模型转换/校准阶段）参与一次浮点乘法运算，运行时VDSP侧完全看不到这个浮点数，只看到预先算好的定点乘子 $M_{2,0}$（Q31定点表示）和对应的移位量 $shift_2$，满足：

$$
M_2 \approx M_{2,0} \cdot 2^{-(31-shift_2)}
$$

$M_1$ 的分解方式相同。因此运行时的整数化流程是：

1. **零点减法+位宽扩展**：向量化地把 $q_1,q_2$ 减去各自zero point并widen到int16/int32，避免后续溢出。
2. **定点乘子缩放**：对 $(q_1-z_1)$ 用 $M_{1,0}$ 做定点高位乘（如`VQRDMULH`）并按$shift_1$舍入右移；对 $(q_2-z_2)$ 用 $M_{2,0}$、$shift_2$做同样操作。注意这一步只是"整数×定点常数"，不涉及两个变量之间的乘法，因此比"weight也是int8"的情形少了一次向量对向量的int×int MAC。
3. **整数域加法**：把两路缩放后的int32结果直接向量相加。
4. **加输出zero point+饱和窄化**：加 $z_o$ 后用饱和指令截断到int8范围，打包输出。

#### 两条路径的专业对比

| 维度         | 浮点路径                     | 量化路径（weight为float标量）                                |
| ------------ | ---------------------------- | ------------------------------------------------------------ |
| 数据位宽     | float32，SIMD并行度低        | int8/int16为主，SIMD并行度高（同宽度下可提升约4倍通道数）    |
| 权重参与方式 | 直接以float乘法参与运行时FMA | 离线阶段融合进定点乘子$M_2$，运行时不出现浮点乘法            |
| 核心算子     | 向量FMA（float MAC）         | 向量widen减法 + 定点高位乘（fixed-point multiply-high）+ 舍入右移 + 饱和窄化 |
| 精度控制点   | 由浮点单元原生保证           | 依赖$M_0$/shift分解精度及round-to-nearest的舍入策略          |
| VDSP吞吐瓶颈 | 浮点MAC单元吞吐与位宽        | 定点MAC单元吞吐远高于浮点，瓶颈通常在饱和/打包等窄化指令     |

值得强调的是，因为weight在这个场景里始终是标量而非逐通道（per-channel）向量，$M_2$、$shift_2$只需要离线算一次、运行时作为标量广播给所有向量lane即可，不需要像weight被量化且per-channel时那样维护一张按通道索引的乘子表，这也是"weight保持浮点标量"相比"weight也int8量化"在实现上更简洁的地方——本质上是省掉了一次真正的乘法运算（weight量化后需要int×int的MAC，而标量浮点权重只需离线一次乘法就能吸收进整数域的定点乘子）。

### **重量化为int8输出的核心公式是先算出真实值域结果，再除以输出scale、加输出zero point、四舍五入并饱和截断到int8范围，用定点乘子实现时写作 $q_o = \mathrm{round}(M_1\cdot(q_1-z_1) + M_2\cdot(q_2-z_2)) + z_o$。**

下面把这个公式从头到尾按标准的仿射量化（affine quantization）定义重新推导一遍，并修正之前的排版问题。

#### 仿射量化的基本定义

对任意一个量化张量/标量，真实值 $r$ 与量化整数值 $q$ 的关系是：

$$
r = s \cdot (q - z)
$$

其中 $s$ 是scale（浮点正数），$z$ 是zero point（整数，对称量化时 $z=0$）。反过来，从真实值得到量化整数值的公式是：

$$
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{r}{s}\right) + z,\ q_{min},\ q_{max}\right)
$$

对int8而言，$q_{min}=-128$，$q_{max}=127$（若是无符号int8则是$0\sim255$）。

#### 本例中的真实值域计算

设input_1、input_2的量化参数分别是 $(s_1,z_1,q_1)$、$(s_2,z_2,q_2)$，weight是浮点标量 $w$，输出量化参数是 $(s_o,z_o,q_o)$。真实值域上的运算是：

$$
r_{out} = s_1 (q_1 - z_1) + w \cdot s_2 (q_2 - z_2)
$$

#### 重量化为int8输出的完整公式

把 $r_{out}$ 代入前面的量化公式，就得到重量化到int8输出 $q_o$ 的完整表达式：

$$
q_o = \mathrm{clip}\left(\mathrm{round}\left(\frac{r_{out}}{s_o}\right) + z_o,\ -128,\ 127\right)
$$

展开 $r_{out}$ 并把缩放系数提出来，定义：

$$
M_1 = \frac{s_1}{s_o}, \qquad M_2 = \frac{w \cdot s_2}{s_o}
$$

则最终的重量化公式为：

$$
q_o = \mathrm{clip}\Big(\mathrm{round}\big(M_1\cdot(q_1-z_1) + M_2\cdot(q_2-z_2)\big) + z_o,\ -128,\ 127\Big)
$$

这就是标准形式：**两路输入各自减去zero point后，分别乘以对应的缩放系数 $M_1$、$M_2$，相加，四舍五入，加上输出zero point，最后饱和截断到int8的表示范围**。

#### 在VDSP上对应的定点化写法

由于 $M_1$、$M_2$ 都是普通浮点数，运行时不能直接做浮点乘法，所以离线阶段要把它们分解成"int32定点乘子 + 右移位数"的组合，即：

$$
M_i \approx M_{i,0} \cdot 2^{-n_i}, \qquad M_{i,0} \in [2^{30}, 2^{31}) \ (\text{Q31定点表示})
$$

于是运行时的整数化公式变成：

$$
q_o = \mathrm{clip}\Big(\mathrm{RoundShift}\big(M_{1,0}\cdot(q_1-z_1),\ n_1\big) + \mathrm{RoundShift}\big(M_{2,0}\cdot(q_2-z_2),\ n_2\big) + z_o,\ -128,\ 127\Big)
$$

这里 $\mathrm{RoundShift}(x, n)$ 表示"取乘积的高位部分并做带舍入的右移 $n$ 位"，对应VDSP上一条定点高位乘指令加一条舍入右移指令；最外层的$\mathrm{clip}$对应一条饱和窄化指令，把int32结果截断打包回int8。整条链路从头到尾都不出现浮点运算，这也是它比直接用float计算更快的根本原因。

### **下面给出一个基于Cadence Tensilica Vision（P6/Q7系列，如Vision 230）IVP向量intrinsic风格的典型int8量化MAC算子实现，核心是widen减零点、定点乘子高位乘、舍入右移、饱和窄化四步，全程不出现浮点运算。**

Cadence的Vision系列DSP（包括Vision P6、Q6、Q7，"230"这类型号通常指某个具体的向量宽度/主频配置）在C层面主要通过Tensilica的**IVP（Image/Vision Processing）intrinsic**编程，向量寄存器类型如`xb_vec2Nx8`（2N个int8）、`xb_vecNx16`（N个int16）等，配合`IVP_*`前缀的intrinsic完成向量化的减法、乘加、移位、饱和操作。下面按照前面推导的重量化公式实现这个算子，并附带说明每一步对应的硬件行为。

#### 算子代码

```c
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
    int8_t   z1, z2, zo;   /* 三路zero point */
    int32_t  M1_0, M2_0;   /* Q31定点乘子 */
    int8_t   shift1, shift2; /* 对应右移位数(>=0) */
} vdsp_quant_param_t;

void vdsp_quant_madd_int8(
    const int8_t *in1,     /* input_1 量化数据 */
    const int8_t *in2,     /* input_2 量化数据 */
    int8_t       *out,     /* 输出量化数据 */
    int           len,     /* 元素个数 */
    const vdsp_quant_param_t *p)
{
    /* 把zero point、乘子广播成向量寄存器，循环外只做一次 */
    xb_vec2Nx8   vz1  = IVP_ADDN2X8(0, p->z1);
    xb_vec2Nx8   vz2  = IVP_ADDN2X8(0, p->z2);
    xb_vec2Nx8   vzo  = IVP_ADDN2X8(0, p->zo);
    xb_vecNx32   vM1  = IVP_ADDN_2X32(0, p->M1_0);
    xb_vecNx32   vM2  = IVP_ADDN_2X32(0, p->M2_0);

    int step = 2 * XCHAL_IVPN_SIMD_WIDTH;  /* 每次处理的向量宽度(int8通道数) */

    for (int i = 0; i < len; i += step) {
        /* 1. 向量加载int8数据 */
        xb_vec2Nx8 v1 = IVP_LV2NX8_I((xb_vec2Nx8 *)(in1 + i), 0);
        xb_vec2Nx8 v2 = IVP_LV2NX8_I((xb_vec2Nx8 *)(in2 + i), 0);

        /* 2. 减零点 + widen 到int16，避免后续乘法溢出 */
        xb_vecNx16 v1_lo, v1_hi, v2_lo, v2_hi;
        IVP_SUBWN2X8(v1_lo, v1_hi, v1, vz1);
        IVP_SUBWN2X8(v2_lo, v2_hi, v2, vz2);

        /* 3. int16 -> int32，做定点高位乘 (等价于 x * M_0 取高32位)，
              这一步在硬件上对应一条乘加流水，输出即为 M*(q-z) 的近似值 */
        xb_vecNx32 r1_lo = IVP_MULNX16PACKL_0(v1_lo, vM1);
        xb_vecNx32 r1_hi = IVP_MULNX16PACKL_0(v1_hi, vM1);
        xb_vecNx32 r2_lo = IVP_MULNX16PACKL_0(v2_lo, vM2);
        xb_vecNx32 r2_hi = IVP_MULNX16PACKL_0(v2_hi, vM2);

        /* 4. 按各自shift做带舍入的右移 (round-to-nearest) */
        r1_lo = IVP_SRARIN_2X32(r1_lo, p->shift1);
        r1_hi = IVP_SRARIN_2X32(r1_hi, p->shift1);
        r2_lo = IVP_SRARIN_2X32(r2_lo, p->shift2);
        r2_hi = IVP_SRARIN_2X32(r2_hi, p->shift2);

        /* 5. 两路结果在int32精度上相加，保证不提前截断丢精度 */
        xb_vecNx32 sum_lo = IVP_ADDN_2X32(r1_lo, r2_lo);
        xb_vecNx32 sum_hi = IVP_ADDN_2X32(r1_hi, r2_hi);

        /* 6. 加输出zero point */
        sum_lo = IVP_ADDN_2X32(sum_lo, (xb_vecNx32)vzo);
        sum_hi = IVP_ADDN_2X32(sum_hi, (xb_vecNx32)vzo);

        /* 7. 饱和窄化回int8并打包 (SSAT到[-128,127]后narrow) */
        xb_vec2Nx8 vout = IVP_PACKVRNR2NX24_0(sum_hi, sum_lo);

        /* 8. 向量存储输出 */
        IVP_SV2NX8_I(vout, (xb_vec2Nx8 *)(out + i), 0);
    }
}
```

#### 每一步对应的硬件语义说明

第1步的向量加载指令`IVP_LV2NX8`一次能读入一整个向量寄存器宽度的int8数据（宽度由具体型号的SIMD lane数决定，Vision 230这类配置通常支持64或128字节的向量宽度），这是VDSP相比标量CPU吞吐更高的第一个来源。

第2步的`IVP_SUBWN2X8`是"宽化减法"（widening subtract）指令，一次完成减zero point和位宽从int8扩展到int16两件事，避免了先做减法再单独做sign-extend的额外开销，这类合并指令是Cadence Vision IVP指令集里针对量化推理专门优化的地方。

第3、4步是整条链路里最关键的部分：`IVP_MULNX16PACKL`族指令对应的是"定点乘法取高位"（等价于ARM里的`SMMUL`/`VQRDMULH`语义），把int16数据和Q31格式的定点乘子相乘后只保留高32位有效部分，这正好实现了 $M_{i,0}\cdot(q_i-z_i)$ 这一步而不需要任何浮点单元参与；随后的`IVP_SRARIN`是带舍入的算术右移（round arithmetic shift right），对应公式里"除以$2^{shift_i}$并四舍五入"这一操作。

第5\~6步把两路缩放后的int32结果相加、再加输出zero point，全部保持在int32精度上进行，避免中间截断造成的精度损失，这与之前推导的公式 $\mathrm{round}(M_1\cdot(q_1-z_1)+M_2\cdot(q_2-z_2))+z_o$ 完全对应。

第7步的`IVP_PACKVRNR`族指令是饱和打包指令（saturating pack narrow rounding），把int32结果先做\[-128,127\]范围的饱和截断，再窄化打包回int8并合并两路（lo/hi）向量为一个完整宽度的输出向量，这一步就是公式里最外层的$\mathrm{clip}$操作。

需要说明的是，上面代码中的具体intrinsic名称（如`IVP_SUBWN2X8`、`IVP_MULNX16PACKL_0`、`IVP_PACKVRNR2NX24_0`等）是按照Cadence Tensilica Vision系列IVP intrinsic的一般命名规律和典型用法给出的示意性实现，实际到具体型号（如Vision 230对应的具体xtensa配置）落地时，需要对照该型号随芯片/工具链发布的`xt_ivpn.h`头文件和用户手册核对精确的intrinsic签名、向量宽度常量（`XCHAL_IVPN_SIMD_WIDTH`）以及是否存在更贴合这个MAC模式的融合指令（很多Vision型号会针对"widen-mul-add-narrow"这种量化推理常见模式提供单条融合intrinsic，若存在应优先使用以进一步减少指令数和寄存器压力）。

### **下面在之前的算子代码基础上，为每一条IVP intrinsic指令都补充了逐行含义注释，说明该指令在向量硬件上具体做了什么操作，便于逐步核对每一步与量化公式的对应关系。**



需要再次强调的是，这里每条intrinsic的命名和精确行为（例如`IVP_SUBWN2X8`是否严格按sign-extend方式widen、`IVP_MULNX16PACKL_0`取的是乘积的哪一段高位、`IVP_PACKVRNR2NX24_0`的舍入模式细节等）是按照Cadence Tensilica Vision系列IVP intrinsic的通用命名规律和典型行为给出的示意性说明；落地到Vision 230具体型号时，务必对照该芯片配套发布的`xt_ivpn.h`头文件注释和Xtensa ISA用户手册逐条核对真实的位宽、舍入规则和溢出处理方式，因为不同Vision子型号（P6/Q6/Q7）的向量宽度和部分融合指令集是有差异的。



第一部分`quantize`/`dequantize`实现最基础的仿射量化与反量化，`quantize`里特意用`np.round`而不是直接截断，是为了和真实量化推理里round-to-nearest的做法一致；返回值用`int32`容器承载int8数值范围，方便后续中间计算不必频繁做类型提升。

第二部分`quantize_multiplier`是整段代码里最关键的函数，它把之前推导公式里的浮点缩放系数 $M_1=s_1/s_o$、$M_2=w\cdot s_2/s_o$ 分解成"Q31定点整数$M_0$ + 右移位数shift"的组合，这一步严格对应真实VDSP/CMSIS-NN实现里的乘子生成逻辑；`round_shift`函数则模拟了VDSP上"定点高位乘+带舍入右移"这一复合操作，用`int64`避免中间乘积溢出，同时用`rounding = 1 << (total_shift-1)`实现四舍五入而非简单截断，这与之前代码里`IVP_SRARIN`指令的语义一致。

第三部分`quant_madd_int8`就是把前面推导的公式 $q_o=\mathrm{clip}(\mathrm{round}(M_1(q_1-z_1)+M_2(q_2-z_2))+z_o,-128,127)$ 直接翻译成整数运算代码，全程不出现浮点乘法，只在离线阶段（生成M1_0/shift1/M2_0/shift2时）用了浮点。

第四部分是验证环节：先算出浮点参考结果`output_ref_fp`作为"金标准"，再走一遍完整的量化路径得到`q_o`，反量化回浮点后与参考结果逐样本比较绝对误差和相对误差，并统计"误差落在1个LSB（即输出scale$s_o$，是int8量化能表达的最小分辨单位）以内"的样本占比——这是量化推理里最常用的精度验收标准，如果这个比例很高（通常要求接近或等于100%），说明定点乘子分解和移位舍入策略是准确的，量化路径的精度损失基本只来自int8本身的量化粒度，而不是定点运算近似引入的额外误差。

运行这段代码时，你会看到M1、M2被分解出的具体Q31乘子和shift值，以及最终的MAE、最大误差和"误差在1\~2个LSB以内"的比例，可以据此判断当前的weight取值、输入数据范围、量化scale选择是否会带来超出预期的精度损失，如果发现误差偏大，通常可以先检查`s1`、`s2`、`s_o`的取值是否让有效数据范围覆盖了int8的大部分动态范围（即校准阶段的scale是否设置合理），这往往比定点乘子分解本身的精度更容易成为误差来源。

*内容由 AI 生成仅供参考*