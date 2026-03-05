import numpy as np
import random
import onnx
from collections import defaultdict
from typing import Optional


class TensorInfo:
    """
    描述计算图中的一个 Tensor。
    持有对其 producer Operator 和所有 consumer Operator 的直接引用。
    """
    def __init__(self, name: str, size_bytes: int):
        self.name = name
        self.size_bytes = size_bytes
        self.producer: Optional["Operator"] = None       # 产生此 tensor 的算子
        self.consumers: list["Operator"] = []            # 消费此 tensor 的算子列表
        self.is_model_input: bool = False                # 是否为模型外部输入
        self.is_model_output: bool = False               # 是否为模型最终输出

    def __repr__(self):
        prod_name = self.producer.name if self.producer else "external"
        return (f"Tensor(name={self.name!r}, "
                f"size={self.size_bytes // 1024}KB, "
                f"producer={prod_name!r}, "
                f"consumers={[c.name for c in self.consumers]})")


class Operator:
    """
    描述计算图中的一个算子(Op)。
    直接持有输入/输出 TensorInfo 对象，以及前驱/后继 Operator 引用。
    """
    def __init__(self, name: str, op_type: str = "Unknown"):
        self.name = name
        self.op_type = op_type
        self.inputs: list[TensorInfo] = []               # 输入 tensor 列表
        self.outputs: list[TensorInfo] = []              # 输出 tensor 列表
        self.predecessors: list["Operator"] = []         # 前驱算子（必须先于本算子执行）
        self.successors: list["Operator"] = []           # 后继算子（本算子执行后才能执行）

    def add_input(self, tensor: TensorInfo):
        if tensor not in self.inputs:
            self.inputs.append(tensor)
            if self not in tensor.consumers:
                tensor.consumers.append(self)

    def add_output(self, tensor: TensorInfo):
        if tensor not in self.outputs:
            self.outputs.append(tensor)
            tensor.producer = self

    #添加后继算子
    def add_successor(self, op: "Operator"):
        if op not in self.successors:
            self.successors.append(op)
        if self not in op.predecessors:
            op.predecessors.append(self)

    @property
    def in_degree(self) -> int:
        return len(self.predecessors)

    def __repr__(self):
        return (f"Operator(name={self.name!r}, type={self.op_type!r}, "
                f"inputs={[t.name for t in self.inputs]}, "
                f"outputs={[t.name for t in self.outputs]})")



class ComputeGraph:
    """
    由 Operator 节点和 TensorInfo 边构成的 DAG 计算图。

    关系结构：
        Operator.outputs  ──►  TensorInfo.producer (反向引用)
        TensorInfo        ──►  Operator.inputs      (被消费)
        Operator.predecessors / successors           (拓扑依赖)
    """
    def __init__(self):
        self.operators: dict[str, Operator] = {}         # name -> Operator
        self.tensors: dict[str, TensorInfo] = {}         # name -> TensorInfo
        self.model_inputs: list[TensorInfo] = []         # 外部输入 tensor
        self.model_outputs: list[TensorInfo] = []        # 最终输出 tensor

    # ── 构建接口 ──────────────────────────────────────────────────

    def add_operator(self, op: Operator) -> Operator:
        self.operators[op.name] = op
        return op

    def add_tensor(self, tensor: TensorInfo) -> TensorInfo:
        self.tensors[tensor.name] = tensor
        return tensor

    def get_or_create_tensor(self, name: str, size_bytes: int) -> TensorInfo:
        if name not in self.tensors:
            self.tensors[name] = TensorInfo(name, size_bytes)
        return self.tensors[name]

    def get_or_create_operator(self, name: str, op_type: str = "Unknown") -> Operator:
        if name not in self.operators:
            self.operators[name] = Operator(name, op_type)
        return self.operators[name]

    def connect(self, producer_op: Operator, tensor: TensorInfo,
                consumer_op: Operator):
        """
        建立完整的连接：
          producer_op 产生 tensor → consumer_op 消费 tensor
          同时自动建立算子间的拓扑依赖。
        """
        producer_op.add_output(tensor)
        consumer_op.add_input(tensor)
        producer_op.add_successor(consumer_op)

    # ── 查询接口 ──────────────────────────────────────────────────

    def get_temp_tensors(self) -> list[TensorInfo]:
        """
        返回所有需要计入 temp buffer 的 tensor：
        排除模型外部输入和最终输出。
        """
        excluded = set(self.model_inputs + self.model_outputs)
        return [t for t in self.tensors.values() if t not in excluded]

    def topological_sort_default(self) -> list[Operator]:
        """Kahn 算法生成一个默认的拓扑排序（按名称字典序打破平局）"""
        in_deg = {op: len(op.predecessors) for op in self.operators.values()}
        queue = sorted(
            [op for op, d in in_deg.items() if d == 0],
            key=lambda x: x.name
        )
        result = []
        while queue:
            op = queue.pop(0)
            result.append(op)
            for succ in sorted(op.successors, key=lambda x: x.name):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)
        return result



    def print_graph(self, show_tensor_size: bool = True, show_lifetime: bool = False,
                    execution_order: list["Operator"] | None = None):
        """
        以可视化文本格式输出计算图的完整结构。

        参数：
        - show_tensor_size  : 是否显示每个 tensor 的大小
        - show_lifetime     : 是否显示每个 tensor 的生命周期区间（需提供 execution_order）
        - execution_order   : 算子执行顺序列表，show_lifetime=True 时必须提供
        """

        # ── 预计算 lifetime（可选）──────────────────────────────────
        lifetime_map: dict[str, tuple[int, int]] = {}
        if show_lifetime and execution_order:
            lifetime_map = compute_tensor_lifetimes(execution_order, self)

        # ── 工具函数 ────────────────────────────────────────────────
        def _fmt_size(size_bytes: int) -> str:
            if size_bytes >= 1024 * 1024:
                return f"{size_bytes / 1024 / 1024:.2f}MB"
            elif size_bytes >= 1024:
                return f"{size_bytes / 1024:.1f}KB"
            return f"{size_bytes}B"

        def _fmt_tensor(t: "TensorInfo") -> str:
            tag = ""
            if t.is_model_input:
                tag = " [MODEL INPUT]"
            elif t.is_model_output:
                tag = " [MODEL OUTPUT]"
            size_str = f"  ({_fmt_size(t.size_bytes)})" if show_tensor_size else ""
            lt_str = ""
            if show_lifetime and t.name in lifetime_map:
                born, die = lifetime_map[t.name]
                lt_str = f"  lifetime=[{born}→{die}]"
            return f"{t.name}{size_str}{lt_str}{tag}"

        # ── 确定打印顺序：优先用传入的执行顺序，否则默认拓扑序 ──
        ordered_ops: list["Operator"] = (
            execution_order if execution_order
            else self.topological_sort_default()
        )
        op_index: dict[str, int] = {op.name: i for i, op in enumerate(ordered_ops)}

        # ── 统计信息 ────────────────────────────────────────────────
        total_temp_bytes = sum(t.size_bytes for t in self.get_temp_tensors())
        sep = "═" * 62

        print(f"\n{sep}")
        print(f"  COMPUTE GRAPH")
        print(f"{sep}")
        print(f"  Operators    : {len(self.operators)}")
        print(f"  Tensors      : {len(self.tensors)}")
        print(f"  Temp Tensors : {len(self.get_temp_tensors())}  "
            f"(total {_fmt_size(total_temp_bytes)})")
        print(f"  Model Inputs : {[t.name for t in self.model_inputs]}")
        print(f"  Model Outputs: {[t.name for t in self.model_outputs]}")
        print(f"{sep}\n")

        # ── 逐算子输出 ──────────────────────────────────────────────
        for step, op in enumerate(ordered_ops):
            # 算子标题行
            step_str = f"[{step:02d}]" if execution_order else "    "
            print(f"  {step_str} ┌─ {op.name}  <{op.op_type}>")

            # 输入 tensor
            if op.inputs:
                print(f"       │  ── inputs ──")
                for i, t in enumerate(op.inputs):
                    is_last = (i == len(op.inputs) - 1) and not op.outputs
                    connector = "└" if is_last else "├"
                    print(f"       │  {connector}─ ◄ {_fmt_tensor(t)}")
            else:
                print(f"       │  (no inputs)")

            # 输出 tensor
            if op.outputs:
                print(f"       │  ── outputs ──")
                for i, t in enumerate(op.outputs):
                    is_last = i == len(op.outputs) - 1
                    connector = "└" if is_last else "├"
                    # 标注该 tensor 被哪些算子消费
                    consumer_names = [c.name for c in t.consumers]
                    consumer_str = ""
                    if consumer_names:
                        consumer_str = f"  → consumed by {consumer_names}"
                    print(f"       │  {connector}─ ► {_fmt_tensor(t)}{consumer_str}")

            # 后继算子
            if op.successors:
                succ_names = sorted(op.successors, key=lambda x: op_index.get(x.name, 999))
                print(f"       │  ── next ops ──")
                for i, succ in enumerate(succ_names):
                    is_last = i == len(succ_names) - 1
                    connector = "└" if is_last else "├"
                    print(f"       │  {connector}─ ↓ {succ.name}")

            print(f"       └{'─' * 40}")

        # ── Tensor 生命周期总览（可选）──────────────────────────────
        if show_lifetime and lifetime_map:
            print(f"\n{'─' * 62}")
            print(f"  TENSOR LIFETIME OVERVIEW  (step 0 → {len(ordered_ops)-1})")
            print(f"{'─' * 62}")

            max_steps = len(ordered_ops)
            bar_width = min(max_steps, 40)
            scale = max_steps / bar_width

            # 表头：步骤刻度
            header = "  {:<28}  {}".format("Tensor", "Lifetime")
            print(header)
            print(f"  {'─'*28}  {'─'*bar_width}")

            for t in self.get_temp_tensors():
                if t.name not in lifetime_map:
                    continue
                born, die = lifetime_map[t.name]
                # 绘制生命周期条形图
                bar = [" "] * bar_width
                b_pos = int(born / scale)
                d_pos = min(int(die / scale), bar_width - 1)
                for pos in range(b_pos, d_pos + 1):
                    bar[pos] = "█"
                if b_pos < bar_width:
                    bar[b_pos] = "▶"
                if d_pos < bar_width:
                    bar[d_pos] = "◀" if d_pos != b_pos else "■"
                bar_str = "".join(bar)

                size_str = _fmt_size(t.size_bytes)
                label = f"{t.name} ({size_str})"
                print(f"  {label:<28}  {bar_str}  [{born}→{die}]")

            print(f"{'─' * 62}")





def parse_onnx_to_graph(onnx_path: str) -> ComputeGraph:
    """将 ONNX 模型解析为 ComputeGraph（Operator + TensorInfo 结构）"""
    model = onnx.load(onnx_path)
    onnx_graph = model.graph
    cg = ComputeGraph()

    # ── 收集所有 tensor 的 shape/size 信息 ──
    def _infer_size(type_proto) -> int:
        try:
            shape = type_proto.tensor_type.shape
            dims = [max(d.dim_value, 1) for d in shape.dim]
            bytes_per_elem = {1: 4, 10: 2, 11: 8, 6: 4, 7: 8}.get(
                type_proto.tensor_type.elem_type, 4
            )
            total = 1
            for d in dims:
                total *= d
            return total * bytes_per_elem
        except Exception:
            return 4096

    shape_map: dict[str, int] = {}
    for vi in list(onnx_graph.value_info) + \
               list(onnx_graph.input) + \
               list(onnx_graph.output):
        shape_map[vi.name] = _infer_size(vi.type)

    # ── 标记模型输入/输出 ──
    model_input_names = {inp.name for inp in onnx_graph.input}
    model_output_names = {out.name for out in onnx_graph.output}

    for name in model_input_names:
        t = cg.get_or_create_tensor(name, shape_map.get(name, 4096))
        t.is_model_input = True
        cg.model_inputs.append(t)

    # ── 遍历 ONNX 节点，构建 Operator 和 TensorInfo ──
    for node in onnx_graph.node:
        op_name = node.name if node.name else f"{node.op_type}_{id(node)}"
        op = cg.get_or_create_operator(op_name, node.op_type)

        # 处理输出 tensor
        for out_name in node.output:
            if not out_name:
                continue
            t = cg.get_or_create_tensor(out_name, shape_map.get(out_name, 4096))
            op.add_output(t)
            if out_name in model_output_names:
                t.is_model_output = True
                cg.model_outputs.append(t)

        # 处理输入 tensor（建立与 producer op 的连接）
        for inp_name in node.input:
            if not inp_name:
                continue
            t = cg.get_or_create_tensor(inp_name, shape_map.get(inp_name, 4096))
            op.add_input(t)
            # 如果该 tensor 有 producer，建立算子间依赖
            if t.producer is not None and t.producer is not op:
                t.producer.add_successor(op)

    return cg




def compute_tensor_lifetimes(
    execution_order: list[Operator],
    cg: ComputeGraph
) -> dict[str, tuple[int, int]]:
    """
    给定算子执行顺序，计算每个 temp tensor 的生命周期区间 [born, die]。

    born：产生该 tensor 的 Operator 在 execution_order 中的 index
    die ：最后一个消费该 tensor 的 Operator 的 index
          若无 consumer，则 die = born（产生后即可释放）
    """
    op_index: dict[Operator, int] = {op: i for i, op in enumerate(execution_order)}
    lifetimes: dict[str, tuple[int, int]] = {}

    for tensor in cg.get_temp_tensors():
        # producer 必须在执行序列中
        if tensor.producer is None or tensor.producer not in op_index:
            continue

        born = op_index[tensor.producer]

        # 找所有 consumer 中执行最晚的那个
        active_consumers = [
            c for c in tensor.consumers if c in op_index
        ]
        die = max((op_index[c] for c in active_consumers), default=born)

        lifetimes[tensor.name] = (born, die)

    return lifetimes


def compute_peak_memory(
    execution_order: list[Operator],
    cg: ComputeGraph
) -> int:
    """计算给定执行顺序下的峰值 temp buffer 内存（字节）"""
    lifetimes = compute_tensor_lifetimes(execution_order, cg)
    n_steps = len(execution_order)
    peak = 0

    for t in range(n_steps):
        mem_at_t = sum(
            cg.tensors[name].size_bytes
            for name, (born, die) in lifetimes.items()
            if born <= t <= die
        )
        peak = max(peak, mem_at_t)

    return peak




# ── 工具：获取当前就绪算子 ─────────────────────────────────────
def get_ready_ops(
    executed: set[Operator],
    all_ops: list[Operator]
) -> list[Operator]:
    """返回所有前驱均已执行、自身尚未执行的算子"""
    return [
        op for op in all_ops
        if op not in executed and all(p in executed for p in op.predecessors)
    ]


# ── 策略 1：贪心（内存增量最小优先）────────────────────────────
def greedy_min_memory_order(cg: ComputeGraph) -> tuple[list[Operator], int]:
    """
    每步从就绪队列中选择执行后"净内存增量"最小的算子。
    净增量 = 新产生 tensor 总大小 - 因此 op 执行后可立即释放的 tensor 总大小
    """
    all_ops = list(cg.operators.values())
    executed: set[Operator] = set()
    order: list[Operator] = []

    # 记录每个 tensor 还有多少 consumer 尚未执行
    remaining_consumers: dict[TensorInfo, int] = {
        t: len(t.consumers) for t in cg.get_temp_tensors()
    }

    while len(order) < len(all_ops):
        ready = get_ready_ops(executed, all_ops)
        if not ready:
            break

        best_op, best_delta = None, float('inf')

        for op in ready:
            # 执行 op 新产生的 temp tensor 内存
            produced = sum(
                t.size_bytes for t in op.outputs
                if t in remaining_consumers
            )
            # 执行 op 后可以立即释放的 tensor（op 是其最后一个 consumer）
            freed = sum(
                t.size_bytes for t in op.inputs
                if t in remaining_consumers and remaining_consumers[t] == 1
            )
            delta = produced - freed
            if delta < best_delta:
                best_delta, best_op = delta, op

        order.append(best_op)
        executed.add(best_op)

        # 更新 remaining_consumers
        for t in best_op.inputs:
            if t in remaining_consumers:
                remaining_consumers[t] -= 1

    return order, compute_peak_memory(order, cg)


# ── 策略 2：束搜索（Beam Search）────────────────────────────────
def beam_search_optimal_order(
    cg: ComputeGraph,
    beam_width: int = 16
) -> tuple[list[Operator], int]:
    """
    维护 beam_width 个候选序列，每步扩展所有就绪算子，保留最优的候选。
    """
    all_ops = list(cg.operators.values())
    # beam 中每个元素：(partial_peak, order_so_far, executed_set)
    beam: list[tuple[int, list[Operator], set[Operator]]] = [
        (0, [], set())
    ]

    for _ in range(len(all_ops)):
        candidates = []
        for _, order, executed in beam:
            ready = get_ready_ops(executed, all_ops)
            for op in ready:
                new_order = order + [op]
                new_executed = executed | {op}
                partial_peak = compute_peak_memory(new_order, cg)
                candidates.append((partial_peak, new_order, new_executed))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    best_peak, best_order, _ = beam[0]
    return best_order, best_peak


# ── 策略 3：模拟退火（Simulated Annealing）───────────────────────
def _swap_adjacent_independent(
    order: list[Operator]
) -> list[Operator] | None:
    """
    随机找一对相邻且相互独立的算子并交换，产生新的合法拓扑排序。
    两个算子相互独立 = 彼此不在对方的 predecessors/successors 中。
    """
    indices = list(range(len(order) - 1))
    random.shuffle(indices)
    for i in indices:
        a, b = order[i], order[i + 1]
        if b not in a.successors and a not in b.successors:
            new_order = list(order)
            new_order[i], new_order[i + 1] = b, a
            return new_order
    return None


def simulated_annealing_optimize(
    cg: ComputeGraph,
    initial_order: list[Operator] | None = None,
    T_start: float = 1e6,
    T_end: float = 1.0,
    cooling_rate: float = 0.995,
    max_iter: int = 50000,
    verbose: bool = True
) -> tuple[list[Operator], int]:
    """
    模拟退火全局搜索最优执行顺序，以峰值 temp buffer 内存为优化目标。
    """
    if initial_order is None:
        current_order, _ = greedy_min_memory_order(cg)
    else:
        current_order = list(initial_order)

    current_peak = compute_peak_memory(current_order, cg)
    best_order = list(current_order)
    best_peak = current_peak
    T = T_start

    for iteration in range(1, max_iter + 1):
        if T < T_end:
            break

        new_order = _swap_adjacent_independent(current_order)
        if new_order is None:
            T *= cooling_rate
            continue

        new_peak = compute_peak_memory(new_order, cg)
        delta = new_peak - current_peak

        # Metropolis 准则
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_order = new_order
            current_peak = new_peak

        if current_peak < best_peak:
            best_peak = current_peak
            best_order = list(current_order)

        T *= cooling_rate

        if verbose and iteration % 5000 == 0:
            print(f"  [SA] iter={iteration:6d}  T={T:8.1f}  "
                  f"current={current_peak//1024}KB  best={best_peak//1024}KB")

    return best_order, best_peak







cg = parse_onnx_to_graph("Unet-Segmentation.onnx")
cg.print_graph()

# 2. 带执行顺序编号（step 00, 01...）
order, peak1 = greedy_min_memory_order(cg)
cg.print_graph(execution_order=order)

# 3. 完整输出：执行顺序 + 每个 tensor 的生命周期区间 + 条形图
cg.print_graph(
    show_tensor_size=True,
    show_lifetime=True,
    execution_order=order
)
print(peak1/1024/1024)


from dataclasses import dataclass, field

@dataclass
class MemoryBlock:
    """内存池中的一个已分配块"""
    tensor_name: str
    offset: int        # 起始地址（字节）
    size: int          # 大小（字节）
    born: int          # 分配时刻（执行步骤）
    die: int           # 释放时刻（执行步骤）

    @property
    def end(self) -> int:
        return self.offset + self.size


class MemoryPoolAllocator:
    """
    贪心 best-fit 内存池分配器。
    给定所有 tensor 的生命周期，静态计算每个 tensor 在内存池中的 offset，
    使得生命周期不重叠的 tensor 可以复用同一段内存，
    最终确定内存池的最小总大小。
    """

    def __init__(self, lifetimes: dict[str, tuple[int, int]],
                 tensors: dict[str, "TensorInfo"]):
        self.lifetimes = lifetimes   # tensor_name -> (born, die)
        self.tensors = tensors       # tensor_name -> TensorInfo
        self.blocks: list[MemoryBlock] = []
        self.pool_size: int = 0

    def _overlaps(self, name_a: str, name_b: str) -> bool:
        """判断两个 tensor 的生命周期是否重叠（重叠则不能复用内存）"""
        a0, a1 = self.lifetimes[name_a]
        b0, b1 = self.lifetimes[name_b]
        return not (a1 < b0 or b1 < a0)

    def allocate(self) -> int:
        """
        按 tensor 大小从大到小排序，依次用 best-fit 策略分配 offset。
        返回内存池总大小（字节）。
        """
        # 只分配有生命周期信息的 tensor
        names = [n for n in self.lifetimes]
        # 按大小降序（大 tensor 优先分配，减少碎片）
        names.sort(key=lambda n: self.tensors[n].size_bytes, reverse=True)

        allocated: list[MemoryBlock] = []  # 已分配的块

        for name in names:
            size = self.tensors[name].size_bytes
            born, die = self.lifetimes[name]

            # 找出所有与当前 tensor 生命周期重叠的已分配块
            conflicting_ends = sorted(set(
                b.end for b in allocated if self._overlaps(name, b.tensor_name)
            ))

            # best-fit：从 offset=0 开始，找第一个能放下且不与任何冲突块重叠的位置
            candidate_offsets = [0] + conflicting_ends
            best_offset = None

            for offset in candidate_offsets:
                end = offset + size
                # 检查 [offset, end) 是否与所有冲突块都不重叠
                conflict = any(
                    b.offset < end and offset < b.end
                    for b in allocated
                    if self._overlaps(name, b.tensor_name)
                )
                if not conflict:
                    best_offset = offset
                    break

            if best_offset is None:
                # 追加到内存池末尾
                best_offset = max((b.end for b in allocated), default=0)

            block = MemoryBlock(name, best_offset, size, born, die)
            allocated.append(block)

        self.blocks = allocated
        self.pool_size = max((b.end for b in allocated), default=0)
        return self.pool_size



def visualize_memory_pool(
    cg: "ComputeGraph",
    execution_order: list["Operator"],
    bar_width: int = 60,
    animate: bool = True,
    animate_interval: float = 0.6,
    save_frames: bool = False
) -> "MemoryPoolAllocator":
    """
    动态逐帧展示内存池在每个执行步骤的占用情况。

    参数：
    - bar_width       : 内存池可视化宽度（字符数）
    - animate         : True=逐帧动态刷新（终端动画），False=静态逐帧打印
    - animate_interval: 动画帧间隔（秒）
    - save_frames     : 是否将每帧保存为字符串列表并返回

    返回：MemoryPoolAllocator（含分配结果）
    """
    import time, sys, math

    # ── 1. 计算生命周期 & 分配内存 ──────────────────────────────
    lifetimes = compute_tensor_lifetimes(execution_order, cg)
    allocator = MemoryPoolAllocator(lifetimes, cg.tensors)
    pool_size = allocator.allocate()

    n_steps = len(execution_order)
    blocks = allocator.blocks

    # ── 2. 颜色/符号方案（每个 tensor 分配一个唯一符号）──────────
    SYMBOLS = "▓░▒█▄▀■□●○◆◇▪▫★☆▲△▼▽◀▶"
    tensor_symbol: dict[str, str] = {}
    for i, b in enumerate(sorted(blocks, key=lambda x: x.offset)):
        tensor_symbol[b.tensor_name] = SYMBOLS[i % len(SYMBOLS)]

    # ── 3. 工具函数 ──────────────────────────────────────────────
    def _fmt_size(size_bytes: int) -> str:
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes/1024/1024:.1f}MB"
        elif size_bytes >= 1024:
            return f"{size_bytes/1024:.0f}KB"
        return f"{size_bytes}B"

    def _render_frame(step: int) -> list[str]:
        """渲染第 step 步的内存池状态，返回行列表"""
        op = execution_order[step]
        lines = []

        # ── 标题 ──
        lines.append(f"{'═'*66}")
        lines.append(
            f"  Step [{step:02d}/{n_steps-1}]  Op: {op.name:<16} <{op.op_type}>"
        )
        lines.append(f"  Pool Size: {_fmt_size(pool_size):<10}  "
                     f"Bar width = {bar_width} chars")
        lines.append(f"{'─'*66}")

        # ── 内存池条形图 ──
        # 每个字符代表 pool_size / bar_width 字节
        bytes_per_char = pool_size / bar_width
        bar = [" "] * bar_width

        # 当前步骤存活的 tensor（born <= step <= die）
        alive_blocks = [
            b for b in blocks
            if b.born <= step <= b.die
        ]
        # 刚被分配（born == step）
        just_born = [b for b in blocks if b.born == step]
        # 刚被释放（die == step - 1，即本步开始时已释放）
        just_freed = [b for b in blocks if b.die == step - 1]

        for b in alive_blocks:
            start_char = int(b.offset / bytes_per_char)
            end_char = min(int(math.ceil((b.offset + b.size) / bytes_per_char)),
                           bar_width)
            sym = tensor_symbol[b.tensor_name]
            for c in range(start_char, end_char):
                bar[c] = sym

        lines.append(f"  ┌{'─'*bar_width}┐")
        lines.append(f"  │{''.join(bar)}│  ← Memory Pool")
        lines.append(f"  └{'─'*bar_width}┘")
        lines.append(f"  0{' '*(bar_width-len(_fmt_size(pool_size))-1)}"
                     f"{_fmt_size(pool_size)}")

        # ── 当前存活 tensor 详情 ──
        lines.append(f"{'─'*66}")
        # 计算当前实际占用内存
        alive_bytes = sum(b.size for b in alive_blocks)
        utilization = alive_bytes / pool_size * 100 if pool_size > 0 else 0
        lines.append(
            f"  Alive tensors: {len(alive_blocks):2d}  |  "
            f"Used: {_fmt_size(alive_bytes):<10}  |  "
            f"Utilization: {utilization:5.1f}%"
        )
        lines.append(f"{'─'*66}")

        if alive_blocks:
            lines.append(f"  {'Sym':<4} {'Tensor':<20} {'Offset':>10} "
                         f"{'Size':>8}  {'Lifetime':>10}")
            lines.append(f"  {'─'*4} {'─'*20} {'─'*10} {'─'*8}  {'─'*10}")
            for b in sorted(alive_blocks, key=lambda x: x.offset):
                sym = tensor_symbol[b.tensor_name]
                marker = ""
                if b.born == step:
                    marker = " ◄ NEW"
                elif b.die == step:
                    marker = " ► LAST"
                lines.append(
                    f"  {sym:<4} {b.tensor_name:<20} "
                    f"{_fmt_size(b.offset):>10} "
                    f"{_fmt_size(b.size):>8}  "
                    f"[{b.born:02d}→{b.die:02d}]{marker}"
                )

        # ── 本步事件：新分配 & 即将释放 ──
        if just_born:
            lines.append(f"{'─'*66}")
            lines.append(f"  ▲ Allocated : "
                         + ", ".join(f"{b.tensor_name}({_fmt_size(b.size)})"
                                     for b in just_born))
        if just_freed:
            lines.append(f"  ▼ Freed     : "
                         + ", ".join(f"{b.tensor_name}({_fmt_size(b.size)})"
                                     for b in just_freed))

        # ── 峰值内存水位线 ──
        lines.append(f"{'─'*66}")
        peak_at_step = max(
            (sum(b.size for b in blocks if b.born <= s <= b.die)
             for s in range(n_steps)),
            default=0
        )
        current_pct = alive_bytes / peak_at_step * 100 if peak_at_step > 0 else 0
        watermark_bar = int(current_pct / 100 * 30)
        wbar = "█" * watermark_bar + "░" * (30 - watermark_bar)
        lines.append(
            f"  Peak: {_fmt_size(peak_at_step):<10}  "
            f"Current/Peak: [{wbar}] {current_pct:5.1f}%"
        )
        lines.append(f"{'═'*66}")

        return lines

    # ── 4. 图例 ──────────────────────────────────────────────────
    def _render_legend() -> list[str]:
        lines = [f"  LEGEND:", f"  {'─'*40}"]
        for b in sorted(blocks, key=lambda x: x.offset):
            sym = tensor_symbol[b.tensor_name]
            lines.append(
                f"  {sym}  {b.tensor_name:<22} "
                f"offset={_fmt_size(b.offset):<10} "
                f"size={_fmt_size(b.size)}"
            )
        lines.append(f"  {'─'*40}")
        return lines

    # ── 5. 渲染全流程 ────────────────────────────────────────────
    all_frames = []

    for step in range(n_steps):
        frame_lines = _render_frame(step)
        all_frames.append(frame_lines)

        if animate:
            # 清屏后重绘（终端动画）
            sys.stdout.write("\033[2J\033[H")  # ANSI 清屏
            sys.stdout.write("\n".join(frame_lines) + "\n")
            # 打印图例
            for l in _render_legend():
                sys.stdout.write(l + "\n")
            sys.stdout.flush()
            time.sleep(animate_interval)
        else:
            # 静态模式：逐帧打印，帧间加分隔
            print("\n".join(frame_lines))
            for l in _render_legend():
                print(l)
            print()

    # ── 6. 最终汇总帧 ────────────────────────────────────────────
    summary_lines = [
        f"{'═'*66}",
        f"  MEMORY POOL ALLOCATION SUMMARY",
        f"{'─'*66}",
        f"  Pool Size (min required) : {_fmt_size(pool_size)}",
        f"  Peak Live Memory         : {_fmt_size(max(sum(b.size for b in blocks if b.born <= s <= b.die) for s in range(n_steps)))}",
        f"  Total Tensors Allocated  : {len(blocks)}",
        f"  Execution Steps          : {n_steps}",
        f"{'─'*66}",
        f"  Offset Map (sorted by offset):",
        f"  {'Tensor':<22} {'Offset':>10} {'Size':>10} {'Lifetime':>12}",
        f"  {'─'*22} {'─'*10} {'─'*10} {'─'*12}",
    ]
    for b in sorted(blocks, key=lambda x: x.offset):
        summary_lines.append(
            f"  {b.tensor_name:<22} {_fmt_size(b.offset):>10} "
            f"{_fmt_size(b.size):>10}   [{b.born:02d} → {b.die:02d}]"
        )
    summary_lines.append(f"{'═'*66}")

    if animate:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("\n".join(summary_lines) + "\n")
        sys.stdout.flush()
    else:
        print("\n".join(summary_lines))

    if save_frames:
        allocator.frames = all_frames  # 将帧数据挂载到 allocator 上方便外部使用

    return allocator





# 先用贪心或模拟退火得到最优执行顺序
order, peak = greedy_min_memory_order(cg)
print(f"贪心执行顺序峰值内存: {peak / 1024:.1f} KB")
print(f"执行顺序: {[op.name for op in order]}\n")

# ── 动画模式（在支持 ANSI 的终端中运行）──
allocator = visualize_memory_pool(
    cg,
    execution_order=order,
    bar_width=56,
    animate=True,          # 终端动态刷新
    animate_interval=0.8,  # 每帧停留 0.8 秒
)

# ── 静态模式（Jupyter / 日志输出）──
# allocator = visualize_memory_pool(
#     cg, execution_order=order,
#     bar_width=56, animate=False
# )

print(f"\n最终内存池大小: {allocator.pool_size / 1024:.1f} KB")



import json

def export_memory_pool_html_plotly(
    cg: "ComputeGraph",
    execution_order: list["Operator"],
    output_path: str = "memory_pool.html",
    title: str = "Memory Pool Visualization",
) -> "MemoryPoolAllocator":
    """
    使用 Plotly.js 将内存池分配过程导出为交互式 HTML 动画。

    可视化内容：
    - 上图：内存池布局 Gantt 图（X轴=执行步骤, Y轴=内存地址）
            每个 tensor 是一个矩形色块，支持 hover 详情
    - 下图：逐步内存占用折线图 + 峰值标注
    - 动画帧：逐步播放，高亮当前步骤的存活 tensor
    - 右侧滑块 + 播放按钮控制
    """

    # ── 1. 计算生命周期 & 分配内存 ──────────────────────────────
    lifetimes = compute_tensor_lifetimes(execution_order, cg)
    allocator = MemoryPoolAllocator(lifetimes, cg.tensors)
    pool_size = allocator.allocate()
    blocks = allocator.blocks
    n_steps = len(execution_order)

    def fmt_size(b: int) -> str:
        if b >= 1024 * 1024:
            return f"{b/1024/1024:.2f} MB"
        elif b >= 1024:
            return f"{b/1024:.1f} KB"
        return f"{b} B"

    # ── 2. 颜色方案 ──────────────────────────────────────────────
    # Plotly 内置高对比度调色板
    PALETTE = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]
    tensor_color = {
        b.tensor_name: PALETTE[i % len(PALETTE)]
        for i, b in enumerate(sorted(blocks, key=lambda x: x.offset))
    }

    # ── 3. 构建 Plotly traces ────────────────────────────────────
    #
    # 上图（subplot 1）：每个 tensor 一条 scatter trace，
    # 用 fill='toself' 画矩形（X=步骤区间，Y=地址区间）
    #
    # 下图（subplot 2）：内存占用折线

    # 计算每步存活内存
    memory_curve = []
    for step in range(n_steps):
        alive_bytes = sum(
            b.size for b in blocks if b.born <= step <= b.die
        )
        memory_curve.append(alive_bytes)

    peak_memory = max(memory_curve) if memory_curve else 0
    peak_step = memory_curve.index(peak_memory)

    # ── 上图 traces：每个 tensor 一个矩形 trace ──────────────────
    pool_traces = []
    for b in blocks:
        color = tensor_color[b.tensor_name]
        # 矩形的四个角点（顺时针闭合），X=步骤，Y=地址（KB）
        x_rect = [b.born - 0.4, b.die + 0.4, b.die + 0.4, b.born - 0.4, b.born - 0.4]
        y_rect = [
            b.offset / 1024,
            b.offset / 1024,
            (b.offset + b.size) / 1024,
            (b.offset + b.size) / 1024,
            b.offset / 1024,
        ]
        hover_text = (
            f"<b>{b.tensor_name}</b><br>"
            f"Offset : {fmt_size(b.offset)}<br>"
            f"Size   : {fmt_size(b.size)}<br>"
            f"End    : {fmt_size(b.offset + b.size)}<br>"
            f"Lifetime: step {b.born} → {b.die}"
        )
        pool_traces.append({
            "type": "scatter",
            "x": x_rect,
            "y": y_rect,
            "fill": "toself",
            "fillcolor": color,
            "line": {"color": color, "width": 1},
            "opacity": 0.85,
            "name": b.tensor_name,
            "text": hover_text,
            "hoverinfo": "text",
            "showlegend": True,
            "legendgroup": b.tensor_name,
            "xaxis": "x",
            "yaxis": "y",
        })

    # ── 下图 traces：内存占用折线 ────────────────────────────────
    curve_trace = {
        "type": "scatter",
        "x": list(range(n_steps)),
        "y": [v / 1024 for v in memory_curve],
        "mode": "lines+markers",
        "line": {"color": "#636EFA", "width": 2.5, "shape": "hv"},
        "marker": {"size": 5, "color": "#636EFA"},
        "fill": "tozeroy",
        "fillcolor": "rgba(99,110,250,0.15)",
        "name": "Live Memory",
        "hovertemplate": "Step %{x}<br>Memory: %{y:.1f} KB<extra></extra>",
        "showlegend": False,
        "xaxis": "x2",
        "yaxis": "y2",
    }

    # 峰值标注点
    peak_trace = {
        "type": "scatter",
        "x": [peak_step],
        "y": [peak_memory / 1024],
        "mode": "markers+text",
        "marker": {"size": 10, "color": "#EF553B", "symbol": "star"},
        "text": [f"Peak {fmt_size(peak_memory)}"],
        "textposition": "top center",
        "textfont": {"color": "#EF553B", "size": 11},
        "name": "Peak",
        "hovertemplate": f"Peak Memory: {fmt_size(peak_memory)}<extra></extra>",
        "showlegend": False,
        "xaxis": "x2",
        "yaxis": "y2",
    }

    # ── 当前步骤指示线（垂直线，两个子图各一条）────────────────
    # 用 shape 实现，初始在 step=0

    # ── 4. 构建动画 frames ───────────────────────────────────────
    # 每帧更新：
    #   - 上图中高亮当前步骤存活的 tensor（调整 opacity）
    #   - 下图中当前步骤的标记点
    #   - 垂直指示线位置

    frames = []
    for step in range(n_steps):
        op = execution_order[step]
        alive_names = {b.tensor_name for b in blocks if b.born <= step <= b.die}
        just_born = [b.tensor_name for b in blocks if b.born == step]
        just_freed = [b.tensor_name for b in blocks if b.die == step - 1]

        # 更新每个 tensor trace 的透明度
        frame_traces = []
        for b in blocks:
            is_alive = b.tensor_name in alive_names
            is_new = b.tensor_name in just_born
            opacity = 0.92 if is_alive else 0.08
            line_width = 2.5 if is_new else 1
            frame_traces.append({
                "opacity": opacity,
                "line": {
                    "color": tensor_color[b.tensor_name],
                    "width": line_width
                }
            })

        # 当前步骤高亮点（下图）
        frame_traces.append({})  # curve_trace 不变
        frame_traces.append({})  # peak_trace 不变

        # 构建 hover 注释文字
        alive_list_str = "<br>".join(
            f"{'🆕 ' if n in just_born else '   '}{n} ({fmt_size(next(b.size for b in blocks if b.tensor_name == n))})"
            for n in sorted(alive_names)
        )
        freed_str = ", ".join(just_freed) if just_freed else "none"
        born_str = ", ".join(just_born) if just_born else "none"

        frames.append({
            "name": str(step),
            "data": frame_traces,
            "layout": {
                "shapes": [
                    # 上图垂直指示线
                    {
                        "type": "line",
                        "xref": "x", "yref": "paper",
                        "x0": step, "x1": step,
                        "y0": 0, "y1": 0.58,
                        "line": {"color": "#f97316", "width": 2, "dash": "dot"},
                    },
                    # 下图垂直指示线
                    {
                        "type": "line",
                        "xref": "x2", "yref": "paper",
                        "x0": step, "x1": step,
                        "y0": 0, "y1": 0.38,
                        "line": {"color": "#f97316", "width": 2, "dash": "dot"},
                    },
                ],
                "annotations": [
                    {
                        "xref": "paper", "yref": "paper",
                        "x": 1.01, "y": 1.0,
                        "xanchor": "left", "yanchor": "top",
                        "text": (
                            f"<b>Step {step}/{n_steps-1}</b><br>"
                            f"Op: <b>{op.name}</b> &lt;{op.op_type}&gt;<br>"
                            f"Live: <b>{fmt_size(memory_curve[step])}</b> "
                            f"({memory_curve[step]/peak_memory*100:.1f}% of peak)<br>"
                            f"<br><b>Allocated:</b> {born_str}<br>"
                            f"<b>Freed:</b> {freed_str}<br>"
                            f"<br><b>Alive tensors:</b><br>{alive_list_str}"
                        ),
                        "showarrow": False,
                        "align": "left",
                        "bgcolor": "#1a2332",
                        "bordercolor": "#334155",
                        "borderwidth": 1,
                        "borderpad": 8,
                        "font": {"size": 11, "color": "#e2e8f0", "family": "monospace"},
                    }
                ],
            },
        })

    # ── 5. 播放按钮 & 滑块配置 ───────────────────────────────────
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": -0.06,
            "xanchor": "left", "yanchor": "top",
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 800, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 200},
                        },
                    ],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]

    sliders = [
        {
            "active": 0,
            "steps": [
                {
                    "label": str(i),
                    "method": "animate",
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 150},
                        },
                    ],
                }
                for i in range(n_steps)
            ],
            "x": 0.0, "y": -0.02,
            "len": 0.88,
            "xanchor": "left", "yanchor": "top",
            "currentvalue": {
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "center",
                "font": {"size": 13, "color": "#94a3b8"},
            },
            "transition": {"duration": 150},
            "pad": {"t": 40},
            "bgcolor": "#1e2d3d",
            "bordercolor": "#334155",
            "tickcolor": "#475569",
            "font": {"color": "#64748b", "size": 9},
        }
    ]

    # ── 6. Layout ────────────────────────────────────────────────
    op_tickvals = list(range(n_steps))
    op_ticktext = [f"{i}<br>{execution_order[i].name}" for i in range(n_steps)]

    layout = {
        "title": {
            "text": title,
            "font": {"size": 20, "color": "#e2e8f0"},
            "x": 0.04,
        },
        "paper_bgcolor": "#0f1117",
        "plot_bgcolor": "#0f1117",
        "font": {"color": "#94a3b8", "family": "monospace"},
        "height": 820,
        "margin": {"l": 70, "r": 260, "t": 60, "b": 120},

        # 上图坐标轴
        "xaxis": {
            "title": "Execution Step",
            "domain": [0, 0.88],
            "tickvals": op_tickvals,
            "ticktext": op_ticktext,
            "tickfont": {"size": 9},
            "gridcolor": "#1e2d3d",
            "zerolinecolor": "#334155",
            "range": [-0.6, n_steps - 0.4],
        },
        "yaxis": {
            "title": "Memory Address (KB)",
            "domain": [0.42, 1.0],
            "gridcolor": "#1e2d3d",
            "zerolinecolor": "#334155",
            "range": [-pool_size * 0.02 / 1024, pool_size * 1.05 / 1024],
        },

        # 下图坐标轴
        "xaxis2": {
            "title": "Execution Step",
            "domain": [0, 0.88],
            "anchor": "y2",
            "gridcolor": "#1e2d3d",
            "zerolinecolor": "#334155",
            "range": [-0.6, n_steps - 0.4],
        },
        "yaxis2": {
            "title": "Live Memory (KB)",
            "domain": [0.0, 0.36],
            "anchor": "x2",
            "gridcolor": "#1e2d3d",
            "zerolinecolor": "#334155",
        },

        "legend": {
            "x": 1.01, "y": 0.38,
            "xanchor": "left",
            "bgcolor": "#1a2332",
            "bordercolor": "#334155",
            "borderwidth": 1,
            "font": {"size": 10},
            "title": {"text": "Tensors", "font": {"size": 11}},
        },

        # 初始 shapes（步骤指示线）
        "shapes": [
            {
                "type": "line",
                "xref": "x", "yref": "paper",
                "x0": 0, "x1": 0,
                "y0": 0, "y1": 0.58,
                "line": {"color": "#f97316", "width": 2, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x2", "yref": "paper",
                "x0": 0, "x1": 0,
                "y0": 0, "y1": 0.38,
                "line": {"color": "#f97316", "width": 2, "dash": "dot"},
            },
            # pool size 上限线
            {
                "type": "line",
                "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1,
                "y0": pool_size / 1024, "y1": pool_size / 1024,
                "line": {"color": "#ef4444", "width": 1.5, "dash": "dash"},
            },
        ],

        # 初始 annotations（右侧信息面板 + pool size 标注）
        "annotations": [
            {
                "xref": "paper", "yref": "y",
                "x": 0.89, "y": pool_size / 1024,
                "xanchor": "left",
                "text": f"Pool Size<br>{fmt_size(pool_size)}",
                "showarrow": False,
                "font": {"size": 10, "color": "#ef4444"},
                "align": "left",
            },
            {
                "xref": "paper", "yref": "paper",
                "x": 1.01, "y": 1.0,
                "xanchor": "left", "yanchor": "top",
                "text": f"<b>Step 0/{n_steps-1}</b><br>Press ▶ Play to start",
                "showarrow": False,
                "align": "left",
                "bgcolor": "#1a2332",
                "bordercolor": "#334155",
                "borderwidth": 1,
                "borderpad": 8,
                "font": {"size": 11, "color": "#e2e8f0", "family": "monospace"},
            },
        ],

        "updatemenus": updatemenus,
        "sliders": sliders,
    }

    # ── 7. 组装所有 traces ───────────────────────────────────────
    all_traces = pool_traces + [curve_trace, peak_trace]

    # ── 8. 生成 HTML ─────────────────────────────────────────────
    fig_json = json.dumps(
        {"data": all_traces, "layout": layout, "frames": frames},
        ensure_ascii=False
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; background: #0f1117; }}
    #graph {{ width: 100vw; height: 820px; }}
  </style>
</head>
<body>
  <div id="graph"></div>
  <script>
    const fig = {fig_json};
    Plotly.newPlot('graph', fig.data, fig.layout, {{
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {{
        format: 'png', filename: 'memory_pool', scale: 2
      }}
    }}).then(() => {{
      Plotly.addFrames('graph', fig.frames);
    }});
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ 已导出: {output_path}")
    print(f"  Pool size : {fmt_size(pool_size)}")
    print(f"  Peak live : {fmt_size(peak_memory)}  (step {peak_step})")
    print(f"  Tensors   : {len(blocks)},  Steps: {n_steps}")








# 用贪心 or 模拟退火得到执行顺序
order, peak = greedy_min_memory_order(cg)
print(f"执行顺序峰值内存: {peak / 1024:.1f} KB")

# 导出 HTML
allocator = export_memory_pool_html_plotly(
    cg,
    execution_order=order,
    output_path="memory_pool.html",
    title="UNet Memory Pool Visualization",
)

