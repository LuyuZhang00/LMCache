# mem_kernels.cu 算子详解与昇腾适配指南

> 本文档详细分析 `csrc/mem_kernels.cu` 中所有 CUDA 算子的实现原理，并为昇腾（Ascend NPU）适配提供具体的参考建议。
>
> 版本基准：LMCache `dev` 分支

---

## 目录

1. [算子全景图](#1-算子全景图)
2. [核心算子详解：multi_layer_kv_transfer](#2-核心算子详解multi_layer_kv_transfer)
3. [其他算子详解](#3-其他算子详解)
4. [昇腾适配建议：应该参考哪个算子](#4-昇腾适配建议应该参考哪个算子)
5. [昇腾适配的伪代码实现](#5-昇腾适配的伪代码实现)
6. [关键设计决策与注意事项](#6-关键设计决策与注意事项)
7. [附录 A：完整五层调用链路分析（vLLM → CUDA Kernel）](#附录-a完整的五层调用链路分析vllm--cuda-kernel)
8. [附录 B：关键代码文件索引](#附录-b关键代码文件索引)

---

## 1. 算子全景图

`csrc/mem_kernels.cu` 包含 **8 个算子**，分为三类：

### 1.1 算子分类表

| 类别 | 算子名称 | Kernel 函数 | 用途 | 状态 | 推荐适配 |
|------|---------|------------|------|------|---------|
| **全层传输（vLLM）** | `multi_layer_kv_transfer` | `load_and_reshape_multi_layer_kernel` | vLLM 全层 KV 传输 | **主算子** | ⭐⭐⭐ **首选** |
| **全层传输（SGLang）** | `multi_layer_kv_transfer_unilateral` | `load_and_reshape_multi_layer_kernel_unilateral` | SGLang K/V 分离传输 | 活跃 | ⭐⭐ |
| **逐层传输（vLLM）** | `single_layer_kv_transfer` | `single_layer_kv_transfer_kernel` | vLLM 单层 KV 传输 | 活跃 | ⭐⭐ |
| **逐层传输（SGLang）** | `single_layer_kv_transfer_sgl` | `single_layer_kv_transfer_sgl_kernel` | SGLang 单层 KV 传输 | 活跃 | ⭐ |
| **块级传输** | `multi_layer_block_kv_transfer` | `multi_layer_block_transfer_kernel` | 多进程/块级传输 | 活跃 | ⭐⭐ |
| **异步内存拷贝** | `lmcache_memcpy_async` | 无 Kernel（cudaMemcpyAsync） | 钉扎内存异步拷贝 | 活跃 | ⭐ |
| **旧版 Flash 加载** | `load_and_reshape_flash` | `load_and_reshape_flash_kernel` | 旧版单层加载 | **已废弃** | ❌ |
| **旧版 Flash 回写** | `reshape_and_cache_back_flash` | `reshape_and_cache_back_flash_kernel` | 旧版单层回写 | **已废弃** | ❌ |

### 1.2 算子调用关系

```
GPUConnector (Python)
  │
  ├── VLLMPagedMemGPUConnectorV2/V3
  │     └── lmc_ops.multi_layer_kv_transfer()        ← ⭐ 主算子
  │           └── load_and_reshape_multi_layer_kernel
  │
  ├── SGLangGPUConnector
  │     └── lmc_ops.multi_layer_kv_transfer_unilateral()
  │           └── load_and_reshape_multi_layer_kernel_unilateral
  │
  ├── VLLMPagedMemLayerwiseGPUConnector
  │     └── lmc_ops.single_layer_kv_transfer()        ← 逐层算子
  │           └── single_layer_kv_transfer_kernel
  │
  ├── SGLangLayerwiseGPUConnector
  │     └── lmc_ops.single_layer_kv_transfer_sgl()
  │           └── single_layer_kv_transfer_sgl_kernel
  │
  └── TRTLLMGPUConnector / MP 路径
        └── lmc_ops.multi_layer_block_kv_transfer()   ← 块级算子
              └── multi_layer_block_transfer_kernel (mp_mem_kernels.cu)
```

---

## 2. 核心算子详解：multi_layer_kv_transfer

### 2.1 算子定位

**这是 LMCache 最核心的算子，与 vLLM 适配时必须实现的算子。**

它负责在 LMCache 的连续内存 `[2, NL, T, H]` 和 vLLM 的分页 GPU 缓冲区之间进行散射（scatter）/ 聚集（gather）传输，一次内核调用处理**所有层、所有 token、K 和 V**。

### 2.2 函数签名

```cpp
// csrc/mem_kernels.cu 第 620 行
void multi_layer_kv_transfer(
    torch::Tensor& key_value,              // [2, NL, T, H] LMCache 连续张量
    const torch::Tensor& key_value_ptrs,   // [NL] GPU 上的 int64 指针数组
    const torch::Tensor& slot_mapping,     // [T] token→slot 映射
    const torch::Device& paged_memory_device,
    const int page_buffer_size,            // NB * BS
    const TransferDirection direction,     // H2D 或 D2H
    const GPUKVFormat gpu_kv_format,       // 7 种 vLLM 格式之一
    const int block_size,                  // BS，通常 16
    const int head_size,                   // HS，通常 128
    const int skip_prefix_n_tokens = 0     // 跳过前 N 个 token
)
```

### 2.3 参数详解

| 参数 | 类型 | 含义 |
|------|------|------|
| `key_value` | Tensor `[2, NL, T, H]` | LMCache 的连续 KV 张量。dim0=2 表示 K(0) 和 V(1)，dim1=层数，dim2=token 数，dim3=hidden_dim |
| `key_value_ptrs` | Tensor `[NL]` (int64) | vLLM 每层 KV 缓冲区的设备指针。每个元素是一个 `data_ptr()`，指向该层的分页缓冲区 |
| `slot_mapping` | Tensor `[T]` (int64) | 每个 token 对应的物理 slot 索引。`slot = block_id * block_size + offset_in_block`。值为 -1 表示无效 |
| `direction` | enum | `H2D` = LMCache→vLLM（检索/恢复），`D2H` = vLLM→LMCache（存储/卸载） |
| `gpu_kv_format` | enum | vLLM 的 KV 缓冲区物理布局格式（7 种，见下表） |
| `block_size` | int | 分页块大小，通常 16 |
| `head_size` | int | 每个注意力头的维度，通常 128。HND 格式必须提供 |
| `skip_prefix_n_tokens` | int | 跳过前 N 个 token 的传输（用于前缀缓存场景） |

### 2.4 内核实现详解

#### 2.4.1 类型分派（Host 端）

```cpp
// 第 637-645 行
int copy_size = num_origin_elements * key_value.element_size();
if (copy_size % 8 == 0)      → launch_kernel<int64_t>(...)   // 8 字节向量化
else if (copy_size % 4 == 0)  → launch_kernel<int32_t>(...)   // 4 字节
else if (copy_size % 2 == 0)  → launch_kernel<int16_t>(...)   // 2 字节
else                           → launch_kernel<int8_t>(...)    // 1 字节
```

**目的**：最大化向量化内存访问。对于 fp16（2 字节/元素）且 head_size=128 的典型情况，`copy_size = 128 * 2 = 256` 字节，256 % 8 == 0，使用 `int64_t`（8 字节），每次 load/store 处理 4 个 fp16 值。

#### 2.4.2 Grid/Block 配置

```cpp
// 第 554-555 行
dim3 grid(num_transfer_tokens, num_layers, k_or_v_size);
dim3 block(std::min(num_xwords, 128));
```

- **Grid**：`(T, NL, 2)` —— 每个线程块处理一个 (token, layer, K/V) 组合
  - `blockIdx.x` = token_id
  - `blockIdx.y` = layer_id
  - `blockIdx.z` = k_or_v (0=Key, 1=Value)
  - 对于 MLA，k_or_v_size=1（无 K/V 分离）
- **Block**：`(min(scalars_per_token, 128),)` —— 每个线程块最多 128 线程

#### 2.4.3 内核核心逻辑（Device 端）

```cpp
// load_and_reshape_multi_layer_kernel（第 368-411 行）
template <typename scalar_t, bool DIRECTION, GPUKVFormat format>
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* key_value,           // [2, NL, T, H] 连续张量
    scalar_t** paged_buffer_ptrs,  // [NL] 指针数组，指向 vLLM 分页缓冲区
    int64_t* slot_mapping,         // [T] slot 映射
    int scalars_per_token,         // 每个 token 的标量数（H / sizeof(scalar_t)）
    int num_tokens, int num_layers,
    int page_buffer_size, int block_size, int head_size,
    int skip_prefix_n_tokens)
{
    // 1. 确定当前线程块处理的 token/layer/k_or_v
    const int token_id = blockIdx.x;
    const int layer_id = blockIdx.y;
    const int k_or_v  = blockIdx.z;
    const int tid     = threadIdx.x;
    const int num_threads = blockDim.x;

    // 2. 应用 skip_prefix_n_tokens 偏移
    const int kv_token_id = token_id + skip_prefix_n_tokens;

    // 3. 读取 slot_mapping 得到物理 slot 索引
    const int64_t slot_idx = slot_mapping[kv_token_id];
    if (slot_idx < 0) return;  // 跳过无效 slot

    // 4. 获取该层的 vLLM 分页缓冲区指针
    scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];

    // 5. 循环处理该 token 的所有标量
    for (int i = tid; i < scalars_per_token; i += num_threads) {
        // 计算 LMCache 连续张量中的偏移
        const int64_t lmcache_offset =
            key_value_offset(k_or_v, layer_id, kv_token_id, i,
                             scalars_per_token, num_tokens, num_layers);
        // = k_or_v * NL * T * H + layer_id * T * H + kv_token_id * H + i

        // 计算 vLLM 分页缓冲区中的偏移（根据 GPUKVFormat）
        const int64_t vllm_offset =
            page_buffer_offset<format>(k_or_v, slot_idx, i,
                                        scalars_per_token, page_buffer_size,
                                        block_size, head_size);

        // 6. 根据方向执行拷贝
        if (DIRECTION)  // D2H: vLLM → LMCache
            key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
        else            // H2D: LMCache → vLLM
            paged_buffer_ptr[vllm_offset] = key_value[lmcache_offset];
    }
}
```

#### 2.4.4 page_buffer_offset——核心寻址函数

这是算子最关键的函数，负责将逻辑参数转换为不同格式的物理地址：

```cpp
template <GPUKVFormat format>
__device__ int64_t page_buffer_offset(
    int k_or_v,          // 0=Key, 1=Value
    int token_idx,       // slot_mapping[t] 的值
    int scalar_offset,   // 线程循环索引 i
    int scalars_per_token,  // NH * HS / sizeof(scalar_t)
    int page_buffer_size,   // NB * BS
    int block_size,         // BS
    int head_size)          // HS / sizeof(scalar_t)
```

**各格式的寻址逻辑：**

**格式 0: NB_NL_TWO_BS_NH_HS（vLLM 跨层池）**
```
物理布局: [NB, NL, 2, BS, NH, HS]
偏移 = k_or_v * page_buffer_size * scalars_per_token
     + token_idx * scalars_per_token
     + scalar_offset
```
解释：K 和 V 在 dim2 上分离，token 直接索引。

**格式 1: NL_X_TWO_NB_BS_NH_HS（vLLM FlashAttention NHD）**
```
物理布局: 每层 [2, NB, BS, NH, HS]
偏移 = k_or_v * page_buffer_size * scalars_per_token
     + token_idx * scalars_per_token
     + scalar_offset
```
解释：与格式 0 相同，因为每层独立，指针数组已选中层。

**格式 2: NL_X_NB_TWO_BS_NH_HS（vLLM FlashInfer NHD）**
```
物理布局: 每层 [NB, 2, BS, NH, HS]
偏移 = block_idx * 2 * block_size * scalars_per_token
     + k_or_v * block_size * scalars_per_token
     + block_offset * scalars_per_token
     + scalar_offset
```
解释：K 和 V 在每个块内交错存储。

**格式 3: NL_X_NB_BS_HS（vLLM MLA）**
```
物理布局: 每层 [NB, BS, HS]
偏移 = token_idx * scalars_per_token + scalar_offset
```
解释：MLA 无 K/V 分离，直接索引。

**格式 6: NL_X_TWO_NB_NH_BS_HS（vLLM FlashAttention HND）**
```
物理布局: 每层 [2, NB, NH, BS, HS]
需要 head_size 来分解 scalar_offset:
  head_idx   = scalar_offset / head_size
  head_offset = scalar_offset % head_size
偏移 = k_or_v * page_buffer_size * scalars_per_token
     + block_idx * NH * BS * HS
     + head_idx * BS * HS
     + block_offset * HS
     + head_offset
```
解释：HND 布局中 head 维度在外层，需要额外的维度分解。

**格式 7: NL_X_NB_TWO_NH_BS_HS（vLLM FlashInfer HND）**
```
物理布局: 每层 [NB, 2, NH, BS, HS]
偏移 = block_idx * 2 * NH * BS * HS
     + k_or_v * NH * BS * HS
     + head_idx * BS * HS
     + block_offset * HS
     + head_offset
```

#### 2.4.5 key_value_offset——LMCache 连续张量寻址

```cpp
__device__ int64_t key_value_offset(
    int k_or_v, int layer_idx, int token_idx, int scalar_offset,
    int scalars_per_token, int num_tokens, int num_layers)
{
    return k_or_v * num_layers * num_tokens * scalars_per_token
         + layer_idx * num_tokens * scalars_per_token
         + token_idx * scalars_per_token
         + scalar_offset;
}
```

这是标准的行优先（row-major）寻址，对应张量形状 `[2, NL, T, H]`。

### 2.5 支持的 vLLM 格式（7 种）

| 枚举值 | 整数 | vLLM 使用场景 | 物理布局（每层） |
|--------|------|--------------|----------------|
| `NB_NL_TWO_BS_NH_HS` | 0 | 跨层 KV 池 | 单张量 `[NB, NL, 2, BS, NH, HS]` |
| `NL_X_TWO_NB_BS_NH_HS` | 1 | FlashAttention (NHD) | `[2, NB, BS, NH, HS]` |
| `NL_X_NB_TWO_BS_NH_HS` | 2 | FlashInfer (NHD) | `[NB, 2, BS, NH, HS]` |
| `NL_X_NB_BS_HS` | 3 | MLA | `[NB, BS, HS]` |
| `NL_X_TWO_NB_NH_BS_HS` | 6 | FlashAttention (HND) | `[2, NB, NH, BS, HS]` |
| `NL_X_NB_TWO_NH_BS_HS` | 7 | FlashInfer (HND) | `[NB, 2, NH, BS, HS]` |
| `NB_NL_TWO_NH_BS_HS` | 8 | TRT-LLM 跨层 (HND) | 单张量 `[NB, NL, 2, NH, BS, HS]` |

**注意**：格式 4、5、9 是 SGLang 专用，由 `multi_layer_kv_transfer_unilateral` 处理。

---

## 3. 其他算子详解

### 3.1 multi_layer_kv_transfer_unilateral（SGLang 专用）

**用途**：SGLang 的 K 和 V 存储在**独立的缓冲区**中（不是交错的）。

**关键区别**：
- 指针数组大小为 `[NL * 2]`，前 NL 个是 K 层，后 NL 个是 V 层
- `key_ptr = paged_buffer_ptrs[layer_id]`
- `value_ptr = paged_buffer_ptrs[layer_id + num_layers]`
- MLA 情况回退到 `multi_layer_kv_transfer`

### 3.2 single_layer_kv_transfer（逐层传输）

**用途**：layerwise 模式下每次传输一层，与 vLLM 的 forward pass 逐层交织。

**关键区别**：
- Grid = `(num_tokens,)`，不含 layer 和 k_or_v 维度
- 支持 NHD 和 HND 两种布局
- 支持 MLA（kv_size=1）

### 3.3 single_layer_kv_transfer_sgl（SGLang 逐层）

**用途**：SGLang 的逐层传输，K/V 分离。

**关键区别**：
- 接受独立的 `sgl_key_cache` 和 `sgl_value_cache` 张量

### 3.4 lmcache_memcpy_async（异步内存拷贝）

**用途**：在钉扎主机内存和设备内存之间执行异步 `cudaMemcpyAsync`。

**关键特性**：
- 按 `cudaHostRegister` 对齐粒度分块拷贝
- 处理 `LazyMemoryAllocator` 路径中的内存对齐问题

### 3.5 load_and_reshape_flash / reshape_and_cache_back_flash（已废弃）

**用途**：旧版单层单格式内核，仅用于单元测试。**不应作为适配参考。**

---

## 4. 昇腾适配建议：应该参考哪个算子

### 4.1 推荐方案

**首选参考：`multi_layer_kv_transfer` + `load_and_reshape_multi_layer_kernel`**

理由：

| 维度 | multi_layer_kv_transfer | single_layer_kv_transfer |
|------|------------------------|-------------------------|
| 覆盖场景 | 全层同时传输（默认路径） | 逐层传输（layerwise 路径） |
| vLLM 默认使用 | ✅ 是 | ❌ 仅 layerwise=True 时 |
| 效率 | 更高（一次内核处理所有层） | 较低（每层一次内核启动） |
| 复杂度 | 中等（需处理 7 种格式） | 中等（需处理 5 种格式） |
| 适配优先级 | ⭐⭐⭐ 最高 | ⭐⭐ 次高 |

### 4.2 昇腾适配的分阶段建议

**第一阶段（必须）：实现 `multi_layer_kv_transfer`**
- 这是 vLLM 默认路径使用的算子
- 覆盖 7 种 vLLM 格式
- 一次内核处理所有层，效率最高

**第二阶段（推荐）：实现 `single_layer_kv_transfer`**
- 用于 layerwise 模式（`use_layerwise=True`）
- 与 vLLM forward pass 逐层交织，降低峰值内存

**第三阶段（可选）：实现 `multi_layer_kv_transfer_unilateral`**
- 仅在使用 SGLang 时需要
- 如果只适配 vLLM，可以跳过

**第四阶段（可选）：实现 `lmcache_memcpy_async`**
- 用于 `LazyMemoryAllocator` 路径
- 可以用昇腾的异步内存拷贝 API 替代

### 4.3 与 Python 层的接口

算子通过 pybind11 暴露给 Python（`csrc/pybind.cpp`）：

```cpp
PYBIND11_MODULE(c_ops, m) {
    m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
          py::call_guard<py::gil_scoped_release>());
    m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
          py::call_guard<py::gil_scoped_release>());
    // ...
}
```

**昇腾适配时需要**：
1. 创建一个新的 `ascend_ops` 模块（类似 `c_ops`）
2. 实现同名函数，保持相同的 Python 接口
3. 在 `lmcache/__init__.py` 中根据设备类型选择 `c_ops` 或 `ascend_ops`

### 4.4 Python 调用路径

```
vllm_v1_adapter.py
  → lmcache_engine.store_layer()
    → gpu_connector.batched_from_gpu()
      → lmc_ops.multi_layer_kv_transfer(key_value, ptrs, slot_mapping, D2H, format, ...)
        → CUDA kernel / Ascend kernel
```

---

## 5. 昇腾适配的伪代码实现

### 5.1 昇腾 CANN 算子伪代码

以下是 `load_and_reshape_multi_layer_kernel` 的昇腾适配伪代码，使用 Ascend C 编程范式：

```cpp
// ascend_mem_kernels.cpp
// 昇腾 Ascend C 实现

#include "kernel_operator.h"

// 数据搬运：从 vLLM 分页缓冲区读取 → LMCache 连续缓冲区写入
// 或反向

// 昇腾 TBE/AICPU 实现思路：
// 1. 使用 DataCopy 从 GM（Global Memory）搬入 UB（Unified Buffer）
// 2. 在 UB 中完成地址重映射（scatter/gather）
// 3. 使用 DataCopy 从 UB 搬回 GM

// 伪代码（基于 Ascend C 编程模型）:
template <typename T>
__aicore__ void multi_layer_kv_transfer_kernel(
    __gm__ T* key_value,              // [2, NL, T, H] GM
    __gm__ T** paged_buffer_ptrs,     // [NL] 指针数组 GM
    __gm__ int64_t* slot_mapping,     // [T] GM
    int scalars_per_token,
    int num_tokens, int num_layers,
    int page_buffer_size, int block_size, int head_size,
    int skip_prefix_n_tokens,
    bool direction)  // true=D2H, false=H2D
{
    // 每个 AI Core 处理多个 (token, layer, kv) 组合
    // block_idx_x = token_id, block_idx_y = layer_id, block_idx_z = k_or_v

    int token_id = GetBlockIdx();  // 简化，实际需要 3D 索引
    int layer_id = ...;
    int k_or_v = ...;

    int kv_token_id = token_id + skip_prefix_n_tokens;
    int64_t slot_idx = slot_mapping[kv_token_id];
    if (slot_idx < 0) return;

    T* paged_buffer = paged_buffer_ptrs[layer_id];

    // 计算源和目标地址
    // LMCache 连续张量偏移
    int64_t lmcache_base = k_or_v * num_layers * num_tokens * scalars_per_token
                         + layer_id * num_tokens * scalars_per_token
                         + kv_token_id * scalars_per_token;

    // vLLM 分页缓冲区偏移（以 NHD 格式为例）
    int64_t vllm_base = k_or_v * page_buffer_size * scalars_per_token
                      + slot_idx * scalars_per_token;

    // 使用 DataCopy 搬运数据
    // 从 GM 搬入 UB
    T ubuffer[scalars_per_token];  // Unified Buffer

    if (direction) {  // D2H: vLLM → LMCache
        DataCopy(ubuffer, paged_buffer + vllm_base, scalars_per_token);
        PipeBarrier<PIPE_MTE2>();  // 等待 GM→UB 完成
        DataCopy(key_value + lmcache_base, ubuffer, scalars_per_token);
        PipeBarrier<PIPE_MTE3>();  // 等待 UB→GM 完成
    } else {          // H2D: LMCache → vLLM
        DataCopy(ubuffer, key_value + lmcache_base, scalars_per_token);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(paged_buffer + vllm_base, ubuffer, scalars_per_token);
        PipeBarrier<PIPE_MTE3>();
    }
}
```

### 5.2 关键适配点

| CUDA 概念 | 昇腾对应 | 说明 |
|----------|---------|------|
| `__global__` kernel | `__aicore__` kernel | 昇腾 AI Core 函数 |
| `blockIdx.x/y/z` | `GetBlockIdx()` + 手动 3D 映射 | 昇腾支持 1D/2D block 索引 |
| `threadIdx.x` | `GetThreadId()` | AI Core 内的线程 ID |
| `cudaMemcpyAsync` | `DataCopy` + `PipeBarrier` | 昇腾的 GM↔UB 数据搬运 |
| `__ldg()` (只读缓存) | `DataCopy` 默认行为 | 昇腾 GM 读取默认经 L2 缓存 |
| `cudaStream` | `AiCoreQueue` | 昇腾的异步队列 |
| `cudaHostRegister` | 昇腾的主机内存注册 API | 用于 DMA 传输 |
| 模板参数 `GPUKVFormat` | 运行时 if-else 或编译期宏 | 昇腾不支持 CUDA 模板特化，需用条件分支 |

### 5.3 格式处理策略

CUDA 版本使用编译期模板特化消除运行时分支。昇腾适配有两种策略：

**策略 A：运行时分派（推荐）**
```cpp
// 在 kernel 内使用运行时分支
__aicore__ int64_t page_buffer_offset(
    int format, int k_or_v, int token_idx, int scalar_offset,
    int scalars_per_token, int page_buffer_size, int block_size, int head_size)
{
    if (format == 0) {  // NB_NL_TWO_BS_NH_HS
        return k_or_v * page_buffer_size * scalars_per_token
             + token_idx * scalars_per_token + scalar_offset;
    } else if (format == 1) {  // NL_X_TWO_NB_BS_NH_HS
        return k_or_v * page_buffer_size * scalars_per_token
             + token_idx * scalars_per_token + scalar_offset;
    } else if (format == 2) {  // NL_X_NB_TWO_BS_NH_HS
        int block_idx = token_idx / block_size;
        int block_offset = token_idx % block_size;
        return block_idx * 2 * block_size * scalars_per_token
             + k_or_v * block_size * scalars_per_token
             + block_offset * scalars_per_token + scalar_offset;
    }
    // ... 其他格式
}
```

**策略 B：编译期宏（性能更好）**
```cpp
#define PAGE_BUFFER_OFFSET_FORMAT0(k_or_v, token_idx, scalar_offset, ...) \
    (k_or_v * page_buffer_size * scalars_per_token + token_idx * scalars_per_token + scalar_offset)

#define PAGE_BUFFER_OFFSET_FORMAT2(k_or_v, token_idx, scalar_offset, ...) \
    (token_idx / block_size * 2 * block_size * scalars_per_token + ...)

// 在 kernel 中根据格式选择宏
switch (format) {
    case 0: offset = PAGE_BUFFER_OFFSET_FORMAT0(...); break;
    case 2: offset = PAGE_BUFFER_OFFSET_FORMAT2(...); break;
}
```

---

## 6. 关键设计决策与注意事项

### 6.1 为什么选择 multi_layer_kv_transfer 而非 single_layer_kv_transfer

| 维度 | multi_layer | single_layer |
|------|------------|-------------|
| 内核启动次数 | 1 次（所有层） | NL 次（每层 1 次） |
| 内核启动开销 | 极低 | NL × 启动开销（约 32μs） |
| 与 forward 的交互 | 不交错 | 逐层交错（layerwise） |
| 内存峰值 | 需要所有层的缓冲区 | 只需 1-2 层的缓冲区 |
| vLLM 默认路径 | ✅ | ❌ |

**结论**：`multi_layer_kv_transfer` 是 vLLM 默认路径，一次内核处理所有层，效率最高。

### 6.2 slot_mapping 的处理

```cpp
const int64_t slot_idx = slot_mapping[kv_token_id];
if (slot_idx < 0) return;  // 关键：跳过无效 slot
```

- `slot_mapping[t]` = -1 表示该 token 未分配物理 slot
- 内核必须检查并跳过 -1 值
- `kv_token_id = token_id + skip_prefix_n_tokens` 处理前缀跳过

### 6.3 skip_prefix_n_tokens 的语义

当 vLLM 已经有前 N 个 token 的 KV Cache 时（prefix caching），LMCache 只需传输剩余部分：

```
slot_mapping: [s0, s1, s2, ..., s_{N-1}, s_N, s_{N+1}, ..., s_{T-1}]
              |<-- vLLM 已有 -->|  |<-- LMCache 需传输 -->|
skip_prefix_n_tokens = N
token_id 从 0 开始，但实际传输的 token 从 N 开始
```

### 6.4 get_kernel_ptr——主机/设备指针统一

```cpp
template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
    if (tensor.device().is_cuda()) {
        return static_cast<T*>(tensor.data_ptr());  // GPU 直接使用
    } else if (tensor.device().is_cpu()) {
        T* ptr;
        cudaHostGetDevicePointer(&ptr, tensor.data_ptr(), 0);  // 钉扎内存
        return ptr;
    }
}
```

**昇腾适配**：需要使用昇腾的主机内存注册 API 替代 `cudaHostGetDevicePointer`。

### 6.5 MLA 模式的特殊处理

MLA（Multi-head Latent Attention）模型的 KV 缓冲区没有 K/V 分离：

```cpp
int k_or_v_size = is_mla(gpu_kv_format) ? 1 : 2;
```

- MLA：`k_or_v_size = 1`，Grid 的 z 维度为 1
- 非 MLA：`k_or_v_size = 2`，Grid 的 z 维度为 2（K 和 V）

### 6.6 HND 布局的额外维度分解

HND 布局需要 `head_size` 来将 `scalar_offset` 分解为 `(head_idx, head_offset)`：

```cpp
// 仅 HND 格式需要
const int head_idx = scalar_offset / head_size;
const int head_offset = scalar_offset % head_size;
const int num_heads = scalars_per_token / head_size;
```

NHD 布局不需要这种分解，因为 head 维度在最内层，天然连续。

---

## 附录 A：完整的五层调用链路分析（vLLM → CUDA Kernel）

本节以 vLLM 为例，追踪 `multi_layer_kv_transfer` 从 vLLM 到 CUDA 内核的完整调用链路。系统有两条并行路径：**非逐层路径（batch path）**直接调用 `multi_layer_kv_transfer`，**逐层路径（layerwise path）**使用 `single_layer_kv_transfer`。

### A.1 调用链路全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: vLLM Adapter (vllm_v1_adapter.py)                              │
│                                                                         │
│  Store 路径 (GPU→CPU):                                                   │
│    wait_for_save() ──→ lmcache_engine.store()                           │
│    save_kv_layer() ──→ lmcache_engine.store_layer() [生成器]             │
│                                                                         │
│  Load 路径 (CPU→GPU):                                                    │
│    start_load_kv() ──→ lmcache_engine.retrieve()                        │
│    wait_for_layer_load() ──→ next(retriever) [逐层推进]                  │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: Cache Engine (cache_engine.py)                                  │
│                                                                         │
│  store() ──→ gpu_connector.batched_from_gpu(memory_objs, **kwargs)      │
│  retrieve() ──→ gpu_connector.batched_to_gpu(memory_objs, **kwargs)     │
│  store_layer() ──→ 生成器包装 batched_from_gpu()                         │
│  retrieve_layer() ──→ 生成器包装 batched_to_gpu()                        │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: GPU Connector (gpu_connectors.py)                               │
│                                                                         │
│  VLLMPagedMemGPUConnectorV2 (非逐层):                                    │
│    batched_from_gpu() ──→ from_gpu() ──→ lmc_ops.multi_layer_kv_transfer│
│    batched_to_gpu() ──→ to_gpu() ──→ lmc_ops.multi_layer_kv_transfer   │
│                                                                         │
│  VLLMPagedMemLayerwiseGPUConnector (逐层):                               │
│    batched_from_gpu() ──→ lmc_ops.single_layer_kv_transfer [逐层]       │
│    batched_to_gpu() ──→ lmc_ops.single_layer_kv_transfer [逐层]         │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3.5: Python Binding (pybind.cpp)                                   │
│  GIL 释放，传递原始指针到 C++                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 4: CUDA Operator (mem_kernels.cu)                                  │
│                                                                         │
│  multi_layer_kv_transfer()          ← 类型分派 (int64/int32/int16/int8) │
│    └→ multi_layer_kv_transfer_templated()  ← 提取原始指针，计算 Grid     │
│         └→ load_and_reshape_multi_layer_kernel<<<(T,L,2),(128)>>>        │
│              ├→ key_value_offset()        ← LMCache 连续张量寻址         │
│              └→ page_buffer_offset<FMT>() ← vLLM 分页缓冲区寻址         │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.2 LEVEL 1：vLLM Adapter（入口层）

**文件**：`lmcache/integration/vllm/vllm_v1_adapter.py`

#### Store 路径（GPU→CPU，卸载 KV Cache）

**非逐层路径**（默认）：`wait_for_save()` 一次性调用 `lmcache_engine.store()`：

```python
# vllm_v1_adapter.py 第 1186-1195 行
self.lmcache_engine.store(
    token_ids,                        # 输入 token IDs
    mask=store_mask,                  # 布尔掩码，标记哪些 token 需要存储
    kvcaches=kvcaches,                # vLLM 的 KV 缓冲区张量列表
    slot_mapping=slot_mapping,        # token→物理 slot 映射
    offset=skip_leading_tokens,       # 跳过前缀
    transfer_spec=request.disagg_spec,
    request_configs=request.request_configs,
    req_id=request.req_id,
)
```

**逐层路径**（`use_layerwise=True`）：`save_kv_layer()` 每层调用一次，通过生成器推进：

```python
# vllm_v1_adapter.py 第 1061-1074 行
# 首次调用：创建生成器
layerwise_storer = self.lmcache_engine.store_layer(
    token_ids, mask=store_mask, kvcaches=kvcaches,
    slot_mapping=slot_mapping, offset=skip_leading_tokens,
)
# 后续调用：每层推进一次
next(layerwise_storer)
```

#### Load 路径（CPU→GPU，恢复 KV Cache）

**非逐层路径**（默认）：`start_load_kv()` 一次性调用 `lmcache_engine.retrieve()`：

```python
# vllm_v1_adapter.py 第 835-843 行
ret_token_mask = self.lmcache_engine.retrieve(
    tokens[:lmcache_cached_tokens],
    token_mask[:lmcache_cached_tokens],
    kvcaches=kvcaches,
    slot_mapping=slot_mapping[:lmcache_cached_tokens],
    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
    request_configs=request.request_configs,
)
```

**逐层路径**（`use_layerwise=True`）：`start_load_kv()` 创建生成器，预取前 2 层：

```python
# vllm_v1_adapter.py 第 822-833 行
layerwise_retriever = self.lmcache_engine.retrieve_layer(
    tokens[:lmcache_cached_tokens], token_mask[:lmcache_cached_tokens],
    kvcaches=kvcaches, slot_mapping=slot_mapping[:lmcache_cached_tokens],
    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
)
next(layerwise_retriever)  # 预取 layer 0
next(layerwise_retriever)  # 预取 layer 1
```

后续每层由 `wait_for_layer_load()` 推进：

```python
# vllm_v1_adapter.py 第 961-962 行
for layerwise_retriever in self.layerwise_retrievers:
    ret_token_mask = next(layerwise_retriever)
```

#### 关键参数传递

| 参数 | 来源 | 含义 |
|------|------|------|
| `kvcaches` | vLLM Worker 注册 | 每层的 KV 缓冲区张量，如 `[(K_layer0, V_layer0), ...]` |
| `slot_mapping` | Scheduler 分配 | 每个 token 的物理 slot 索引，`slot = block_id * block_size + offset` |
| `mask` | Adapter 计算 | 布尔掩码，标记哪些 token 需要存储/检索 |
| `vllm_cached_tokens` | LoadSpec | vLLM 已缓存的 token 数，用于计算 `skip_prefix_n_tokens` |

### A.3 LEVEL 2：Cache Engine（引擎层）

**文件**：`lmcache/v1/cache_engine.py`

Cache Engine 负责 token 分块、内存分配、存储编排，然后调用 GPUConnector 执行实际的数据传输。

#### Store 路径

```python
# cache_engine.py 第 534 行
self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
```

`**kwargs` 透传自 Adapter 层：`kvcaches`、`slot_mapping`、`vllm_cached_tokens` 等。

#### Retrieve 路径

```python
# cache_engine.py 第 855 行
self.gpu_connector.batched_to_gpu(
    list(memory_objs), list(starts), list(ends), **kwargs
)
```

#### Layerwise 路径（生成器模式）

`store_layer()` 和 `retrieve_layer()` 返回 Python 生成器，每层 `yield` 一次：

```python
# cache_engine.py 第 720-731 行（store_layer 生成器）
mem_obj_generator = self.gpu_connector.batched_from_gpu(
    memory_objs, starts, ends, **kwargs
)
next(mem_obj_generator)  # 初始化

for layer_id in range(self.num_layers):
    yield                           # 让 vLLM 的 forward pass 执行该层
    next(mem_obj_generator)         # 传输该层的 KV 数据
    self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])
```

### A.4 LEVEL 3：GPU Connector（连接器层）

**文件**：`lmcache/v1/gpu_connector/gpu_connectors.py`

这是 Python 层和 CUDA 层的桥梁。有两个关键类：

- **`VLLMPagedMemGPUConnectorV2`**（非逐层）→ 调用 `multi_layer_kv_transfer`
- **`VLLMPagedMemLayerwiseGPUConnector`**（逐层）→ 调用 `single_layer_kv_transfer`

#### VLLMPagedMemGPUConnectorV2 的核心流程

**Step 1：收集指针（`_initialize_pointers`）**

```python
# gpu_connectors.py 第 240-270 行
def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
    # 1. 发现 KV 缓冲区的物理布局格式
    self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
        kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
    )

    # 2. 提取每层的 data_ptr()，存入 CPU int64 张量
    self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]

    # 3. 拷贝到 GPU（CUDA 内核需要在 GPU 上读取指针）
    self.kv_cache_pointers_on_gpu[idx] = torch.empty(
        self.num_layers, dtype=torch.int64, device=self.device
    )
    self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

    # 4. 提取布局参数
    self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
    self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
    self.page_buffer_size = self.num_blocks * self.block_size
    self.head_size = get_head_size(kv_caches, self.gpu_kv_format)
```

**这是最关键的桥梁**：将 Python 层的张量对象转换为 CUDA 内核可以使用的原始设备指针。

**Step 2：调用 CUDA 算子（`to_gpu` / `from_gpu`）**

```python
# gpu_connectors.py 第 264-300 行（to_gpu，CPU→GPU）
def to_gpu(self, memory_obj, start, end, **kwargs):
    kv_cache_pointers = self._initialize_pointers(self.kvcaches)
    slot_mapping = kwargs["slot_mapping"]

    # 计算跳过前缀的 token 数
    vllm_cached = kwargs.get("vllm_cached_tokens", 0)
    skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

    lmc_ops.multi_layer_kv_transfer(
        memory_obj.tensor,            # [2, NL, T, H] LMCache 连续张量
        kv_cache_pointers,            # [NL] GPU int64 指针数组
        slot_mapping[start:end],      # [T] slot 映射切片
        self.device,                  # GPU 设备
        self.page_buffer_size,        # NB * BS
        lmc_ops.TransferDirection.H2D,# 方向：LMCache→vLLM
        self.gpu_kv_format,           # 物理布局格式枚举
        block_size=self.block_size,
        head_size=self.head_size,
        skip_prefix_n_tokens=skip_prefix_n_tokens,
    )
```

```python
# gpu_connectors.py 第 329-370 行（from_gpu，GPU→CPU）
def from_gpu(self, memory_obj, start, end, **kwargs):
    kv_cache_pointers = self._initialize_pointers(self.kvcaches)
    slot_mapping = kwargs["slot_mapping"]

    with torch.cuda.stream(self.store_stream):  # 在专用 stream 上执行
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches[0].device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.D2H,      # 方向：vLLM→LMCache
            self.gpu_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
        )
```

**Step 3：批量调用（`batched_to_gpu` / `batched_from_gpu`）**

```python
# gpu_connectors.py 第 401-415 行
def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
    with torch.cuda.stream(self.load_stream):
        for memory_obj, start, end in zip(memory_objs, starts, ends):
            self.to_gpu(memory_obj, start, end, **kwargs)
    self.load_stream.synchronize()  # 等待所有传输完成

def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
    for memory_obj, start, end in zip(memory_objs, starts, ends):
        self.from_gpu(memory_obj, start, end, **kwargs)
```

### A.5 LEVEL 3.5：Python Binding（绑定层）

**文件**：`csrc/pybind.cpp` 第 36-42 行

```cpp
m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
      py::arg("key_value"), py::arg("key_value_ptrs"),
      py::arg("slot_mapping"), py::arg("paged_memory_device"),
      py::arg("page_buffer_size"), py::arg("direction"),
      py::arg("gpu_kv_format"), py::arg("block_size") = 0,
      py::arg("head_size") = 0, py::arg("skip_prefix_n_tokens") = 0,
      py::call_guard<py::gil_scoped_release>());  // 释放 GIL
```

**关键设计**：`py::call_guard<py::gil_scoped_release>()` 在 CUDA 内核执行期间释放 Python GIL，允许其他 Python 线程并发执行。

### A.6 LEVEL 4：CUDA Operator（内核层）——深度展开

**文件**：`csrc/mem_kernels.cu`

CUDA 层采用**三级分派**架构：入口函数 → 模板函数 → CUDA 内核。每一级都有明确的职责。

#### A.6.1 三级分派架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 第 1 级：multi_layer_kv_transfer()（第 620-647 行）                      │
│   职责：根据元素大小选择最优标量类型（int64/int32/int16/int8）            │
│   输入：PyTorch Tensor 对象                                              │
│   输出：调用第 2 级                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ 第 2 级：multi_layer_kv_transfer_templated<T>()（第 519-613 行）         │
│   职责：提取原始指针、计算 Grid/Block、根据 direction+format 分派内核     │
│   输入：PyTorch Tensor → 原始指针                                        │
│   输出：启动 CUDA 内核                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ 第 3 级：load_and_reshape_multi_layer_kernel<T, DIR, FMT>()（第 368 行） │
│   职责：每个线程块处理一个 (token, layer, K/V)，执行实际的数据拷贝        │
│   输入：原始指针、标量参数                                                │
│   输出：数据在 LMCache 连续张量和 vLLM 分页缓冲区之间移动                 │
└─────────────────────────────────────────────────────────────────────────┘
```

#### A.6.2 第 1 级：入口函数 `multi_layer_kv_transfer`（第 620-647 行）

```cpp
void multi_layer_kv_transfer(
    torch::Tensor& key_value,           // [2, NL, T, H] LMCache 连续张量
    const torch::Tensor& key_value_ptrs,// [NL] GPU int64 指针数组
    const torch::Tensor& slot_mapping,  // [T] token→slot 映射
    const torch::Device& paged_memory_device,
    const int page_buffer_size,         // NB * BS
    const TransferDirection direction,  // H2D 或 D2H
    const GPUKVFormat gpu_kv_format,    // 7 种格式之一
    const int block_size,               // BS，通常 16
    const int head_size,                // HS，通常 128
    const int skip_prefix_n_tokens)     // 跳过前 N 个 token
{
    int num_origin_elements = key_value.size(3);  // H = NH * HS
    int copy_size = num_origin_elements * key_value.element_size();

    // 根据元素大小选择最优标量类型
    // 目的：最大化向量化内存访问（8 字节 load/store 最优）
    if (copy_size % 8 == 0) {
        multi_layer_kv_transfer_templated<int64_t>(...);  // 8 字节 = 4 个 fp16
    } else if (copy_size % 4 == 0) {
        multi_layer_kv_transfer_templated<int32_t>(...);  // 4 字节 = 2 个 fp16
    } else if (copy_size % 2 == 0) {
        multi_layer_kv_transfer_templated<int16_t>(...);  // 2 字节 = 1 个 fp16
    } else {
        multi_layer_kv_transfer_templated<int8_t>(...);   // 1 字节
    }
}
```

**类型分派的数值示例**：

| 场景 | dtype | head_size | copy_size | 选择类型 | 每次处理元素数 |
|------|-------|-----------|-----------|---------|-------------|
| 标准 fp16 | fp16 (2B) | 128 | 128×2=256B | int64_t | 4 个 fp16 |
| 标准 bf16 | bf16 (2B) | 128 | 128×2=256B | int64_t | 4 个 bf16 |
| MLA fp16 | fp16 (2B) | 512 | 512×2=1024B | int64_t | 4 个 fp16 |
| fp8 量化 | fp8 (1B) | 128 | 128×1=128B | int64_t | 8 个 fp8 |
| 小 head | fp16 (2B) | 32 | 32×2=64B | int64_t | 4 个 fp16 |
| 小 head fp8 | fp8 (1B) | 32 | 32×1=32B | int64_t | 8 个 fp8 |

#### A.6.3 第 2 级：模板函数 `multi_layer_kv_transfer_templated<T>`（第 519-613 行）

```cpp
template <typename T>
void multi_layer_kv_transfer_templated(
    torch::Tensor& key_value,
    const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping,
    const torch::Device& paged_memory_device,
    const int page_buffer_size,
    const TransferDirection direction,
    const GPUKVFormat gpu_kv_format,
    const int block_size,
    const int head_size,
    const int skip_prefix_n_tokens)
{
    // ──────────────────────────────────────────────
    // Step 1: 提取原始设备指针
    // ──────────────────────────────────────────────
    T* key_value_ptr = get_kernel_ptr<T, torch::Tensor>(key_value);
    // key_value_ptr 指向 LMCache 连续张量 [2, NL, T, H] 的起始地址
    // 可以是 GPU 指针或钉扎 CPU 指针（通过 cudaHostGetDevicePointer）

    T** page_buffer_ptrs = get_kernel_ptr<T*, const torch::Tensor>(key_value_ptrs);
    // page_buffer_ptrs 是一个 GPU 上的指针数组，每个元素指向 vLLM 某层的 KV 缓冲区
    // 例如：page_buffer_ptrs[0] = K_layer0 的 data_ptr()
    //       page_buffer_ptrs[1] = V_layer0 的 data_ptr() （如果非 MLA）
    //       ... 或者 page_buffer_ptrs[layer_id] = 该层的 data_ptr()

    const int64_t* slot_mapping_ptr = get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
    // slot_mapping_ptr 指向 slot 映射数组 [T]
    // slot_mapping_ptr[t] = 该 token 在 vLLM 分页缓冲区中的物理 slot 索引

    // ──────────────────────────────────────────────
    // Step 2: 计算内核参数
    // ──────────────────────────────────────────────
    int num_layers = key_value.size(1);      // NL，例如 32
    int num_tokens = key_value.size(2);      // T，例如 256（一个 chunk）
    int num_transfer_tokens = num_tokens - skip_prefix_n_tokens;
    // 例如 skip_prefix_n_tokens=128，则只传输后 128 个 token

    int num_origin_elements = key_value.size(3);  // H = NH * HS，例如 32*128=4096

    // 将元素数转换为"标量字"（xword）数
    // int64_t 类型时：elements_per_xword = 8 / 2 = 4（每个 xword 包含 4 个 fp16）
    int elements_per_xword = sizeof(T) / key_value.element_size();
    int num_xwords = num_origin_elements / elements_per_xword;
    // 例如：4096 个 fp16 元素 / 4 = 1024 个 xword

    // head_size 也需要转换为 xword 单位
    int head_size_xword = head_size > 0 ? head_size / elements_per_xword : 0;
    // 例如：128 个 fp16 / 4 = 32 个 xword

    // ──────────────────────────────────────────────
    // Step 3: 配置 Grid 和 Block
    // ──────────────────────────────────────────────
    int k_or_v_size = is_mla(gpu_kv_format) ? 1 : 2;

    dim3 grid(num_transfer_tokens, num_layers, k_or_v_size);
    // 例如：grid(128, 32, 2) = 128 * 32 * 2 = 8192 个线程块
    // 每个线程块处理一个 (token, layer, K/V) 组合

    dim3 block(std::min(num_xwords, 128));
    // 例如：block(128) —— 128 个线程
    // 每个线程处理一个或多个 xword

    // ──────────────────────────────────────────────
    // Step 4: 根据 direction + format 分派内核
    // ──────────────────────────────────────────────
    const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // H2D (LMCache → vLLM)
    if (direction == TransferDirection::H2D) {
        switch (gpu_kv_format) {
            case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
                LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
                break;
            case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
                LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
                break;
            // ... 其他 5 种格式
        }
    }
    // D2H (vLLM → LMCache)
    else {
        switch (gpu_kv_format) {
            case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
                LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
                break;
            // ... 其他 5 种格式
        }
    }
}
```

**LAUNCH_KERNEL_WITH_FORMAT 宏展开**（第 511-517 行）：

```cpp
#define LAUNCH_KERNEL_WITH_FORMAT(T, DIRECTION, FORMAT)                      \
    lmc::load_and_reshape_multi_layer_kernel<T, DIRECTION, FORMAT>           \
        <<<grid, block, 0, stream>>>(                                        \
            key_value_ptr, page_buffer_ptrs, slot_mapping_ptr,               \
            num_xwords, num_tokens, num_layers, page_buffer_size,            \
            block_size, head_size_xword, skip_prefix_n_tokens);              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();
```

宏将 `T`、`DIRECTION`、`FORMAT` 作为编译期模板参数传入内核，编译器为每种组合生成特化代码。

#### A.6.4 第 3 级：CUDA 内核 `load_and_reshape_multi_layer_kernel`（第 368-411 行）

这是实际在 GPU 上执行的代码。逐行详解：

```cpp
template <typename scalar_t, bool DIRECTION, GPUKVFormat format>
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* __restrict__ key_value,           // [2, NL, T, H] LMCache 连续张量
    scalar_t** __restrict__ paged_buffer_ptrs,  // [NL] vLLM 每层 KV 缓冲区指针
    const int64_t* __restrict__ slot_mapping,   // [T] token→slot 映射
    const int scalars_per_token,                // 每个 token 的 xword 数
    const int num_tokens,                       // T
    const int num_layers,                       // NL
    const int page_buffer_size,                 // NB * BS
    const int block_size,                       // BS
    const int head_size,                        // HS（xword 单位）
    const int skip_prefix_n_tokens)             // 跳过前 N 个 token
{
    // ═══════════════════════════════════════════════════
    // Step 1: 确定当前线程块处理的 (token, layer, K/V)
    // ═══════════════════════════════════════════════════
    const int token_id = blockIdx.x;   // token 维度
    const int layer_id = blockIdx.y;   // 层维度
    const int k_or_v   = blockIdx.z;   // 0=Key, 1=Value（MLA 时只有 0）
    const int tid      = threadIdx.x;  // 线程在块内的索引
    const int num_threads = blockDim.x;// 块内线程数（128）

    // ═══════════════════════════════════════════════════
    // Step 2: 应用前缀跳过
    // ═══════════════════════════════════════════════════
    const int kv_token_id = token_id + skip_prefix_n_tokens;
    // 例如 skip_prefix_n_tokens=128:
    //   token_id=0 → kv_token_id=128（实际传输的第 128 个 token）
    //   token_id=1 → kv_token_id=129
    // 这样 vLLM 已有的前 128 个 token 的 KV Cache 不会被覆盖

    // ═══════════════════════════════════════════════════
    // Step 3: 读取 slot_mapping 得到物理 slot 索引
    // ═══════════════════════════════════════════════════
    const int64_t slot_idx = slot_mapping[kv_token_id];
    // slot_idx 是该 token 在 vLLM 分页缓冲区中的全局物理位置
    // 例如 slot_idx=48 表示该 token 位于第 3 个块的第 0 个位置（block_size=16 时）

    if (slot_idx < 0) {
        return;  // slot_idx=-1 表示该 token 未分配物理 slot，跳过
    }

    // ═══════════════════════════════════════════════════
    // Step 4: 获取该层的 vLLM 缓冲区指针
    // ═══════════════════════════════════════════════════
    scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];
    // paged_buffer_ptrs[layer_id] 是 vLLM 第 layer_id 层的 KV 缓冲区的设备指针
    // 由 Python 层的 _initialize_pointers() 收集并拷贝到 GPU

    // ═══════════════════════════════════════════════════
    // Step 5: 循环处理该 token 的所有标量（xword）
    // ═══════════════════════════════════════════════════
    // 每个线程处理 scalars_per_token / num_threads 个 xword
    // 例如 scalars_per_token=1024, num_threads=128 → 每线程处理 8 个 xword
    for (int i = tid; i < scalars_per_token; i += num_threads) {

        // ──────────────────────────────────────────
        // Step 5a: 计算 LMCache 连续张量中的偏移
        // ──────────────────────────────────────────
        const int64_t lmcache_offset =
            key_value_offset(k_or_v, layer_id, kv_token_id, i,
                             scalars_per_token, num_tokens, num_layers);
        // 公式：k_or_v * NL * T * H + layer_id * T * H + kv_token_id * H + i
        // 这是标准的行优先寻址，对应张量 [2, NL, T, H]

        // ──────────────────────────────────────────
        // Step 5b: 计算 vLLM 分页缓冲区中的偏移
        // ──────────────────────────────────────────
        const int64_t vllm_offset =
            page_buffer_offset<format>(k_or_v, slot_idx, i,
                                       scalars_per_token, page_buffer_size,
                                       block_size, head_size);
        // 这是内核的核心——根据 GPUKVFormat 计算不同的物理地址
        // 编译期模板特化，零运行时分支

        // ──────────────────────────────────────────
        // Step 5c: 根据方向执行拷贝
        // ──────────────────────────────────────────
        if (DIRECTION) {  // D2H: vLLM → LMCache（存储/卸载）
            key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
        } else {          // H2D: LMCache → vLLM（检索/恢复）
            paged_buffer_ptr[vllm_offset] = key_value[lmcache_offset];
        }
    }
}
```

#### A.6.5 关键辅助函数 1：`key_value_offset`（第 301-308 行）

LMCache 连续张量 `[2, NL, T, H]` 的行优先寻址：

```cpp
__device__ __forceinline__ int64_t
key_value_offset(const int k_or_v,      // 0=Key, 1=Value
                 const int layer_idx,    // 层索引
                 const int token_idx,    // token 索引（已加 skip_prefix）
                 const int scalar_offset,// xword 索引
                 const int scalars_per_token,  // H / sizeof(scalar_t)
                 const int num_tokens,   // T
                 const int num_layers)   // NL
{
    return k_or_v * num_layers * num_tokens * scalars_per_token +
           layer_idx * num_tokens * scalars_per_token +
           token_idx * scalars_per_token +
           scalar_offset;
}
```

**地址计算的物理含义**：

```
key_value 张量内存布局（行优先）：
[0                    ... NL*T*H-1]  ← Key (k_or_v=0)
[NL*T*H               ... 2*NL*T*H-1]  ← Value (k_or_v=1)

Key 内部：
[layer=0: T*H 个元素]
[layer=1: T*H 个元素]
...
[layer=NL-1: T*H 个元素]

每层内部：
[token=0: H 个元素]
[token=1: H 个元素]
...
[token=T-1: H 个元素]
```

**数值示例**（fp16, NH=32, HS=128, NL=32, T=256）：
```
scalars_per_token = 4096 / 2 = 2048（int64_t 单位，每个 int64 包含 4 个 fp16）
lmcache_offset(Key, layer=5, token=100, scalar=500) =
    0 * 32 * 256 * 2048 +    // k_or_v=0 (Key)
    5 * 256 * 2048 +          // layer=5
    100 * 2048 +              // token=100
    500                       // scalar=500
    = 0 + 2621440 + 204800 + 500
    = 2826740                 // 第 2826740 个 int64 元素
```

#### A.6.6 关键辅助函数 2：`page_buffer_offset`（第 216-294 行）

这是内核最核心的函数——根据 vLLM 的 KV 缓冲区物理布局格式，将逻辑参数转换为物理地址。

**通用参数语义**：
```
k_or_v:            0=Key, 1=Value
token_idx:         slot_mapping[t] 的值 = block_id * block_size + offset_in_block
scalar_offset:     线程循环索引 i，在 [0, scalars_per_token) 范围内
scalars_per_token: NH * HS / sizeof(scalar_t)（xword 单位）
page_buffer_size:  NB * BS（分页缓冲区总 slot 数）
block_size:        BS（每块的 slot 数）
head_size:         HS / sizeof(scalar_t)（xword 单位，仅 HND 格式使用）
```

##### 格式 0：NB_NL_TWO_BS_NH_HS（vLLM 跨层池）

```
物理布局: [NB, NL, 2, BS, NH, HS]
                  ↑
            K/V 在 dim2 分离

偏移公式:
  return k_or_v * page_buffer_size * scalars_per_token
       + token_idx * scalars_per_token
       + scalar_offset;

内存排列:
  [Key_all_blocks | Value_all_blocks]
  每个 block 内: [layer0_data | layer1_data | ...]
  每层内: [token0_data | token1_data | ...]
```

##### 格式 1：NL_X_TWO_NB_BS_NH_HS（vLLM FlashAttention NHD）

```
物理布局: 每层 [2, NB, BS, NH, HS]
                ↑
          K/V 在 dim0 分离

偏移公式（与格式 0 相同，因为指针数组已选中层）:
  return k_or_v * page_buffer_size * scalars_per_token
       + token_idx * scalars_per_token
       + scalar_offset;

内存排列（每层独立）:
  [Key: NB*BS*NH*HS | Value: NB*BS*NH*HS]
  Key 内: [block0: BS*NH*HS | block1: BS*NH*HS | ...]
  每 block: [token0: NH*HS | token1: NH*HS | ...]
```

##### 格式 2：NL_X_NB_TWO_BS_NH_HS（vLLM FlashInfer NHD）

```
物理布局: 每层 [NB, 2, BS, NH, HS]
                  ↑
            K/V 在每个 block 内交错

偏移公式:
  block_idx = token_idx / block_size;
  block_offset = token_idx % block_size;
  return block_idx * 2 * block_size * scalars_per_token  // 跳过前面的 block
       + k_or_v * block_size * scalars_per_token          // 在 block 内选择 K 或 V
       + block_offset * scalars_per_token                  // 在 K/V 内选择 token
       + scalar_offset;                                    // 在 token 内选择 xword

内存排列（每层独立）:
  [block0: [Key: BS*NH*HS, Value: BS*NH*HS],
   block1: [Key: BS*NH*HS, Value: BS*NH*HS],
   ...]
```

**关键区别**：格式 1 的 K/V 在块外层分离，格式 2 的 K/V 在块内层交错。

##### 格式 3：NL_X_NB_BS_HS（vLLM MLA）

```
物理布局: 每层 [NB, BS, HS]
            无 K/V 分离！

偏移公式:
  return token_idx * scalars_per_token + scalar_offset;

内存排列（每层独立）:
  [block0: [token0: HS, token1: HS, ...],
   block1: [token0: HS, token1: HS, ...],
   ...]
```

**MLA 特殊性**：`k_or_v_size=1`，Grid 的 z 维度为 1，`page_buffer_offset` 不使用 `k_or_v` 参数。

##### 格式 6：NL_X_TWO_NB_NH_BS_HS（vLLM FlashAttention HND）

```
物理布局: 每层 [2, NB, NH, BS, HS]
                       ↑
                 head 维度在 block 维度外层

偏移公式:
  block_idx = token_idx / block_size;
  block_offset = token_idx % block_size;
  head_idx = scalar_offset / head_size;      // 需要分解！
  head_offset = scalar_offset % head_size;   // 需要分解！
  num_heads = scalars_per_token / head_size;
  return k_or_v * page_buffer_size * scalars_per_token  // K/V 分离
       + block_idx * num_heads * block_size * head_size  // 跳过前面的 block
       + head_idx * block_size * head_size               // 选择 head
       + block_offset * head_size                        // 选择 token
       + head_offset;                                    // 选择 head 内元素

内存排列（每层独立）:
  [Key: [block0: [head0: BS*HS, head1: BS*HS, ...],
          block1: [head0: BS*HS, head1: BS*HS, ...], ...],
   Value: [...同上...]]
```

**HND 的性能问题**：同一 warp 的线程在跨 head 边界时访问非连续地址（stride = BS*HS），会降低内存合并访问效率。源码注释中提到了这个 TODO。

##### 格式 7：NL_X_NB_TWO_NH_BS_HS（vLLM FlashInfer HND）

```
物理布局: 每层 [NB, 2, NH, BS, HS]
                  ↑
            K/V 在 block 内交错，head 在 block 内外层

偏移公式:
  block_idx = token_idx / block_size;
  block_offset = token_idx % block_size;
  head_idx = scalar_offset / head_size;
  head_offset = scalar_offset % head_size;
  num_heads = scalars_per_token / head_size;
  return block_idx * 2 * num_heads * block_size * head_size  // 跳过前面的 block
       + k_or_v * num_heads * block_size * head_size          // K/V 交错
       + head_idx * block_size * head_size                     // 选择 head
       + block_offset * head_size                              // 选择 token
       + head_offset;                                          // 选择 head 内元素

内存排列（每层独立）:
  [block0: [Key: [head0: BS*HS, head1: BS*HS, ...],
            Value: [head0: BS*HS, head1: BS*HS, ...]],
   block1: [...同上...], ...]
```

#### A.6.7 具体数值示例：完整走一遍内核

**场景**：vLLM FlashAttention NHD 格式，fp16，Llama-7B（NL=32, NH=32, HS=128, BS=16）

**参数**：
```
key_value:        [2, 32, 256, 4096] fp16（一个 chunk，256 tokens）
slot_mapping:     [256] int64，例如 [0,1,2,...,15, 16,17,...,31, ...]
gpu_kv_format:    NL_X_TWO_NB_BS_NH_HS (=1)
block_size:       16
head_size:        128
direction:        D2H（vLLM → LMCache，存储）
skip_prefix:      0
```

**类型分派**：
```
copy_size = 4096 * 2 = 8192 字节
8192 % 8 == 0 → 选择 int64_t
elements_per_xword = 8 / 2 = 4（每个 int64 包含 4 个 fp16）
num_xwords = 4096 / 4 = 1024
head_size_xword = 128 / 4 = 32
```

**Grid/Block**：
```
k_or_v_size = 2（非 MLA）
grid = (256, 32, 2) = 16384 个线程块
block = min(1024, 128) = 128 个线程
总线程数 = 16384 * 128 = 2,097,152
```

**内核执行（以 token_id=5, layer_id=3, k_or_v=0 为例）**：
```
token_id = 5, layer_id = 3, k_or_v = 0 (Key)
kv_token_id = 5 + 0 = 5
slot_idx = slot_mapping[5] = 5
paged_buffer_ptr = paged_buffer_ptrs[3]  // 第 3 层的 KV 缓冲区

对每个线程 tid (0..127)，循环 i = tid, tid+128, tid+256, ...:
  例如 tid=0, i=0:
    lmcache_offset = key_value_offset(0, 3, 5, 0, 1024, 256, 32)
                   = 0*32*256*1024 + 3*256*1024 + 5*1024 + 0
                   = 786432 + 5120 + 0
                   = 791552

    vllm_offset = page_buffer_offset<1>(0, 5, 0, 1024, NB*16, 16, 32)
                = 0 * NB*16 * 1024 + 5 * 1024 + 0
                = 5120

    D2H: key_value[791552] = paged_buffer_ptr[5120]
    // 从 vLLM 第 3 层的第 5 个 token 的第 0 个 xword 拷贝到 LMCache

  例如 tid=1, i=1:
    lmcache_offset = 791552 + 1 = 791553
    vllm_offset = 5120 + 1 = 5121
    D2H: key_value[791553] = paged_buffer_ptr[5121]

  ... 每个线程处理 8 个 xword (1024/128=8)
```

#### A.6.8 `key_value` 的连续内存布局：到底有多长？

这是一个关键问题：`key_value` 张量的连续内存**不是整个模型的 KV Cache**，而是**一个 chunk（默认 256 tokens）的 KV Cache**。

##### 核心结论

```
key_value 不是 [2, NL, 整个序列长度, H]
而是     [2, NL, chunk_size, H]    ← 通常 chunk_size = 256
```

**LMCache 的设计哲学是"分块管理"**：将长序列切分为固定大小的 chunk，每个 chunk 独立分配、独立存储、独立传输。

##### 从代码追踪 `key_value` 的分配过程

**Step 1：TokenDatabase 将序列切分为 chunk**

```python
# cache_engine.py 第 461-470 行
for start, end, key in self.token_database.process_tokens(tokens, mask, ...):
    num_tokens = end - start  # 通常 = chunk_size = 256
```

`ChunkedTokenDatabase` 按 `chunk_size`（默认 256）将 token 序列切分：
```
输入: tokens[0:1024]
输出: chunk_0 = tokens[0:256], chunk_1 = tokens[256:512], ...
      每个 chunk 的 num_tokens = 256
```

**Step 2：根据 num_tokens 计算形状**

```python
# metadata.py 第 84-112 行
def get_shapes(self, num_tokens):
    return [torch.Size([
        self.kv_shape[1],                    # kv_size: 2 (K+V) 或 1 (MLA)
        self.kv_shape[0],                    # num_layers: 例如 32
        num_tokens,                          # chunk_size: 例如 256
        self.kv_shape[3] * self.kv_shape[4], # hidden_dim: NH * HS
    ])]
```

**Step 3：分配 MemoryObj**

```python
# cache_engine.py 第 475-482 行
memory_obj = self.storage_manager.allocate(kv_shapes, kv_dtypes, fmt=self.fmt)
```

分配器从预分配的钉扎 CPU 内存池中切出一块连续内存，形状为 `[2, NL, chunk_size, H]`。

**Step 4：传入 CUDA 内核**

```python
# gpu_connectors.py 第 316 行
lmc_ops.multi_layer_kv_transfer(
    memory_obj.tensor,  # shape = [2, NL, chunk_size, H]，例如 [2, 32, 256, 4096]
    ...
)
```

##### 各模型的具体内存大小

| 模型 | NL | NH | HS | H | dtype | chunk_size | key_value 形状 | 内存大小 |
|------|----|----|----|----|-------|-----------|---------------|---------|
| Llama-7B | 32 | 32 | 128 | 4096 | fp16 | 256 | [2, 32, 256, 4096] | **128 MB** |
| Llama-13B | 40 | 40 | 128 | 5120 | fp16 | 256 | [2, 40, 256, 5120] | **200 MB** |
| Llama-70B | 80 | 64 | 128 | 8192 | fp16 | 256 | [2, 80, 256, 8192] | **640 MB** |
| DeepSeek-V3 (MLA) | 61 | 1 | 512 | 512 | fp16 | 256 | [1, 61, 256, 512] | **15.6 MB** |
| Llama-7B | 32 | 32 | 128 | 4096 | fp16 | 128 | [2, 32, 128, 4096] | **64 MB** |

**计算公式**：
```
内存大小 = kv_size × NL × chunk_size × H × dtype_size
         = 2 × 32 × 256 × 4096 × 2 字节
         = 128 MB (Llama-7B)
```

##### 为什么是 chunk 而不是整个序列？

**原因 1：内存效率**
- 整个序列的 KV Cache 可能非常大（128K tokens × 4096 hidden × 2 bytes × 32 layers × 2 = 64 GB）
- 一次性分配 64 GB 连续内存不现实
- 分块后每块只需 128 MB，可以灵活分配和释放

**原因 2：缓存粒度**
- 不同 chunk 可以独立缓存、独立淘汰
- 前缀匹配在 chunk 粒度进行
- 部分命中时只需传输命中的 chunk

**原因 3：逐层流水线**
- layerwise 模式下，每层只需一个 chunk 的缓冲区
- 峰值内存从 32 × 128 MB = 4 GB 降到 1 × 128 MB = 128 MB

##### 内存布局的物理视图

以 Llama-7B fp16 为例，`key_value` 张量的物理内存布局：

```
key_value = [2, 32, 256, 4096] fp16
总大小 = 2 × 32 × 256 × 4096 × 2 字节 = 128 MB

物理内存（行优先连续排列）：
┌──────────────────────────────────────────────────────────┐
│ Key (k_or_v=0): 64 MB                                    │
│ ├── Layer 0:  256 tokens × 4096 elements × 2B = 2 MB     │
│ │   ├── Token 0:   4096 elements × 2B = 8 KB             │
│ │   ├── Token 1:   4096 elements × 2B = 8 KB             │
│ │   ├── ...                                               │
│ │   └── Token 255: 4096 elements × 2B = 8 KB             │
│ ├── Layer 1:  2 MB                                       │
│ ├── ...                                                   │
│ └── Layer 31: 2 MB                                       │
├──────────────────────────────────────────────────────────┤
│ Value (k_or_v=1): 64 MB                                  │
│ ├── Layer 0:  2 MB                                       │
│ ├── ...                                                   │
│ └── Layer 31: 2 MB                                       │
└──────────────────────────────────────────────────────────┘
```

##### 一个 chunk 的数据从哪来？

`key_value` 的一个 chunk（256 tokens）是从 vLLM 的分页 GPU 缓冲区中"聚集"而来的：

```
vLLM 的分页缓冲区（GPU）：
  [num_blocks, block_size, NH, HS]  例如 [1024, 16, 32, 128]
  物理上不连续——token 可能分散在不同 block 中

slot_mapping: [0, 1, 2, ..., 15, 16, 17, ..., 31, ...]
              block_0 的 16 个 token   block_1 的 16 个 token

LMCache 的连续缓冲区（CPU 钉扎内存）：
  [2, 32, 256, 4096]  256 个 token 连续排列
  物理上完全连续——通过 CUDA 内核从分页缓冲区聚集而来
```

##### 与 vLLM 分页缓冲区的大小对比

| 维度 | vLLM 分页缓冲区 | LMCache 连续缓冲区 |
|------|----------------|-------------------|
| 形状 | [NB, BS, NH, HS] 每层 | [2, NL, chunk_size, H] |
| token 数 | NB × BS（整个 KV 池） | chunk_size（通常 256） |
| 层数 | 每层独立张量 | 所有层在一个张量中 |
| 连续性 | 物理上不连续（分页） | 物理上完全连续 |
| 位置 | GPU 显存 | CPU 钉扎内存 |
| 典型大小 | 数十 GB | 128 MB（Llama-7B） |

##### 完整请求的 KV Cache 由多个 chunk 组成

一个 1024 token 的请求在 LMCache 中被切分为 4 个 chunk：

```
请求: 1024 tokens
  │
  ├── Chunk 0: tokens[0:256]
  │   key_value shape = [2, 32, 256, 4096], 128 MB
  │   CacheEngineKey = CacheEngineKey("llama-7b", 1, 0, hash_0, fp16)
  │
  ├── Chunk 1: tokens[256:512]
  │   key_value shape = [2, 32, 256, 4096], 128 MB
  │   CacheEngineKey = CacheEngineKey("llama-7b", 1, 0, hash_1, fp16)
  │
  ├── Chunk 2: tokens[512:768]
  │   key_value shape = [2, 32, 256, 4096], 128 MB
  │   CacheEngineKey = CacheEngineKey("llama-7b", 1, 0, hash_2, fp16)
  │
  └── Chunk 3: tokens[768:1024]
      key_value shape = [2, 32, 256, 4096], 128 MB
      CacheEngineKey = CacheEngineKey("llama-7b", 1, 0, hash_3, fp16)

每个 chunk 独立分配、独立传输、独立存储
4 个 chunk 的 key_value 是 4 个独立的 MemoryObj，不共享内存
```

##### chunk_size 对性能的影响

| chunk_size | 优点 | 缺点 |
|-----------|------|------|
| 128 | 内存粒度细，淘汰灵活 | 管理开销大，CUDA 内核启动次数多 |
| **256（默认）** | 平衡点 | - |
| 512 | 管理开销小 | 内存浪费（部分命中也需加载整块） |
| 1024 | 内核效率高 | 内存浪费严重 |

##### 昇腾适配的关键启示

1. **`key_value` 是 chunk 级别的**，不是整个序列——这意味着昇腾只需要分配 chunk_size 大小的连续缓冲区
2. **每个 chunk 独立传输**——可以逐 chunk 调用昇腾内核
3. **连续缓冲区在 CPU 侧**——昇腾内核需要支持从 NPU 读取分页数据写入 CPU 连续缓冲区（或反过来）
4. **chunk_size 通常为 256**——内核的 Grid.x 维度通常是 256

#### A.6.9 `get_kernel_ptr`——主机/设备指针统一（第 467-484 行）

**注意**：原 A.6.8 已上移为 A.6.8（key_value 连续内存布局），本节编号顺延。

```cpp
template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
    torch::Device device = tensor.device();
    if (device.is_cuda()) {
        return static_cast<T*>(tensor.data_ptr());
        // GPU 张量：直接返回设备指针
    } else if (device.is_cpu()) {
        T* ptr;
        auto st = cudaHostGetDevicePointer(
            (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
        TORCH_CHECK(st == cudaSuccess,
                    "Host tensor not registered/pinned (or bad ptr)");
        return ptr;
        // 钉扎 CPU 张量：返回设备可访问的主机指针
        // CUDA 内核可以通过 DMA 直接读写这个地址
    }
}
```

**这个函数的意义**：使得 CUDA 内核可以统一处理两种情况：
1. **key_value 在 GPU 上**：直接使用 `data_ptr()`，GPU→GPU 拷贝
2. **key_value 在钉扎 CPU 上**：通过 `cudaHostGetDevicePointer` 获取设备可访问指针，GPU→CPU DMA

**昇腾适配**：需要使用昇腾的主机内存注册 API（如 `aclrtMallocHost` 或 `aclrtRegisterMemory`）替代 `cudaHostGetDevicePointer`，使 NPU 可以通过 DMA 访问主机内存。

### A.7 两条路径的对比

| 维度 | 非逐层路径（默认） | 逐层路径（layerwise） |
|------|-------------------|---------------------|
| 入口 | `wait_for_save()` / `start_load_kv()` | `save_kv_layer()` / `wait_for_layer_load()` |
| Engine 方法 | `store()` / `retrieve()` | `store_layer()` / `retrieve_layer()`（生成器） |
| Connector 类 | `VLLMPagedMemGPUConnectorV2` | `VLLMPagedMemLayerwiseGPUConnector` |
| CUDA 算子 | `multi_layer_kv_transfer` ⭐ | `single_layer_kv_transfer` |
| 内核启动次数 | 1 次（所有层） | NL 次（每层 1 次） |
| CPU 内存峰值 | 需要所有层的缓冲区 | 只需 1-2 层的缓冲区 |
| 与 forward 的关系 | 不交错 | 逐层交错 |

### A.8 Python Fallback（无 CUDA 时的备选实现）

**文件**：`lmcache/python_ops_fallback.py` 第 505-636 行

当 CUDA 不可用时，提供纯 Python 向量化实现：

```python
def multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping, ...):
    # 1. 过滤无效 slot
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]

    # 2. 对每层执行向量化传输
    for layer_id in range(num_layers):
        paged_tensor = _tensor_from_ptr(key_value_ptrs[layer_id], shape, dtype, device)

        if direction == TransferDirection.H2D:
            # LMCache → vLLM：index_copy_ 将连续数据散射到分页位置
            lmc_valid = key_value[:, layer_id, valid_mask_kv, :]
            paged_tensor.index_copy_(1, valid_slots, lmc_valid.to(device))
        else:
            # vLLM → LMCache：index_select 从分页位置聚集到连续数据
            gathered = paged_tensor.index_select(1, valid_slots)
            key_value[:, layer_id, valid_mask_kv, :] = gathered.to(kv_device)
```

**昇腾适配参考**：这个 fallback 实现展示了算法的纯逻辑，可以作为昇腾算子实现的参考。核心就是 `index_copy_`（scatter）和 `index_select`（gather）操作。

### A.9 数据流参数变化追踪

以 Store 路径（GPU→CPU）为例，追踪 `slot_mapping` 在各层的变化：

```
vLLM Worker:
  slot_mapping = [0, 1, 2, ..., 15, 16, 17, ..., 31, ...]  # 全局 slot 映射
  │
  ▼
vllm_v1_adapter.py (wait_for_save):
  slot_mapping = slot_mapping  # 透传
  │
  ▼
cache_engine.py (store):
  # 按 chunk 切分
  for start, end, key in token_database.process_tokens(tokens, mask):
      slot_mapping_chunk = slot_mapping[start:end]  # [chunk_size]
  │
  ▼
gpu_connectors.py (from_gpu):
  slot_mapping[start:end]  # 传入 CUDA 算子
  │
  ▼
mem_kernels.cu (kernel):
  slot_idx = slot_mapping[token_id + skip_prefix_n_tokens]
  block_idx = slot_idx / block_size
  block_offset = slot_idx % block_size
  // 根据 GPUKVFormat 计算 vllm_offset
```

以 `key_value` 张量的变化追踪：

```
vLLM Worker:
  kvcaches = [(K_layer0, V_layer0), (K_layer1, V_layer1), ...]
  每层形状: [num_blocks, block_size, num_heads, head_size]
  │
  ▼
gpu_connectors.py (_initialize_pointers):
  # 提取每层的 data_ptr()，存入 GPU 指针数组
  kv_cache_pointers = [K0.data_ptr(), V0.data_ptr(), K1.data_ptr(), V1.data_ptr(), ...]
  │
  ▼
cache_engine.py (store):
  # 分配 LMCache 连续张量
  memory_obj.tensor shape = [2, num_layers, chunk_size, hidden_dim]
  │
  ▼
mem_kernels.cu (kernel):
  # LMCache 连续张量偏移
  lmcache_offset = k_or_v * NL * T * H + layer_id * T * H + token_id * H + i
  # vLLM 分页缓冲区偏移（取决于格式）
  vllm_offset = page_buffer_offset<format>(k_or_v, slot_idx, i, ...)
```

### A.10 get_kernel_ptr——主机/设备指针统一

```cpp
// mem_kernels.cu 第 468-484 行
template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
    if (tensor.device().is_cuda()) {
        return static_cast<T*>(tensor.data_ptr());  // GPU 张量：直接使用
    } else if (tensor.device().is_cpu()) {
        T* ptr;
        cudaHostGetDevicePointer(&ptr, tensor.data_ptr(), 0);  // 钉扎内存：获取设备可访问指针
        return ptr;
    }
}
```

这个函数使得 CUDA 内核可以统一处理 GPU 张量和钉扎 CPU 张量：
- GPU 张量：直接使用 `data_ptr()`
- 钉扎 CPU 张量：通过 `cudaHostGetDevicePointer` 获取设备可访问的指针

**昇腾适配**：需要使用昇腾的主机内存注册 API 替代，使 NPU 可以通过 DMA 访问主机内存。

---

## 附录 B：关键代码文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 742-868 | `start_load_kv()` Load 路径入口 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 948-972 | `wait_for_layer_load()` 逐层等待 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 975-1074 | `save_kv_layer()` 逐层保存 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 1077-1219 | `wait_for_save()` 保存完成 |
| `lmcache/v1/cache_engine.py` | 364-567 | `store()` 非逐层存储 |
| `lmcache/v1/cache_engine.py` | 569-751 | `store_layer()` 逐层存储生成器 |
| `lmcache/v1/cache_engine.py` | 755-906 | `retrieve()` 非逐层检索 |
| `lmcache/v1/cache_engine.py` | 908-1061 | `retrieve_layer()` 逐层检索生成器 |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | 142-415 | `VLLMPagedMemGPUConnectorV2` |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | 240-270 | `_initialize_pointers()` 指针收集 |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | 264-300 | `to_gpu()` CPU→GPU |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | 329-370 | `from_gpu()` GPU→CPU |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | 1052-1421 | `VLLMPagedMemLayerwiseGPUConnector` |
| `lmcache/v1/gpu_connector/gpu_ops.py` | 1-80 | `lmcache_memcpy_async_h2d/d2h` |
| `lmcache/python_ops_fallback.py` | 505-636 | Python fallback 实现 |
| `csrc/pybind.cpp` | 36-42 | Python 绑定 |
| `csrc/mem_kernels.cuh` | 1-148 | GPUKVFormat 枚举 + 函数声明 |
| `csrc/mem_kernels.cu` | 216-294 | `page_buffer_offset()` 格式寻址 |
| `csrc/mem_kernels.cu` | 301-308 | `key_value_offset()` 连续张量寻址 |
| `csrc/mem_kernels.cu` | 368-411 | `load_and_reshape_multi_layer_kernel` 内核 |
| `csrc/mem_kernels.cu` | 519-613 | `multi_layer_kv_transfer_templated` 分派 |
| `csrc/mem_kernels.cu` | 620-647 | `multi_layer_kv_transfer` 入口 |

---

> 文档生成时间：2026-05-30
> 基于 LMCache `csrc/mem_kernels.cu` 源码深度分析
