# LMCache 架构深度解析

> 本文档全面剖析 LMCache 项目的系统架构、核心组件、数据流转全流程、CUDA 算子实现、vLLM 集成机制以及 Mooncake 存储集成。
>
> 版本基准：`dev` 分支，2026 年 5 月

---

## 目录

1. [系统架构概述](#1-系统架构概述)
2. [目录结构与模块职责](#2-目录结构与模块职责)
3. [核心组件关系](#3-核心组件关系)
4. [KVCache 数据流转全流程](#4-kvcache-数据流转全流程)
5. [multi_layer_kv_transfer 算子深度解析](#5-multi_layer_kv_transfer-算子深度解析)
6. [vLLM 到 LMCache 的 KVCache 转换](#6-vllm-到-lmcache-的-kvcache-转换)
7. [LMCache 到 Mooncake 的存储过程](#7-lmcache-到-mooncake-的存储过程)
8. [完整端到端流程示例](#8-完整端到端流程示例)
9. [关键优化技术总结](#9-关键优化技术总结)

---

## 1. 系统架构概述

### 1.1 项目定位

LMCache 是一个面向 LLM 推理服务的 **KV Cache 管理引擎**，核心目标是通过在多级存储（GPU → CPU → 磁盘 → 远程存储）之间智能缓存和传输 KV Cache，降低首 Token 延迟（TTFT），提升吞吐量，尤其是在长上下文场景下效果显著。

### 1.2 架构全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM 推理引擎 (vLLM / SGLang / TRT-LLM)        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Paged KV    │  │ Slot        │  │ Scheduler   │                 │
│  │ Buffer      │  │ Mapping     │  │ (控制面)     │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          │ KV Connector 协议                        │
├──────────────────────────┼──────────────────────────────────────────┤
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Integration Layer (集成层)                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
│  │  │ vLLM Adapter │  │ SGLang Adapter│  │ TRT-LLM      │       │   │
│  │  │ (进程内/MP)   │  │              │  │ Adapter      │       │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │   │
│  └─────────┼─────────────────┼─────────────────┼───────────────┘   │
│            └─────────────────┼─────────────────┘                   │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              GPU Connector Layer (GPU 连接层)                 │   │
│  │  ┌──────────────────┐  ┌──────────────────────┐             │   │
│  │  │ VLLMPagedMem     │  │ multi_layer_kv_       │             │   │
│  │  │ GPUConnectorV2/V3 │  │ transfer CUDA Kernel  │             │   │
│  │  └──────────────────┘  └──────────────────────┘             │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              LMCache Engine (核心引擎)                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Token DB │  │ Memory   │  │ Storage  │  │ GPU      │   │   │
│  │  │ (分块+   │  │ Allocator│  │ Manager  │  │ Connector│   │   │
│  │  │  前缀哈希)│  │ (钉扎内存)│  │ (多后端)  │  │ (传输)   │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Storage Backend Layer (存储后端层)                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Local    │  │ Local    │  │ Remote   │  │ P2P /    │   │   │
│  │  │ CPU      │  │ Disk     │  │ (Mooncake│  │ NIXL /   │   │   │
│  │  │ (热缓存)  │  │ (冷缓存)  │  │  Redis..)│  │ GDS      │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 设计哲学

LMCache 的核心设计理念是 **"连续存储 + 散射/聚集传输"**：

- **LMCache 侧**：KV Cache 以连续的 `[2, num_layers, num_tokens, hidden_dim]` 张量存储，便于管理、序列化、压缩
- **推理引擎侧**：KV Cache 以分页块（paged blocks）存储，通过 `slot_mapping` 索引
- **CUDA 算子**：负责两者之间的散射（scatter，CPU→GPU）和聚集（gather，GPU→CPU），通过编译期模板消除运行时分支

---

## 2. 目录结构与模块职责

### 2.1 顶层目录

| 目录 | 职责 |
|---|---|
| `lmcache/` | Python 主包——核心引擎、存储后端、集成层、CLI、工具 |
| `csrc/` | C++/CUDA/HIP/SYCL 原生扩展——GPU 内存内核、存储后端（Redis、文件系统、Mooncake）、存储管理器 |
| `docs/` | Sphinx 用户文档、设计文档、编码标准 |
| `tests/` | 测试套件，镜像 `lmcache/` 包结构 |
| `examples/` | 25+ 示例目录：disagg_prefill、blend_kv、cache_controller、multi_process 等 |
| `benchmarks/` | 性能基准测试 |
| `operator/` | Kubernetes Operator（Go 语言） |
| `docker/` | Docker 构建文件（CUDA、ROCm、轻量级） |
| `rust/` | Rust 原生块存储后端插件 |
| `requirements/` | 按用途拆分的依赖文件 |

### 2.2 核心 Python 包：`lmcache/v1/`

这是项目的灵魂所在，包含 2000+ 行的缓存引擎和完整的存储后端体系。

#### 2.2.1 核心引擎文件

| 文件 | 职责 |
|---|---|
| `cache_engine.py` (2104 行) | `LMCacheEngine`——中央协调器，管理 tokenization、内存管理、存储、GPU 传输的全生命周期 |
| `config.py` (1075 行) | `LMCacheEngineConfig`——50+ 配置参数（chunk_size、local_cpu、远程 URL 等） |
| `token_database.py` | `ChunkedTokenDatabase`、`SegmentTokenDatabase`——将 token 序列映射为缓存键 |
| `memory_management.py` | `TensorMemoryObj`、`PagedTensorMemoryAllocator`、`MixedMemoryAllocator`——内存分配与管理 |
| `metadata.py` | `LMCacheMetadata`——模型名称、world_size、worker_id、kv_shape、kv_dtype |
| `kv_layer_groups.py` | KV 层分组管理（用于异构 KV 布局，如 DeepSeek V4） |

#### 2.2.2 存储后端：`lmcache/v1/storage_backend/`

```
storage_backend/
├── abstract_backend.py          # StorageBackendInterface 基类
├── storage_manager.py           # StorageManager——编排所有后端
├── local_cpu_backend.py         # CPU 内存后端（热缓存 + 内存分配器）
├── local_disk_backend.py        # 本地磁盘后端
├── remote_backend.py            # 远程存储后端（通用包装器）
├── gds_backend.py               # GPU Direct Storage 后端
├── nixl_storage_backend.py      # NIXL 存储后端
├── p2p_backend.py               # 点对点 GPU 缓存共享
├── pd_backend.py                # Prefill-Decode 分离后端
├── maru_backend.py              # Maru 存储后端
├── connector/                   # 16 个连接器实现
│   ├── mooncakestore_connector.py   # Mooncake 分布式存储
│   ├── redis_connector.py           # Redis/Valkey
│   ├── s3_connector.py              # AWS S3
│   ├── fs_connector.py              # 本地/共享文件系统
│   ├── infinistore_connector.py     # InfiniStore RDMA
│   └── ...                          # 更多连接器
├── cache_policy/                # 缓存淘汰策略
│   ├── lru.py                       # LRU
│   ├── lfu.py                       # LFU
│   ├── fifo.py                      # FIFO
│   └── mru.py                       # MRU
└── naive_serde/                 # 序列化/反序列化
    ├── naive_serde.py               # 直通（不压缩）
    └── kivi_serde.py                # 量化压缩
```

#### 2.2.3 GPU 连接层：`lmcache/v1/gpu_connector/`

```
gpu_connector/
├── gpu_connectors.py (89K)     # GPUConnectorInterface + 7 个具体实现
├── gpu_ops.py                  # GPU 操作包装器（导入 c_ops）
├── utils.py (62K)              # 格式发现、指针收集、形状描述
├── xpu_connectors.py           # Intel XPU 连接器
├── hpu_connector.py            # Habana HPU 连接器
└── mock_gpu_connector.py       # 测试用 Mock
```

#### 2.2.4 集成层：`lmcache/integration/`

```
integration/
├── vllm/
│   ├── vllm_v1_adapter.py (72K)        # 进程内 vLLM 适配器
│   ├── vllm_multi_process_adapter.py (49K) # 多进程 vLLM 适配器
│   ├── lmcache_connector_v1.py          # vLLM KV Connector 入口
│   ├── lmcache_mp_connector.py          # 多进程 KV Connector 入口
│   └── vllm_service_factory.py          # 服务工厂
├── sglang/
│   ├── sglang_adapter.py                # SGLang 适配器
│   └── multi_process_adapter.py         # SGLang 多进程适配器
└── tensorrt_llm/                        # TRT-LLM 集成
```

#### 2.2.5 多进程架构：`lmcache/v1/multiprocess/`

```
multiprocess/
├── server.py                # 多进程服务器
├── mq.py (31K)              # ZeroMQ 消息队列实现
├── gpu_context.py           # GPU 上下文管理
├── non_gpu_context.py       # CPU-only 工作进程
├── transfer_context.py      # 传输上下文（CUDA IPC / Data Transfer）
├── session.py               # 会话管理
├── http_server.py           # HTTP API 服务器
└── modules/                 # 可插拔模块
```

### 2.3 C++/CUDA 原生扩展：`csrc/`

```
csrc/
├── mem_kernels.cu           # 主 KV 传输 CUDA 内核（1090 行）
├── mem_kernels.cuh          # GPUKVFormat 枚举、TransferDirection 枚举
├── mp_mem_kernels.cu        # 多进程块级 KV 传输内核（391 行）
├── mp_mem_kernels.cuh       # PageBufferShapeDesc 结构体
├── pos_kernels.cu           # 融合旋转位置编码内核
├── mem_alloc.cpp            # 钉扎内存、NUMA、大页、共享内存分配器
├── pybind.cpp               # c_ops Python 绑定
├── ac_enc.cu / ac_dec.cu    # CacheGen 算术编码压缩
├── storage_backends/
│   ├── connector_base.h     # 连接器基类模板
│   ├── mooncake/            # Mooncake C++ 连接器
│   ├── redis/               # Redis C++ 连接器
│   └── fs/                  # 文件系统 C++ 连接器
└── sycl/                    # Intel XPU SYCL 移植
```

---

## 3. 核心组件关系

### 3.1 组件依赖图

```
                    LMCacheEngine
                   /    |    |    \
                  /     |    |     \
         TokenDB  MemoryAlloc  StorageManager  GPUConnector
            |         |              |               |
            |         |         ┌────┴────┐          |
            |         |         |         |          |
            |         |    Allocator   Backend       |
            |         |    Backend    Interfaces     |
            |         |    (CPU)      /    |    \    |
            |         |         CPU  Disk  Remote   |
            |         |              |      |       |
            |         |              |   Connector   |
            |         |              |   (Mooncake,  |
            |         |              |    Redis, S3) |
            |         |              |              CUDA Kernels
            |         |              |         (multi_layer_kv_transfer)
            v         v              v              v
        CacheEngineKey    MemoryObj    StorageBackend   PagedKVBuffer
```

### 3.2 CacheEngineKey——缓存键

定义在 `lmcache/utils.py`：

```python
@dataclass
class CacheEngineKey:
    model_name: str       # 模型名称，如 "llama-7b"
    world_size: int       # 张量并行 world size
    worker_id: int        # 当前 TP rank
    chunk_hash: int       # token 块的前缀哈希
    dtype: torch.dtype    # KV 张量数据类型
    request_configs: dict # 可选的每请求标签（如 LoRA ID）
```

序列化格式：`model_name@world_size@worker_id@chunk_hash_hex@dtype[@tag%value...]`

对于 layerwise 模式，`LayerCacheEngineKey` 在 `dtype` 之后追加 `layer_id` 字段。

### 3.3 MemoryObj——内存对象

```
MemoryObj (抽象基类)
├── TensorMemoryObj     # 包装 torch.Tensor，支持引用计数和 Pin
└── BytesBufferMemoryObj # 包装 bytes，用于序列化/压缩数据
```

`TensorMemoryObj` 内部持有一个 `raw_data`（连续的 torch.Tensor）和 `MemoryObjMetadata`（形状、数据类型、地址、引用计数、Pin 计数、内存格式）。`tensor` 属性将扁平缓冲区重塑为逻辑 KV 形状。

### 3.4 内存分配器层次

```
MemoryAllocatorInterface
├── TensorMemoryAllocator       # 通用分配器，使用 AddressManager 空闲链表
├── PagedTensorMemoryAllocator  # 固定页大小，O(1) 分配/释放
├── MixedMemoryAllocator        # 生产默认：PinMemory + BufferAllocator
├── GPUMemoryAllocator          # GPU 内存分配器
└── CuFileMemoryAllocator       # GDS 分配器
```

### 3.5 StorageManager——存储编排

`StorageManager` 持有一个 `OrderedDict[str, StorageBackendInterface]`，按固定顺序创建后端：

1. `PDBackend`（如果启用了 PD 分离）
2. `LocalCPUBackend`（始终创建，作为分配器后端）
3. `P2PBackend`（如果启用 P2P）
4. `NixlStorageBackend`（如果启用 NIXL）
5. `LocalDiskBackend`（如果配置了磁盘路径）
6. `GdsBackend`（如果配置了 GDS 路径）
7. `RemoteBackend` 实例（每个远程插件一个）
8. 动态存储插件

---

## 4. KVCache 数据流转全流程

### 4.1 Store 流程：从 GPU 到存储

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Store 全流程                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① Token 处理                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ ChunkedTokenDatabase.process_tokens(tokens, mask)               │ │
│  │   ├── 按 chunk_size (默认256) 分块                              │ │
│  │   ├── 前缀哈希: hash_N = hash_func((hash_{N-1}, chunk_N, ()))   │ │
│  │   └── 产出: (start_idx, end_idx, CacheEngineKey)                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ② 内存分配                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ StorageManager.allocate(kv_shapes, kv_dtypes, fmt)              │ │
│  │   └── MixedMemoryAllocator.allocate()                           │ │
│  │       └── 从预分配的钉扎 CPU 缓冲区中切出一块 TensorMemoryObj     │ │
│  │           形状: [2, num_layers, num_tokens, hidden_dim]          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ③ GPU→CPU 传输                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ GPUConnector.batched_from_gpu(memory_objs, starts, ends)        │ │
│  │   ├── 收集每层的 data_ptr() 到 GPU 指针张量                      │ │
│  │   ├── 启动 multi_layer_kv_transfer(D2H) CUDA 内核               │ │
│  │   │   ├── 读取 slot_mapping[token_id] 得到物理 slot 索引         │ │
│  │   │   ├── 根据 GPUKVFormat 计算分页缓冲区偏移                    │ │
│  │   │   └── 将数据从分页 GPU 缓冲区聚集到连续 CPU 张量              │ │
│  │   └── store_stream.synchronize() 等待传输完成                    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ④ 异步写入存储后端                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ StorageManager.batched_put(keys, memory_objs)                   │ │
│  │   ├── LocalCPUBackend: 插入 hot_cache 字典 (key → memory_obj)   │ │
│  │   ├── LocalDiskBackend: 异步写入 .pt 文件                       │ │
│  │   ├── RemoteBackend:                                            │ │
│  │   │   ├── 序列化 MemoryObj (naive/kivi/cachegen)                │ │
│  │   │   └── connector.put(key, compressed_obj)                    │ │
│  │   │       └── Mooncake: store.put_from(key, ptr, size)          │ │
│  │   │           或 store.batch_put_from(keys, ptrs, sizes)        │ │
│  │   └── ref_count_down() 所有 memory_obj                          │ │
│  │       └── 引用计数归零时，内存归还分配器空闲链表                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Retrieve 流程：从存储到 GPU

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Retrieve 全流程                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① Token 处理（同 Store）                                             │
│  ② 查找并获取                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ StorageManager.batched_get(keys, location)                      │ │
│  │   ├── 按顺序搜索: CPU 热缓存 → 磁盘 → 远程                      │ │
│  │   ├── CPU 命中: 零拷贝返回 MemoryObj（已在 CPU 内存中）           │ │
│  │   ├── 磁盘命中: 分配 CPU MemoryObj，从磁盘文件读取               │ │
│  │   ├── 远程命中: connector.get()，反序列化                        │ │
│  │   │   └── Mooncake: store.batch_get_into(keys, ptrs, sizes)     │ │
│  │   │       直接写入预分配的钉扎内存缓冲区（零拷贝 RDMA）           │ │
│  │   └── 自动回写: 非 CPU 命中的数据自动缓存到 LocalCPUBackend      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ③ CPU→GPU 传输                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ GPUConnector.batched_to_gpu(memory_objs, starts, ends)          │ │
│  │   ├── 启动 multi_layer_kv_transfer(H2D) CUDA 内核               │ │
│  │   │   ├── 从连续 CPU 张量读取数据                                │ │
│  │   │   ├── 根据 slot_mapping 和 GPUKVFormat 散射到分页 GPU 缓冲区 │ │
│  │   │   └── skip_prefix_n_tokens 跳过已缓存的前缀                  │ │
│  │   └── load_stream.synchronize() 等待传输完成                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ④ 清理                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ ref_count_down() 所有 memory_obj                                 │ │
│  │ 返回 ret_mask (布尔张量，标记哪些 token 位置成功检索)             │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Layerwise Store/Retrieve

当 `config.use_layerwise=True` 时，引擎使用生成器模式逐层处理，降低峰值 CPU 内存使用：

**Store（逐层）：**
1. Token 处理和内存分配一次性完成，但 key 按层拆分：`key.split_layers(num_layers)`
2. key/memory_obj 从 chunk-major 转置为 layer-major
3. `GPUConnector.batched_from_gpu()` 返回生成器，每次产出一层
4. 每层：推进生成器（GPU→CPU 拷贝该层），然后 `storage_manager.batched_put()` 写入存储

**Retrieve（逐层）：**
1. key 按层拆分并转置
2. `storage_manager.layerwise_batched_get()` 返回每层 future 的生成器
3. 每层：等待 future 完成，通过 `gpu_connector.batched_to_gpu()` 发送到 GPU
4. 使用 **ping-pong 双缓冲** 重叠计算和加载

### 4.4 Lookup 流程

`LMCacheEngine.lookup()` 检查 KV Cache 是否存在，但不实际加载：

1. Token 处理得到每个 chunk 的 `(start, end, key)`
2. `storage_manager.batched_contains()` 检查所有后端
3. 对于 layerwise 模式：chunk 必须在所有层都存在于同一位置才算命中
4. 返回前缀连续命中的 token 数量
5. 如果 `pin=True`，命中 key 被钉扎以防止淘汰

---

## 5. multi_layer_kv_transfer 算子深度解析

### 5.1 算子概述

`multi_layer_kv_transfer` 是 LMCache 最核心的 CUDA 算子，负责在 LMCache 的连续内存布局和推理引擎的分页 GPU 缓冲区之间进行高效的数据传输。它处理的是一个**散射/聚集问题**：

- **LMCache 侧**：连续的 `[2, num_layers, num_tokens, hidden_dim]` 张量
- **推理引擎侧**：分页块，通过 `slot_mapping` 索引

### 5.2 GPUKVFormat 枚举——10 种物理内存布局

定义在 `csrc/mem_kernels.cuh`，这是算子的核心抽象：

| 枚举值 | 整数 | 使用者 | 物理布局 |
|---|---|---|---|
| `NB_NL_TWO_BS_NH_HS` | 0 | vLLM 跨层 | 单张量 `[NB, NL, 2, BS, NH, HS]` |
| `NL_X_TWO_NB_BS_NH_HS` | 1 | vLLM FlashAttention (NHD) | 每层 `[2, NB, BS, NH, HS]` |
| `NL_X_NB_TWO_BS_NH_HS` | 2 | vLLM FlashInfer (NHD) | 每层 `[NB, 2, BS, NH, HS]` |
| `NL_X_NB_BS_HS` | 3 | vLLM MLA | 每层 `[NB, BS, HS]` |
| `TWO_X_NL_X_NBBS_NH_HS` | 4 | SGLang MHA (进程内) | `2 x NL x [PBS, NH, HS]` |
| `NL_X_NBBS_ONE_HS` | 5 | SGLang MLA | 每层 `[PBS, 1, HS]` |
| `NL_X_TWO_NB_NH_BS_HS` | 6 | vLLM FlashAttention (HND) | 每层 `[2, NB, NH, BS, HS]` |
| `NL_X_NB_TWO_NH_BS_HS` | 7 | vLLM FlashInfer (HND) | 每层 `[NB, 2, NH, BS, HS]` |
| `NB_NL_TWO_NH_BS_HS` | 8 | TRT-LLM 跨层 (HND) | 单张量 `[NB, NL, 2, NH, BS, HS]` |
| `TWO_X_NL_X_NB_BS_NH_HS` | 9 | SGLang MHA via MP daemon | `2 x NL x [NB, BS, NH, HS]` |

其中：NB=num_blocks, NL=num_layers, BS=block_size, NH=num_heads, HS=head_size, PBS=page_buffer_size

### 5.3 multi_layer_kv_transfer 核心内核

**文件**：`csrc/mem_kernels.cu`，第 620 行

**函数签名**：
```cpp
void multi_layer_kv_transfer(
    torch::Tensor key_value,           // LMCache 连续张量 [2, NL, T, H]
    torch::Tensor key_value_ptrs,      // 推理引擎分页缓冲区指针（GPU 上的 int64 张量）
    torch::Tensor slot_mapping,        // token 位置 → 分页 slot 映射
    torch::Device paged_memory_device,
    int64_t page_buffer_size,
    TransferDirection direction,        // H2D 或 D2H
    GPUKVFormat gpu_kv_format,         // 10 种格式之一
    int64_t block_size,
    int64_t head_size,
    int64_t skip_prefix_n_tokens = 0
)
```

**内核启动参数**：
```
Grid:  (num_tokens, num_layers, k_or_v_size)
       其中 k_or_v_size = 2 (MHA) 或 1 (MLA)
Block: (128,)  // 每个线程块 128 个线程
```

**每个线程块的处理逻辑**：

```
┌─────────────────────────────────────────────────────────────────┐
│  线程块 (token_id, layer_id, k_or_v)                            │
│                                                                  │
│  1. 读取 slot_idx = slot_mapping[token_id]                       │
│     └── 如果 slot_idx == -1，跳过                                │
│                                                                  │
│  2. 分解 slot 索引:                                              │
│     block_idx   = slot_idx / block_size                          │
│     block_offset = slot_idx % block_size                         │
│                                                                  │
│  3. 计算分页缓冲区偏移 (page_buffer_offset<format>):             │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ NHD 格式:                                            │     │
│     │   offset = block_idx * stride +                      │     │
│     │            k_or_v * block_size * scalars_per_token + │     │
│     │            block_offset * scalars_per_token          │     │
│     │                                                      │     │
│     │ HND 格式:                                            │     │
│     │   head_idx   = scalar_offset / head_size             │     │
│     │   head_offset = scalar_offset % head_size            │     │
│     │   offset = block_idx * stride +                      │     │
│     │            head_idx * block_size * head_size +       │     │
│     │            block_offset * head_size                  │     │
│     │                                                      │     │
│     │ MLA 格式:                                            │     │
│     │   offset = token_idx * scalars_per_token             │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│  4. 复制 scalars_per_token 个元素:                               │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ D2H: paged_buffer[offset+i] → key_value[kv,l,t,i]  │     │
│     │ H2D: key_value[kv,l,t,i] → paged_buffer[offset+i]  │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│  5. 类型分派: 根据元素大小选择 int64/int32/int16/int8            │
│     └── 最大化向量化内存访问（8 字节 load/store）                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 编译期模板优化

关键优化：**方向（H2D/D2H）和格式（GPUKVFormat）都是编译期模板参数**，编译器为每种组合生成特化代码，消除所有运行时分支：

```cpp
template <bool DIRECTION, GPUKVFormat FORMAT, typename scalar_t>
__global__ void load_and_reshape_multi_layer_kernel(...) {
    // 编译期已知 FORMAT，page_buffer_offset 直接内联
    // 编译期已知 DIRECTION，copy 方向直接内联
    // 零运行时分支
}
```

### 5.5 类型分派

```cpp
// 根据元素大小选择最优标量类型
if (copy_size % sizeof(int64_t) == 0) {
    AT_DISPATCH_ALL_TYPES_AND(..., at::ScalarType::Long, ...);
} else if (copy_size % sizeof(int32_t) == 0) {
    AT_DISPATCH_ALL_TYPES_AND(..., at::ScalarType::Int, ...);
} else if (copy_size % sizeof(int16_t) == 0) {
    AT_DISPATCH_ALL_TYPES_AND(..., at::ScalarType::Short, ...);
} else {
    AT_DISPATCH_ALL_TYPES_AND(..., at::ScalarType::Byte, ...);
}
```

### 5.6 SGLang 专用变体

**`multi_layer_kv_transfer_unilateral()`**（第 674 行）：

SGLang 的 K 和 V 存储在**独立的缓冲区**中（不是交错的）。指针数组顺序为 `[K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]`，内核通过 `paged_buffer_ptrs[layer_id]` 和 `paged_buffer_ptrs[layer_id + num_layers]` 分别索引 K 和 V。

### 5.7 单层传输变体

**`single_layer_kv_transfer()`**（第 728 行）：

用于逐层（layerwise）连接器，每次传输一层。Grid 维度为 `(num_tokens,)`。支持 NHD 和 HND 两种布局。

**`single_layer_kv_transfer_sgl()`**（第 974 行）：

SGLang 专用单层变体，接受独立的 `key_cache` 和 `value_cache` 张量。

### 5.8 多进程块级传输

**`multi_layer_block_kv_transfer()`**（`csrc/mp_mem_kernels.cu`，第 371 行）：

这是更高级的块级内核，用于多进程路径和 TRT-LLM：

- **块级寻址**：接受 `engine_block_ids`（块索引张量），而非 `slot_mapping`
- **多个 LMCache 对象**：通过 `MemoryObj4<ScalarType>` 支持 1-4 个内存对象
- **PageBufferShapeDesc**：详细的形状描述符，包含 `block_stride_elems` 处理填充池
- **Per-head warp 组织**：每个 warp 处理一个 head
- **向量化拷贝**：使用 `uint4` 的 128 位 cache-streaming load/store
- **skip_prefix_n_blocks**：跳过前 N 个块（用于前缀缓存）

### 5.9 异步内存拷贝

**`lmcache_memcpy_async()`**（第 1056 行）：

在钉扎主机内存和设备内存之间执行异步 `cudaMemcpyAsync`，按 `cudaHostRegister` 对齐粒度分块拷贝。这对 `LazyMemoryAllocator` 路径至关重要。

---

## 6. vLLM 到 LMCache 的 KVCache 转换

### 6.1 两种集成架构

LMCache 与 vLLM 的集成有两种截然不同的架构：

| 特性 | Path A: 进程内连接器 | Path B: 多进程连接器 |
|---|---|---|
| 入口 | `LMCacheConnectorV1Dynamic` | `LMCacheMPConnector` |
| 实现 | `vllm_v1_adapter.py` | `vllm_multi_process_adapter.py` |
| 引擎位置 | vLLM worker 进程内 | 独立 LMCache 服务器进程 |
| 通信 | 直接函数调用 | ZeroMQ 消息队列 |
| GPU 访问 | 直接访问 vLLM 的 KV 缓冲区 | 通过 CUDA IPC 句柄 |
| 逐层流水线 | 支持（生成器模式） | 不支持（块级批量传输） |

### 6.2 vLLM KV Connector 协议

vLLM 的 KV Connector 协议将操作分为**调度器角色**（控制面）和**工作进程角色**（数据面）：

**调度器侧（每个 forward pass）：**
```
① get_num_new_matched_tokens(request, num_computed_tokens)
   └── vLLM 询问：有多少 token 在外部缓存中？

② update_state_after_alloc(request, blocks, num_external_tokens)
   └── vLLM 确认：已为外部 token 分配块

③ build_connector_meta(scheduler_output)
   └── vLLM 要求：构建发送给 worker 的元数据

④ request_finished(request, block_ids)
   └── vLLM 通知：请求已完成
```

**工作进程侧（每个 forward pass）：**
```
① register_kv_caches(kv_caches)
   └── vLLM 提供物理 KV 缓存张量

② start_load_kv(forward_context)
   └── forward pass 前：启动从 LMCache 加载

③ wait_for_layer_load(layer_name)
   └── 每层调用：逐层流水线加载

④ save_kv_layer(layer_name, kv_layer, attn_metadata)
   └── 每层调用：逐层保存 KV 数据

⑤ wait_for_save()
   └── forward pass 后：确保所有保存完成

⑥ get_finished(finished_req_ids)
   └── 返回已完成异步传输的请求 ID 集合
```

### 6.3 进程内逐层传输（Path A）

#### 6.3.1 初始化

```python
# vllm_v1_adapter.py 第 452-498 行
class LMCacheConnectorV1Impl:
    def __init__(self, ...):
        config = lmcache_get_or_create_config()
        factory = VllmServiceFactory(config, ...)
        self.lmcache_engine = LMCacheManager.create_engine(config, metadata, gpu_connector)
```

GPU 连接器选择逻辑（`CreateGPUConnector()`）：
- `use_layerwise=True` → `VLLMPagedMemLayerwiseGPUConnector`（或 blending 时用 `VLLMBufferLayerwiseGPUConnector`）
- `use_layerwise=False, use_gpu_connector_v3=True` → `VLLMPagedMemGPUConnectorV3`（异构 KV 层组）
- `use_layerwise=False, use_gpu_connector_v3=False` → `VLLMPagedMemGPUConnectorV2`

#### 6.3.2 Slot Mapping 构建

```python
# vllm_v1_adapter.py 第 296-434 行
block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
block_offsets = torch.arange(0, block_size, dtype=torch.long)
slot_mapping = (
    block_offsets.reshape((1, block_size))
    + block_ids.reshape((num_blocks, 1)) * block_size
)
slot_mapping = slot_mapping.flatten()[:len(token_ids)]
```

这将每个 token 位置映射到 vLLM 分页 KV 缓冲区中的物理 slot：`slot = block_id * block_size + offset_within_block`

#### 6.3.3 逐层 Save（存储）

```
save_kv_layer(layer_name) 被每层调用
    │
    ├── 首次调用: 创建 layerwise_storer 生成器
    │   └── lmcache_engine.store_layer(token_ids, mask, kvcaches, slot_mapping)
    │
    └── 后续调用: next(layerwise_storer) 推进一层
        │
        └── 生成器内部:
            ├── 对每个 chunk:
            │   ├── 检查是否已缓存 (storage_manager.contains)
            │   ├── 分配所有层的内存对象
            │   ├── key.split_layers(num_layers) 按层拆分 key
            │   └── 创建 gpu_connector.batched_from_gpu() 生成器
            │
            └── 每次 next():
                ├── GPUConnector: multi_layer_kv_transfer(D2H) 该层
                └── StorageManager: batched_put(keys[layer], objs[layer])
```

#### 6.3.4 逐层 Retrieve（检索）

```
start_load_kv()
    │
    ├── 创建 layerwise_retriever 生成器
    │   └── lmcache_engine.retrieve_layer(tokens, kvcaches, slot_mapping)
    │
    ├── 预取前 2 层: next(retriever) x 2
    │
    └── 存储所有 retriever 到 self.layerwise_retrievers

wait_for_layer_load(layer_name) 每层调用
    │
    └── next(layerwise_retriever) 推进一层
        │
        └── 生成器内部:
            ├── StorageManager: layerwise_batched_get() 每层 future
            ├── 等待 future 完成
            └── GPUConnector: multi_layer_kv_transfer(H2D) 该层
```

### 6.4 多进程传输（Path B）

#### 6.4.1 架构概述

```
┌──────────────────┐     ZeroMQ      ┌──────────────────┐
│   vLLM Worker    │ ◄──────────────► │  LMCache Server  │
│                  │   消息队列        │                  │
│  ┌────────────┐  │                  │  ┌────────────┐  │
│  │ KV Buffer  │  │   CUDA IPC      │  │ LMCache    │  │
│  │ (GPU)      │◄─┼─────────────────┼─►│ Engine     │  │
│  └────────────┘  │   句柄           │  └────────────┘  │
│                  │                  │                  │
│  ┌────────────┐  │                  │  ┌────────────┐  │
│  │ Transfer   │  │                  │  │ Storage    │  │
│  │ Context    │  │                  │  │ Backends   │  │
│  └────────────┘  │                  │  └────────────┘  │
└──────────────────┘                  └──────────────────┘
```

#### 6.4.2 关键流程

**注册 KV Cache：**
1. vLLM Worker 将 KV 缓存张量包装为 `CudaIPCWrapper`
2. 通过 ZeroMQ 发送 `REGISTER_KV_CACHE` 消息到 LMCache Server
3. Server 通过 CUDA IPC 句柄直接访问 vLLM 的 GPU 内存

**存储（Store）：**
1. vLLM Scheduler 通过 `build_connector_meta()` 生成 STORE 元数据
2. Worker 记录 CUDA event（`interprocess=True`）
3. 发送 `STORE` 消息：`[key, instance_id, block_ids, event.ipc_handle()]`
4. Server 等待 CUDA event 完成，然后从 vLLM 的 GPU 内存读取数据
5. Server 调用 `multi_layer_block_kv_transfer(D2H)` 聚集到 CPU 内存
6. Server 写入存储后端

**检索（Retrieve）：**
1. Scheduler 通过 `get_num_new_matched_tokens()` 发送 LOOKUP 消息
2. Worker 记录 CUDA event
3. 发送 `RETRIEVE` 消息：`[key, instance_id, block_ids, event.ipc_handle()]`
4. Server 从存储后端读取数据到 CPU 内存
5. Server 调用 `multi_layer_block_kv_transfer(H2D)` 散射到 vLLM 的 GPU 内存
6. Server 通过 CUDA event 通知 vLLM 传输完成

#### 6.4.3 请求状态机

```
PREFETCHING ──── update_state_after_alloc ────► WAITING_FOR_LOAD
                                                      │
                                         process_loading_requests
                                                      │
                                                      ▼
                                                    READY
```

---

## 7. LMCache 到 Mooncake 的存储过程

### 7.1 Mooncake 集成架构

Mooncake 是一个分布式 KV 存储系统，LMCache 通过两种路径集成：

1. **Python 连接器**：`mooncakestore_connector.py`，包装 `mooncake.store.MooncakeDistributedStore`
2. **C++ 连接器**：`csrc/storage_backends/mooncake/`，直接使用 `mooncake::RealClient`

### 7.2 Python 连接器详解

**文件**：`lmcache/v1/storage_backend/connector/mooncakestore_connector.py`

#### 7.2.1 配置

```python
class MooncakeStoreConfig:
    # 来自 MOONCAKE_CONFIG_PATH 环境变量（JSON 文件）或 LMCacheEngineConfig.extra_config
    local_hostname: str         # 本机主机名
    metadata_server: str        # 元数据服务器地址
    global_segment_size: int    # 全局段大小（默认 3.125GB）
    local_buffer_size: int      # 本地缓冲区大小（默认 1GB）
    protocol: str               # 传输协议（默认 "tcp"）
    device_name: str            # 网络设备名
    master_server_address: str  # 主服务器地址
    # LMCache 专用
    transfer_timeout: float     # 传输超时
    prefer_local_alloc: bool    # 优先本地分配
```

#### 7.2.2 初始化

```python
class MooncakestoreConnector:
    async def initialize(self):
        # 1. NUMA 绑定
        numa_id = detect_numa_node_for_gpu()
        mooncake.store.bind_to_numa_node(numa_id)

        # 2. 设置 Mooncake Store
        store.setup(local_hostname, metadata_server, global_segment_size, ...)

        # 3. 注册 CPU 缓冲区（启用零拷贝 RDMA）
        store.register_buffer(buffer_ptr, buffer_size)

        # 4. 创建副本配置
        replica_config = ReplicateConfig(replica_num=1)
        if prefer_local_alloc:
            replica_config.preferred_segment = local_hostname
```

#### 7.2.3 零拷贝模式 (`save_chunk_meta=False`)

```
Put 流程:
┌─────────────────────────────────────────────────────────────────┐
│ store.put_from(key_str, buffer_ptr, buffer_size, replica_config)│
│   └── 直接传递原始指针，无数据拷贝                              │
│       Mooncake 通过 RDMA 从钉扎内存直接读取                     │
└─────────────────────────────────────────────────────────────────┘

批量 Put 流程:
┌─────────────────────────────────────────────────────────────────┐
│ store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes, ...)  │
│   └── 单次 RPC 传输多个 chunk，减少网络往返                     │
└─────────────────────────────────────────────────────────────────┘

Get 流程:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 预分配 CPU MemoryObj（从 LocalCPUBackend 的钉扎内存）        │
│ 2. store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)    │
│    └── Mooncake 通过 RDMA 直接写入预分配的缓冲区（零拷贝）      │
└─────────────────────────────────────────────────────────────────┘
```

#### 7.2.4 带元数据模式 (`save_chunk_meta=True`)

```
Put 流程:
┌─────────────────────────────────────────────────────────────────┐
│ store.put_parts(key_str, metadata_bytes, kv_bytes)              │
│   └── 发送元数据头 + 原始 KV 字节（两部分）                     │
└─────────────────────────────────────────────────────────────────┘

Get 流程:
┌─────────────────────────────────────────────────────────────────┐
│ 1. store.batch_get_buffer(key_strs) 返回字节缓冲区              │
│ 2. 反序列化 RemoteMetadata 头                                   │
│ 3. 拷贝数据到 MemoryObj                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 C++ 连接器详解

**目录**：`csrc/storage_backends/mooncake/`

C++ 连接器使用 `mooncake::RealClient` 直接操作，性能更高：

```cpp
class MooncakeConnector : public ConnectorBase<WorkerMooncakeConn> {
    // 构造函数
    MooncakeConnector(config) {
        client = RealClient::create();
        setup_internal(config);
        // 可选：预注册 L1 内存区域用于 RDMA
        if (l1_config) preregister_l1_memory(l1_config);
        start_worker_threads();
    }

    // 零拷贝读取
    void do_single_get(key, buf, len) {
        client->get_into(key, buf, len);  // 直接写入预注册缓冲区
    }

    // 零拷贝写入
    void do_single_set(key, buf, len) {
        client->put_from(key, buf, len);  // 直接从预注册缓冲区读取
    }

    // 批量读取（单次 RPC）
    void do_batch_get(keys, buf_ptrs, buf_lens) {
        client->batch_get_into(keys, buf_ptrs, buf_lens);
    }

    // 批量写入（单次 RPC）
    void do_batch_set(keys, buf_ptrs, buf_lens) {
        client->batch_put_from(keys, buf_ptrs, buf_lens);
    }

    // 预注册内存区域
    void preregister_l1_memory(config) {
        // 按 max_mr_size 分段注册
        for (segment : memory_segments) {
            client->register_buffer(segment.ptr, segment.size);
        }
    }
};
```

### 7.4 NUMA 感知优化

Mooncake 连接器在初始化时检测当前 GPU 的 NUMA 节点：

```python
numa_id = detect_numa_node_for_gpu()  # 通过 PCI 总线 ID 推断
mooncake.store.bind_to_numa_node(numa_id)
```

这确保 RDMA 操作从最近的 NUMA 节点分配内存，减少跨 NUMA 访问延迟。

### 7.5 完整的 LMCache → Mooncake 数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│              LMCache → Mooncake 存储流程                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LMCacheEngine.store()                                               │
│       │                                                              │
│       ▼                                                              │
│  StorageManager.batched_put(keys, memory_objs)                       │
│       │                                                              │
│       ├── LocalCPUBackend: 热缓存（零拷贝插入）                       │
│       │                                                              │
│       └── RemoteBackend (mooncakestore://):                          │
│            │                                                         │
│            ├── 序列化 (naive 模式: 直通)                              │
│            │                                                         │
│            └── connector.put(key, memory_obj)                        │
│                 │                                                    │
│                 ├── 零拷贝模式:                                       │
│                 │   store.put_from(key, buffer_ptr, buffer_size)     │
│                 │   └── Mooncake RDMA 从钉扎内存直接读取              │
│                 │                                                    │
│                 └── 批量模式:                                         │
│                     store.batch_put_from(keys, ptrs, sizes)          │
│                     └── 单次 RPC 传输多个 chunk                      │
│                                                                      │
│  Mooncake 存储层:                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  全局段 (Global Segment)                                       │ │
│  │  ├── 段 0 (本机 NUMA 节点 0)                                  │ │
│  │  │   ├── key_0: [2, NL, T, H] 连续字节                        │ │
│  │  │   ├── key_1: [2, NL, T, H] 连续字节                        │ │
│  │  │   └── ...                                                   │ │
│  │  ├── 段 1 (本机 NUMA 节点 1)                                  │ │
│  │  └── 段 N (远程节点)                                           │ │
│  │  副本策略: replica_num=1, prefer_local_alloc=true              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. 完整端到端流程示例

### 8.1 场景设定

- 模型：Llama-7B，32 层，32 heads，head_dim=128
- 推理引擎：vLLM（FlashAttention，NHD 布局）
- 存储：CPU 热缓存 + Mooncake 远程存储
- chunk_size=256，block_size=16
- 使用 layerwise 模式

### 8.2 第一次 Prefill（无缓存命中）

```
用户输入: "The quick brown fox jumps over the lazy dog..." (1024 tokens)

┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: vLLM Scheduler 调度                                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ vLLM → LMCache Connector:                                           │
│   get_num_new_matched_tokens(request, num_computed_tokens=0)         │
│     └── lookup: 1024 tokens, chunk_size=256                         │
│         ├── chunk 0: hash(tokens[0:256])                             │
│         ├── chunk 1: hash(hash_0, tokens[256:512])                   │
│         ├── chunk 2: hash(hash_1, tokens[512:768])                   │
│         └── chunk 3: hash(hash_2, tokens[768:1024])                  │
│         结果: 0 tokens 缓存命中                                      │
│     └── 返回: num_new_matched = 0                                   │
│                                                                      │
│ vLLM 分配 1024/16 = 64 个 KV blocks                                  │
│                                                                      │
│ build_connector_meta() → ReqMeta:                                    │
│   slot_mapping = [0,1,2,...,15, 16,17,...,31, ..., 1008,...,1023]    │
│   存储方向: STORE (新请求)                                           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: vLLM Worker Forward Pass + 逐层存储                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ register_kv_caches(kv_caches):                                       │
│   kv_caches = [(K_layer0, V_layer0), ..., (K_layer31, V_layer31)]   │
│   每层形状: [num_blocks, 16, 32, 128]                               │
│                                                                      │
│ ── 逐层 Forward + Save ──                                            │
│                                                                      │
│ Layer 0:                                                             │
│   forward: 计算 layer 0 的 K, V                                      │
│   save_kv_layer("layer_0"):                                          │
│     ├── 首次调用，创建 layerwise_storer 生成器                        │
│     ├── store_layer() 内部:                                          │
│     │   ├── 分配 memory_obj: [2, 32, 1024, 4096] 钉扎 CPU 内存      │
│     │   ├── key_0.split_layers(32) → 32 个 LayerCacheEngineKey       │
│     │   └── 创建 batched_from_gpu 生成器                             │
│     └── next(storer):                                                │
│         ├── GPUConnector: multi_layer_kv_transfer(D2H, layer=0)      │
│         │   ├── Grid: (1024, 1, 2)                                   │
│         │   ├── 每个线程块处理一个 token 的 K 或 V                   │
│         │   ├── 读取 slot_mapping[t] → 物理 slot                    │
│         │   └── 从 vLLM 分页缓冲区聚集到连续 CPU 张量               │
│         └── StorageManager: batched_put(key_layer_0, obj_layer_0)    │
│             ├── LocalCPUBackend: 插入 hot_cache                      │
│             └── RemoteBackend → Mooncake:                            │
│                 store.put_from(key, buffer_ptr, buffer_size)         │
│                                                                      │
│ Layer 1:                                                             │
│   forward: 计算 layer 1 的 K, V                                      │
│   save_kv_layer("layer_1"):                                          │
│     └── next(storer):                                                │
│         ├── multi_layer_kv_transfer(D2H, layer=1)                    │
│         └── batched_put(key_layer_1, obj_layer_1)                    │
│                                                                      │
│ ... (Layer 2-30 类似) ...                                            │
│                                                                      │
│ Layer 31:                                                            │
│   forward: 计算 layer 31 的 K, V                                     │
│   save_kv_layer("layer_31"):                                         │
│     └── next(storer):                                                │
│         ├── multi_layer_kv_transfer(D2H, layer=31)                   │
│         └── batched_put(key_layer_31, obj_layer_31)                  │
│                                                                      │
│ wait_for_save():                                                     │
│     └── next(storer) [最终 yield，清理资源]                          │
│                                                                      │
│ ── 存储完成 ──                                                       │
│ 结果: 1024 tokens 的 KV Cache 已存储到 CPU + Mooncake               │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.3 第二次请求（缓存命中 + 新 token）

```
用户输入: "The quick brown fox jumps over the lazy dog... [新增 512 tokens]"

┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: vLLM Scheduler 调度                                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ get_num_new_matched_tokens(request, num_computed_tokens=0):          │
│   ├── chunk 0-3: 与第一次请求完全相同 → 缓存命中！                   │
│   ├── chunk 4: hash(hash_3, tokens[1024:1280]) → 未命中              │
│   └── chunk 5: hash(hash_4, tokens[1280:1536]) → 未命中              │
│   结果: 1024 tokens 缓存命中 (4 个完整 chunk)                        │
│   返回: num_new_matched = 1024                                       │
│                                                                      │
│ vLLM 分配 1536/16 = 96 个 KV blocks                                  │
│   ├── 64 blocks 用于缓存的 1024 tokens                               │
│   └── 32 blocks 用于新计算的 512 tokens                              │
│                                                                      │
│ build_connector_meta():                                              │
│   ├── RETRIEVE: tokens[0:1024], slot_mapping[0:64]                   │
│   └── STORE: tokens[1024:1536], slot_mapping[64:96]                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: vLLM Worker Forward Pass + 逐层检索/存储                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ start_load_kv():                                                     │
│   ├── retrieve_layer(tokens[0:1024], ...) → 生成器                   │
│   ├── next(retriever) x 2 → 预取 layer 0, 1                         │
│   │   └── 从 Mooncake 获取: store.batch_get_into(keys, ptrs, sizes)  │
│   │       └── RDMA 直接写入预分配的钉扎内存                          │
│   └── 存储 retriever                                                │
│                                                                      │
│ ── 逐层 Forward ──                                                   │
│                                                                      │
│ Layer 0:                                                             │
│   wait_for_layer_load("layer_0"):                                    │
│     └── next(retriever):                                             │
│         ├── StorageManager: 获取 layer 0 的 memory_obj               │
│         ├── GPUConnector: multi_layer_kv_transfer(H2D, layer=0)      │
│         │   ├── Grid: (1024, 1, 2)                                   │
│         │   ├── skip_prefix_n_tokens = 0                             │
│         │   └── 从连续 CPU 张量散射到 vLLM 分页缓冲区               │
│         └── ret_mask[0:1024] = True                                  │
│                                                                      │
│   forward: 计算 layer 0（使用已恢复的 KV Cache）                     │
│   save_kv_layer("layer_0"):                                          │
│     └── next(storer): 保存新 token 的 layer 0 KV                    │
│                                                                      │
│ Layer 1-31: 类似处理                                                 │
│                                                                      │
│ ── 结果 ──                                                           │
│ 前 1024 tokens: 从缓存恢复，跳过 prefill 计算                       │
│ 后 512 tokens: 正常 prefill 计算 + 存储到缓存                       │
│ TTFT 大幅降低！                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.4 多进程模式的差异

```
┌──────────────────────────────────────────────────────────────────────┐
│ 多进程模式的关键差异                                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ 1. 注册阶段:                                                         │
│    vLLM Worker → LMCache Server (via ZMQ):                           │
│      REGISTER_KV_CACHE [CUDA IPC handles for all layers]             │
│    Server 通过 IPC 句柄直接访问 vLLM 的 GPU 内存                     │
│                                                                      │
│ 2. 存储:                                                             │
│    vLLM Worker → LMCache Server (via ZMQ):                           │
│      STORE [key, block_ids, event.ipc_handle()]                      │
│    Server:                                                           │
│      ├── 等待 CUDA event 完成                                        │
│      ├── multi_layer_block_kv_transfer(D2H) 从 vLLM GPU→CPU         │
│      └── batched_put() 写入存储后端                                  │
│    └── vLLM 不等待，继续 forward pass                                │
│                                                                      │
│ 3. 检索:                                                             │
│    vLLM Worker → LMCache Server (via ZMQ):                           │
│      RETRIEVE [key, block_ids, event.ipc_handle()]                   │
│    Server:                                                           │
│      ├── batched_get() 从存储后端读取                                │
│      ├── multi_layer_block_kv_transfer(H2D) CPU→vLLM GPU            │
│      └── 记录 CUDA event 完成                                        │
│    └── vLLM 在需要时等待 event                                       │
│                                                                      │
│ 优势: vLLM 和 LMCache 完全解耦，可独立扩展                           │
│ 劣势: 不支持逐层流水线，延迟略高                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键优化技术总结

### 9.1 编译期模板消除运行时分支

CUDA 内核使用编译期模板参数处理**传输方向**（H2D/D2H）和**内存格式**（10 种 GPUKVFormat），编译器为每种组合生成特化代码，消除所有 `if/switch` 分支。

### 9.2 向量化内存访问

类型分派逻辑根据元素大小选择 `int64_t`/`int32_t`/`int16_t`/`int8_t`，最大化向量化 load/store 带宽。块级内核使用 `uint4` 的 128 位 cache-streaming 操作。

### 9.3 钉扎内存 + 零拷贝传输

CPU 内存使用 `cudaHostAlloc` 或 `cudaHostRegister` 钉扎，避免 CUDA 驱动的隐式拷贝。Mooncake 通过 `register_buffer()` 注册钉扎内存区域，实现 RDMA 零拷贝传输。

### 9.4 NUMA 感知分配

内存分配器支持 NUMA 感知模式（`alloc_pinned_numa_ptr`），确保内存分配在 GPU 所在的 NUMA 节点上，减少跨 NUMA 访问延迟。

### 9.5 大页内存支持

支持 2MB 大页内存（`alloc_hugepage_pinned_ptr`），通过 `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` 分配，减少 TLB 缺失。

### 9.6 Ping-Pong 双缓冲

逐层传输使用双缓冲（`compute_gpu_buffer_obj` / `load_gpu_buffer_obj`），重叠当前层的计算和下一层的加载。

### 9.7 前缀哈希避免重复存储

`ChunkedTokenDatabase` 使用前缀哈希链，确保相同的 token 前缀产生相同的 key，避免重复存储。支持前缀缓存（prefix caching）场景。

### 9.8 块级批量传输

多进程路径使用 `multi_layer_block_kv_transfer` 内核，一次处理整个块（而非单个 token），减少内核启动开销。Mooncake 支持 `batch_put_from` / `batch_get_into` 单次 RPC 传输多个 chunk。

### 9.9 CUDA IPC 跨进程共享

多进程路径通过 CUDA IPC 句柄实现跨进程 GPU 内存共享，避免 GPU→CPU→GPU 的冗余拷贝。

### 9.10 异步 I/O 与事件驱动

存储后端的 put/get 操作完全异步，通过事件循环（asyncio）和 CUDA stream 管理并发。磁盘后端使用线程池（max 4 workers），远程后端使用 async/await。

### 9.11 多级缓存自动回写

从磁盘或远程后端读取的数据自动回写到 CPU 热缓存，加速后续访问。StorageManager 按 CPU → 磁盘 → 远程的顺序搜索。

### 9.12 智能缓存淘汰

支持 4 种淘汰策略（LRU、LFU、FIFO、MRU），所有策略都尊重引用计数和 Pin 状态：只有 `ref_count == 0` 且未被 Pin 的条目才可被淘汰。

### 9.13 层组异构支持

`KVLayerGroupsManager` 将 KV 层分为传输内核调度单元，支持不同层有不同 KV 缓存几何形状（如 DeepSeek V4 的压缩 + 密集组）。每个组使用独立的 `PageBufferShapeDesc` 和内核启动。

### 9.14 MLA 优化

对于 Multi-head Latent Attention（MLA）模型，KV 维度从 2 压缩为 1，`save_only_first_rank` 确保只有 rank 0 写入远程存储，减少存储和网络开销。

### 9.15 SYCL/XPU 移植优化

Intel XPU 移植版本使用：
- 工作组大小 256（适配 Intel EU 调度）
- 子组大小锁定为 16（PVC/DG2/BMG 原生）
- 融合 K+V 内核（减少一半调度开销）
- 循环不变量提升（避免重复整数除法）

---

## 附录：关键配置参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `chunk_size` | 256 | 每个缓存块的 token 数 |
| `max_local_cpu_size` | 5.0 GB | 钉扎 CPU 内存池大小 |
| `local_cpu` | True | 启用 CPU 热缓存 |
| `local_disk` | None | 磁盘后端路径 |
| `remote_url` | None | 远程后端 URL（如 `mooncakestore://`） |
| `use_layerwise` | False | 逐层 store/retrieve |
| `save_unfull_chunk` | False | 缓存不完整的尾部 chunk |
| `enable_blending` | False | 使用 SegmentTokenDatabase |
| `enable_pd` | False | 启用 Prefill-Decode 分离 |
| `enable_async_loading` | False | 异步预取内存对象 |
| `cache_policy` | "LRU" | 淘汰策略 |
| `remote_serde` | "naive" | 远程序列化方式 |
| `enable_kv_events` | False | 发送缓存事件 |

---

> 文档生成时间：2026-05-30
> 基于 LMCache `dev` 分支分析
