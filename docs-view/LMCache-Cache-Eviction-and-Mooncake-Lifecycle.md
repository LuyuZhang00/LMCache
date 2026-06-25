# LMCache 缓存淘汰策略与 Mooncake 后端生命周期详解

> 本文档详细分析 LMCache 的多级缓存淘汰机制、KV Cache 在各层级的生命周期、以及 Mooncake 后端的存储与淘汰策略。
>
> 版本基准：LMCache `dev` 分支

---

## 目录

1. [核心问题概述](#1-核心问题概述)
2. [缓存淘汰的前置条件：can_evict 守卫](#2-缓存淘汰的前置条件can_evict-守卫)
3. [四种缓存淘汰策略详解](#3-四种缓存淘汰策略详解)
4. [CPU 热缓存的淘汰机制](#4-cpu-热缓存的淘汰机制)
5. [KV Cache 的生命周期：从 GPU 到 Mooncake](#5-kv-cache-的生命周期从-gpu-到-mooncake)
6. [Mooncake 后端的存储与淘汰](#6-mooncake-后端的存储与淘汰)
7. [多级缓存的交互与独立性](#7-多级缓存的交互与独立性)
8. [完整流程图与数值示例](#8-完整流程图与数值示例)

---

## 1. 核心问题概述

LMCache 的缓存管理涉及三个核心问题：

```
问题 1：KV Cache 通过 multi_layer_kv_transfer 从 GPU 传到 CPU 后，
       它在 LMCache Engine 侧是如何管理的？什么时候会被淘汰？

问题 2：KV Cache 什么时候会被写入 Mooncake 后端？
       是立即写入还是延迟写入？

问题 3：Mooncake 后端有自己的淘汰策略吗？
       LMCache 能控制 Mooncake 的淘汰吗？
```

**一句话回答**：

1. KV Cache 存入 CPU 热缓存（`hot_cache`），由 LRU/LFU 等策略在**内存不足时按需淘汰**
2. KV Cache 在 `batched_put()` 时**同时写入** CPU 热缓存和 Mooncake，没有延迟
3. Mooncake 有**自己独立的淘汰机制**，LMCache **无法控制**也**不感知** Mooncake 的淘汰

---

## 2. 缓存淘汰的前置条件：can_evict 守卫

### 2.1 MemoryObj 的两个引用计数

每个 `MemoryObj` 有两个计数器控制其生命周期：

```python
# memory_management.py 第 772-777 行
@property
def can_evict(self) -> bool:
    return not self.is_pinned and self.get_ref_count() == 1
```

| 计数器 | 含义 | 初始值 | 增加时机 | 减少时机 |
|--------|------|--------|---------|---------|
| `ref_count` | 引用计数 | 1 | hot_cache 存储时 +1，消费者读取时 +1 | 消费者用完时 -1，hot_cache 移除时 -1 |
| `pin_count` | 钉扎计数 | 0 | vLLM lookup 时 pin（防止淘汰） | 请求完成后 unpin |

### 2.2 can_evict 的含义

```
can_evict = True 当且仅当：
  ① pin_count == 0（没有活跃请求在使用）
  ② ref_count == 1（只有 hot_cache 自己持有引用）
```

**ref_count 的生命周期**：

```
allocate() 分配 MemoryObj
  └── ref_count = 1

submit_put_task() 存入 hot_cache
  └── ref_count_up() → ref_count = 2

get_blocking() 消费者读取
  └── ref_count_up() → ref_count = 3

消费者用完后 ref_count_down()
  └── ref_count = 2

此时 can_evict = (ref_count == 1) = False
因为 hot_cache 还持有引用

hot_cache 淘汰时 batched_remove()
  └── hot_cache.pop(key)
  └── ref_count_down() → ref_count = 1
  └── can_evict = True → MemoryObj 被回收
```

### 2.3 什么情况下 cannot_evict

| 场景 | ref_count | pin_count | can_evict | 原因 |
|------|-----------|-----------|-----------|------|
| 刚分配，尚未存入 hot_cache | 1 | 0 | True | 但此时还没有被缓存 |
| 存入 hot_cache，无消费者 | 2 | 0 | **False** | hot_cache 持有引用 |
| 存入 hot_cache，有消费者读取中 | 3 | 0 | **False** | 消费者持有引用 |
| 存入 hot_cache，被 vLLM pin | 2 | 1 | **False** | pin_count>0 |
| 消费者用完，未被 pin | 2 | 0 | **False** | hot_cache 仍持有引用 |

---

## 3. 四种缓存淘汰策略详解

### 3.1 策略接口

```python
# cache_policy/base_policy.py
class BaseCachePolicy:
    def init_mutable_mapping(): ...        # 创建底层数据结构
    def update_on_hit(key, cache_dict): ...# 缓存命中时调用
    def update_on_put(key): ...            # 缓存写入时调用
    def update_on_force_evict(key): ...    # 强制淘汰时调用
    def get_evict_candidates(cache_dict, num_candidates): ...  # 返回淘汰候选
```

### 3.2 LRU（Least Recently Used，默认策略）

```python
# cache_policy/lru.py
class LRUCachePolicy:
    def __init__(self):
        self.cache_dict = OrderedDict()  # 有序字典

    def update_on_hit(self, key, cache_dict):
        cache_dict.move_to_end(key)  # 命中时移到末尾（最近使用）

    def update_on_put(self, key):
        pass  # 新 key 自动在末尾（OrderedDict 插入顺序）

    def get_evict_candidates(self, cache_dict, num_candidates):
        candidates = []
        for key in cache_dict:           # 从头部（最久未使用）遍历
            if cache_dict[key].can_evict:
                candidates.append(key)
            if len(candidates) >= num_candidates:
                break
        return candidates
```

**LRU 的淘汰顺序**：
```
OrderedDict: [A(最久) → B → C → D → E(最近)]
淘汰顺序: A 先淘汰，然后 B，然后 C...
命中 D 后: [A → B → C → E → D]（D 移到末尾）
```

**额外功能**：LRU 还跟踪 `chunk_hash_to_init_timestamp`，记录每个 chunk 首次出现的时间，用于统计复用间隔。当字典达到 1250 万条时自动清理。

### 3.3 LFU（Least Frequently Used）

```python
# cache_policy/lfu.py
class LFUCachePolicy:
    def __init__(self):
        self.freq_to_keys = SortedDict()  # 频率 → {key: None}
        self.key_to_freq = {}             # key → 频率

    def update_on_hit(self, key, cache_dict):
        old_freq = self.key_to_freq[key]
        new_freq = old_freq + 1
        # 从旧频率桶移除
        del self.freq_to_keys[old_freq][key]
        if not self.freq_to_keys[old_freq]:
            del self.freq_to_keys[old_freq]
        # 插入新频率桶
        if new_freq not in self.freq_to_keys:
            self.freq_to_keys[new_freq] = {}
        self.freq_to_keys[new_freq][key] = None
        self.key_to_freq[key] = new_freq

    def update_on_put(self, key):
        self.key_to_freq[key] = 1
        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = {}
        self.freq_to_keys[1][key] = None

    def get_evict_candidates(self, cache_dict, num_candidates):
        candidates = []
        for freq in self.freq_to_keys:  # 从最低频率开始
            for key in self.freq_to_keys[freq]:
                if cache_dict[key].can_evict:
                    candidates.append(key)
                if len(candidates) >= num_candidates:
                    return candidates
        return candidates
```

**LFU 的淘汰顺序**：
```
频率 1: {A, B}     ← 最先淘汰
频率 2: {C}
频率 5: {D, E}     ← 最后淘汰
```

### 3.4 FIFO（First In First Out）

```python
# cache_policy/fifo.py
class FIFOCachePolicy:
    def __init__(self):
        self.cache_dict = dict()  # Python 3.7+ 保持插入顺序

    def update_on_hit(self, key, cache_dict):
        pass  # 命中不改变顺序

    def update_on_put(self, key):
        pass  # 新 key 在末尾

    def get_evict_candidates(self, cache_dict, num_candidates):
        candidates = []
        for key in cache_dict:  # 从头部（最早插入）遍历
            if cache_dict[key].can_evict:
                candidates.append(key)
            if len(candidates) >= num_candidates:
                break
        return candidates
```

### 3.5 MRU（Most Recently Used）

```python
# cache_policy/mru.py
class MRUCachePolicy:
    def __init__(self):
        self.cache_dict = OrderedDict()

    def update_on_hit(self, key, cache_dict):
        cache_dict.move_to_end(key)  # 命中时移到末尾

    def get_evict_candidates(self, cache_dict, num_candidates):
        candidates = []
        for key in reversed(cache_dict):  # 从尾部（最近使用）遍历！
            if cache_dict[key].can_evict:
                candidates.append(key)
            if len(candidates) >= num_candidates:
                break
        return candidates
```

**MRU 与 LRU 的唯一区别**：`get_evict_candidates` 从尾部开始遍历，淘汰最近使用的条目。适用于扫描型访问模式（一次扫描后不再访问）。

### 3.6 策略选择

```python
# cache_policy/__init__.py
POLICY_MAPPING = {
    "LRU": LRUCachePolicy,
    "LFU": LFUCachePolicy,
    "FIFO": FIFOCachePolicy,
    "MRU": MRUCachePolicy,
}

def get_cache_policy(policy_name):
    return POLICY_MAPPING[policy_name.upper()]
```

配置方式：`config.cache_policy = "LRU"`（默认）

---

## 4. CPU 热缓存的淘汰机制

### 4.1 hot_cache 的结构

```python
# local_cpu_backend.py 第 60 行
self.hot_cache = self.cache_policy.init_mutable_mapping()
# 对于 LRU: OrderedDict[CacheEngineKey, MemoryObj]
```

`hot_cache` 是一个字典，key 是 `CacheEngineKey`，value 是 `MemoryObj`。它同时被缓存策略的数据结构（如 LRU 的 `OrderedDict`）和 `hot_cache` 字典引用。

### 4.2 写入 hot_cache

```python
# local_cpu_backend.py 第 147-184 行
def submit_put_task(self, key, memory_obj, **kwargs):
    if key in self.hot_cache:
        return  # 已存在，跳过

    memory_obj.ref_count_up()  # ref_count: 1 → 2
    self.hot_cache[key] = memory_obj
    self.cache_policy.update_on_put(key)  # 更新策略（LRU: 插入末尾）
```

### 4.3 从 hot_cache 读取

```python
# local_cpu_backend.py 第 208-220 行
def get_blocking(self, key, **kwargs):
    if key not in self.hot_cache:
        return None
    memory_obj = self.hot_cache[key]
    memory_obj.ref_count_up()  # ref_count: 2 → 3（消费者持有）
    return memory_obj
    # 消费者用完后调用 ref_count_down() → ref_count: 3 → 2
```

### 4.4 Pin 机制（防止活跃请求的 KV 被淘汰）

```python
# local_cpu_backend.py 第 124-132 行
def contains(self, key, pin=False):
    if key in self.hot_cache:
        if pin:
            self.hot_cache[key].pin()  # pin_count: 0 → 1
            self.keys_in_request.append(key)
        return True
    return False
```

vLLM 在 `lookup` 时会 `pin=True`，确保正在使用的 KV Cache 不会被淘汰。请求完成后调用 `touch_cache()` 更新策略并 unpin。

### 4.5 淘汰触发点：allocate() 方法

**淘汰不是后台线程触发的，而是在分配内存时按需触发。**

```python
# local_cpu_backend.py 第 577-677 行
def allocate(self, shapes, dtypes, fmt, eviction=True, busy_loop=False):
    # Step 1: 尝试直接分配
    memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
    if memory_obj is not None:
        return memory_obj

    # Step 2: 分配失败，进入淘汰循环
    if eviction:
        while True:
            # 2a: 从策略中获取淘汰候选
            evict_keys = self.cache_policy.get_evict_candidates(
                self.hot_cache, num_candidates=1
            )

            # 2b: 淘汰找到的候选
            if evict_keys:
                self.batched_remove(evict_keys, force=False)
                # remove 内部: hot_cache.pop(key), memory_obj.ref_count_down()
                # ref_count: 2 → 1 → MemoryObj 可被回收

                # 2c: 重新尝试分配
                memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
                if memory_obj is not None:
                    return memory_obj
            else:
                # 2d: 没有可淘汰的候选
                if busy_loop:
                    time.sleep(0.1)  # 等待 100ms，让进行中的操作完成
                    continue
                else:
                    return None  # 放弃分配

    return None
```

**busy_loop 参数的关键作用**：
- `busy_loop=True`（检索路径）：会持续等待，因为检索必须成功
- `busy_loop=False`（存储路径）：不会阻塞，避免并发存储死锁

### 4.6 淘汰的完整流程图

```
allocate() 被调用（需要分配新的 MemoryObj）
    │
    ▼
memory_allocator.allocate() 尝试分配
    │
    ├── 成功 → 返回 MemoryObj
    │
    └── 失败（内存不足）
        │
        ▼
    cache_policy.get_evict_candidates(hot_cache, 1)
        │
        ├── LRU: 从 OrderedDict 头部找 can_evict=True 的条目
        ├── LFU: 从最低频率桶找 can_evict=True 的条目
        ├── FIFO: 从 dict 头部找 can_evict=True 的条目
        └── MRU: 从 OrderedDict 尾部找 can_evict=True 的条目
        │
        ├── 找到候选 key
        │   │
        │   ▼
        │   batched_remove([key])
        │   ├── hot_cache.pop(key)
        │   ├── memory_obj.ref_count_down()  → ref_count: 2→1→自动回收
        │   └── cache_policy.update_on_force_evict(key)
        │   │
        │   ▼
        │   memory_allocator.allocate() 重新尝试
        │   └── 成功 → 返回 MemoryObj
        │
        └── 未找到候选（所有条目都被 pin 或正在使用）
            │
            ├── busy_loop=True → sleep(0.1) → 重试
            └── busy_loop=False → 返回 None（放弃）
```

---

## 5. KV Cache 的生命周期：从 GPU 到 Mooncake

### 5.1 完整生命周期时间线

```
时刻 T0: vLLM 完成 Prefill，KV Cache 在 GPU 分页缓冲区中
    │     形状: [num_blocks, block_size, NH, HS] 每层
    │
    ▼
时刻 T1: multi_layer_kv_transfer(D2H)
    │     CUDA 内核将 KV Cache 从 GPU 分页缓冲区聚集到 CPU 钉扎内存
    │     MemoryObj 形状: [2, NL, chunk_size, H]
    │     ref_count = 1
    │
    ▼
时刻 T2: StorageManager.batched_put() 被调用
    │     同时写入 CPU 热缓存和 Mooncake（无延迟）
    │
    ├── T2a: LocalCPUBackend.submit_put_task()
    │   ├── memory_obj.ref_count_up()  → ref_count = 2
    │   ├── hot_cache[key] = memory_obj
    │   └── cache_policy.update_on_put(key)
    │
    └── T2b: RemoteBackend.batched_submit_put_task()
        ├── serialize(memory_obj)  → compressed_bytes
        ├── mooncake.put_from(key, buffer_ptr, buffer_size)
        │   └── RDMA/TCP 零拷贝写入 Mooncake
        └── （异步完成）
    │
    ▼
时刻 T3: ref_count_down() 清理
    │     StorageManager 对所有分配的 MemoryObj 调用 ref_count_down()
    │     消费者持有的引用释放后，ref_count 回到 2（hot_cache 持有）
    │
    ▼
时刻 T4: MemoryObj 在 CPU 热缓存中"活着"
    │     ref_count = 2（hot_cache 自身的引用）
    │     can_evict = (ref_count == 1) = False
    │     数据同时存在于 CPU 热缓存和 Mooncake 中
    │
    ▼
时刻 T5: 新请求到来，需要分配新的 MemoryObj
    │     allocate() 发现内存不足
    │     调用 cache_policy.get_evict_candidates()
    │     找到 can_evict=True 的旧条目（消费者已释放，ref_count=1 的条目）
    │     或者等待当前条目的消费者释放引用
    │
    ▼
时刻 T6: 旧条目被淘汰
    │     hot_cache.pop(key)
    │     memory_obj.ref_count_down() → ref_count: 2→1→自动回收到分配器
    │     CPU 热缓存释放内存
    │     Mooncake 中的数据仍然存在（不受影响）
    │
    ▼
时刻 T7: 后续请求需要已被淘汰的 KV Cache
    │     StorageManager.get() 搜索: CPU 热缓存 → 未命中
    │     继续搜索: Mooncake → 命中！
    │     从 Mooncake 读取数据到新的 MemoryObj
    │     自动回写到 CPU 热缓存（auto write-back）
    │
    ▼
时刻 T8: 数据重新出现在 CPU 热缓存中
          ref_count = 2（hot_cache 持有）
          等待下一次淘汰周期
```

### 5.2 关键时间点分析

**T1→T2：数据同时写入 CPU 和 Mooncake**

```python
# storage_manager.py 第 383-432 行
def batched_put(self, keys, memory_objs, ...):
    # 对于每个后端：
    for backend in self.storage_backends.values():
        backend.batched_submit_put_task(keys, objs)
    # LocalCPUBackend 和 RemoteBackend 在同一个循环中被调用
    # 数据同时写入两者，没有先后之分
```

**T4→T5：淘汰是按需触发的**

没有后台淘汰线程。只有当 `allocate()` 发现内存不足时才触发淘汰。

**T6→T7：Mooncake 是"永久"存储**

CPU 热缓存淘汰不影响 Mooncake。Mooncake 中的数据在 LMCache 看来是"永久"的（除非 Mooncake 自己淘汰）。

**T7→T8：自动回写机制**

```python
# storage_manager.py 第 449-456 行
def get(self, key, ...):
    for backend in self.storage_backends.values():
        memory_obj = backend.get_blocking(key)
        if memory_obj is not None:
            # 自动回写到 CPU 热缓存
            if not isinstance(backend, LocalCPUBackend):
                self.local_cpu_backend.submit_put_task(key, memory_obj)
            return memory_obj
```

---

## 6. Mooncake 后端的存储与淘汰

### 6.1 Mooncake 在 LMCache 中的位置

```
LMCache Engine
    │
    ▼
StorageManager
    ├── LocalCPUBackend（CPU 热缓存，LMCache 管理淘汰）
    └── RemoteBackend（包装 Mooncake Connector）
            │
            ▼
        MooncakestoreConnector
            │
            ▼
        MooncakeDistributedStore（外部服务，独立管理淘汰）
```

### 6.2 Mooncake 的写入方式

**零拷贝模式（`save_chunk_meta=False`，推荐）**：

```python
# mooncakestore_connector.py 第 680-720 行
def put(self, key, memory_obj):
    key_str = key.to_string()
    buffer_ptr = memory_obj.data_ptr
    buffer_size = memory_obj.get_size()
    self.store.put_from(key_str, buffer_ptr, buffer_size, self.replica_config)
    # put_from 直接从注册的 CPU 缓冲区读取，零拷贝
```

**带元数据模式（`save_chunk_meta=True`）**：

```python
def put(self, key, memory_obj):
    key_str = key.to_string()
    metadata_bytes = self._serialize_metadata(memory_obj)
    kv_bytes = memory_obj.raw_data.numpy().tobytes()
    self.store.put_parts(key_str, metadata_bytes, kv_bytes)
    # put_parts 发送元数据头 + KV 数据
```

**批量写入**：

```python
def batched_put(self, keys, memory_objs):
    key_strs = [key.to_string() for key in keys]
    buffer_ptrs = [obj.data_ptr for obj in memory_objs]
    buffer_sizes = [obj.get_size() for obj in memory_objs]
    self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes, self.replica_config)
    # 单次 RPC 传输多个 chunk
```

### 6.3 Mooncake 的内存配置

```python
# mooncakestore_connector.py 第 325-439 行
class MooncakeStoreConfig:
    global_segment_size: int = 3.125 * 1024 * 1024 * 1024  # 3.125 GB
    local_buffer_size: int = 1 * 1024 * 1024 * 1024        # 1 GB
    protocol: str = "tcp"
    device_name: str = ""
    prefer_local_alloc: bool = True
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `global_segment_size` | Mooncake 全局段大小 | 3.125 GB |
| `local_buffer_size` | 本地缓冲区大小 | 1 GB |
| `protocol` | 传输协议 | tcp |
| `prefer_local_alloc` | 优先本地分配 | True |

### 6.4 Mooncake 的淘汰策略

**关键结论：LMCache 无法控制 Mooncake 的淘汰策略。**

Mooncake 是一个独立的分布式 KV 存储系统，有自己的内存管理和淘汰机制。LMCache 只是 Mooncake 的一个客户端，通过 `put_from`/`batch_put_from` 写入数据，通过 `batch_get_into`/`batch_get_buffer` 读取数据。

```
┌─────────────────────────────────────────────────────────┐
│ LMCache 视角                                            │
│                                                         │
│  put_from(key, ptr, size)  →  Mooncake 存储了数据       │
│  batch_get_into(keys, ptrs) → Mooncake 返回了数据       │
│  LMCache 不知道 Mooncake 内部如何管理内存                │
│  LMCache 不知道 Mooncake 何时淘汰数据                    │
│  LMCache 无法告诉 Mooncake "不要淘汰这个 key"            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Mooncake 视角                                           │
│                                                         │
│  收到 put_from 请求 → 存储到全局段                       │
│  收到 batch_get_into 请求 → 从全局段读取                 │
│  全局段满了 → Mooncake 自己决定淘汰哪些 key              │
│  淘汰策略由 Mooncake 运行时控制，LMCache 无法干预        │
└─────────────────────────────────────────────────────────┘
```

### 6.5 Mooncake 的 ReplicateConfig

```python
# mooncakestore_connector.py 第 429-435 行
self.replica_config = ReplicateConfig()
self.replica_config.replica_num = 1
if self.config.prefer_local_alloc:
    self.replica_config.preferred_segment = self.store.get_hostname()
```

| 参数 | 含义 |
|------|------|
| `replica_num = 1` | 每个 key 只存一份（无副本） |
| `preferred_segment` | 优先存储在本机的内存段 |

### 6.6 Mooncake 的 NUMA 绑定

```python
# mooncakestore_connector.py 第 400-410 行
numa_id = detect_numa_node_for_gpu()
if numa_id >= 0:
    mooncake.store.bind_to_numa_node(numa_id)
```

Mooncake 在初始化时绑定到 GPU 所在的 NUMA 节点，确保 RDMA 传输的内存访问本地化。

### 6.7 Mooncake 的缓冲区注册

```python
# mooncakestore_connector.py 第 437 行
self.store.register_buffer(buffer.data_ptr(), buffer.numel())
```

LMCache 将 CPU 钉扎内存的缓冲区注册到 Mooncake，使 Mooncake 可以通过 RDMA 直接访问这块内存，实现零拷贝传输。

---

## 7. 多级缓存的交互与独立性

### 7.1 各层级的独立性

```
┌─────────────────────────────────────────────────────────────┐
│                    缓存层级独立性                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CPU 热缓存（LMCache 管理）                                  │
│  ├── 淘汰策略：LRU/LFU/FIFO/MRU（可配置）                    │
│  ├── 淘汰触发：allocate() 内存不足时按需触发                  │
│  ├── 淘汰粒度：单个 chunk（256 tokens）                      │
│  └── 淘汰后：数据可能仍在 Mooncake 中                        │
│                                                             │
│  Mooncake（独立管理）                                        │
│  ├── 淘汰策略：Mooncake 运行时决定（LMCache 不感知）          │
│  ├── 淘汰触发：Mooncake 全局段满时                           │
│  ├── 淘汰粒度：由 Mooncake 决定                              │
│  └── 淘汰后：LMCache 不会被通知                              │
│                                                             │
│  两层之间没有协调协议                                        │
│  CPU 淘汰不会触发 Mooncake 淘汰                              │
│  Mooncake 淘汰不会触发 CPU 淘汰                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 数据流向图

```
GPU KV Cache
    │
    ▼ multi_layer_kv_transfer(D2H)
    │
CPU 钉扎内存（MemoryObj）
    │
    ▼ StorageManager.batched_put()
    │
    ├──→ LocalCPUBackend.submit_put_task()
    │    └── hot_cache[key] = memory_obj
    │        数据在 CPU 热缓存中
    │        受 LRU/LFU 等策略管理
    │        内存不足时按需淘汰
    │
    └──→ RemoteBackend.batched_submit_put_task()
         └── mooncake.put_from(key, ptr, size)
             数据在 Mooncake 中
             受 Mooncake 运行时管理
             Mooncake 段满时自行淘汰

检索时：
    StorageManager.get()
        │
        ├── 先查 CPU 热缓存 → 命中则直接返回
        │
        └── 未命中则查 Mooncake → 命中则：
            ├── 返回数据
            └── 自动回写到 CPU 热缓存（auto write-back）
```

### 7.3 关键交互场景

**场景 1：CPU 热缓存淘汰，Mooncake 保留**

```
T0: chunk_A 在 CPU 热缓存和 Mooncake 中
T1: 内存不足，chunk_A 被 LRU 淘汰出 CPU 热缓存
T2: chunk_A 仍在 Mooncake 中
T3: 新请求需要 chunk_A
T4: StorageManager.get() → CPU 未命中 → Mooncake 命中
T5: chunk_A 从 Mooncake 读取，自动回写到 CPU 热缓存
```

**场景 2：Mooncake 淘汰，CPU 热缓存保留**

```
T0: chunk_B 在 CPU 热缓存和 Mooncake 中
T1: Mooncake 全局段满，chunk_B 被 Mooncake 内部淘汰
T2: chunk_B 仍在 CPU 热缓存中（LMCache 不知道 Mooncake 淘汰了它）
T3: 新请求需要 chunk_B
T4: StorageManager.get() → CPU 命中 → 直接返回
T5: Mooncake 中没有 chunk_B，但 LMCache 不知道也不需要知道
```

**场景 3：两层都被淘汰**

```
T0: chunk_C 在 CPU 热缓存和 Mooncake 中
T1: CPU 内存不足，chunk_C 被 LRU 淘汰出 CPU 热缓存
T2: Mooncake 全局段满，chunk_C 被 Mooncake 内部淘汰
T3: chunk_C 在两层中都不存在了
T4: 新请求需要 chunk_C
T5: StorageManager.get() → CPU 未命中 → Mooncake 未命中
T6: vLLM 必须重新计算 chunk_C 的 KV Cache
```

---

## 8. 完整流程图与数值示例

### 8.1 Llama-7B 的缓存淘汰数值示例

**配置**：
```
模型: Llama-7B (NL=32, NH=32, HS=128, H=4096)
chunk_size: 256
max_local_cpu_size: 5 GB
cache_policy: LRU
Mooncake global_segment_size: 3.125 GB
```

**单个 chunk 的内存大小**：
```
key_value = [2, 32, 256, 4096] fp16
大小 = 2 × 32 × 256 × 4096 × 2 字节 = 128 MB
```

**CPU 热缓存容量**：
```
max_local_cpu_size = 5 GB
可容纳 chunk 数 = 5 GB / 128 MB ≈ 39 个 chunk
```

**Mooncake 容量**：
```
global_segment_size = 3.125 GB
可容纳 chunk 数 = 3.125 GB / 128 MB ≈ 24 个 chunk
```

### 8.2 淘汰过程的数值示例

```
初始状态: CPU 热缓存有 39 个 chunk，Mooncake 有 24 个 chunk

新请求到达，需要分配 1 个新的 chunk（128 MB）
    │
    ▼
allocate() 尝试分配 128 MB
    │
    └── 失败（5 GB 已满）
        │
        ▼
    LRU.get_evict_candidates(hot_cache, 1)
        │
        └── 找到 chunk_X（最久未使用，can_evict=True）
            │
            ▼
        batched_remove([chunk_X])
        ├── hot_cache.pop(chunk_X)
        ├── chunk_X.ref_count_down()  → ref_count: 2→1→自动回收
        └── 释放 128 MB 内存
            │
            ▼
        allocate() 重新尝试 → 成功
        │
        ▼
    新 chunk 存入 CPU 热缓存
    同时写入 Mooncake（如果 Mooncake 也满了，Mooncake 自己决定淘汰谁）
```

### 8.3 淘汰策略的选择建议

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 多轮对话 | LRU | 最近使用的 KV Cache 最可能被再次使用 |
| RAG（随机文档访问） | LFU | 频繁访问的文档应保留 |
| 批处理（一次性扫描） | MRU | 扫描后不再访问，淘汰最近的 |
| 简单场景 | FIFO | 最简单的策略，开销最小 |

### 8.4 配置参数速查

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `local_cpu` | True | 启用 CPU 热缓存 |
| `max_local_cpu_size` | 5.0 GB | CPU 热缓存大小 |
| `cache_policy` | "LRU" | 淘汰策略 |
| `remote_url` | None | Mooncake URL（如 `mooncakestore://`） |
| `remote_serde` | "naive" | 远程序列化方式 |

### 8.5 Mooncake 配置参数速查

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `global_segment_size` | 3.125 GB | Mooncake 全局段大小 |
| `local_buffer_size` | 1 GB | 本地缓冲区大小 |
| `protocol` | "tcp" | 传输协议 |
| `prefer_local_alloc` | True | 优先本地分配 |
| `save_chunk_meta` | False | 是否保存 chunk 元数据 |

---

> 文档生成时间：2026-05-30
> 基于 LMCache `dev` 分支源码深度分析
