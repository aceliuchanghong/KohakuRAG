# KohakuRAG 项目概览

这是一个生产级、领域无关的 RAG（检索增强生成）框架，专为层次化文档索引和智能检索设计。

```
┌─────────────────────────────────────────────────────────┐
│  第一阶段：索引构建                                │
│  PDF/MD/TXT 文档 → Parser → DocumentPayload            │
│                  → DocumentIndexer → TreeNode 层次结构   │
│                  → Jina v3/v4 嵌入句子                  │
│                  → 向上传播嵌入向量到父节点              │
│                  → 存储到 SQLite 数据库                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  第二阶段：检索                                 │
│  用户问题 → LLMQueryPlanner → 生成多个搜索查询          │
│           → 向量相似度搜索 (top_k)                       │
│           → 去重 + 重排序 + 截断                         │
│           → 上下文扩展 (父/子节点)                       │
│           → 可选: BM25 稀疏搜索 + 图片检索               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  第三阶段：回答                               │
│  上下文片段 → 构建 Prompt                               │
│             → LLM (OpenAI/OpenRouter)                    │
│             → 解析结构化 JSON 响应                       │
│             → 验证和标准化答案                           │
└─────────────────────────────────────────────────────────┘
```

```python
# 节点层级：doc_id:sec{N}:p{N}:s{N}（文档 → 章节 → 段落 → 句子）
StoredNode:
    node_id: str           # 层次化ID
    parent_id: str | None  # 父节点引用
    kind: NodeKind         # DOCUMENT/SECTION/PARAGRAPH/SENTENCE/ATTACHMENT
    text: str              # 实际内容
    embedding: np.ndarray  # 向量嵌入
    child_ids: list[str]   # 子节点引用
```

## 最精髓的设计

从代码分析来看，这个项目最核心的设计思想是 层次化索引 + 嵌入传播，体现在 indexer.py 中：

### 1️⃣ 核心：嵌入传播机制

```
句子（叶子节点）→ 直接嵌入
         ↓
段落 = 平均句子嵌入（或直接嵌入）
         ↓
章节 = 平均段落嵌入
         ↓
文档 = 平均章节嵌入
```

```python
def _propagate_embeddings(self, node: TreeNode, ...):
    if node.embedding is not None:
        return node.embedding  # 叶子节点已有嵌入

    # 先递归获取子节点嵌入
    child_vectors = [self._propagate_embeddings(child) for child in node.children]
    
    # 计算子节点的平均嵌入
    averaged_embedding = average_embeddings(child_vectors)
    node.embedding = averaged_embedding
    return node.embedding
```

- 只需对句子做嵌入计算（批量高效）
- 父节点的嵌入是语义聚合，可以代表段落/章节的主题
- 检索时可以匹配不同粒度的节点（句子精度高，段落上下文好）

#### 第一步：只有句子被"直接嵌入"
```
文档片段示例：
├── 段落1："光伏电池的效率通常在15-22%之间。"
│   └── 句子1："光伏电池的效率通常在15-22%之间。"
│
├── 段落2："钙钛矿电池效率已突破25%。单晶硅电池效率可达26%。"
│   ├── 句子1："钙钛矿电池效率已突破25%。"
│   └── 句子2："单晶硅电池效率可达26%。"
```

**indexer 只对句子调用嵌入模型**：

```
# 所有句子文本 → 嵌入模型 → 向量
句子嵌入结果：
  s1: "光伏电池的效率..." → [0.12, -0.34, 0.56, ...]  (768维向量)
  s2: "钙钛矿电池效率..."  → [0.23, 0.45, -0.12, ...]
  s3: "单晶硅电池效率..."  → [0.18, 0.41, -0.08, ...]
```

#### 第二步：父节点用"平均嵌入"

段落嵌入 = 它所有句子嵌入的平均值

```
# 段落1 只有1个句子
段落1嵌入 = 句子1嵌入
         = [0.12, -0.34, 0.56, ...]

# 段落2 有2个句子
段落2嵌入 = (句子2嵌入 + 句子3嵌入) / 2
         = ([0.23, 0.45, -0.12] + [0.18, 0.41, -0.08]) / 2
         = [0.205, 0.43, -0.10, ...]

同理，章节嵌入 = 它所有段落嵌入的平均值：

章节嵌入 = (段落1嵌入 + 段落2嵌入) / 2
```

段落嵌入虽然是用平均值"合成"的，但仍然能代表段落的主题方向，让检索可以跨粒度匹配。


### 2️⃣ 检索时的上下文扩展

层次化节点 ID 设计：doc:sec1:p2:s3（文档:章节:段落:句子）

这让上下文扩展变得简单：
- 匹配到句子 s3 → 可以轻松获取父段落 p2 的完整文本
- 用户问题可能匹配多个相关句子，扩展后提供完整上下文

```python
snippets = await matches_to_snippets(
    all_matches,
    store,
    parent_depth=self._parent_depth,  # 向上扩展几层
    child_depth=self._child_depth,     # 向下扩展几层
    dedup=self._snippet_dedup,          # 树去重
)
```

### 3️⃣ 多查询 + 智能重排序

LLMQueryPlanner 把一个问题拆成多个查询角度，然后：

```python
# 收集所有查询的结果
all_matches = []
for vector in query_vectors:
    matches = await store.search(vector, k=top_k)
    all_matches.extend(matches)

# 重排序策略：频率优先（多查询共识） + 分数加权
unique_matches.sort(key=lambda m: (
    node_stats[m.node.node_id]["frequency"],  # 出现次数
    node_stats[m.node.node_id]["total_score"], # 总分
), reverse=True)
```
