# D3：Agentic RAG 实战

> Easy Data x AI 课程 · 术篇 · 第三期
>
> D1 你学会了 Tool Use——Agent 与外部数据之间的桥梁。D2 你搭建了数据层——桥对面的目的地。这一期，我们把桥和目的地连起来，构建一个真正能用你的数据回答问题的知识库助手。

## 开场：该把两件事连起来了

回顾一下你目前拥有的能力。

D1 给了你一个机制：Tool Use。模型不再只能“说话”，它可以声明“我需要调用某个工具”，你的代码去执行，结果传回来，模型继续推理。你跑通了一个完整的循环——定义工具、模型调用、处理结果。

D2 给了你一个基座：seekdb 数据层。你用三行代码建了一个支持向量搜索和全文搜索的知识库，体验了混合检索的效果，也直观看到了纯向量搜索在遇到精确匹配需求时的软肋。

现在的问题是：**这两样东西怎么接到一起？**

说白了就是一句话：让 Agent 通过 Tool Use 去调用 seekdb 的检索能力，拿到结果后基于真实数据回答用户的问题。这就是 Agentic RAG——Agent 自主决定“要不要查”、“查什么”、“查到的结果够不够用”，然后基于检索到的内容生成回答。

这一期有两个目标：

1. **跑通 Agentic RAG 的完整代码链路**——从用户提问到 Agent 回答，中间经历 Tool Use 调用 seekdb 检索，整个过程完整串通
2. **通过对比实验亲眼看到差距**——同一组查询，纯向量检索和混合检索的结果差距到底有多大

第二个目标是这节课的重头戏。不需要看论文，不需要听人讲道理——跑一次实验，数据替你说话。

如果你身边有人还在说“用向量数据库做 RAG 就够了”，这节课结束后你会有一组实验数据可以回应他——不是观点对观点，是数据对观点。

## 第一部分：知识库构建的基本流程

在构建 Agentic RAG 之前，我们需要先有一个“装满了知识的数据库”供 Agent 查询。知识库构建的完整流程是：

**文档解析 → 分块（Chunking） → 向量化（Embedding） → 存储**

简单解释一下每个环节做了什么：

- **文档解析**：把 PDF、Word、网页等格式的文档转成纯文本。你的原始知识可能散落在各种格式里，第一步是统一提取出文字内容
- **分块**：大模型的上下文窗口有限（F1 讲过），一篇几万字的文档没法整篇塞给模型。所以需要把长文档切成较小的段落（chunk），每个 chunk 是一个独立的检索单元
- **向量化**：用 Embedding 模型把每个 chunk 转成一个向量，用于后续的语义搜索
- **存储**：把 chunk 的文本内容、向量和元数据（来源、版本号、更新时间等）一起存入数据库

分块策略的选择（按固定长度切还是按语义边界切）对检索质量有直接影响，但这属于偏深的工程决策——本模块的延伸阅读会提供参考，主线不在这里展开。

**好消息是**：本模块提供了一个预处理好的示例数据集，你可以跳过文档解析和分块的环节，直接从“存入数据库”开始。这样你可以把精力集中在 Agentic RAG 的核心逻辑和对比实验上。

### 把示例数据集加载到 seekdb

我们的示例数据集模拟了一个技术产品的知识库——包含产品文档、错误码手册、版本发布说明、性能调优指南等内容。这些内容已经完成了分块处理，每条记录就是一个 chunk。

```python
from pyseekdb import SeekDB

db = SeekDB()

# 创建知识库集合
db.create_collection(
    name="product_kb",
    vector_column="content",
    fulltext_columns=["content"]
)

# 示例数据集：模拟真实产品知识库的文档分块
knowledge_chunks = [
    {"content": "OB-4.2.1 版本兼容性说明：OB-4.2.1 与 OB-4.1.x 保持向后兼容，但不兼容 OB-3.x 系列。升级前请确认所有客户端驱动已更新至 4.x 版本。已知问题：在 ARM 架构下 JIT 编译存在偶发性能退化。",
     "doc_type": "release_notes", "version": "4.2.1"},
    {"content": "OB-4.1.0 版本兼容性说明：OB-4.1.0 为大版本升级，不兼容 OB-3.x 的数据格式，需要执行数据迁移工具。与 OB-4.0.x 保持兼容。",
     "doc_type": "release_notes", "version": "4.1.0"},
    {"content": "错误码 E-4012：数据库连接池耗尽。当并发连接数超过 max_connections 配置值时触发。解决方案：(1) 增大 max_connections 参数，(2) 检查应用是否存在连接泄漏，(3) 考虑使用连接池中间件。",
     "doc_type": "error_codes", "version": "4.2"},
    {"content": "错误码 E-4013：认证握手超时。客户端在 10 秒内未完成认证流程时触发。解决方案：(1) 检查网络延迟，(2) 确认 SSL 证书配置正确，(3) 排查防火墙是否拦截了认证端口。",
     "doc_type": "error_codes", "version": "4.2"},
    ...
]

db.insert(collection_name="product_kb", documents=knowledge_chunks)
print(f"已写入 {len(knowledge_chunks)} 个知识片段")
```

数据就位了。接下来，我们让 Agent 用起来。

## 第二部分：Agentic RAG 的代码结构

这一步的本质是把 D1 和 D2 连接起来：Agent 通过 Tool Use（D1 的机制）将检索请求转交给 seekdb（D2 的数据层），检索结果传回 Agent 进行推理，生成最终回答。

整个流程用一张图来表示：

```
用户提问
    ↓
Agent（大模型）分析问题
    ↓
Agent 决定调用 search_knowledge_base 工具
    ↓
你的代码收到 tool_call，调用 seekdb 执行混合检索
    ↓
seekdb 返回检索结果（最相关的文档片段）
    ↓
检索结果作为 tool 消息传回 Agent
    ↓
Agent 基于检索到的真实内容生成最终回答
```

是不是看着很眼熟？这就是 D1 里 Tool Use 的五步循环，只不过这次“工具”不是一个模拟的函数，而是真实的知识库检索。

这里有一个关键的区别值得强调：传统 RAG 是一个固定流程——用户提问，系统自动检索，结果塞给模型，模型回答。整个过程像流水线一样从头到尾执行一遍。而 Agentic RAG 中，**Agent 自己决定要不要检索**。它会先分析用户的问题：这个问题我已经知道答案了吗？还是需要查知识库？查了之后结果够不够用？需不需要换个关键词再查一次？这种主动判断的能力，就是 “Agentic” 这个词的含义——Agent 不是被动执行流程，而是主动做决策。

### 完整代码：从提问到回答

```python
from openai import OpenAI
import json

client = OpenAI()

# 第一步：定义检索工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "从产品知识库中检索相关信息。当用户询问产品功能、错误码、版本信息、性能优化、营收数据等问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索知识库的查询文本，应尽量保留用户问题中的关键词和专有名词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 第二步：定义工具执行函数——调用 seekdb 混合检索
def execute_search(query: str) -> str:
    results = db.hybrid_search(
        collection_name="product_kb",
        query_text=query,
        top_k=3
    )
    if not results:
        return "知识库中未找到相关内容。"

    formatted = []
    for i, r in enumerate(results):
        formatted.append(f"[结果 {i+1}] (相关度: {r['score']:.3f})\n{r['content']}")
    return "\n\n".join(formatted)

# 第三步：Agentic RAG 主循环
def ask_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": (
            "你是一个产品技术文档助手。回答用户问题时，请先查询知识库获取准确信息，"
            "然后基于查询结果回答。如果知识库中没有相关信息，请诚实告知用户。"
            "回答时请引用具体的数据和版本号，不要猜测。"
        )},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    # Agent 决定调用工具
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        query_text = arguments["query"]

        print(f"  🔍 Agent 决定检索: \"{query_text}\"")

        search_results = execute_search(query_text)

        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": search_results
        })

        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )

        return final_response.choices[0].message.content

    # Agent 决定不调用工具，直接回答
    print("  ✅ Agent 决定直接回答")
    return message.content
```

来试一下：

```python
answer = ask_agent("OB-4.2.1 版本和旧版本兼容吗？")
print(answer)
```

你会看到这样的过程：Agent 收到用户问题，判断需要查知识库，通过 Tool Use 声明调用 `search_knowledge_base`，你的代码执行 seekdb 混合检索，结果返回给 Agent，Agent 基于检索到的 OB-4.2.1 兼容性文档生成一个准确的回答——引用了具体的版本号、兼容性关系和已知问题。

这就是 Agentic RAG 的核心链路。模型负责理解用户意图和组织语言，seekdb 负责提供准确的数据，Tool Use 负责把两者连接起来。**三者各司其职，缺一不可。**

![](https://raw.githubusercontent.com/ob-labs/easy-data-x-ai/main/docs/public/images/dev/D3/01-agentic-rag-flow.png)

## 第三部分：对比实验——数据说话

到目前为止，你可能对“混合检索比纯向量检索好”这个判断已经有了直觉印象——D2 的简单示例给了你一些感觉。但直觉不够。这一节，我们用一组系统性的对比实验来验证这个判断。

实验设计很简单：**同一组查询，分别用纯向量检索和混合检索，对比 Top-1 结果是否命中了用户真正需要的内容。**

### 构建对比实验

```python
def compare_search(query: str):
    """对比纯向量检索和混合检索的结果"""
    print(f"查询: \"{query}\"")
    print("-" * 60)

    # 纯向量检索
    vector_results = db.vector_search(
        collection_name="product_kb",
        query_text=query,
        top_k=3
    )

    # 混合检索
    hybrid_results = db.hybrid_search(
        collection_name="product_kb",
        query_text=query,
        top_k=3
    )

    print("纯向量检索 Top-3:")
    for i, r in enumerate(vector_results):
        snippet = r["content"][:60].replace("\n", " ")
        print(f"  {i+1}. [{r['score']:.3f}] {snippet}...")

    print("混合检索 Top-3:")
    for i, r in enumerate(hybrid_results):
        snippet = r["content"][:60].replace("\n", " ")
        print(f"  {i+1}. [{r['score']:.3f}] {snippet}...")
    print()
```

### 运行实验

我们特意选了三类最能体现差距的查询——都是真实场景中高频出现的类型：

```python
test_queries = [
    "OB-4.2.1版本的兼容性",          # 包含精确版本号
    "error code E-4012",             # 包含精确错误码
    "2024年Q3营收数据",               # 包含精确时间和数据类型
    "怎么优化数据库的查询性能",          # 纯语义查询（作为对照）
    "DBMS_HYBRID_SEARCH 函数怎么用",  # 包含精确函数名
]

for query in test_queries:
    compare_search(query)
```

### 实验结果

跑完上面的代码，你会得到一张类似这样的对比表：

| 查询 | 纯向量检索 Top-1 | 混合检索 Top-1 | 命中正确？ |
| --- | --- | --- | --- |
| “OB-4.2.1版本的兼容性” | OB-4.1.0 兼容性说明 ❌ | **OB-4.2.1 兼容性说明** ✅ | 向量 ❌ / 混合 ✅ |
| “error code E-4012” | E-4013 认证握手超时 ❌ | **E-4012 连接池耗尽** ✅ | 向量 ❌ / 混合 ✅ |
| “2024年Q3营收数据” | 2024年Q2营收数据 ❌ | **2024年Q3营收数据** ✅ | 向量 ❌ / 混合 ✅ |
| “怎么优化数据库的查询性能” | 并行查询优化指南 ✅ | 并行查询优化指南 ✅ | 向量 ✅ / 混合 ✅ |
| “DBMS_HYBRID_SEARCH 函数怎么用” | 索引设计最佳实践 ❌ | **DBMS_HYBRID_SEARCH 函数说明** ✅ | 向量 ❌ / 混合 ✅ |

仔细看这张表。

前三条查询都包含需要**精确匹配**的内容——版本号“4.2.1”、错误码“E-4012”、时间“Q3”。纯向量检索在这三条上全部失手：它返回的是语义上“差不多”的内容，而不是用户真正需要的那条。

- 用户问的是 4.2.1 的兼容性，向量检索返回了 4.1.0 的——因为在向量空间中“版本兼容性说明”这个语义概念几乎一样，模型分不清 4.2.1 和 4.1.0
- 用户问的是 E-4012，向量检索返回了 E-4013——D2 已经演示过这个问题，这里再次验证
- 用户问的是 Q3 的营收，向量检索返回了 Q2 的——“营收数据”的语义向量高度相似，Q2 和 Q3 在向量空间中几乎无差别

第四条查询是纯语义查询——“怎么优化数据库的查询性能”。这条没有任何需要精确匹配的关键词。两种方式都能正确命中。这说明混合检索并没有牺牲语义能力——它只是在语义的基础上**补上了精确匹配**。

第五条查询包含一个精确的函数名 `DBMS_HYBRID_SEARCH`。纯向量检索完全找不到它——因为这个函数名在语义空间中没有特殊含义，它就是一个“看起来像技术术语”的字符串。但全文搜索能精确匹配到它。

### 关键发现

5 条查询中，纯向量检索只在 1 条上给出了正确的 Top-1 结果（命中率 20%），混合检索在全部 5 条上都正确（命中率 100%）。

而且这不是我们刻意挑的“刁钻查询”。你回想一下自己日常工作中查文档的场景：查某个版本的功能、查某个错误码的解决方案、查某个季度的数据、查某个 API 的用法——**这些就是最常见的查询类型，而它们恰恰是纯向量检索最容易翻车的地方**。

### 这意味着什么

如果你的 Agentic RAG 系统只用纯向量检索，那么上面这些场景中，Agent 拿到的就是**错误的检索结果**。模型再聪明，基于错误的数据推理出的答案也是错的——甚至比没有检索更危险，因为 Agent 会信心满满地给出一个“看起来很对但其实答非所问”的回答。

用 P2 的归因框架来说：这不是模型层的问题（模型的推理能力没问题），这是**数据层的问题**（检索策略选错了，导致模型拿到了错误的数据）。

而且这类错误特别隐蔽。如果 Agent 直接说“我不知道”，用户至少知道需要换个方式问。但当 Agent 基于“语义相近但实际错误”的检索结果回答时，它的语气完全是自信的——因为它确实拿到了“来源数据”，只不过拿错了。用户很难察觉回答是错的，除非他自己去核实原始数据。这种“自信的错误”比“坦诚的无知”危害大得多。

![](https://raw.githubusercontent.com/ob-labs/easy-data-x-ai/main/docs/public/images/dev/D3/02-search-comparison-result.png)

把实验结果放到 Agentic RAG 的上下文中更容易理解。我们拿“2024年Q3营收数据”这条查询举个例子：

```python
# 用纯向量检索的 Agentic RAG
answer_vector = ask_agent_with_vector_only("2024年Q3的总营收是多少？")
# Agent 可能基于 Q2 的数据回答："2024年Q3总营收为 2.41 亿元"
# → 错误！2.41 亿是 Q2 的数据，Q3 是 2.87 亿

# 用混合检索的 Agentic RAG
answer_hybrid = ask_agent_with_hybrid("2024年Q3的总营收是多少？")
# Agent 基于 Q3 的数据回答："2024年Q3总营收为 2.87 亿元，同比增长 34%"
# → 正确
```

同一个 Agent，同一个模型，同一个 Prompt，唯一的差别是检索策略——**结果一个答对了，一个答错了。** 这就是数据层选型的影响力。

## 第四部分：从实验到生产——你需要知道的几件事

跑完对比实验之后，你对混合检索的价值应该已经有了直觉判断。在把这个 Agentic RAG 系统推向更真实的场景之前，有几个工程层面的要点值得提一下。

### 检索结果的质量直接决定回答质量

这听起来像废话，但很多团队在 RAG 系统出问题时的第一反应是调 Prompt 或者换模型。对照实验告诉你：如果 Agent 拿到的检索结果本身就是错的，Prompt 调得再好也没用。**先确保检索层给出了正确的结果，再考虑模型层的优化。**

### 工具描述（Tool Description）的质量也很关键

你在 D1 学过这一点：如果工具的 `description` 写得不清楚，模型不知道什么时候该调用它。在 Agentic RAG 中，这意味着模型可能在应该查知识库的时候没有查——直接“猜”了一个回答。这就是幻觉的一个常见来源。

所以 Prompt Engineering 在 Agentic RAG 里仍然重要——但它解决的是“Agent 要不要查”和“怎么组织答案”的问题，不是“查到什么”的问题。后者是数据层的事。

### 检索数量（top_k）的取舍

top_k 设太小，可能遗漏关键信息；设太大，会给模型传入大量不相关内容，影响推理质量（噪声太多反而干扰判断）。实践中，top_k = 3~5 是一个常见的起点，可以根据你的数据密度和查询类型调整。

### 数据更新与一致性

知识库不是一次性构建完就不管了。文档会更新，错误码会新增，版本会迭代。你需要一个机制来保持知识库数据和源文档的同步。最简单的方案是定期全量重建，但更实际的做法是增量更新——只处理变更的文档。seekdb 支持对已有集合追加和更新数据，你不需要每次都删库重建。

在生产环境中，“数据新鲜度”往往比“检索算法的精细调优”更影响用户体验——Agent 回答了一个三个月前就已经修复的 bug 的解决方案，用户的信任会立刻崩塌。这也是为什么 P2 讲知识库是产品决策：**更新频率本身就是一个需要 PM 和开发共同决定的产品策略**。

## 我们的思考

对比实验是我们说服自己（和客户）“混合检索不是可选优化”的方式。

每次有人问“纯向量搜索够用吗”，我们都不争论——我们建议他拿自己的业务数据跑一次对比。特别是那些包含产品型号、版本号、错误码、精确数值的查询——这些在几乎每个企业的知识库中都大量存在，而且恰恰是用户最关心精确性的场景。跑完之后，结果替我们说话。

seekdb 的混合检索通过单条查询完成（`DBMS_HYBRID_SEARCH`），向量搜索和全文搜索在引擎内部执行并通过 RRF 算法融合排序——不需要在应用层维护两套检索系统，也不需要自己写结果合并逻辑。这让做对比实验变得非常简单：你只需要把 `hybrid_search` 换成 `vector_search`，其他代码一行不用改，就能直接看到两种策略的差异。

这也是我们做 seekdb 时反复验证的一个设计原则：**正确的做法应该同时是最简单的做法。** 如果“做得对”需要你额外维护两套系统、写大量胶水代码、处理复杂的分数归一化——那大多数团队都会“先凑合用向量搜索”。但当混合检索只是一个参数的差别时，没有理由不用它。

![](https://raw.githubusercontent.com/ob-labs/easy-data-x-ai/main/docs/public/images/dev/D3/03-hybrid-search-advantage.png)

回到课程的主线来看这件事：F1 讲了大模型的三个局限本质上都是数据问题，F2 讲了 Agent 的每一项能力拆到底都是数据的存储与检索。D3 的对比实验是这条主线的一次具体验证——同样的模型、同样的 Prompt、同样的数据，仅仅因为检索策略不同，最终结果就是“对”和“错”的区别。这不是理论推演，是你刚刚亲手跑出来的实验结论。

## 这节课要留下的印象

如果这节课的所有内容你只记住一段话，记住这段：

> **同样是 RAG，检索策略的差异导致肉眼可见的结果差距——不需要看论文，跑一次对比实验就明白了。混合检索不是高级优化，而是生产级 RAG 的基本要求。**

## 课后行动

1. **用你自己的数据重复这个实验**。本模块提供了示例数据集，但真正有说服力的实验是用你自己业务场景的数据来跑。把你的产品文档、API 文档、内部 FAQ 导入 seekdb，然后用你和同事日常真正会问的问题作为查询。

2. **记录 3 个最能体现差距的查询案例**。特别关注那些包含专有名词、版本号、精确术语的查询——这些是纯向量检索最容易翻车、混合检索优势最明显的场景。

3. **分享给觉得“纯向量搜索够用了”的同事**。不需要争论，把对比实验的结果发给他——数据说话比任何论点都有说服力。

## 延伸阅读

如果你对本期提到的概念想做进一步了解，以下是一些推荐资源：

- **分块策略（Chunking Strategies）**：固定长度分块（按 Token 数或字符数切分）实现简单但可能切断上下文；语义分块（按段落、章节或主题边界切分）保留了上下文完整性但实现更复杂。实践中，很多团队从固定长度分块开始，在遇到检索质量问题后再切换到语义分块——这是一个典型的“先跑通再优化”的工程决策
- **RRF（Reciprocal Rank Fusion）算法**：混合检索中，如何将向量搜索的排序结果和全文搜索的排序结果融合成一个统一排名？RRF 是业界最常用的方案。核心思想：不直接比较分数（因为两种搜索的分数含义不同），而是基于**排名**做融合——排名靠前的结果得到更高权重。公式简洁、效果稳定，seekdb 的 `DBMS_HYBRID_SEARCH` 内部使用的就是这个算法
- **LangChain RAG 文档**：[RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)，LangChain 的 RAG 教程，展示了如何用框架组织检索和生成流程。本课程选择不依赖框架而是用原生 API，是为了让你看清每一步发生了什么——理解原理之后，框架只是效率工具

> **下一期预告**：D4 · Agent 开发与记忆系统——你的 Agent 现在能查知识库了，但它还没有”记忆”。每次对话都是从零开始，不知道你是谁、之前聊过什么。D4 会给 Agent 加上记忆系统——用 PowerMem 实现记忆的存储、检索和遗忘。你会亲眼看到”有记忆”和”没记忆”的 Agent 在对话体验上的差距。

---

欢迎各位老师在 https://github.com/ob-labs/easy-data-x-ai 参与课程共建。

也欢迎各位老师加入 Data x AI 交流群~

<div align="center">
  <img src="https://raw.githubusercontent.com/ob-labs/easy-data-x-ai/main/docs/public/images/base_knowledge/F0/F0-20.png" width="200" />
</div>