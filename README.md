<h1 align="center">Easy Data X AI（Alpha 内测版，欢迎各位老师参与共建）</h1>

<p align="center">
  <em>面向所有 AI 爱好者的 Data 与 AI 基础知识入门教程</em>
</p>

<p align="center">
  <a href="https://ob-labs.github.io/easy-data-x-ai/">在线阅读</a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://open.oceanbase.com/course/760">社区在线课堂</a>
</p>

---

## 📚 这门课程适合谁？

> 双轨并行，论道与习术

为了满足不同角色的学习需求，我们将课程精心设计为两条路径："道篇"与"术篇"。

### 道篇：悟其道（零基础 AI 爱好者和产品决策者的"心法篇"）

**适合人群：** 零基础 AI 爱好者、产品决策者

**学完收获：**
- 🎯 场景判断力：学会评估"这个需求适不适合做 Agent"，避免在立项之初就走上弯路
- 🔍 归因决策力：获得一套"三层度量框架"，精准定位问题出在数据层、模型层还是业务层
- 🏗️ 系统设计力：理解 RAG、MCP、Skill、Memory 背后的产品设计哲学

### 术篇：用其术（开发者的"功法篇"）

**适合人群：** 已能调用 LLM API 的开发者

**学完收获：**
- 💪 坚实的工程基础：掌握流式输出（Streaming）和工具调用（Tool Use）
- 🗄️ 完整的数据层构建经验：基于轻量级 AI Native 数据库从零搭建数据层
- 📊 看得见的性能差距：通过对比实验见证"混合检索"与"纯向量检索"的效果差异
- 🤖 从零到一的 Agent 构建：为 Agent 加上记忆系统，教会它使用技能

## 📖 课程目录

```
公共基础篇
├── F0：课前闲聊 —— OpenClaw 为什么越用越好用？
├── F1：大模型的本质与边界
└── F2：AI Agent 的完整图景

道篇（P1-P5）
├── P1：找准 Agent 的用武之地 —— AI Agent 场景识别 ✅
├── P2：让 Agent 会查资料 —— RAG 产品设计 🚧
├── P3：让 Agent 真正记住你 —— 记忆系统设计 🚧
├── P4：把经验变可复用 —— Skill 与知识管理 🚧
└── P5：用数据验证价值 —— 案例与度量 🚧

术篇（D1-D5）
├── D1：打通 Agent 与数据 —— 大模型 API 入门 ✅
├── D2：一个系统搞定 —— 统一 AI Native 数据层实战 🚧
├── D3：实践出真知 —— Agentic RAG 实战 🚧
├── D4：记哪些、忘哪些？—— Agent 记忆系统开发 🚧
└── D5：授 AI 以渔 —— 综合实战，从 Skill 开发到 MCP 标准化 🚧
```

## 📅 课程安排

<table>
  <thead>
    <tr><th>篇章</th><th>课程编号</th><th>课程标题</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">公共基础篇</td><td>F1</td><td>大模型的本质与边界</td></tr>
    <tr><td>F2</td><td>AI Agent 的完整图景</td></tr>
    <tr><td rowspan="5">道篇</td><td>P1</td><td>找准 Agent 的用武之地 —— AI Agent 场景识别</td></tr>
    <tr><td>P2</td><td>让 Agent 会查资料 —— RAG 产品设计</td></tr>
    <tr><td>P3</td><td>让 Agent 真正记住你 —— 记忆系统设计</td></tr>
    <tr><td>P4</td><td>把经验变可复用 —— Skill 与知识管理</td></tr>
    <tr><td>P5</td><td>用数据验证价值 —— 案例与度量</td></tr>
    <tr><td rowspan="5">术篇</td><td>D1</td><td>打通 Agent 与数据 —— 大模型 API 入门</td></tr>
    <tr><td>D2</td><td>一个系统搞定 —— 统一 AI Native 数据层实战</td></tr>
    <tr><td>D3</td><td>实践出真知 —— Agentic RAG 实战</td></tr>
    <tr><td>D4</td><td>记哪些、忘哪些？—— Agent 记忆系统开发</td></tr>
    <tr><td>D5</td><td>授 AI 以渔 —— 综合实战，从 Skill 开发到 MCP 标准化</td></tr>
    <tr><td>结营仪式</td><td></td><td>结营仪式</td></tr>
  </tbody>
</table>

> 💡 **核心理念**：洞察先行，自然跟随。当你看懂了数据，才是真正看懂了 AI 的未来。

## 🚀 快速开始

### 在线阅读

访问 [https://ob-labs.github.io/easy-data-x-ai](https://ob-labs.github.io/easy-data-x-ai) 在线阅读课程内容。

### 本地阅读

```bash
# 克隆仓库
git clone https://github.com/ob-labs/easy-data-x-ai.git

cd easy-data-x-ai

# 安装依赖
npm install

# 本地预览
npm run docs:dev
```

### 本地运行示例代码

```bash
cd code

# 安装 Python 依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填写你的 API Key

# 运行示例
cd D1
python3 d1_1_base.py
```

## 🤝 参与贡献

- 如果你发现了一些问题，可以提 [Issue](https://github.com/ob-labs/easy-data-x-ai/issues) 进行反馈
- 如果你想参与贡献本项目，欢迎提 [Pull Request](https://github.com/ob-labs/easy-data-x-ai/pulls)
- 欢迎加入课程共建，一起完善内容

## 👥 贡献者名单

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/liboyang0730">
          <img src="https://github.com/liboyang0730.png" width="100px;" alt="liboyang0730"/>
          <br />
          <sub><b>Zlatan (liboyang0730)</b></sub>
        </a>
        <br />
        <sub>项目维护者</sub>
      </td>
      <td align="center">
        <a href="https://github.com/webup">
          <img src="https://github.com/webup.png" width="100px;" alt="webup"/>
          <br />
          <sub><b>Haili Zhang (webup)</b></sub>
        </a>
        <br />
        <sub>课程共建者</sub>
      </td>
    </tr>
  </table>
</div>

## 关注我们
<div align=center>
<p>欢迎扫描下方二维码加入 Data x AI 课程交流群</p>
<img src="https://raw.githubusercontent.com/ob-labs/easy-data-x-ai/main/docs/public/images/base_knowledge/F0/F0-20.png" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
