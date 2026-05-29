# Paper-Reproduce Skill

## 项目目标

开发一个 Claude Code skill，帮助研究人员快速复现尚未开源的论文方法。聚焦 CV、NLP、多模态领域，端到端覆盖从论文解析到结果验证的完整流程。

## 核心设计决策

| 决策项 | 结论 |
|--------|------|
| 推理引擎 | Claude 自身负责推理，Python 工具脚本处理 PDF 解析、指标对比等任务 |
| 交互模式 | 分 5 阶段推进，每阶段完成后展示结果，等待用户确认再进入下一阶段 |
| 上下文管理 | 每个复现任务维护 `reproduction_state.yaml` 作为外部记忆，防上下文窗口溢出 |
| 组件粒度 | 细粒度 + 可拼装（如 `ResNetBlock` 而非完整 `ResNet50`） |
| 组件来源 | 内置组件 + 用户扩展 + 复现过程中沉淀 |
| 组件规范 | 统一接口规范 + 三级自动检查 + 规范化重写 + 用户确认后入库 |
| LLM 支持 | 改造方式组件（LoRA/Adapter/Attention 替换）+ 基础设施组件（并行/内存/评估）双维度覆盖 |

## 整体架构（5 阶段流水线）

```
PDF链接/文件 → [1.论文获取与解析] → [2.复现方案生成] → [3.代码实现] → [4.实验运行] → [5.结果验证]
```

### 阶段 1：论文获取与多模态解析
- arXiv API / PDF 文件读取
- PyMuPDF 提取文本 + 图表
- 多模态交叉印证（图与文字描述比对）
- 输出结构化 YAML：架构、训练配方、数据集、超参数、模糊区域

### 阶段 2：复现方案生成
- 依赖分析、缺失信息清单、分步实施计划
- "坑点预警"（已知复现难点提示）
- 产出复现计划书供用户确认

### 阶段 3：代码实现
- 基于组件库拼装生成模型代码、数据 pipeline、训练/评估脚本
- 生成 `requirements.txt` / `environment.yml`

### 阶段 4：实验运行
- 辅助用户配置实验、管理超参数组合
- 可选：直接运行实验并监控 loss

### 阶段 5：结果验证
- 复现结果与论文报告指标自动对比
- 差异分析和可能原因提示

## 组件库设计

### 目录结构

```
~/.claude/skills/paper-reproduce/
├── SKILL.md
├── tools/
│   ├── pdf_parser.py
│   ├── arxiv_fetcher.py
│   └── metrics_compare.py
├── components/
│   ├── builtin/
│   │   ├── cv/
│   │   │   ├── backbones/
│   │   │   ├── necks/
│   │   │   ├── heads/
│   │   │   ├── losses/
│   │   │   └── augmentations/
│   │   └── nlp/
│   │       ├── backbones/
│   │       ├── tokenizers/
│   │       ├── heads/
│   │       └── losses/
│   ├── user/                   # 用户自定义，不被更新覆盖
│   │   ├── cv/
│   │   └── nlp/
│   └── .registry.yaml
└── examples/
```

### 组件注册规范

每个组件需包含 `component.yaml` 描述文件：
- name、type、domain
- interface（inputs/outputs 规格）
- 来源论文
- spec_version

### 客制组件管理全流程

```
用户编写组件
    │
    ▼
┌─────────────┐    有问题    ┌──────────────┐
│  L1/L2 检查  │ ──────────► │ 意图识别+重写  │
└─────────────┘              └──────┬───────┘
    │ 通过                          │
    │                               ▼
    │                        ┌──────────────┐
    │                        │  Diff 展示    │
    │                        │  等待用户确认  │
    │                        └──────┬───────┘
    │                               │ 确认
    ▼                               ▼
┌─────────────┐              ┌──────────────┐
│  L3 兼容检查 │ ◄─────────── │  二次检查     │
└──────┬──────┘              └──────────────┘
       │ 通过
       ▼
┌─────────────┐
│  注册入库    │
│  更新registry│
└─────────────┘
```

- **L1 结构检查**：描述文件完整性、必填字段
- **L2 行为检查**：实际 import + dummy 输入验证输出形状/类型
- **L3 兼容检查**：与已注册组件的组合兼容性
- **重写原则**：保留核心算法逻辑和设计意图，规范化外围接口和命名

## 记忆文档（Reproduction State File）

每个复现任务维护 `reproduction_state.yaml`：

```yaml
task_id: "resnet-50-repro"
paper: "Deep Residual Learning for Image Recognition"
status: stage-1

parsed_architecture: {...}
reproduction_plan: {...}
decisions:           # 每次用户确认/修改的记录
  - timestamp: "..."
    stage: "..."
    decision: "..."
    reason: "..."
context_summary:     # 防窗口爆炸的上下文摘要
  what_was_done: "..."
  what_is_next: "..."
  key_findings: [...]
```

---

## 开发计划

### Phase 1: Skill 骨架 + 阶段 1（论文获取与解析）

**状态**: 🟢 已完成

- [x] 创建 `SKILL.md` 主定义文件 → `SKILL.md`
- [x] 创建目录结构（tools/、components/、examples/） → 18 个目录
- [x] 实现 `tools/arxiv_fetcher.py`：arXiv API 下载 + 元数据获取
- [x] 实现 `tools/pdf_parser.py`：文本提取（section 检测）+ 图表提取（渲染为 PNG）
- [x] 实现 `tools/reproduction_state.py`：schema 定义 + state 文件 I/O + 验证
- [x] 实现 `tools/cross_validator.py`：图文交叉验证记录生成 + claims 提取

### Phase 2: 阶段 2（复现方案生成）

**状态**: 🟢 已完成

- [x] 实现 `tools/pitfall_kb.yaml`：18 条 CV/NLP/LLM/通用坑点记录，含严重程度、解决方案、来源
- [x] 实现 `tools/dependency_mapper.py`：混合架构（规则匹配 Tier 1 + LLM 推理 Tier 2），自动生成 requirements.txt
- [x] 增强 `SKILL.md` Stage 2 引导 prompt：7 个子步骤的详细流程，含工具调用指令和输出格式模板

### Phase 3: 组件库骨架 + 阶段 3（代码实现）

**状态**: 🟢 已完成

- [x] 实现 `tools/component_schema.py`：`component.yaml` schema 定义 + 验证 + registry 索引/搜索/注册
- [x] 实现 `tools/component_checker.py`：L1 结构检查 + L2 行为检查（import → forward → shape验证）+ L3 兼容性检查
- [x] 实现 `tools/component_normalizer.py`：代码结构提取 → 意图识别 prompt → 规范化重写 → Diff 展示
- [x] 创建 4 个内置组件：`resnet_bottleneck`, `focal_loss`, `transformer_encoder_block`, `lora_linear`
- [x] 初始化 `components/.registry.yaml`（4 个组件已注册）
- [x] 增强 `SKILL.md` Stage 3 引导 prompt：组件发现 → 拼装 → 生成 → 检查 → 呈现的完整流程

### Phase 4: 阶段 4 + 5（实验运行 + 结果验证）

**状态**: 🔴 待开始

- [ ] 实现实验配置管理
- [ ] 实现实验运行与监控（可选）
- [ ] 实现结果与论文指标自动对比
- [ ] 实现差异分析报告生成

### Phase 5: 文档、示例、发布准备

**状态**: 🔴 待开始

- [ ] 编写完整复现示例（如 ResNet、BERT 复现）
- [ ] 编写用户文档
- [ ] 发布至 Claude Code skill registry

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-05-29 | 初始文档，确立设计决策和开发计划 |
| 2026-05-29 | **Phase 1 完成**：SKILL.md + 4 个工具脚本（arxiv_fetcher, pdf_parser, reproduction_state, cross_validator） |
| 2026-05-29 | **Phase 2 完成**：pitfall_kb.yaml + dependency_mapper.py + SKILL.md Stage 2 prompt 增强 |
| 2026-05-29 | **Phase 3 完成**：component_schema + checker + normalizer + 4 个内置组件 + registry 初始化 + SKILL.md Stage 3 增强 |
