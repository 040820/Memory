# 实验目录索引

## 目录规则

- 每项实验使用两位序号和见名知意的名称，例如 `01_evidence_grounded_memory_evolution`、`02_实验名称`；序号递增，已有实验不改号、不覆盖。
- 实验开始前先在对应目录写简要 `实验方案.md`，说明目标、内容、数据、对照、指标及验收；实现和运行时再逐步补充配置、日志、检查点、原始结果和分析。
- 每次重复运行单独保存，不覆盖历史结果；未执行、预检、失败和完成分别如实记录。
- 不擅自设置整体运行时限或其他会中断实验的限制。运行前记录实际模型和输入，保留可续跑的检查点。
- 标准答案与隐藏证据只用于评测，不进入构图、检索或回答流程。

## 已规划实验

| 序号 | 目录 | 内容 | 当前状态 |
|---|---|---|---|
| 01 | [01_evidence_grounded_memory_evolution](01_evidence_grounded_memory_evolution/实验方案.md) | 证据驱动记忆演化与回答 | 组件开发与影子预检，未完成端到端评测 |
| 02 | [02_evidence_relation_and_answer_diagnostics](02_evidence_relation_and_answer_diagnostics/实验方案.md) | 新旧事实关系与回答诊断 | 开发案例验证，尚无真实数据效果结论 |
| 03 | [03_longmemeval_update_preflight](03_longmemeval_update_preflight/实验方案.md) | LongMemEval 更新题预检 | 一题开发预检已完成，尚非正式评测 |
| 04 | [04_fact_attribute_pair_retrieval](04_fact_attribute_pair_retrieval/实验方案.md) | 基于事实属性的新旧配对检索 | 一题开发诊断已完成，尚无独立效果评测 |

## 已有历史材料

`evidence_memory_pilot/` 是编号规则确立前的预检材料，保留原位置，不当作实验 01 已完成的结果。

## 阶段总结与计划

- [2026-11-01 工作总结](2026-11-01_工作总结.md)
- [2026-11-01 后续工作计划](2026-11-01_后续工作计划.md)
