# 长期对话记忆：小规模人工标注试验

只使用 Python 标准库。脚本生成候选材料，不调用模型、不修改原始数据、不产生自动金标准。

```powershell
python annotation_pilot/make_pilot.py --output annotation_pilot/outputs/pilot_20261101
```

输出目录必须不存在，防止覆盖已填写的标注。可以通过 `--seed`、`--mem-per-stratum`、`--long-count` 和 `--locomo-count` 调整采样；通过 `--help` 查看参数。默认目标是 MemConflict 四组各 5 条、LongMemEval 8 条、LoCoMo 8 条；候选不足时实际数量较少，见 `manifest.json`。一个 MemConflict 会话可以属于多个组，采样时跨组去重。三组采样规则不同，不能将数量比例当成原数据中的发生率。

## 输出

* `annotation.jsonl`：供人工标注的盲化文件。每行一个案例，包含原始对话和空白 `annotation`；只填写空白字段，保留 `case_id`、原始文本及定位信息。建议两位标注员各自复制一份独立填写。
* `references.jsonl`：独立保存数据集的参考信息、抽样类别及 LongMemEval 的问题、答案和标注答案会话。**盲标完成前不要打开或向标注员展示**；它也不能输入记忆系统、检索器、判别器或回答器。
* `manifest.json`：种子、候选池与各组数量、上下文策略、输入文件绝对路径及 SHA-256。用于复现和审计。

这些文件用于**人工数据分析**；它们不是模型运行输入协议，也不是已经完成的标注结果。不要把 MemConflict 的 `Updated_Attributes` 或 LongMemEval 的答案会话当作证据充分性的金标准。

## 上下文与选样

* MemConflict：用更新元数据选择 Residence、Career_Status、Marital_Status，以及带静态冲突标记的会话；盲标材料保留该人物从第一会话到目标会话的全部原始对话。目标会话 ID 指明待审查位置，但不显示所选字段及参考标签。
* LongMemEval：从非 `_abs` 的 `knowledge-update` 类别抽样；保留完整 haystack，屏蔽 `has_answer`、问题、标准答案和答案会话。此组不自动标明旧/新证据，标注成本较高；允许跳过并写明原因。仅问题类别用于离线抽样。
* LoCoMo：在对话消息中按脚本中的英文关键词选取候选，保留对话到命中轮次为止。关键词仅用于寻找候选，不表示必然发生状态变化；可能存在词义误命中和漏召回。

会话按照**文件原始顺序**保留，不自动按时间重排或按提问时间截取；保留原始日期字符串，同日不推断先后。LongMemEval 完整历史可能含问题时间之后的同日会话；盲标中没有问题时间，严格在线评测必须之后单独审计，不得把此材料直接当成在线输入。

## 标注说明

`old_evidence` 和 `new_evidence` 为数组，每项推荐写 `{ "session_id": "...", "turn_id": "...", "message_index": 0, "quote": "原文连续片段" }`；没有旧或新证据时留空并解释。LongMemEval 的 `turn_id` 是原会话内从 0 开始的消息索引；LoCoMo 是原始 `dia_id`；MemConflict 是 `dialogue_turn_N` 加消息序号。不要把助手的猜测当作用户确认。

`relation` 建议填写 `change`（明确变化）、`correction`（明确纠正）、`duplicate`、`complement`、`coexist`、`unresolved_conflict`、`insufficient_evidence` 或 `other`；无法判断保持为 `insufficient_evidence` 并填写 `uncertainty`。先依据可见原文独立判断，再填动作。`allowed_actions` 可填写多个，例如 `ADD`、`MERGE`、`REPLACE`、`CORRECT`、`RETAIN`、`DEFER`；动作不是数据集自带真值。

记录 `subject`、`attribute`、值、模态及时间时允许为空；不能把“计划”写成“已经发生”，不能因新消息更晚就断言纠正。时间写出原文表述和精度，不确定就说明。完成后将 `status` 改为 `completed`；无法标注可写 `skipped` 并说明原因。`quote` 必须是对应消息中的连续原文，不能用释义冒充引用。

标注完成后，按 `case_id` 与参考文件核对；差异属于待复核，不自动视为标注员或数据集错误。分析时报告抽样偏差、跳过原因、分组数量和人工一致性，不将词面命中或模型输出称作人工标签。

## 测试

```powershell
python -m unittest discover -s annotation_pilot -p 'test_*.py' -v
```
