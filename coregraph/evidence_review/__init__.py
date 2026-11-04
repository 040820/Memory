"""从服务器判断链（实验 01/04）逐字节复制来的纯函数模块（vendor）。

来源（2026-11-04 随提交 20efbdb 同步；文件与源逐字节一致，
将来再同步服务器代码时可直接 diff 对照，请勿本地随意改动）：

- fact_extraction_v2.py, semantic_review_v2.py, semantic_review_v21.py,
  semantic_review_v22.py, longmemeval.py
    <- experiments/01_evidence_grounded_memory_evolution/implementation/evidence_memory/
- normalization.py, pair_retrieval.py, relation_judge.py
    <- experiments/04_fact_attribute_pair_retrieval/src/

只复制了无平台文件锁、不自带 LLM 客户端的模块：
服务器的 StateStore（store.py，Linux 文件锁）留在实验目录，不进主项目。
LLM 调用统一由 coregraph/evidence_bridge 注入，不写死在这些模块里。
"""
