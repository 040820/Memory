# 离线实验准备

先阅读 [实验方案.md](实验方案.md)。该目录当前只有固定的开发样本清单、受控输入/评测期望、校验器和测试；**不是一次已完成的模型实验**。

```powershell
conda run -n licomemory python -m unittest discover -s experiments/evidence_memory_pilot -p 'test_*.py' -v
conda run -n licomemory python experiments/evidence_memory_pilot/validate_cases.py
```

`controlled_inputs.json` 是可供系统读取的内容；`controlled_gold.json` 是独立评测数据，不能交给构图、检索、Judge 或 Reader。校验器只检查数据格式、ID 和引用存在，不判断语义正确性，也不对系统打分。期望结果仍需人工复核。

## 本地连接远程模型（实验适配，尚未完成端到端运行）

`remote_embedding.py` 使用 OpenAI 兼容 `/v1/embeddings`，检验数量、索引、维度及有限数值，错误直接抛出；默认绕过系统代理访问显式配置的服务地址。`run_local_baseline.py` 仅在当前进程临时替换现有 `EmbeddingManager` 的初始化及编码方法，保留原余弦相似度方法，不修改源项目文件。[baseline_qwen.yaml](baseline_qwen.yaml) 包含服务地址和模型名；可用 `LICO_EMBEDDING_URL`、`LICO_EMBEDDING_MODEL` 临时覆盖 embedding 地址和模型名，服务需要认证时可用 `LICO_EMBEDDING_API_KEY` 环境变量。LLM 地址和模型名以配置文件为准。

启动器针对配置的 LLM 模型在每次调用中加入 `chat_template_kwargs.enable_thinking=false`，包括构图、查询及启用时的评测 LLM 调用。仅直接调用原项目 `main.py` 不会启用该实验适配。`llm.temperature` 和 `llm.max_token` 等配置不保证全部实际生效：原 `LLMManager._generate_internal` 使用每次调用的参数与默认值，正式实验需记录实际请求参数。

已确认当前服务为 Qwen3-8B 与 bge-m3，后者返回 1024 维；**当前 Conda 环境缺少项目依赖，尚未通过原始 main 的启动和端到端检查**。在项目根目录使用 `conda activate licomemory` 后执行 `python -m pip install -r requirements.txt`。这会安装到当前 Conda 环境，PyTorch 如果已有兼容版本会被复用；受网络、CUDA、驱动及环境现状影响仍需预检。示例配置中 `data_root` 指向尚需生成的处理后数据目录，并非原始 JSON 路径。不要将 `demo-key` 作为 LLM key，因为原代码会将其判为模拟模式。不要将密码或密钥写入配置和产物。
