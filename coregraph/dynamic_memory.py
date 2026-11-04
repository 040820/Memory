"""动态记忆图谱（DynamicMemory）模块。

整个 GraphRAG 的"建图中枢"，负责：
1. 会话摘要的生成与增量更新（可选，由配置 retriever.enable_summary 控制）；
2. 从分块（chunks）中抽取实体和关系（对话模式 / 传统模式两种路径）；
3. 实体与关系的去重、三元组生成、图谱构建与增量合并；
4. 证据记忆（EvidenceMemory，V2）：在关系去重前先登记每条原始抽取结果及其来源分片；
5. 实体/关系/会话摘要的向量（embedding）预计算；
6. 图谱与分片存储的 pickle 持久化与加载（支持全量重建 / 加载已有 / 动态追加三种模式）。
"""
from typing import List, Dict, Any
import asyncio
import sys
import os
import pickle
import networkx as nx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from .graph_builder import GraphBuilder
from .evidence_memory import EvidenceMemory, namespace_chunks, chunk_conflicts  # V2: 独立的带来源事实存储
from .entity_extractor import EntityExtractor
from .dialogue_extractor import DialogueExtractor
from .session_summarizer import SessionSummarizer
from utils.time_statistic import GraphBuildingTimeStatistic
from utils.cost_manager import GraphBuildingCostManager

class DynamicMemory:
    """Dynamic Memory Graph for knowledge representation.

    动态记忆图谱：以有向图形式表示从对话/文档中抽取的知识，
    并维护分片存储、实体索引、证据记忆、时间与成本统计等附属状态。
    """

    def __init__(self, config: Config, base_dir: str = "./results"):
        """Initialize dynamic memory.

        初始化动态记忆：各抽取器延迟到 set_extractors 时才创建
        （因为它们依赖 LLM 管理器）。
        """
        self.config = config
        self.base_dir = base_dir
        self.graph_builder = GraphBuilder(config)
        self.entity_extractor = None       # 实体抽取器（依赖 LLM，稍后注入）
        self.dialogue_extractor = None     # 对话抽取器（依赖 LLM，稍后注入）
        self.session_summarizer = None     # 会话摘要器（依赖 LLM，稍后注入）
        self.chunk_processor = None        # 分块处理器（由外部注入）
        self.time_manager = GraphBuildingTimeStatistic()   # 建图各阶段耗时统计
        self.cost_manager = None           # 建图 token/成本统计
        self.chunk_storage = {}            # chunk_id -> 完整 chunk 字典（含 session_time 等元数据）
        self.evidence_memory = EvidenceMemory()
        # V2: 只有本次构建显式开启了证据记忆采集，离线存储才可视为"完整"；
        # 否则按不完整处理，避免下游误以为证据齐全。
        self.evidence_memory.complete = bool(getattr(config.graph, 'enable_evidence_memory', False))
        self.entity_name_to_index = {}     # 实体名 -> 索引 的查询用映射

        logger.info("Dynamic Memory initialized")

    async def generate_session_summaries(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate summaries for all sessions in the documents.

        为语料中出现的所有会话生成摘要：
        1. 按 session_id 聚合文档（同一会话的 context 拼接去重）；
        2. 批量调用 LLM 生成摘要（带进度条，阶段 1）；
        3. 统计摘要生成的 token 消耗并记录耗时；
        4. 摘要保存到 base_dir 下的 session_summaries.json。
        """
        logger.info("Starting session summary generation...")

        self.time_manager.start_summary_generation()
        cost_before_summary = self.cost_manager.get_costs()

        # 按 session_id 聚合：同一会话的多个文档合并为一个"会话"条目
        sessions_map = {}
        for doc in documents:
            session_id = doc.get('session_id', 'unknown')
            if session_id not in sessions_map:
                sessions_map[session_id] = {
                    'session_id': session_id,
                    'session_time': str(doc.get('session_time', 'unknown')),  # 转成字符串避免序列化问题
                    'context': doc.get('context', '')
                }
            else:
                # 同会话新上下文若未出现过，则追加拼接
                existing_context = sessions_map[session_id]['context']
                new_context = doc.get('context', '')
                if new_context and new_context not in existing_context:
                    sessions_map[session_id]['context'] += '\n\n' + new_context
        
        sessions = list(sessions_map.values())
        logger.info(f"Found {len(sessions)} unique sessions to summarize")
        
        # 阶段 1：会话摘要生成 —— 按实际会话数创建进度条
        from tqdm import tqdm
        summary_progress_bar = tqdm(total=len(sessions), desc="Stage 1: Session Summary Generation", unit="calls")
        
        logger.info(f"Stage 1: Processing {len(sessions)} session summaries")
        
        summaries = await self.session_summarizer.summarize_sessions(sessions, progress_bar=summary_progress_bar)
        summary_progress_bar.close()
        # 用"摘要前后的 token 差值"统计摘要生成的消耗
        cost_after_summary = self.llm_manager.cost_manager.get_costs()
        summary_prompt_tokens = cost_after_summary.total_prompt_tokens - cost_before_summary.total_prompt_tokens
        summary_completion_tokens = cost_after_summary.total_completion_tokens - cost_before_summary.total_completion_tokens
        self.cost_manager.update_summary_generation_cost(summary_prompt_tokens, summary_completion_tokens, self.llm_manager.model)
        
        self.time_manager.end_summary_generation()
        
        # 保存摘要到 JSON 文件
        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        self.session_summarizer.save_summaries(summaries, summary_path)
        
        logger.info(f"Generated and saved {len(summaries)} session summaries")
        return summaries

    def set_extractors(self, llm_manager, embedding_manager=None):
        """Set LLM-based extractors and embedding manager.

        注入 LLM 管理器并据此创建各抽取器，同时创建成本统计器。
        必须在建图/抽摘要之前调用。
        """
        self.entity_extractor = EntityExtractor(llm_manager)
        self.relationship_extractor = None
        # 依据配置的数据集类型选择对话抽取提示词（LongmemEval / LOCOMO）
        data_type = getattr(self.config, 'data_type', 'LongmemEval')
        self.dialogue_extractor = DialogueExtractor(llm_manager, data_type=data_type)
        self.session_summarizer = SessionSummarizer(llm_manager)
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager
        # 建图成本统计器与 LLM 管理器共用同一个预算上限
        self.cost_manager = GraphBuildingCostManager(
            max_budget=llm_manager.cost_manager.max_budget
        )
        
        logger.info("Extractors initialized with LLM manager and cost tracking")

    async def build_graph(self, chunks: List[Dict[str, Any]], force: bool = None, add: bool = None) -> None:
        """建图总入口，按模式分发：
        - add=True  ：动态追加模式，先加载已有图，再把新分块增量并入；
        - force=True 或图不存在：从零全量重建；
        - 其他情况  ：仅从 pkl 文件加载已有图谱。
        参数缺省时分别回退到配置项 graph.force / graph.add。
        """

        if force is None:
            force = getattr(self.config.graph, 'force', False)
        if add is None:
            add = getattr(self.config.graph, 'add', False)

        logger.info(f"Graph build mode: force={force}, add={add}")

        graph_exists = hasattr(self, 'graph_builder') and self.graph_builder.graph is not None

        if add:
            # 动态追加模式 —— 总是先尝试加载已有图
            logger.info("🔄 Dynamic addition mode: loading existing graph and adding new chunks")
            try:
                await self._load_existing_graph()
                logger.info("✅ Successfully loaded existing graph")
            except FileNotFoundError:
                # 没有历史图可加载，退化为全量构建
                logger.info("📝 No existing graph found, building from scratch")
                await self._build_graph(chunks)
                return
            await self._dynamic_add_chunks(chunks)
        elif force or not graph_exists:
            # 全量重建模式
            mode_reason = "force rebuild" if force else "graph doesn't exist"
            logger.info(f"🏗️ Build from scratch mode: {mode_reason}")
            await self._build_graph(chunks)
        else:
            # 加载模式 —— 直接从 pkl 文件恢复图谱
            logger.info("📂 Load existing mode: loading graph from pkl file")
            await self._load_existing_graph()

    async def _manage_session_summaries(self, chunks: List[Dict[str, Any]], force: bool) -> None:
        """管理会话摘要文件的生成（仅在配置开启摘要时生效）。

        - force 重建时删除旧摘要文件；
        - 摘要文件缺失或 force 时重新生成；
        - 已存在且非 force 时跳过。
        """
        if getattr(self.config.retriever, 'enable_summary', False):
            summaries_path = os.path.join(self.base_dir, 'session_summaries.json')
            if force and os.path.exists(summaries_path):
                logger.info(f"Removing existing session summaries file due to forced rebuild with summary enabled: {summaries_path}")
                os.remove(summaries_path)
            if force or not os.path.exists(summaries_path):
                logger.info("Generating session summaries at the start of the process")
                await self.generate_session_summaries(chunks)
            else:
                logger.info("Session summaries file already exists, skipping generation")
        else:
            logger.info("Summary generation is disabled, skipping session summaries management")

    async def _load_existing_graph(self) -> None:
        """Load existing graph from pkl file based on base_dir and index_name.

        根据 base_dir 和配置的 index_name 定位 pkl 文件并加载图谱；
        文件不存在时抛出 FileNotFoundError，加载后尝试打印图谱统计。
        """
        try:
            pkl_filename = f"{self.config.index_name}.pkl"
            pkl_path = os.path.join(self.base_dir, pkl_filename)
            
            if not os.path.exists(pkl_path):
                logger.error(f"PKL file not found: {pkl_path}")
                raise FileNotFoundError(f"Graph file not found: {pkl_path}")
            
            logger.info(f"Loading graph from: {pkl_path}")
            self.load_graph(pkl_path)
            
            try:
                stats = self.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            except Exception as stats_error:
                # 加载成功但统计失败不算致命，降级为警告
                logger.warning(f"Could not retrieve graph statistics after loading: {stats_error}")
                logger.info("Graph loaded successfully, but statistics unavailable")
            
        except Exception as e:
            logger.error(f"Failed to load existing graph: {e}")
            raise

    async def _build_graph(self, chunks: List[Dict[str, Any]]) -> None:
        """Internal method to build graph from scratch.

        从零开始全量建图的内部实现，主流程：
        1. 清理旧的 pkl / 备份文件，清空内存中的图与分片存储；
        2. 把原始 chunk 存入 chunk_storage；
        3. 阶段 2：抽取实体与关系（对话模式一次调用同时抽两者；
           传统模式只抽实体，关系列表留空）；
        4. 实体去重、（开启证据记忆时）先登记证据、再关系去重；
        5. 生成三元组并用 GraphBuilder 建图；
        6. 预计算实体/关系/摘要的 embedding；
        7. 记录总耗时并打印建图汇总日志。
        """
        logger.info(f"Starting to build graph from {len(chunks)} chunks")

        self.time_manager.start_total_graph_building()
        # 若尚未创建成本统计器（例如未经 set_extractors），此处补建
        self.cost_manager = GraphBuildingCostManager() if self.cost_manager is None else self.cost_manager

        # force 重建前清理已存在的旧文件
        pkl_filename = f"{self.config.index_name}.pkl"
        pkl_path = os.path.join(self.base_dir, pkl_filename)
        if os.path.exists(pkl_path):
            logger.info(f"Removing existing graph file: {pkl_path}")
            os.remove(pkl_path)
        
        # 同时删除备份文件（若存在）
        backup_path = pkl_path + ".backup"
        if os.path.exists(backup_path):
            logger.info(f"Removing existing backup file: {backup_path}")
            os.remove(backup_path)

        logger.info("Clearing existing graph and chunk storage...")
        self.graph_builder.graph = nx.DiGraph()
        self.chunk_storage.clear()
        self.evidence_memory = EvidenceMemory()  # V2: 全量重建意味着重新开始一份证据存储
        self.evidence_memory.complete = bool(getattr(self.config.graph, 'enable_evidence_memory', False))
        logger.info("✅ Cleared existing graph and chunk storage")

        logger.info("Storing original chunk data...")
        for chunk in chunks:
            # 仅以唯一的 chunk_id 作为存储键
            chunk_id = str(chunk.get('chunk_id', 0))
            # 完整保存 chunk 字典（包括 session_time 等元数据），供查询期溯源
            if chunk.get('text', ''):
                self.chunk_storage[chunk_id] = chunk
                logger.debug(f"Stored chunk {chunk_id}: {len(chunk.get('text', ''))} characters")
        logger.info(f"Stored {len(self.chunk_storage)} chunks in chunk_storage")

        # 阶段 2：实体与关系抽取
        dialogue_mode = getattr(self.config.chunk, 'dialogue_input', False)
        cost_before_add = self.llm_manager.cost_manager.get_costs()
        
        # 按实际 chunk 数创建进度条（每个 chunk 对应一次 LLM 调用）
        from tqdm import tqdm
        
        if dialogue_mode:
            # ---------- 对话模式：一次 LLM 调用同时抽取实体和关系 ----------
            logger.info("Extracting entities and relationships using dialogue mode...")
            cost_before_dialogue = self.llm_manager.cost_manager.get_costs()
            
            dialogue_progress_bar = tqdm(total=len(chunks), desc="Stage 2: Dialogue Entity & Relationship Extraction", unit="calls")
            logger.info(f"Stage 2: Processing {len(chunks)} chunks for entity and relationship extraction")

            self.time_manager.start_entity_extraction()  
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks, progress_bar=dialogue_progress_bar)
            dialogue_progress_bar.close()
            self.time_manager.end_entity_extraction()
            self.time_manager.relationship_extraction_time = 0   # 对话模式中两者合并统计

            # 用前后 token 差值统计本次抽取消耗
            cost_after_dialogue = self.llm_manager.cost_manager.get_costs()
            dialogue_prompt_tokens = cost_after_dialogue.total_prompt_tokens - cost_before_dialogue.total_prompt_tokens
            dialogue_completion_tokens = cost_after_dialogue.total_completion_tokens - cost_before_dialogue.total_completion_tokens

            # 对话模式中实体/关系来自同一次调用，token 消耗对半拆分记账
            entity_tokens = dialogue_prompt_tokens // 2 + dialogue_completion_tokens // 2
            relation_tokens = dialogue_prompt_tokens - (dialogue_prompt_tokens // 2) + dialogue_completion_tokens - (dialogue_completion_tokens // 2)
            
            self.cost_manager.update_entity_extraction_cost(dialogue_prompt_tokens // 2, dialogue_completion_tokens // 2, self.llm_manager.model)
            self.cost_manager.update_relationship_extraction_cost(
                dialogue_prompt_tokens - (dialogue_prompt_tokens // 2), 
                dialogue_completion_tokens - (dialogue_completion_tokens // 2), 
                self.llm_manager.model
            )
            
            # 使用对话模式专属的方法做实体去重
            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            # V2: 在关系去重之前，先把每条原始抽取结果及其来源分片登记进证据记忆
            if getattr(self.config.graph, 'enable_evidence_memory', False):
                self._capture_evidence_relationships(chunks, relationships, entity_mapping)
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )
            
            logger.info(f"Dialogue mode - After deduplication: {len(entities)} entities, {len(relationships)} relationships")
        else:
            # ---------- 传统模式：仅抽取实体，关系抽取已弃用 ----------
            logger.info("Extracting entities...")

            cost_before_entity = self.llm_manager.cost_manager.get_costs()
            
            progress_bar = tqdm(total=len(chunks), desc="Stage 2: Entity Extraction", unit="calls")
            logger.info(f"Stage 2: Processing {len(chunks)} chunks for entity extraction")
            
            self.time_manager.start_entity_extraction()
            entities = await self.entity_extractor.extract_from_chunks(chunks, progress_bar=progress_bar)
            progress_bar.close()
            self.time_manager.end_entity_extraction()

            cost_after_entity = self.llm_manager.cost_manager.get_costs()
            entity_prompt_tokens = cost_after_entity.total_prompt_tokens - cost_before_entity.total_prompt_tokens
            entity_completion_tokens = cost_after_entity.total_completion_tokens - cost_before_entity.total_completion_tokens
            self.cost_manager.update_entity_extraction_cost(entity_prompt_tokens, entity_completion_tokens, self.llm_manager.model)
            
            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )

            relationships = []
            logger.info("Skipping relationship extraction (unused in dialogue mode)")

        # 生成（源实体、目标实体、关系）三元组列表
        triples = self._create_triples(entities, relationships, chunks)

        # 直接用去重后的实体与关系建图
        self.time_manager.start_graph_construction()
        self.graph_builder.build_from_entities_and_relationships(entities, relationships)
        self.time_manager.end_graph_construction()
        
        # 构建实体名 -> 索引 的映射，供查询阶段使用
        self.entity_name_to_index = {entity.get('entity', ''): idx for idx, entity in enumerate(entities)}
        
        # 预计算实体、关系和摘要的 embedding 并存入图结构，加快查询期检索
        await self._precompute_embeddings(entities, relationships)
        
        self.time_manager.end_total_graph_building()
        self._log_graph_building_summary()
        logger.info("Graph built successfully")

    def _log_graph_building_summary(self):
        """打印建图汇总日志：各阶段耗时与 token 成本统计。"""
        logger.info("=" * 80)
        logger.info("📊 GRAPH BUILDING SUMMARY")
        logger.info("=" * 80)
        time_summary = self.time_manager.get_graph_building_summary()
        logger.info(f"⏱️  TIME STATISTICS:")
        logger.info(f"   📄 Chunking Time: {time_summary['chunking_time']}s")
        logger.info(f"   🏷️  Entity and Relation Extraction Time: {time_summary['entity_extraction_time']}s")
        logger.info(f"   🕸️  Graph Construction Time: {time_summary['graph_construction_time']}s")
        logger.info(f"   📝 Summary Generation Time: {time_summary['summary_generation_time']}s")
        logger.info(f"   📊 Total Graph Building Time: {time_summary['total_graph_building_time']}s")

        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_graph_building_summary()
            logger.info(f"💰 COST STATISTICS:")
            total_extraction_tokens = cost_summary['entity_extraction_tokens'] + cost_summary['relationship_extraction_tokens']
            logger.info(f"   🏷️  Entity and Relation Extraction Tokens: {total_extraction_tokens}")
            logger.info(f"   📝 Summary Generation Tokens: {cost_summary['summary_generation_tokens']}")
            logger.info(f"   📊 Total Graph Building Tokens: {cost_summary['total_graph_building_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']}")
        
        logger.info("=" * 80)

    def _log_dynamic_add_summary(self):
        """打印动态追加模式的汇总日志（token 成本统计）。"""
        logger.info("=" * 80)
        logger.info("📊 DYNAMIC ADDITION SUMMARY")
        logger.info("=" * 80)
        
        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_graph_building_summary()
            total_extraction_tokens = cost_summary['entity_extraction_tokens'] + cost_summary['relationship_extraction_tokens']
            logger.info(f"💰 COST STATISTICS:")
            logger.info(f"   🏷️  Entity and Relation Extraction Tokens: {total_extraction_tokens}")
            logger.info(f"   📝 Summary Update Tokens: {cost_summary['summary_generation_tokens']}")
            logger.info(f"   📊 Total Graph Building Tokens: {cost_summary['total_graph_building_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']}")
        
        logger.info("=" * 80)
    
    async def _update_summary_for_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update or create summary for a single session.

        更新或新建单个会话的摘要。
        
        Args:
            session_data: 会话数据，包含 session_id、session_time 与 context
            
        Returns:
            更新后的或新建的摘要
        """
        session_id = session_data.get('session_id', 'unknown')
        
        # 加载已存在的摘要文件
        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        if os.path.exists(summary_path):
            existing_summaries = self.session_summarizer.load_summaries(summary_path)
        else:
            existing_summaries = []
        
        # 查找该会话是否已有摘要
        existing_summary = None
        for summary in existing_summaries:
            if summary.get('session_id') == session_id:
                existing_summary = summary
                break
        
        # 有旧摘要则增量更新（ADDITION_PROMPT），否则新建（SUMMARY_PROMPT）
        if existing_summary:
            logger.info(f"📝 Updating existing summary for session: {session_id} using ADDITION_PROMPT")
            # 构造一个"伪 chunk"用于增量更新
            chunk_data = {
                'session_id': session_id,
                'session_time': session_data.get('session_time', 'unknown'),
                'text': session_data.get('context', '')
            }
            updated_summary = await self.session_summarizer.update_summary_with_chunk(chunk_data, existing_summary)
        else:
            logger.info(f"📝 Creating new summary for session: {session_id} using SUMMARY_PROMPT")
            updated_summary = await self.session_summarizer.summarize_session(session_data)
        
        # 保存全部摘要（与已有摘要合并写回）
        all_summaries_dict = {s.get('session_id', ''): s for s in existing_summaries}
        all_summaries_dict[session_id] = updated_summary
        all_summaries = list(all_summaries_dict.values())
        self.session_summarizer.save_summaries(all_summaries, summary_path)
        
        return updated_summary

    async def add_single_session(self, session_corpus: List[Dict[str, Any]]) -> None:
        """Add a single session to the graph incrementally.

        以增量方式把单个会话加入图谱（add=True 时逐会话调用本方法）。
        对每个会话依次执行：
        1. 检查摘要文件中是否已有该 session_id 的摘要；
        2. 有则增量更新（ADDITION_PROMPT），无则新建（SUMMARY_PROMPT）；
        3. 从该会话抽取实体和关系；
        4. 去重后增量合并进已有图谱。
        
        Args:
            session_corpus: 属于同一个会话的文档列表
        """
        if not session_corpus:
            logger.warning("Empty session corpus, skipping")
            return
        
        # 从第一个文档中取出会话信息
        session_id = session_corpus[0].get('session_id', 'unknown')
        session_time = session_corpus[0].get('session_time', 'unknown')
        
        logger.info("=" * 80)
        logger.info(f"🔄 Processing session: {session_id} (time: {session_time})")
        logger.info("=" * 80)
        
        # 步骤 1：先把会话文档处理成 chunk 并做冲突校验（纯计算 + 只读检查）。
        # 必须放在摘要更新之前：若这里失败，摘要等副作用尚未发生，
        # 不会留下"摘要已改、图和事实没写入"的不一致状态。
        chunks = self.chunk_processor.process_corpus(session_corpus)
        if getattr(self.config.graph, 'enable_evidence_memory', False):
            # V2: 按会话对 chunk_id 命名空间化，避免跨会话 ID 互相覆盖
            chunks = namespace_chunks(chunks)
            self._check_chunk_collisions(chunks)
        logger.info(f"Processed {len(chunks)} chunks from session {session_id}")

        # 步骤 2：更新或新建该会话的摘要（校验通过后才产生副作用）
        if getattr(self.config.retriever, 'enable_summary', False):
            cost_before_summary = self.llm_manager.cost_manager.get_costs()

            # 组装摘要器需要的会话数据
            session_data = {
                'session_id': session_id,
                'session_time': str(session_time),
                'context': '\n\n'.join([doc.get('context', '') for doc in session_corpus])
            }

            await self._update_summary_for_session(session_data)

            # 记录摘要更新的 token 消耗
            cost_after_summary = self.llm_manager.cost_manager.get_costs()
            summary_prompt_tokens = cost_after_summary.total_prompt_tokens - cost_before_summary.total_prompt_tokens
            summary_completion_tokens = cost_after_summary.total_completion_tokens - cost_before_summary.total_completion_tokens
            self.cost_manager.update_summary_generation_cost(
                summary_prompt_tokens,
                summary_completion_tokens,
                self.llm_manager.model
            )

        # 步骤 3：存入分片存储
        for chunk in chunks:
            chunk_id = str(chunk.get('chunk_id', 0))
            if chunk.get('text', ''):
                self.chunk_storage[chunk_id] = chunk
        
        # 步骤 4：抽取实体和关系
        cost_before_extract = self.llm_manager.cost_manager.get_costs()
        
        dialogue_mode = getattr(self.config.chunk, 'dialogue_input', False)
        if dialogue_mode:
            logger.info("Extracting entities and relationships using dialogue mode...")
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks)
            
            # 步骤 5：去重
            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            # V2: 在关系去重之前登记每条原始抽取结果及其来源分片
            if getattr(self.config.graph, 'enable_evidence_memory', False):
                self._capture_evidence_relationships(chunks, relationships, entity_mapping)
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )
            
            logger.info(f"Extracted and deduplicated: {len(entities)} entities, {len(relationships)} relationships")
        else:
            logger.info("Extracting entities...")
            entities = await self.entity_extractor.extract_from_chunks(chunks)
            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            relationships = []
            logger.info(f"Extracted {len(entities)} unique entities")
        
        # 记录抽取阶段的 token 消耗（对话模式对半拆分为实体/关系两笔）
        cost_after_extract = self.llm_manager.cost_manager.get_costs()
        extract_prompt_tokens = cost_after_extract.total_prompt_tokens - cost_before_extract.total_prompt_tokens
        extract_completion_tokens = cost_after_extract.total_completion_tokens - cost_before_extract.total_completion_tokens
        
        self.cost_manager.update_entity_extraction_cost(
            extract_prompt_tokens // 2, 
            extract_completion_tokens // 2, 
            self.llm_manager.model
        )
        self.cost_manager.update_relationship_extraction_cost(
            extract_prompt_tokens - (extract_prompt_tokens // 2), 
            extract_completion_tokens - (extract_completion_tokens // 2), 
            self.llm_manager.model
        )
        
        # 步骤 6：把实体和关系增量合并进图谱
        logger.info("Adding entities and relationships to graph...")
        self.graph_builder.add_entities_and_relationships_incrementally(entities, relationships)
        
        # 步骤 7：更新实体索引（已存在的实体名保持原索引不变）
        for entity in entities:
            entity_name = entity.get('entity', '')
            if entity_name and entity_name not in self.entity_name_to_index:
                self.entity_name_to_index[entity_name] = len(self.entity_name_to_index)
        
        logger.info(f"✅ Session {session_id} processed successfully")
        logger.info("=" * 80)

    async def _dynamic_add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Dynamically add new chunks to existing graph.

        向已有图谱动态追加新分块。
        
        已弃用（DEPRECATED）：保留此方法只为向后兼容，
        新代码应使用 add_single_session() 按会话逐个处理。
        """
        logger.info(f"Dynamically adding {len(chunks)} new chunks to existing graph")

        # V2: 按会话对 chunk_id 命名空间化，防止跨会话 ID 互相覆盖
        if getattr(self.config.graph, 'enable_evidence_memory', False):
            chunks = namespace_chunks(chunks)
            self._check_chunk_collisions(chunks)

        logger.info("Storing new chunk data to chunk_storage...")
        for chunk in chunks:
            # 仅以唯一的 chunk_id 作为存储键
            chunk_id = str(chunk.get('chunk_id', 0))
            # 完整保存 chunk 字典（包括 session_time 等元数据）
            if chunk.get('text', ''):
                self.chunk_storage[chunk_id] = chunk
                logger.debug(f"Stored new chunk {chunk_id}: {len(chunk.get('text', ''))} characters")
        logger.info(f"Stored {len(chunks)} new chunks in chunk_storage")

        dialogue_mode = getattr(self.config.chunk, 'dialogue_input', False)
        cost_before_add = self.llm_manager.cost_manager.get_costs()
        
        if dialogue_mode:
            logger.info("Extracting entities and relationships from new chunks using dialogue mode...")
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks)
            
            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            # V2: 在关系去重之前登记每条原始抽取结果及其来源分片
            if getattr(self.config.graph, 'enable_evidence_memory', False):
                self._capture_evidence_relationships(chunks, relationships, entity_mapping)
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )
            
            logger.info(f"Dialogue mode - Extracted {len(entities)} entities and {len(relationships)} relationships from new chunks")
        else:
            # 传统模式：仅抽取实体
            logger.info("Extracting entities from new chunks...")
            entities = await self.entity_extractor.extract_from_chunks(chunks)
            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            logger.info(f"Extracted {len(entities)} unique entities from new chunks")
            
            # 关系抽取已弃用（仅实体抽取器可用）
            relationships = []
            logger.info("Skipping relationship extraction from new chunks (unused in dialogue mode)")

        # 生成三元组
        triples = self._create_triples(entities, relationships, chunks)
        logger.info(f"Created {len(triples)} triples from new chunks")

        # 把新实体和关系增量合并进已有图谱
        logger.info("Adding new entities and relationships to existing graph incrementally...")
        self.graph_builder.add_entities_and_relationships_incrementally(entities, relationships)
        
        # 用新实体更新实体索引
        start_idx = len(self.entity_name_to_index)
        for idx, entity in enumerate(entities):
            entity_name = entity.get('entity', '')
            if entity_name and entity_name not in self.entity_name_to_index:
                self.entity_name_to_index[entity_name] = start_idx + idx
        logger.info(f"Updated entity index, now has {len(self.entity_name_to_index)} entities")

        # 记录追加阶段的 token 消耗
        cost_after_add = self.llm_manager.cost_manager.get_costs()
        add_prompt_tokens = cost_after_add.total_prompt_tokens - cost_before_add.total_prompt_tokens
        add_completion_tokens = cost_after_add.total_completion_tokens - cost_before_add.total_completion_tokens

        self.cost_manager.update_entity_extraction_cost(add_prompt_tokens // 2, add_completion_tokens // 2, self.llm_manager.model)
        self.cost_manager.update_relationship_extraction_cost(
            add_prompt_tokens - (add_prompt_tokens // 2), 
            add_completion_tokens - (add_completion_tokens // 2), 
            self.llm_manager.model
        )
        self._log_dynamic_add_summary()
        logger.info("Dynamic addition completed")

    def _check_chunk_collisions(self, chunks):
        """V2: reject only genuine payload conflicts.

        V2 冲突检查：只拒绝"真正的载荷冲突"。
        source_chunk_id 是与运行方式相关的（全批处理时按语料顺序编号，
        按会话处理时每个会话从 0 重新编号），因此不参与身份比较；
        同一会话在不同处理模式下重放不会被误判为冲突（幂等）。
        """
        for chunk in chunks:
            prior = self.chunk_storage.get(str(chunk['chunk_id']))
            if prior is not None and chunk_conflicts(prior, chunk):
                raise ValueError('V2 chunk identity collision; refusing to overwrite')

    def _capture_evidence_relationships(self, chunks, relationships, entity_mapping):
        """V2: retain raw extraction candidates; no claim of evidence support.

        V2 证据采集：在关系去重之前，把本次抽取的原始候选逐条
        连同其来源分片登记进证据记忆。此处只做"留存"，不主张
        任何证据支持关系（支持与否由后续审查决定）。

        - 先为每个 chunk 在证据库中登记并建立 raw_id -> 本地存储 ID 的映射；
        - 再为每条关系补充实体规范化后的 canonical_src/canonical_tgt，
          并以来源分片 ID 为锚写入事实。
        """
        source_by_id = {}
        for chunk in chunks:
            raw_id = str(chunk.get('chunk_id'))
            if raw_id in source_by_id:
                raise ValueError('ambiguous chunk id in extraction batch')
            source_by_id[raw_id] = self.evidence_memory.add_chunk(chunk)
        for relationship in relationships:
            source_id = source_by_id.get(str(relationship.get('chunk_id')))
            if source_id is None:
                raise ValueError('relationship has no source chunk in extraction batch')
            raw = dict(relationship)
            # 记录实体去重后的规范名，便于后续按实体检索原始事实
            raw['canonical_src'] = entity_mapping.get(raw.get('src'), raw.get('src', ''))
            raw['canonical_tgt'] = entity_mapping.get(raw.get('tgt'), raw.get('tgt', ''))
            self.evidence_memory.add_fact(raw, source_id)

    def _create_triples(self, entities: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]],
                       chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create triples from entities and relationships.

        依据实体与关系生成三元组列表：
        先建立"实体名 -> 出现过的 chunk_id 列表"映射，
        再为每条关系补充源/目标实体的类型，输出标准三元组结构。
        """
        triples = []

        # 建立 实体名 -> chunk_id 列表 的映射
        entity_chunks = {}
        for entity in entities:
            entity_name = entity.get('entity', '')
            chunk_id = entity.get('chunk_id', 0)
            if entity_name not in entity_chunks:
                entity_chunks[entity_name] = []
            entity_chunks[entity_name].append(chunk_id)

        for relationship in relationships:
            src = relationship.get('src', '')
            tgt = relationship.get('tgt', '')
            relation = relationship.get('relation', '')
            chunk_id = relationship.get('chunk_id', 0)

            # 查找源/目标实体的类型
            src_type = self._get_entity_type(entities, src)
            tgt_type = self._get_entity_type(entities, tgt)

            triple = {
                'src': src,
                'tgt': tgt,
                'src_type': src_type,
                'tgt_type': tgt_type,
                'relation': relation,
                'chunk_id': chunk_id
            }
            triples.append(triple)

        return triples

    def _get_entity_type(self, entities: List[Dict[str, Any]], entity_name: str) -> str:
        """Get entity type from entities list.

        在实体列表中查找指定实体名的类型；找不到时返回 'unknown'。
        """
        for entity in entities:
            if entity.get('entity', '') == entity_name:
                return entity.get('type', 'unknown')
        return 'unknown'
    
    async def _precompute_embeddings(self, entities: List[Dict[str, Any]], 
                                     relationships: List[Dict[str, Any]]) -> None:
        """
        Pre-compute embeddings for all entities, relationships, and summaries.
        Store them in the graph structure for fast retrieval during query time.

        预计算所有实体、关系和会话摘要的向量表示，
        存入图结构中，使查询期无需再实时调用 embedding 服务。
        任一子步骤失败只记录日志，不中断整体流程。
        """
        if not hasattr(self, 'embedding_manager') or not self.embedding_manager:
            logger.warning("⚠️  Embedding manager not available, skipping embedding precomputation")
            return
        
        import time
        start_time = time.time()
        
        try:
            await self._precompute_entity_embeddings(entities)
            await self._precompute_relationship_embeddings(relationships)
            await self._precompute_summary_embeddings()     
            elapsed = time.time() - start_time
            logger.info(f"Embedding precomputation completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error during embedding precomputation: {e}")
            import traceback
            traceback.print_exc()
    
    async def _precompute_entity_embeddings(self, entities: List[Dict[str, Any]]) -> None:
        """预计算全部实体的 embedding 并写入对应图节点。

        向量文本格式为 "实体名 (类型): 描述"（无描述时省略描述部分），
        采用批量请求以大幅减少耗时。
        """
        if not entities:
            return
        
        # 准备用于计算 embedding 的实体文本
        entity_texts = []
        entity_names = []
        for entity in entities:
            entity_name = entity.get('entity', '')
            entity_type = entity.get('type', 'unknown')
            description = entity.get('description', '')
            
            # 拼接实体文本：名称 + 类型 + 描述
            if description:
                entity_text = f"{entity_name} ({entity_type}): {description}"
            else:
                entity_text = f"{entity_name} ({entity_type})"
            
            entity_texts.append(entity_text)
            entity_names.append(entity_name)
        
        # 批量获取 embedding（远快于逐条请求）
        embeddings = await self.embedding_manager.get_embeddings(entity_texts)
        
        if not embeddings or len(embeddings) != len(entity_texts):
            logger.error(f"Failed to get embeddings for entities (got {len(embeddings) if embeddings else 0}, expected {len(entity_texts)})")
            return
        
        # 将 embedding 存入对应图节点
        for entity_name, embedding in zip(entity_names, embeddings):
            if entity_name in self.graph_builder.graph.nodes:
                self.graph_builder.graph.nodes[entity_name]['embedding'] = embedding
    
    async def _precompute_relationship_embeddings(self, relationships: List[Dict[str, Any]]) -> None:
        """Precompute embeddings for all relationships (triples) and store in graph edges.

        预计算全部关系（三元组）的 embedding 并写入对应图的边。
        向量文本格式为 "源 - 关系 - 目标: 描述"。
        """
        if not relationships:
            return
        
        # 准备用于计算 embedding 的三元组文本
        triple_texts = []
        triple_keys = []
        
        for rel in relationships:
            src = rel.get('src', '')
            tgt = rel.get('tgt', '')
            relation = rel.get('relation', '')
            description = rel.get('description', '')
            
            if description:
                triple_text = f"{src} - {relation} - {tgt}: {description}"
            else:
                triple_text = f"{src} - {relation} - {tgt}"
            
            triple_texts.append(triple_text)
            triple_keys.append((src, tgt))
        
        embeddings = await self.embedding_manager.get_embeddings(triple_texts)
        
        if not embeddings or len(embeddings) != len(triple_texts):
            logger.error(f"Failed to get embeddings for relationships (got {len(embeddings) if embeddings else 0}, expected {len(triple_texts)})")
            return
        
        for (src, tgt), embedding in zip(triple_keys, embeddings):
            if self.graph_builder.graph.has_edge(src, tgt):
                self.graph_builder.graph.edges[src, tgt]['embedding'] = embedding
    
    async def _precompute_summary_embeddings(self) -> None:
        """Precompute embeddings for session summaries and store in summary file.

        预计算各会话摘要的 embedding 并直接写回摘要 JSON 文件，
        供查询期做"摘要级"语义检索。
        """
        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        
        if not os.path.exists(summary_path):
            return
        
        import json
        with open(summary_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
        
        if not summaries:
            return
        
        summary_texts = []
        for summary in summaries:
            summary_text = summary.get('summary', '')
            session_id = summary.get('session_id', '')
            
            full_text = f"Session {session_id}: {summary_text}"
            summary_texts.append(full_text)
        
        embeddings = await self.embedding_manager.get_embeddings(summary_texts)
        
        if not embeddings or len(embeddings) != len(summary_texts):
            logger.error(f"Failed to get embeddings for summaries (got {len(embeddings) if embeddings else 0}, expected {len(summary_texts)})")
            return
        
        for summary, embedding in zip(summaries, embeddings):
            summary['embedding'] = embedding
        
        # 写回摘要文件（保留原文本并附上向量）
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics.

        获取图谱统计信息（节点数、边数、类型分布等），委托给 GraphBuilder。
        """
        stats = self.graph_builder.get_graph_stats()
        return stats

    def save_graph(self, path: str) -> None:
        """Save graph and chunk storage to file.

        将图谱、分片存储、实体索引与证据记忆（V2）一起序列化为 pickle 文件。
        """
        graph_data = {
            'graph': self.graph_builder.graph,
            'chunk_storage': self.chunk_storage,
            'entity_name_to_index': self.entity_name_to_index,
            'evidence_memory_v2': self.evidence_memory.to_dict(),
        }
        with open(path, 'wb') as f:
            pickle.dump(graph_data, f)
        logger.info(f"Graph and chunk storage saved to {path}")

    def load_graph(self, path: str) -> None:
        """Load graph and chunk storage from file.

        从 pickle 文件恢复图谱状态。兼容两种历史格式：
        - 新格式（dict）：包含图、分片存储、实体索引和 V2 证据记忆；
          缺少 evidence_memory_v2 时（旧版合并过的事实）无法还原原始证据，
          complete 标记为 False；
        - 旧格式（裸图对象）：仅恢复图本身，其余状态置空。
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
            if isinstance(data, dict):
                self.graph_builder.graph = data.get('graph')
                self.chunk_storage = data.get('chunk_storage', {})
                self.entity_name_to_index = data.get('entity_name_to_index', {})
                if 'evidence_memory_v2' in data:
                    self.evidence_memory = EvidenceMemory.from_dict(data['evidence_memory_v2'])
                else:
                    # 旧版证据记忆为合并后的事实，原始证据无法恢复
                    self.evidence_memory = EvidenceMemory()
                    self.evidence_memory.complete = False  # Legacy merged facts cannot be restored.
                logger.info(f"Graph loaded from {path} with {len(self.chunk_storage)} chunks and {len(self.entity_name_to_index)} entities")
            else:
                # 兼容最老的"仅保存图对象"格式
                self.graph_builder.graph = data
                self.chunk_storage = {}
                self.entity_name_to_index = {}
                self.evidence_memory = EvidenceMemory()
                self.evidence_memory.complete = False
                logger.info(f"Graph loaded from {path} (old format, no chunk storage or entity index)")
        logger.info(f"Graph loaded from {path}")
