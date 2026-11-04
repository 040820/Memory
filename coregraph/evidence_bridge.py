"""V2 桥接层：把服务器判断链（实验 01–04）的输出写入本地证据记忆（05–07）。

三条桥，全部影子运行（从不提交状态、从不读取标准答案 gold）：

1. ingest_sources   -- LongMemEval 风格的轮次 -> 命名空间 chunk + 抽取的事实候选
                       （五维限定词 time/modality/polarity/precision/scope 保留在 raw 里）
2. review_fact      -- semantic_review_v22 审查结论 -> record_review 写到单个事实上
                       （support_status + 只追加的审查历史；fact 身份不变）
3. 配对/关系判断     -- 词面归一键配对 + relation_judge 关系提议 -> add_change
                       （candidate 状态、decision=pending、review_id 重试幂等）

ID 权威：本地 fact_id = hash(src, relation, tgt, 本地 evidence_id) 为准；
服务器 evidence_id 降级保存在 chunk['server_evidence_id'] 和 raw['evidence_id'] 中，
仅作溯源，两套 ID 并存不互相替换。

LLM 传输以 async callable(payload dict) -> 原始字符串 注入：
测试里用 Fake 响应即可全链路离线跑，真实模型（线上/服务器 vLLM）后续接入。
"""
import hashlib
import json

from .evidence_memory import namespace_chunks
from .evidence_review import (fact_extraction_v2, longmemeval, normalization,
                              pair_retrieval, relation_judge,
                              semantic_review_v22)


def server_evidence_id(source):
    """服务器侧证据 ID，公式与实验 01 的 EvidenceStore.put_source 完全一致（stable_id('ev', ...)）。

    由 (dataset_id, conversation_id, session_id, turn_id) 哈希得到，
    保证同一条轮次在重放时得到相同的服务器 ID。
    """
    try:
        values = [str(source[k]) for k in
                  ('dataset_id', 'conversation_id', 'session_id', 'turn_id')]
    except KeyError as exc:
        raise ValueError(f'missing identity field for evidence id: {exc}') from None
    blob = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return 'ev_' + hashlib.sha256(blob.encode('utf8')).hexdigest()


class EvidenceBridge:
    def __init__(self, store, call):
        """store: 本地证据库 EvidenceMemory；call: 异步 LLM 传输 (payload dict) -> 模型原始响应字符串。"""
        self.store = store
        self.call = call

    # ------------------------------------------------------------- 桥 1：抽取写入
    async def ingest_sources(self, sources, *, model, max_tokens=1000):
        """轮次来源 -> 命名空间 chunk + 用户轮次的事实候选（全部 unverified）。

        流程：补全服务器 evidence_id -> 逐轮建成 chunk（命名空间防跨会话覆盖）
              -> 仅对 user 轮次调抽取 -> 词表映射后 add_fact（重复重放幂等）。
        """
        if not isinstance(sources, list) or not sources:
            raise ValueError('sources required')
        prepared = []
        for src in sources:
            item = dict(src)
            if not isinstance(item.get('session_id'), str) or not item['session_id'].strip():
                raise ValueError('stable session_id required')
            if not isinstance(item.get('text'), str) or not item['text'].strip():
                raise ValueError('turn text required')
            item['turn_id'] = str(item.get('turn_id', len(prepared)))
            item['server_evidence_id'] = item.get('evidence_id') or server_evidence_id(item)
            prepared.append(item)

        chunks = [{'session_id': s['session_id'], 'chunk_id': s['turn_id'],
                   'text': s['text'], 'speaker': s.get('speaker', ''),
                   'server_evidence_id': s['server_evidence_id']}
                  for s in prepared]
        for key in ('raw_session_time', 'known_at', 'time_precision',
                    'timezone', 'known_at_basis', 'dataset_id',
                    'conversation_id'):
            if key in prepared[0]:
                for chunk, src in zip(chunks, prepared):
                    if key in src:
                        chunk[key] = src[key]
        chunks = namespace_chunks(chunks)
        local_ids = [self.store.add_chunk(chunk) for chunk in chunks]
        # add_chunk's store key is hash(sid, namespaced chunk_id), not the
        # payload's chunk_id field -- facts must reference the store key.
        chunk_by_server = {}
        for store_id in local_ids:
            data = self.store.chunks[store_id]
            if data.get('server_evidence_id'):
                chunk_by_server[data['server_evidence_id']] = store_id

        facts, skipped = [], []
        for src, fallback_id in zip(prepared, local_ids):
            if src.get('speaker') != 'user':
                continue
            review_source = dict(src)
            review_source['evidence_id'] = src['server_evidence_id']
            payload = fact_extraction_v2.create_payload(
                review_source, model=model, max_tokens=max_tokens)
            raw = await self.call(payload)
            outcome = fact_extraction_v2.parse_candidates(raw, review_source)
            if outcome['status'] != 'parsed':
                skipped.append({'server_evidence_id': src['server_evidence_id'],
                                'reason': outcome['status']})
                continue
            local_id = chunk_by_server.get(src['server_evidence_id'], fallback_id)
            for candidate in outcome['candidates']:
                try:
                    # store vocabulary is src/relation/tgt; the extraction
                    # vocabulary (subject/predicate/value) stays in raw too
                    item = dict(candidate)
                    item.setdefault('src', candidate.get('subject', ''))
                    item.setdefault('relation', candidate.get('predicate', ''))
                    item.setdefault('tgt', candidate.get('value', ''))
                    fid = self.store.add_fact(item, local_id)
                    facts.append(fid)
                except ValueError as exc:        # missing evidence / bad fields
                    skipped.append({'candidate': candidate, 'reason': str(exc)})
            for item in outcome['skipped']:
                skipped.append(item)
        return {'chunk_ids': local_ids,
                'fact_ids': facts, 'skipped': skipped}

    # ------------------------------------------------------------- 桥 2a：单候选审查
    def _source_for(self, fact):
        """构造审查提示词需要的"服务器格式"来源 dict（用服务器 evidence_id，与抽取时一致）。"""
        chunk = self.store.chunks[fact['evidence_id']]
        raw = fact['raw']
        return {'evidence_id': raw.get('evidence_id') or chunk.get('server_evidence_id'),
                'speaker': chunk.get('speaker', ''),
                'text': chunk['text']}

    async def review_fact(self, fact_id, *, model, max_tokens=1000, review_id=None):
        """对单个已存事实跑 semantic_review_v22 -> record_review（影子，不改状态）。

        路由 DEFER / REVIEW_NEXT / TECHNICAL_ERROR；applied 恒为 False。
        review_id 为重试幂等键：同 ID 同结论返回原记录，同 ID 异结论报错。
        """
        if fact_id not in self.store.facts:
            raise ValueError('unknown fact')
        fact = self.store.facts[fact_id]
        source = self._source_for(fact)
        candidate = fact['raw']
        payload = semantic_review_v22.review_payload(
            source, candidate, model=model, max_tokens=max_tokens)
        raw = await self.call(payload)
        result = semantic_review_v22.parse_and_route(raw, source, candidate)
        result['applied'] = False               # shadow: never a state change
        record = self.store.record_review(
            fact_id, result['route'], result['reasons'] or ['UNSPECIFIED_ERROR'],
            review_id=review_id)
        result['review_record'] = record
        return result

    # ------------------------------------------------------------- 桥 2b：配对 + 关系
    def pairing_views(self):
        """从已存事实派生配对检索键（lexical_v1_dev 规范化）；未知保持 UNKNOWN，不猜。"""
        views = []
        for fid, fact in self.store.facts.items():
            chunk = self.store.chunks[fact['evidence_id']]
            raw = fact['raw']
            object_key = str(raw.get('object_key') or 'UNKNOWN').strip() or 'UNKNOWN'
            attribute = normalization.normalize_attribute(
                str(raw.get('predicate') or ''), object_key)
            turn_raw = chunk.get('source_chunk_id')
            try:
                turn_index = int(turn_raw)
            except (TypeError, ValueError):
                turn_index = None
            views.append({
                'fact_id': fid,
                'evidence_id': fact['evidence_id'],       # local chunk id
                'subject_key': 'user' if raw.get('subject') == 'user' else 'UNKNOWN',
                'attribute_key': attribute['key'],
                'object_key': object_key,
                'scope_key': str(raw.get('scope') or 'UNKNOWN').strip() or 'UNKNOWN',
                'value': raw.get('value', ''),
                'quote': raw.get('quote', ''),
                'polarity': raw.get('polarity', 'UNKNOWN'),
                'modality': raw.get('modality', 'UNKNOWN'),
                'session_id': chunk.get('session_id', ''),
                'session_time': chunk.get('raw_session_time'),
                'turn_index': turn_index,
            })
        return views

    def _chunk_source(self, chunk_id):
        chunk = self.store.chunks[chunk_id]
        return {'evidence_id': chunk_id, 'speaker': chunk.get('speaker', ''),
                'text': chunk['text'], 'session_time': chunk.get('raw_session_time')}

    def candidate_pairs(self, *, max_candidates=3):
        views = self.pairing_views()
        by_id = {v['fact_id']: v for v in views}
        outcome = pair_retrieval.retrieve_pairs(views, max_candidates=max_candidates)
        outcome['views'] = by_id
        return outcome

    async def judge_and_record(self, pair, *, views, model, max_tokens=1000,
                               review_id=None):
        """对一对候选事实做关系判断提议 -> add_change（candidate，decision=pending）。

        关系模型说 UNRELATED/UNCERTAIN 等未形成 proposed 的，只记录不落 change。
        """
        old = views.get(pair['old_fact_id'])
        new = views.get(pair['new_fact_id'])
        if old is None or new is None:
            raise ValueError('pair endpoints must be stored facts')
        sources = {old['evidence_id']: self._chunk_source(old['evidence_id']),
                   new['evidence_id']: self._chunk_source(new['evidence_id'])}
        payload = relation_judge.make_payload(
            old, new, sources, model=model, max_tokens=max_tokens,
            time_order=pair.get('time_order', 'ORDER_UNKNOWN'))
        raw = await self.call(payload)
        outcome = relation_judge.parse_relation(raw, old, new, sources)
        outcome['applied'] = False
        if outcome.get('status') != 'proposed':
            outcome['change_id'] = None
            return outcome
        caveats = ','.join(outcome.get('caveats') or []) or 'none'
        reason = f"{outcome['model_reason']} [caveats:{caveats}]"
        change_id = self.store.add_change(
            old['fact_id'], new['fact_id'], outcome['relation'],
            new['evidence_id'], reason,
            status='candidate', decision='pending', review_id=review_id)
        outcome['change_id'] = change_id
        return outcome
