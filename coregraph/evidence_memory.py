"""V2 证据记忆（EvidenceMemory）模块。

保存"独立的带来源事实"与"提议中的变更（change）链接"，是实体图谱之外的
原始证据层：实体图谱只是索引，本存储保留未经图谱合并的原始谓词和证据。
本模块不做文本支持性判断，也不直接提交任何状态更新——
判断与提交由审查流程（evidence_review / evidence_bridge）负责。

核心概念：
- chunk  : 证据分片（带稳定 ID 的会话轮次原文）
- fact   : 从某个 chunk 中抽出的一条原始事实（src/relation/tgt + raw 负载）
- change : 两个事实之间的一条审查事件（关系变更提议，永远只处于 candidate 状态）
"""
import copy
import hashlib
import json


def _id(prefix, values):
    """生成稳定 ID：prefix + sha256(规范化 JSON(values))。"""
    return prefix + hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def namespace_chunks(chunks, ordinal_key=None):
    """Assign stable IDs per session and ordinal; collision checks reject changed inputs.

    按 (session_id, 会话内序号) 为 chunk 生成稳定的命名空间化 ID，
    防止不同会话的 chunk_id 互相覆盖；对内容发生变化的输入，
    后续的冲突检查会直接拒绝。

    原始 chunk_id 会保留在 ``source_chunk_id`` 字段中，保证与原数据集
    的 ID 仍可对得上。序号的两种来源：

    - ``ordinal_key=None``（默认）：按**本批次内**该会话的出现位置从 0 编号。
      适用于整批语料或单会话一次性处理（与原版行为一致）。
    - ``ordinal_key='chunk_id'``（或任意字段名）：用该字段作为序号
      （如数据集的稳定轮次 turn_id）。**同一会话分多批输入时必须用这个**，
      否则每批都从 0 编号会导致不同轮次拿到相同 ID 而冲突。
      字段缺失或为空直接报错，不退回批次位置（避免隐性冲突）。

    由于生成的 ID 只依赖 (session_id, 序号)，对已命名空间化的输入
    重复运行是幂等的。
    """
    counts = {}
    result = []
    for chunk in chunks:
        sid = str(chunk.get('session_id', '')).strip()
        if not sid or sid.lower() == 'unknown':
            raise ValueError('evidence memory needs a stable session_id')
        if ordinal_key is not None:
            raw_ordinal = chunk.get(ordinal_key)
            if raw_ordinal is None or str(raw_ordinal).strip() == '':
                raise ValueError(f'missing stable ordinal {ordinal_key!r} for chunk')
            try:
                index = int(raw_ordinal)
            except (TypeError, ValueError):
                index = str(raw_ordinal)   # 非数字的稳定 ID（如 UUID）原样参与哈希
        else:
            index = counts.get(sid, 0)
            counts[sid] = index + 1
        item = copy.deepcopy(chunk)
        if 'source_chunk_id' not in item:
            item['source_chunk_id'] = chunk.get('chunk_id')
        item['chunk_id'] = _id('session_chunk_', [sid, index])
        result.append(item)
    return result


def chunk_conflicts(a, b):
    """True only when chunk payloads genuinely differ.

    仅当两个 chunk 的载荷"真正不同"时才返回 True。

    ``source_chunk_id`` 是与运行方式相关的溯源信息（全批处理时按语料
    顺序编号，按会话处理时每个会话从 0 重新编号），因此不参与身份
    比较：同一会话在不同处理模式下重放不应被误判为冲突。
    """
    keys = set(a) | set(b)
    keys.discard('source_chunk_id')
    return any(a.get(k) != b.get(k) for k in keys)


class EvidenceMemory:
    """证据记忆存储：chunks（证据分片）/ facts（原始事实）/ changes（审查事件）三张表。"""

    def __init__(self):
        self.chunks = {}        # chunk_id -> chunk 负载
        self.facts = {}         # fact_id -> 事实记录（含原始负载 raw 与审查状态）
        self.changes = {}       # change_id -> 审查事件记录
        self.complete = True    # 离线存储是否可视为"完整"（未开启采集时为 False）
        self._entities = {}     # 实体名(小写) -> 涉及该实体的事实 ID 集合
        self._links = {}        # fact_id -> 关联的 change_id 集合
        self._change_keys = {}   # change_key -> [change_id, ...]，按事件发生顺序
        self._reviews = {}       # hash(change_key, review_id) -> change_id（重试去重）

    def add_chunk(self, chunk):
        """登记一个证据分片，返回其存储 ID。

        ID 由 (session_id, chunk_id) 哈希得到；同 ID 重复写入时若载荷
        完全相同则刷新与运行方式相关的溯源字段（幂等），
        若载荷真正不同则抛出冲突异常。
        """
        sid = str(chunk.get('session_id', '')).strip()
        if not sid or sid.lower() == 'unknown':
            raise ValueError('evidence memory needs a stable session_id')
        if chunk.get('chunk_id') is None or not chunk.get('text'):
            raise ValueError('chunk_id and text are required')
        cid = _id('chunk_', [sid, str(chunk['chunk_id'])])
        data = copy.deepcopy(chunk)
        if cid in self.chunks and chunk_conflicts(self.chunks[cid], data):
            raise ValueError('chunk identity collision; source preserved')
        self.chunks[cid] = data  # 载荷相同：仅刷新与运行方式相关的溯源信息
        return cid

    def add_fact(self, raw, evidence_id):
        """登记一条原始事实，返回其事实 ID。

        - 必须挂在一个已存在的证据分片（evidence_id）上；
        - src/relation/tgt 三元组字段缺一不可；
        - fact_id = hash(src, relation, tgt, evidence_id)，同一来源的
          同一事实重放幂等；
        - 初始 support_status 固定为 'unverified'（是否成立由审查决定）；
        - 同时把 src/tgt（及规范化后的 canonical_src/canonical_tgt）
          登记进实体倒排索引。
        """
        if evidence_id not in self.chunks:
            raise ValueError('missing evidence')
        src, rel, tgt = [str(raw.get(k, '')).strip() for k in ('src', 'relation', 'tgt')]
        if not all((src, rel, tgt)):
            raise ValueError('missing subject, predicate or object')
        fid = _id('fact_', [src, rel, tgt, evidence_id])
        existing = self.facts.get(fid)
        record = {'id': fid, 'src': src, 'relation': rel, 'tgt': tgt,
                  'evidence_id': evidence_id, 'raw': copy.deepcopy(raw),
                  'support_status': 'unverified'}
        if existing is not None:
            # 幂等重放：同一事实重新写入时保留已有审查历史与支持状态，
            # 只刷新 raw 负载（身份不变，审查是针对身份记录的）
            record['support_status'] = existing.get('support_status', 'unverified')
            if existing.get('reviews'):
                record['reviews'] = copy.deepcopy(existing['reviews'])
        self.facts[fid] = record
        for entity in (src, tgt, raw.get('canonical_src', ''), raw.get('canonical_tgt', '')):
            if entity: self._entities.setdefault(str(entity).casefold(), set()).add(fid)
        return fid

    def find_by_entities(self, entities):
        """按实体名集合检索事实：返回所有涉及任一实体的事实副本（按 fact_id 排序）。"""
        ids = set()
        for entity in entities: ids.update(self._entities.get(str(entity).casefold(), ()))
        return [copy.deepcopy(self.facts[x]) for x in sorted(ids)]

    def add_change(self, old_id, new_id, relation, evidence_id, reason,
                   status='candidate', decision='pending', review_id=None,
                   evidence=None):
        """Record one review event between two facts.

        记录两个事实之间的一次审查事件。

        每次调用都追加一条事件：``revision`` 在同一条变更链接内递增
        （因此 keep -> replace -> keep 这类被撤销的决策三步历史全部保留），
        ``seq`` 是全局事件顺序号（据此可识别最新决策）。
        ``review_id`` 是重试幂等键：重复传入相同 review_id 会返回已有记录，
        而相同 review_id 配不同 decision 会被拒绝。不传 review_id 时
        每次调用都是一条新事件（历史优先；审查管线应传入 review_id
        以保证重放安全）。只允许 status='candidate'：未经审查确认的
        链接绝不修改事实状态。
        ``evidence``（可选 dict）：关系判断引用的原文引文与字符区间，
        随事件持久化，保证重载后可恢复"模型依据了哪几段原文"。
        """
        if old_id not in self.facts or new_id not in self.facts or old_id == new_id:
            raise ValueError('two distinct facts required')
        if evidence_id not in self.chunks or not relation or not reason or status != 'candidate':
            raise ValueError('only sourced candidate decisions are supported')
        if evidence is not None and not isinstance(evidence, dict):
            raise ValueError('evidence must be a dict')
        decision = str(decision or '').strip() or 'pending'
        core = [old_id, new_id, relation, evidence_id, reason]
        change_key = _id('change_key_', core)
        if review_id is not None:
            review_id = str(review_id)
            review_key = _id('review_', core + [review_id])
            prior = self._reviews.get(review_key)
            if prior is not None:
                # 同一 review_id 重试：decision 相同则幂等返回原记录
                if self.changes[prior]['decision'] != decision:
                    raise ValueError('review_id reused with a different decision')
                return prior
        revision = len(self._change_keys.get(change_key, ()))
        cid = _id('change_', core + [revision])
        self.changes[cid] = {'id':cid,'old_fact_id':old_id,'new_fact_id':new_id,'relation':relation,
                             'evidence_id':evidence_id,'reason':reason,'status':status,
                             'decision':decision,'change_key':change_key,'revision':revision,
                             'seq':len(self.changes),'review_id':review_id}
        if evidence is not None:
            self.changes[cid]['evidence'] = copy.deepcopy(evidence)
        self._change_keys.setdefault(change_key, []).append(cid)
        if review_id is not None:
            self._reviews[review_key] = cid
        for fid in (old_id,new_id): self._links.setdefault(fid,set()).add(cid)
        return cid

    def trace(self, fact_id):
        """Linked change events in review order (global seq).

        返回某事实关联的全部变更事件，按全局 seq（审查顺序）排序。
        """
        return [copy.deepcopy(self.changes[x]) for x in
                sorted(self._links.get(fact_id, ()), key=lambda x: self.changes[x]['seq'])]

    def record_review(self, fact_id, route, reasons, review_id=None, evidence=None):
        """给单个事实追加一条审查结论（永远不是状态变更）。

        与 add_change 的区别：审查只有一个对象（候选 vs 其原文），没有两端，
        所以不走 change 记录，而是直接更新事实的 support_status 并追加审查历史；
        fact 身份（id）始终不变。
        - route 仅允许 DEFER / REVIEW_NEXT / TECHNICAL_ERROR
        - review_id 是重试幂等键：同 ID 同结论返回原记录；同 ID 异结论报错
        - TECHNICAL_ERROR 只记历史、不改 support_status（崩溃不算审查结论）
        - evidence（可选 dict）：本次审查引用的原文依据（字符区间、组件锚点、
          模型给出的引文等），随历史持久化，保证重载后能恢复"模型当时
          指出了原文哪一段"，而不只是结论
        - 只追加（append-only），历史永不覆盖，随 pickle 往返保留
        """
        if fact_id not in self.facts:
            raise ValueError('unknown fact')
        route = str(route or '').strip()
        if route not in ('DEFER', 'REVIEW_NEXT', 'TECHNICAL_ERROR'):
            raise ValueError('unsupported review route')
        if not isinstance(reasons, (list, tuple)) or not reasons or \
                any(not isinstance(r, str) or not r.strip() for r in reasons):
            raise ValueError('nonempty string reasons required')
        if evidence is not None and not isinstance(evidence, dict):
            raise ValueError('evidence must be a dict')
        reasons = list(dict.fromkeys(reasons))
        reviews = self.facts[fact_id].setdefault('reviews', [])
        if review_id is not None:
            review_id = str(review_id)
            prior = next((r for r in reviews if r.get('review_id') == review_id), None)
            if prior is not None:
                # 同 ID 重试：结论一致则幂等返回，结论不同则报错
                if prior.get('route') != route or prior.get('reasons') != reasons:
                    raise ValueError('review_id reused with a different outcome')
                return prior
        record = {'route': route, 'reasons': reasons,
                  'review_id': review_id, 'seq': len(reviews)}
        if evidence is not None:
            record['evidence'] = copy.deepcopy(evidence)
        reviews.append(record)
        # 只有明确的审查结论才更新支持状态；TECHNICAL_ERROR 不改状态
        if route == 'DEFER':
            self.facts[fact_id]['support_status'] = 'review_defer'
        elif route == 'REVIEW_NEXT':
            self.facts[fact_id]['support_status'] = 'review_next'
        return copy.deepcopy(record)

    def evidence_bundle(self, fact_id):
        """Trace both endpoints and source chunks; links remain unverified candidates.

        汇总某事实的完整证据包：该事实、与它关联的全部变更事件
        （含两端事实）、以及涉及的所有来源分片。
        注意：其中的链接仍是未验证的候选，不构成任何证明。
        """
        if fact_id not in self.facts:
            raise ValueError('unknown fact')
        changes = self.trace(fact_id)
        fact_ids = {fact_id}
        source_ids = set()
        for change in changes:
            fact_ids.update((change['old_fact_id'], change['new_fact_id']))
            source_ids.add(change['evidence_id'])
        facts = [copy.deepcopy(self.facts[fid]) for fid in sorted(fact_ids)]
        source_ids.update(f['evidence_id'] for f in facts)
        return {'facts': facts, 'changes': changes,
                'chunks': {cid: copy.deepcopy(self.chunks[cid]) for cid in sorted(source_ids)}}

    def to_dict(self):
        """序列化为 dict（供 pickle 保存）。

        序列化版本演进：
          v4 = 事实可携带审查历史（support_status/reviews，加载时恢复，身份不变）
          v3 = change 记录改为审查事件（change_key/revision/seq/review_id）
          v2 = change id 含 decision；v1 = 不含 decision
        旧版本加载时先按当年的公式校验完整性，再重编码为当前事件模型。
        """
        return copy.deepcopy({'version':4,'chunks':self.chunks,'facts':self.facts,
                              'changes':self.changes,'complete':self.complete})

    @classmethod
    def from_dict(cls, data):
        """从 dict 反序列化（支持 v1~v4 旧格式，旧版会重编码为当前事件模型）。"""
        version = data.get('version')
        if version not in (1, 2, 3, 4): raise ValueError('unsupported evidence memory version')
        s=cls(); s.complete = data.get('complete',False)
        # 逐条重建 chunk：ID 必须与序列化前一致，否则说明数据被篡改/损坏
        for cid, chunk in data['chunks'].items():
            if s.add_chunk(chunk) != cid: raise ValueError('invalid chunk identity')
        # 逐条重建 fact：ID 由内容决定，必须可复现
        for fid, fact in data['facts'].items():
            if s.add_fact(fact['raw'],fact['evidence_id']) != fid: raise ValueError('invalid fact identity')
            # 审查元数据不属于事实身份，add_fact 重建时会丢，这里从负载恢复
            record=s.facts[fid]
            record['support_status']=fact.get('support_status','unverified')
            reviews=fact.get('reviews',[])
            # 校验审查历史的结构与顺序（seq 必须是 0..n-1 连续）
            if not isinstance(reviews,list) or any(not isinstance(r,dict) or
                    not isinstance(r.get('route'),str) or not isinstance(r.get('reasons'),list)
                    for r in reviews):
                raise ValueError('invalid review history')
            if [r.get('seq') for r in reviews]!=list(range(len(reviews))):
                raise ValueError('invalid review order metadata')
            if reviews: record['reviews']=copy.deepcopy(reviews)
        # 逐条重建 change：按版本选择当年 ID 公式校验后重编码
        for cid, change in data['changes'].items():
            decision = str(change.get('decision') or '').strip() or 'pending'
            core = [change['old_fact_id'],change['new_fact_id'],change['relation'],
                    change['evidence_id'],change['reason']]
            args = (*core, change['status'], decision)
            if version == 1:
                # v1：change id 不含 decision，按当年公式校验
                if _id('change_', core) != cid: raise ValueError('invalid legacy change identity')
                s.add_change(*args)  # 重新按审查事件模型建键
            elif version == 2:
                # v2：change id 含 decision
                if _id('change_', core + [decision]) != cid: raise ValueError('invalid legacy change identity')
                s.add_change(*args)  # 重新按审查事件模型建键
            else:
                # v3/v4：直接复现审查事件，且必须逐一核对 revision/seq/change_key
                new_id = s.add_change(*args, review_id=change.get('review_id'),
                                      evidence=change.get('evidence'))
                if new_id != cid: raise ValueError('invalid change identity')
                record = s.changes[new_id]
                if (record['revision'] != change.get('revision') or
                        record['seq'] != change.get('seq') or
                        record['change_key'] != change.get('change_key')):
                    raise ValueError('invalid change order metadata')
        return s
