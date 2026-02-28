"""Generator, Reflector, and Curator components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .delta import DeltaBatch
from .llm import LLMClient
from .playbook import Playbook
from .prompts import CURATOR_PROMPT, GENERATOR_PROMPT, REFLECTOR_PROMPT


def _safe_json_loads(text: str) -> Dict[str, Any]:
    # 尝试提取 markdown 代码块中的 JSON
    import re
    cleaned_text = text.strip()

    # 匹配 ```json ... ``` 或 ``` ... ``` 代码块
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, cleaned_text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()

    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(f"LLM response is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


def _format_optional(value: Optional[str]) -> str:
    return value or "(none)"

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Literal


# -------------------------
# 数据结构
# -------------------------

@dataclass
class GuidelineEntry:
    source: str
    title: str
    id: Any
    text: str
    overview: str


@dataclass
class RetrievalResult:
    query: str
    method: str                  # "exact" or "hybrid"
    match_fields: str            # "title" or "title+overview"
    entries: List[GuidelineEntry]
    scores: Optional[List[float]] = None  # 对 hybrid 用 cosine 相似度；exact 可以为 None 或 1.0


# -------------------------
# 基础工具函数
# -------------------------

def _load_guideline_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Guideline JSON must be a dict at top level.")
    return data


def _normalize_text(text: str) -> str:
    """通用标准化：小写、去除多余空白和常见标点。"""
    import re

    text = text.strip().lower()
    # 把常见标点去掉，也可以按需调整
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# -------------------------
# Retriever
# -------------------------

class Retriever:
    """
    从 guideline JSON 中检索医疗 guideline。

    JSON 结构:
    {
        "source_a": {
            "title_1": { "id": "...", "text": "...", "overview": "..." },
            "title_2": { ... },
            ...
        },
        "wikidoc": {
            ...
        }
    }

    参数
    ----
    json_path: JSON 文件路径
    source: 只使用某个 source (e.g. "wikidoc")；为 None 则使用全部 source
    match_fields: "title" 或 "title+overview"
    embedding_fn: 可选的 embedding 函数，签名为
                  embedding_fn(texts: List[str]) -> List[List[float]]
    wikidoc_aliases: 可选的 wikidoc 缩写映射:
                     key 为规范化后的 query，value 为 wikidoc 中使用的缩写 title
    """

    def __init__(
        self,
        json_path: str,
        *,
        source: Optional[str] = None,
        match_fields: Literal["title", "title+overview"] = "title",
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        wikidoc_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.source = source
        self.match_fields = match_fields
        self.embedding_fn = embedding_fn
        self.wikidoc_aliases = wikidoc_aliases or {}

        raw = _load_guideline_json(self.json_path)
        self._entries: List[GuidelineEntry] = self._flatten_entries(raw)
        self._title_index: Dict[str, List[GuidelineEntry]] = self._build_title_index()

    # ---------- 数据准备 ----------

    def _flatten_entries(self, raw: Dict[str, Any]) -> List[GuidelineEntry]:
        entries: List[GuidelineEntry] = []
        for src_name, src_dict in raw.items():
            if self.source is not None and src_name != self.source:
                continue
            if not isinstance(src_dict, dict):
                continue
            for title, meta in src_dict.items():
                if not isinstance(meta, dict):
                    continue
                entry = GuidelineEntry(
                    source=src_name,
                    title=str(title),
                    id=meta.get("id"),
                    text=str(meta.get("text", "")),
                    overview=str(meta.get("overview", "")),
                )
                entries.append(entry)
        return entries

    def _build_title_index(self) -> Dict[str, List[GuidelineEntry]]:
        """建立基于 title 的规范化索引，仅用于 exact title 匹配。"""
        index: Dict[str, List[GuidelineEntry]] = {}
        for e in self._entries:
            key = self._normalize_title(e.title, e.source)
            index.setdefault(key, []).append(e)
        return index

    # ---------- 规范化/匹配辅助 ----------

    def _normalize_query(self, query: str) -> str:
        """
        Query 标准化。

        若 source == 'wikidoc'，则允许用别名表把自然语言疾病名映射到缩写。
        """
        q_norm = _normalize_text(query)
        # 如果指定了 source 且为 wikidoc，则尝试别名映射
        if self.source == "wikidoc":
            # 先看是否有直接的 alias
            if q_norm in self.wikidoc_aliases:
                # alias 返回的是 wikidoc 内部 title 形式；我们也归一化一下
                return _normalize_text(self.wikidoc_aliases[q_norm])
            # 否则可以再做一次简单的压缩（只保留字母数字）作为兜底
            import re

            alt = re.sub(r"[^a-z0-9]", "", q_norm)
            if alt and alt in self.wikidoc_aliases:
                return _normalize_text(self.wikidoc_aliases[alt])

        return q_norm

    # def _normalize_query(self, query: str) -> str:
    #     q_norm = _normalize_text(query)

    #     if self.source == "wikidoc":
    #         # 1) 先查 alias（如果你有提供）
    #         if q_norm in self.wikidoc_aliases:
    #             return _normalize_text(self.wikidoc_aliases[q_norm])

    #         # 2) 如果没有 alias，则尝试从 query 构造缩写
    #         #    比如 "acute myocardial infarction" -> "ami"
    #         tokens = q_norm.split()
    #         if len(tokens) >= 2:
    #             acronym = "".join(t[0] for t in tokens if t)
    #             # title_index 的 key 本身就是规范化后的 title（如 "ami"）
    #             if acronym in self._title_index:
    #                 return acronym

    #     # 普通情况：按通用规范化返回
    #     return q_norm


    def _normalize_title(self, title: str, source: str) -> str:
        """
        Title 的规范化规则；对 wikidoc 可以视需要再加额外逻辑。
        """
        norm = _normalize_text(title)
        if source == "wikidoc":
            # 这里可以按需添加 wikidoc 特有的归一化逻辑
            # 例如去掉 '(disease)' 后缀等；你可以按实际数据调整
            if norm.endswith(" disease"):
                norm = norm[: -len(" disease")]
        return norm

    # ---------- 核心匹配函数 ----------

    def _match_title_exact(self, query: str) -> List[GuidelineEntry]:
        """
        title-only 模式下，做 query 和 title 的 exact match（在规范化后）。
        """
        q_norm = self._normalize_query(query)
        return self._title_index.get(q_norm, [])

    def _match_title_overview_exact(self, query: str) -> List[GuidelineEntry]:
        """
        title+overview 模式下，做 query 在 title 或 overview 内的包含匹配（规范化后）。
        """
        q_norm = self._normalize_query(query)
        results: List[GuidelineEntry] = []

        for e in self._entries:
            title_norm = _normalize_text(e.title)
            overview_norm = _normalize_text(e.overview or "")
            if q_norm in title_norm or q_norm in overview_norm:
                results.append(e)
        return results

    # ---------- 对外接口：Exact 检索 ----------

    def retrieve_exact(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        match_fields: Optional[Literal["title", "title+overview"]] = None,
    ) -> RetrievalResult:
        """
        纯 exact 检索（基于字符串规范化的匹配）。

        - 当 match_fields == "title" 时，用 title 的 exact match。
        - 当 match_fields == "title+overview" 时，query 只要出现在 title 或 overview 里即可。
        """
        fields = match_fields or self.match_fields
        if fields == "title":
            entries = self._match_title_exact(query)
        elif fields == "title+overview":
            entries = self._match_title_overview_exact(query)
        else:
            raise ValueError(f"Unknown match_fields: {fields}")

        if top_k is not None:
            entries = entries[:top_k]

        return RetrievalResult(
            query=query,
            method="exact",
            match_fields=fields,
            entries=entries,
            scores=None,
        )

    # ---------- 对外接口：Hybrid 检索 ----------

    def retrieve_hybrid(
        self,
        query: str,
        *,
        top_k: int = 5,
        prefilter_k: int = 50,
        match_fields: Optional[Literal["title", "title+overview"]] = None,
    ) -> RetrievalResult:
        """
        Hybrid 检索：先用 embedding 做粗排，再在前若干条里做 exact 过滤。

        1. 若未提供 embedding_fn，则退化为 retrieve_exact。
        2. 否则：
           - 对所有 entry 构造文本（仅 title 或 title+overview）
           - 用 embedding_fn 计算 query 和所有文本的向量
           - 用 cosine 相似度取 top prefilter_k
           - 在这些候选中再做一次 exact 匹配：
             * title 模式：规范化后完全一致
             * title+overview 模式：规范化 query 出现在 title 或 overview 中
           - 若 exact 过滤为空，则直接使用 embedding 排序的前 top_k 作为结果
        """
        fields = match_fields or self.match_fields

        if self.embedding_fn is None:
            # 没有 embedding，直接 exact
            exact_res = self.retrieve_exact(query, top_k=top_k, match_fields=fields)
            exact_res.method = "exact (fallback-no-embed)"
            return exact_res

        # 1) 准备文本
        if fields == "title":
            corpus = [e.title for e in self._entries]
        elif fields == "title+overview":
            corpus = [f"{e.title}\n\n{e.overview}" for e in self._entries]
        else:
            raise ValueError(f"Unknown match_fields: {fields}")

        # 2) embedding
        query_vec = self.embedding_fn([query])[0]
        doc_vecs = self.embedding_fn(corpus)

        # 3) 计算相似度并排序
        scored: List[Tuple[float, GuidelineEntry]] = []
        for e, vec in zip(self._entries, doc_vecs):
            score = _cosine_similarity(query_vec, vec)
            scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [e for _, e in scored[:prefilter_k]]

        # 4) 在候选中 exact 过滤
        q_norm = self._normalize_query(query)
        exact_filtered: List[Tuple[float, GuidelineEntry]] = []

        for score, e in scored[:prefilter_k]:
            if fields == "title":
                if self._normalize_title(e.title, e.source) == q_norm:
                    exact_filtered.append((score, e))
            else:  # "title+overview"
                title_norm = _normalize_text(e.title)
                overview_norm = _normalize_text(e.overview or "")
                if q_norm in title_norm or q_norm in overview_norm:
                    exact_filtered.append((score, e))

        if exact_filtered:
            exact_filtered.sort(key=lambda x: x[0], reverse=True)
            final = exact_filtered[:top_k]
        else:
            # 如果 exact 匹配不到，就直接用 embedding 的 top_k
            final = scored[:top_k]

        entries = [e for s, e in final]
        scores = [s for s, e in final]

        return RetrievalResult(
            query=query,
            method="hybrid",
            match_fields=fields,
            entries=entries,
            scores=scores,
        )



@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]


class Generator:
    """Produces trajectories using the current playbook."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = GENERATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        base_prompt = self.prompt_template.format(
            # playbook=playbook.as_prompt() or "(empty playbook)",
            playbook=playbook,
            # reflection=_format_optional(reflection),
            question=question,
            context=_format_optional(context),
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                reasoning = str(data.get("reasoning", ""))
                final_answer = str(data.get("final_answer", ""))
                bullet_ids = [
                    str(item)
                    for item in data.get("bullet_ids", [])
                    if isinstance(item, (str, int))
                ]
                return GeneratorOutput(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    bullet_ids=bullet_ids,
                    raw=data,
                )
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt
                    + "\n\n务必仅输出单个有效 JSON 对象，"
                    "请转义所有引号或改用单引号，避免输出额外文本。"
                )
        raise RuntimeError("Generator failed to produce valid JSON.") from last_error


@dataclass
class BulletTag:
    id: str
    tag: str


@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    raw: Dict[str, Any]


class Reflector:
    """Extracts lessons and bullet feedback from trajectories."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> ReflectorOutput:
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)
        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_excerpt or "(no bullets referenced)",
        )
        result: Optional[ReflectorOutput] = None
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(
                    prompt, refinement_round=round_idx, **kwargs
                )
                try:
                    data = _safe_json_loads(response.text)
                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if isinstance(item, dict) and "id" in item and "tag" in item:
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]), tag=str(item["tag"]).lower()
                                    )
                                )
                    candidate = ReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        bullet_tags=bullet_tags,
                        raw=data,
                    )
                    result = candidate
                    # Early exit if we already have actionable output
                    if bullet_tags or candidate.key_insight:
                        return candidate
                    break
                except ValueError as err:
                    last_error = err
                    if attempt + 1 >= self.max_retries:
                        break
                    prompt = (
                        base_prompt
                        + "\n\n请严格输出有效 JSON，对双引号进行转义，"
                        "不要输出额外解释性文本。"
                    )
        if result is None:
            raise RuntimeError("Reflector failed to produce a result.") from last_error
        return result


@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]


class Curator:
    """Transforms reflections into delta updates."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = CURATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def curate(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(playbook.stats()),
            reflection=json.dumps(reflection.raw, ensure_ascii=False, indent=2),
            playbook=playbook.as_prompt() or "(empty playbook)",
            question_context=question_context,
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                delta = DeltaBatch.from_json(data)
                return CuratorOutput(delta=delta, raw=data)
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt
                    + "\n\n提醒：仅输出有效 JSON，所有字符串请转义双引号或改用单引号，"
                    "不要添加额外文本。"
                )
        raise RuntimeError("Curator failed to produce valid JSON.") from last_error


def _make_playbook_excerpt(playbook: Playbook, bullet_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullet = playbook.get_bullet(bullet_id)
        if bullet:
            seen.add(bullet_id)
            lines.append(f"[{bullet.id}] {bullet.content}")
    return "\n".join(lines)
