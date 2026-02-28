# """Playbook storage and mutation logic for ACE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .delta import DeltaBatch, DeltaOperation
from .deduplication import Deduplicator


@dataclass
class Bullet:
    """Single playbook entry."""

    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def apply_metadata(self, metadata: Dict[str, int]) -> None:
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def tag(self, tag: str, increment: int = 1) -> None:
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Playbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._bullets: Dict[str, Bullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ #
    # CRUD utils
    # ------------------------------------------------------------------ #
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        bullet = Bullet(id=bullet_id, section=section, content=content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            bullet.content = content
        if metadata:
            bullet.apply_metadata(metadata)
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        return bullet

    def tag_bullet(self, bullet_id: str, tag: str, increment: int = 1) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment=increment)
        return bullet

    def remove_bullet(self, bullet_id: str) -> None:
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self._sections.get(bullet.section)
        if section_list:
            self._sections[bullet.section] = [
                bid for bid in section_list if bid != bullet_id
            ]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        return self._bullets.get(bullet_id)

    def bullets(self) -> List[Bullet]:
        return list(self._bullets.values())

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {
            "bullets": {bullet_id: asdict(bullet) for bullet_id, bullet in self._bullets.items()},
            "sections": self._sections,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Playbook":
        instance = cls()
        bullets_payload = payload.get("bullets", {})
        if isinstance(bullets_payload, dict):
            for bullet_id, bullet_value in bullets_payload.items():
                if isinstance(bullet_value, dict):
                    instance._bullets[bullet_id] = Bullet(**bullet_value)
        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            instance._sections = {
                section: list(ids) if isinstance(ids, Iterable) else []
                for section, ids in sections_payload.items()
            }
        instance._next_id = int(payload.get("next_id", 0))
        return instance

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "Playbook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    # ------------------------------------------------------------------ #
    # Delta application
    # ------------------------------------------------------------------ #
    def apply_delta(self, delta: DeltaBatch) -> None:
        for operation in delta.operations:
            self._apply_operation(operation)

    def _apply_operation(self, operation: DeltaOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            self.add_bullet(
                section=operation.section,
                content=operation.content or "",
                bullet_id=operation.bullet_id,
                metadata=operation.metadata,
            )
        elif op_type == "UPDATE":
            if operation.bullet_id is None:
                return
            self.update_bullet(
                operation.bullet_id,
                content=operation.content,
                metadata=operation.metadata,
            )
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            for tag, increment in operation.metadata.items():
                self.tag_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_bullet(operation.bullet_id)

    def deduplicate(self, deduplicator: Deduplicator, bullet_ids: List[str]) -> List[str]:
        """
        Finds and removes duplicate bullets from the playbook.

        Args:
            deduplicator: The Deduplicator instance.
            bullet_ids: A list of bullet IDs to check for duplicates.

        Returns:
            A list of bullet IDs that were removed.
        """
        new_bullets = {
            bullet_id: self._bullets[bullet_id].content
            for bullet_id in bullet_ids
            if bullet_id in self._bullets
        }
        existing_bullets = {
            bullet_id: bullet.content
            for bullet_id, bullet in self._bullets.items()
            if bullet_id not in new_bullets
        }

        duplicate_ids = deduplicator.find_duplicates(new_bullets, existing_bullets)

        for bullet_id in duplicate_ids:
            self.remove_bullet(bullet_id)

        return duplicate_ids

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_prompt(self) -> str:
        """Return a human-readable playbook string for prompting LLMs."""
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._bullets[bullet_id]
                counters = (
                    f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                )
                parts.append(f"- [{bullet.id}] {bullet.content} {counters}")
        return "\n".join(parts)

    def stats(self) -> Dict[str, object]:
        return {
            "sections": len(self._sections),
            "bullets": len(self._bullets),
            "tags": {
                "helpful": sum(b.helpful for b in self._bullets.values()),
                "harmful": sum(b.harmful for b in self._bullets.values()),
                "neutral": sum(b.neutral for b in self._bullets.values()),
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"



from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
import json
import re
import math


# ------------------------
# Core data structures
# ------------------------

# @dataclass
# class GuidelineBullet:
#     """Minimal atomic unit of guideline knowledge."""
#     id: str
#     section_id: str
#     content: str                     # raw paragraph / subsection text
#     metadata: Dict[str, Any] = field(default_factory=dict)

#     # Retrieval-oriented fields
#     summary: Optional[str] = None    # short summary for retrieval
#     embedding: Optional[List[float]] = None  # vector representation

#     created_at: str = field(
#         default_factory=lambda: datetime.now(timezone.utc).isoformat()
#     )
#     updated_at: str = field(
#         default_factory=lambda: datetime.now(timezone.utc).isoformat()
    # )

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


@dataclass
class DeltaOperation:
    """
    One atomic change on the playbook.

    type:
      - 'ADD_SECTION'
      - 'UPDATE_SECTION'
      - 'REMOVE_SECTION'
      - 'ADD_BULLET'
      - 'UPDATE_BULLET'
      - 'REMOVE_BULLET'
      - 'TAG_BULLET'  (optional, for self-evo feedback)

    For section ops:
      - section_id / section_title 视情况使用

    For bullet ops:
      - bullet_id (update/remove/tag)
      - section_id + content (add)
    """
    type: str

    # Section-related fields
    section_id: Optional[str] = None
    section_title: Optional[str] = None

    # Bullet-related fields
    bullet_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional: human-readable note
    note: Optional[str] = None


@dataclass
class DeltaBatch:
    """Group of operations; one logical update."""
    operations: List[DeltaOperation]
    base_version: Optional[str] = None      # e.g. "guideline_2019_v1"
    new_version: Optional[str] = None       # e.g. "guideline_2025_v1"
    created_by: Optional[str] = None        # e.g. "self-evo-agent" / "human-expert"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )



@dataclass
class GuidelineBullet:
    id: str
    section_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    summary: Optional[str] = None
    embedding: Optional[List[float]] = None

    # Simple tag counters, e.g. {"helpful": 3, "harmful": 1}
    tags: Dict[str, int] = field(default_factory=dict)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def tag(self, tag: str, increment: int = 1) -> None:
        self.tags[tag] = self.tags.get(tag, 0) + increment
        self.updated_at = datetime.now(timezone.utc).isoformat()

@dataclass
class GuidelineSection:
    """One logical section, containing multiple bullets."""
    id: str
    title: str
    bullet_ids: List[str] = field(default_factory=list)

    # Retrieval-oriented fields
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None


class GuidelinePlaybook:
    """
    Playbook for a single guideline document.

    - Sections are explicit GuidelineSection objects.
    - Each section holds bullet_ids; bullet content lives in GuidelineBullet.
    - Supports:
        * CRUD for sections / bullets
        * Serialization
        * Build summaries / embeddings for retrieval
        * Simple cosine-similarity search
        * from_markdown(...) to parse your guideline text into sections + bullets
    """

    def __init__(self, guideline_id: str, title: str) -> None:
        self.guideline_id = guideline_id
        self.title = title

        self._sections: Dict[str, GuidelineSection] = {}
        self._bullets: Dict[str, GuidelineBullet] = {}

        self._next_section_id = 0
        self._next_bullet_id = 0

    # ------------------------
    # Section CRUD
    # ------------------------

    def add_section(self, title: str, section_id: Optional[str] = None) -> GuidelineSection:
        if section_id is None:
            self._next_section_id += 1
            section_id = f"sec-{self._next_section_id:03d}"
        section = GuidelineSection(id=section_id, title=title)
        self._sections[section_id] = section
        return section

    def get_section(self, section_id: str) -> Optional[GuidelineSection]:
        return self._sections.get(section_id)

    def remove_section(self, section_id: str) -> None:
        section = self._sections.pop(section_id, None)
        if section is None:
            return
        # Also remove all bullets under this section
        for bid in list(section.bullet_ids):
            self._bullets.pop(bid, None)

    def sections(self) -> List[GuidelineSection]:
        return list(self._sections.values())

    # ------------------------
    # Bullet CRUD
    # ------------------------

    def add_bullet(
        self,
        section_id: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuidelineBullet:
        if section_id not in self._sections:
            raise ValueError(f"Section {section_id} does not exist")

        if bullet_id is None:
            self._next_bullet_id += 1
            bullet_id = f"b-{self._next_bullet_id:05d}"

        metadata = metadata or {}
        bullet = GuidelineBullet(
            id=bullet_id,
            section_id=section_id,
            content=content,
            metadata=metadata,
        )
        self._bullets[bullet_id] = bullet
        self._sections[section_id].bullet_ids.append(bullet_id)
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GuidelineBullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            bullet.content = content
        if metadata:
            bullet.metadata.update(metadata)
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        return bullet

    def remove_bullet(self, bullet_id: str) -> None:
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section = self._sections.get(bullet.section_id)
        if section:
            section.bullet_ids = [bid for bid in section.bullet_ids if bid != bullet_id]

    def get_bullet(self, bullet_id: str) -> Optional[GuidelineBullet]:
        return self._bullets.get(bullet_id)

    def bullets(self) -> List[GuidelineBullet]:
        return list(self._bullets.values())

    # ------------------------
    # Serialization
    # ------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guideline_id": self.guideline_id,
            "title": self.title,
            "sections": {sid: asdict(sec) for sid, sec in self._sections.items()},
            "bullets": {bid: asdict(b) for bid, b in self._bullets.items()},
            "next_section_id": self._next_section_id,
            "next_bullet_id": self._next_bullet_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GuidelinePlaybook":
        pb = cls(
            guideline_id=payload.get("guideline_id", ""),
            title=payload.get("title", ""),
        )
        sections_payload = payload.get("sections", {})
        for sid, s_val in sections_payload.items():
            if isinstance(s_val, dict):
                pb._sections[sid] = GuidelineSection(**s_val)

        bullets_payload = payload.get("bullets", {})
        for bid, b_val in bullets_payload.items():
            if isinstance(b_val, dict):
                pb._bullets[bid] = GuidelineBullet(**b_val)

        pb._next_section_id = int(payload.get("next_section_id", 0))
        pb._next_bullet_id = int(payload.get("next_bullet_id", 0))
        return pb

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "GuidelinePlaybook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    # ------------------------
    # Build from guideline text (Markdown-like)
    # ------------------------

    @classmethod
    def from_markdown(cls, guideline_id: str, title: str, text: str) -> "GuidelinePlaybook":
        """
        Parse a guideline like the influenza example into:
          - top level sections from '# ...'
          - within each section, use '## ...' as sub-section bullets
          - if no '##', fall back to paragraphs as bullets
        """
        pb = cls(guideline_id, title)

        lines = text.splitlines()
        sections: List[tuple[str, str]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []

        top_sec_pattern = re.compile(r"^#\s+(.*)")

        # 1) Split into top-level sections by '# ...'
        for line in lines:
            m = top_sec_pattern.match(line)
            if m:
                # flush previous section
                if current_title is not None:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                    current_lines = []
                current_title = m.group(1).strip()
            else:
                current_lines.append(line)

        if current_title is not None:
            sections.append((current_title, "\n".join(current_lines).strip()))

        # 2) For each section, create GuidelineSection + bullets
        for sec_title, sec_text in sections:
            section = pb.add_section(sec_title)

            # Look for level-2 headers '## ...' to form more structured bullets
            subsection_pattern = re.compile(r"^##\s+(.*)")
            subs: List[tuple[Optional[str], str]] = []
            sub_title: Optional[str] = None
            buf: List[str] = []

            for line in sec_text.splitlines():
                m = subsection_pattern.match(line)
                if m:
                    if buf:
                        # flush previous subsection
                        subs.append((sub_title, "\n".join(buf).strip()))
                        buf = []
                    sub_title = m.group(1).strip()
                else:
                    buf.append(line)

            if buf:
                subs.append((sub_title, "\n".join(buf).strip()))

            if subs:
                # Example: Diagnosis section -> each '## History and Symptoms' etc becomes one bullet
                for stitle, scontent in subs:
                    content = scontent.strip()
                    if not content:
                        continue
                    meta: Dict[str, Any] = {}
                    if stitle:
                        meta["subsection_title"] = stitle
                    pb.add_bullet(section.id, content=content, metadata=meta)
            else:
                # No '##' inside this section: use paragraphs as bullets
                paragraphs = [
                    p.strip() for p in re.split(r"\n\s*\n", sec_text) if p.strip()
                ]
                for p in paragraphs:
                    pb.add_bullet(section.id, content=p)

        return pb

    # ------------------------
    # Retrieval signatures: summaries + embeddings
    # ------------------------

    def build_summaries(
        self,
        summarizer: Callable[[str], str],
        level: str = "bullet",
    ) -> None:
        """
        Build short textual summaries for bullets or sections.

        summarizer: a callable, e.g. your LLM wrapper:
            def summarizer(text: str) -> str: ...
        """
        if level == "bullet":
            for bullet in self._bullets.values():
                bullet.summary = summarizer(bullet.content)
        elif level == "section":
            for sec in self._sections.values():
                text = sec.title + "\n" + "\n".join(
                    self._bullets[bid].content
                    for bid in sec.bullet_ids
                    if bid in self._bullets
                )
                sec.summary = summarizer(text)
        else:
            raise ValueError("level must be 'bullet' or 'section'")

    def build_embeddings(
        self,
        encoder: Callable[[str], List[float]],
        level: str = "bullet",
    ) -> None:
        """
        Build vector embeddings for bullets or sections.

        encoder: any embedding function, e.g. a SentenceTransformer wrapper:
            def encoder(text: str) -> List[float]: ...
        """
        if level == "bullet":
            for bullet in self._bullets.values():
                text = bullet.summary or bullet.content
                bullet.embedding = encoder(text)
        elif level == "section":
            for sec in self._sections.values():
                if sec.summary is not None:
                    text = sec.summary
                else:
                    text = sec.title + "\n" + "\n".join(
                        self._bullets[bid].content
                        for bid in sec.bullet_ids
                        if bid in self._bullets
                    )
                sec.embedding = encoder(text)
        else:
            raise ValueError("level must be 'bullet' or 'section'")

    # ------------------------
    # Simple retrieval (cosine similarity)
    # ------------------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(
        self,
        query: str,
        encoder: Callable[[str], List[float]],
        k: int = 5,
        level: str = "bullet",
    ):
        """
        Example retrieval接口：
        - query: 比如从 EHR 抽出来的病历 summary
        - encoder: 同一个 embedding 模型
        - level:
            * 'bullet'：检索最相关的 guideline bullet（适合给 agent 直接当 context）
            * 'section'：检索最相关的章节（适合先 coarse 再 fine）

        返回按相关度排序的对象列表（GuidelineBullet 或 GuidelineSection）。
        """
        query_emb = encoder(query)
        candidates = []

        if level == "bullet":
            for b in self._bullets.values():
                if b.embedding is None:
                    continue
                score = self._cosine(query_emb, b.embedding)
                candidates.append((score, b))
        elif level == "section":
            for s in self._sections.values():
                if s.embedding is None:
                    continue
                score = self._cosine(query_emb, s.embedding)
                candidates.append((score, s))
        else:
            raise ValueError("level must be 'bullet' or 'section'")

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:k]]


    def apply_delta(self, delta: DeltaBatch, strict: bool = False) -> Dict[str, List[str]]:
        """
        Apply a batch of delta operations to the playbook.

        strict:
          - False: silently skip invalid ops (e.g. bullet not found), just记录在 result['skipped'] 里
          - True : 遇到非法操作直接 raise

        Return:
          - {'applied': [...], 'skipped': [...]}  里面是 op 的描述字符串，方便日志记录
        """
        applied: List[str] = []
        skipped: List[str] = []

        for op in delta.operations:
            ok, msg = self._apply_operation(op, strict=strict)
            if ok:
                applied.append(msg)
            else:
                skipped.append(msg)

        # 你也可以在这里记录 version 变化，比如：
        # self.guideline_version = delta.new_version or self.guideline_version

        return {"applied": applied, "skipped": skipped}

    def _apply_operation(self, op: DeltaOperation, strict: bool) -> (bool, str):
        t = op.type.upper()

        # ---- Section-level ops ----
        if t == "ADD_SECTION":
            title = op.section_title or "Untitled Section"
            sec = self.add_section(title=title, section_id=op.section_id)
            return True, f"ADD_SECTION {sec.id}"

        if t == "UPDATE_SECTION":
            if not op.section_id:
                if strict:
                    raise ValueError("UPDATE_SECTION requires section_id")
                return False, "UPDATE_SECTION skipped (no section_id)"
            sec = self.get_section(op.section_id)
            if sec is None:
                if strict:
                    raise ValueError(f"Section {op.section_id} not found")
                return False, f"UPDATE_SECTION skipped (section {op.section_id} not found)"
            if op.section_title:
                sec.title = op.section_title
            return True, f"UPDATE_SECTION {sec.id}"

        if t == "REMOVE_SECTION":
            if not op.section_id:
                if strict:
                    raise ValueError("REMOVE_SECTION requires section_id")
                return False, "REMOVE_SECTION skipped (no section_id)"
            if self.get_section(op.section_id) is None:
                if strict:
                    raise ValueError(f"Section {op.section_id} not found")
                return False, f"REMOVE_SECTION skipped (section {op.section_id} not found)"
            self.remove_section(op.section_id)
            return True, f"REMOVE_SECTION {op.section_id}"

        # ---- Bullet-level ops ----
        if t == "ADD_BULLET":
            if not op.section_id:
                if strict:
                    raise ValueError("ADD_BULLET requires section_id")
                return False, "ADD_BULLET skipped (no section_id)"
            if op.section_id not in self._sections:
                if strict:
                    raise ValueError(f"Section {op.section_id} not found")
                return False, f"ADD_BULLET skipped (section {op.section_id} not found)"
            content = op.content or ""
            bullet = self.add_bullet(
                section_id=op.section_id,
                content=content,
                bullet_id=op.bullet_id,
                metadata=op.metadata,
            )
            return True, f"ADD_BULLET {bullet.id}"

        if t == "UPDATE_BULLET":
            if not op.bullet_id:
                if strict:
                    raise ValueError("UPDATE_BULLET requires bullet_id")
                return False, "UPDATE_BULLET skipped (no bullet_id)"
            bullet = self.get_bullet(op.bullet_id)
            if bullet is None:
                if strict:
                    raise ValueError(f"Bullet {op.bullet_id} not found")
                return False, f"UPDATE_BULLET skipped (bullet {op.bullet_id} not found)"
            self.update_bullet(
                bullet_id=op.bullet_id,
                content=op.content,
                metadata=op.metadata or None,
            )
            return True, f"UPDATE_BULLET {op.bullet_id}"

        if t == "REMOVE_BULLET":
            if not op.bullet_id:
                if strict:
                    raise ValueError("REMOVE_BULLET requires bullet_id")
                return False, "REMOVE_BULLET skipped (no bullet_id)"
            if self.get_bullet(op.bullet_id) is None:
                if strict:
                    raise ValueError(f"Bullet {op.bullet_id} not found")
                return False, f"REMOVE_BULLET skipped (bullet {op.bullet_id} not found)"
            self.remove_bullet(op.bullet_id)
            return True, f"REMOVE_BULLET {op.bullet_id}"

        if t == "TAG_BULLET":
            if not op.bullet_id:
                if strict:
                    raise ValueError("TAG_BULLET requires bullet_id")
                return False, "TAG_BULLET skipped (no bullet_id)"
            bullet = self.get_bullet(op.bullet_id)
            if bullet is None:
                if strict:
                    raise ValueError(f"Bullet {op.bullet_id} not found")
                return False, f"TAG_BULLET skipped (bullet {op.bullet_id} not found)"
            for tag, inc in (op.metadata or {}).items():
                try:
                    increment = int(inc)
                except Exception:
                    increment = 1
                bullet.tag(tag, increment=increment)
            return True, f"TAG_BULLET {op.bullet_id}"

        # Unknown op type
        msg = f"Unknown delta type {op.type}"
        if strict:
            raise ValueError(msg)
        return False, msg
    
        # ------------------------
    # Hierarchical export for LLMs: section -> bullets
    # ------------------------

    def to_hierarchical_dict(
        self,
        *,
        include_embeddings: bool = False,
        include_timestamps: bool = True,
        max_bullets_per_section: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Export the playbook as a hierarchical dict:

        {
          "guideline_id": ...,
          "title": ...,
          "sections": [
            {
              "id": "sec-001",
              "title": "Overview",
              "summary": "...",          # optional
              "bullets": [
                {
                  "id": "b-00001",
                  "content": "...",
                  "summary": "...",      # optional
                  "metadata": {...},
                  "tags": {...},
                  "created_at": "...",
                  "updated_at": "..."
                },
                ...
              ]
            },
            ...
          ]
        }

        This is the recommended format to feed into your LLM so that it can
        refer to section / bullet ids in delta updates.
        """
        sections_payload: List[Dict[str, Any]] = []

        # 保持插入顺序：Python 3.7+ dict 默认有序
        for sec in self._sections.values():
            sec_dict: Dict[str, Any] = {
                "id": sec.id,
                "title": sec.title,
            }
            if sec.summary is not None:
                sec_dict["summary"] = sec.summary
            if include_embeddings and sec.embedding is not None:
                sec_dict["embedding"] = sec.embedding

            bullets_payload: List[Dict[str, Any]] = []
            # 按 section.bullet_ids 的顺序导出，可选截断
            bullet_ids_to_export = sec.bullet_ids
            if max_bullets_per_section is not None:
                bullet_ids_to_export = sec.bullet_ids[:max_bullets_per_section]
            for bid in bullet_ids_to_export:
                bullet = self._bullets.get(bid)
                if bullet is None:
                    continue
                b_dict: Dict[str, Any] = {
                    "id": bullet.id,
                    "content": bullet.content,
                    "metadata": bullet.metadata,
                    "tags": bullet.tags,
                }
                if bullet.summary is not None:
                    b_dict["summary"] = bullet.summary
                if include_embeddings and bullet.embedding is not None:
                    b_dict["embedding"] = bullet.embedding
                if include_timestamps:
                    b_dict["created_at"] = bullet.created_at
                    b_dict["updated_at"] = bullet.updated_at

                bullets_payload.append(b_dict)

            sec_dict["bullets"] = bullets_payload
            sections_payload.append(sec_dict)

        return {
            "guideline_id": self.guideline_id,
            "title": self.title,
            "sections": sections_payload,
        }

    def to_hierarchical_json(
        self,
        *,
        include_embeddings: bool = False,
        include_timestamps: bool = True,
        indent: int = 2,
        ensure_ascii: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Convenience wrapper: export as JSON string with hierarchical structure.

        If max_tokens is specified, will progressively reduce bullets per section
        until the JSON fits within the token limit.
        """
        if max_tokens is None:
            # 无截断
            data = self.to_hierarchical_dict(
                include_embeddings=include_embeddings,
                include_timestamps=include_timestamps,
            )
            return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)

        # 使用 tiktoken 进行截断
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
        except ImportError:
            # 如果没有 tiktoken，使用字符数估算 (约 4 字符 = 1 token)
            enc = None

        def count_tokens(text: str) -> int:
            if enc is not None:
                return len(enc.encode(text))
            return len(text) // 4

        # 找出每个 section 的最大 bullet 数
        max_bullets = max(
            (len(sec.bullet_ids) for sec in self._sections.values()),
            default=0
        )

        # 逐步减少每个 section 的 bullets 数量
        for limit in range(max_bullets, 0, -1):
            data = self.to_hierarchical_dict(
                include_embeddings=include_embeddings,
                include_timestamps=include_timestamps,
                max_bullets_per_section=limit,
            )
            json_str = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
            if count_tokens(json_str) <= max_tokens:
                return json_str

        # 如果即使每个 section 只保留 1 个 bullet 也超限，返回最小版本
        data = self.to_hierarchical_dict(
            include_embeddings=include_embeddings,
            include_timestamps=include_timestamps,
            max_bullets_per_section=1,
        )
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
