"""Observed fact storage, bounded compaction and accepted-image retrieval primitives."""
from __future__ import annotations
import json
import re
import time
from typing import Any, Dict, List, Optional

FACTS_CAP = 250
NAVIGATE_CAP = 1100

def sanitize(text: Any, limit: int = 200) -> str:
    """One-line, length-capped, JSON-string-safe rendering of untrusted text.
    Stored user text is prompt-injection material: it is escaped
    so it reads as quoted data, never as new lines of instructions."""
    one = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(one) > limit:
        one = one[: limit - 1] + "…"
    # json-escape (drops raw quotes/newlines/control chars), then strip the
    # surrounding quotes for inline rendering.
    return json.dumps(one, ensure_ascii=False)[1:-1]

def empty_memory() -> Dict[str, Any]:
    return {"facts": [], "summaries": {}, "fact_seq": 0}

def _next_fact_id(mem: Dict[str, Any]) -> str:
    mem["fact_seq"] = int(mem.get("fact_seq", 0)) + 1
    return f"fact_{mem['fact_seq']:06d}"

def append_fact(
    mem: Dict[str, Any],
    *,
    kind: str,
    fact: str,
    scene_id: str,
    slots: Optional[Dict[str, Any]] = None,
    source: Optional[Dict[str, Any]] = None,
    confidence: Any = 1.0,
    replaces: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Append one durable world-scope fact of one scene (bounded growth, deduped).
    Returns the fact dict, or None when skipped (duplicate / empty)."""
    text = sanitize(fact, 200)
    if not text:
        return None
    facts: List[Dict[str, Any]] = mem["facts"]

    # Dedup: an identical (kind, text) fact is a no-op (bounded growth).
    for f in facts:
        if (f.get("kind"), f.get("fact")) == (kind, text):
            return None

    # Contradiction handling: `replaces` removes superseded fact ids.
    removed_ids = []
    if replaces:
        keep = []
        wanted = {str(r) for r in replaces if r}
        for f in facts:
            if f.get("id") in wanted:
                removed_ids.append(f.get("id"))
            else:
                keep.append(f)
        if removed_ids:
            mem["facts"] = facts = keep

    entry: Dict[str, Any] = {
        "id": _next_fact_id(mem),
        "ts": time.time(),
        "scope": "world",
        "scene_id": scene_id,
        "object_id": None,
        "kind": kind,
        "fact": text,
        "confidence": confidence,
        "supersedes": removed_ids,
    }
    if slots:
        entry["slots"] = dict(slots)
    if source:
        entry["source"] = dict(source)
    facts.append(entry)
    if len(facts) > FACTS_CAP:
        _compact(mem)
    return entry

def _compact(mem: Dict[str, Any]) -> None:
    """Summarize the OLDEST HALF of the facts into memory.summaries["world"] +
    one kind="summary" fact; recent facts stay verbatim. Deterministic."""
    facts = mem["facts"]
    half = facts[: len(facts) // 2]
    half_ids = {f["id"] for f in half}
    anchor = half[-1]

    summary_text = "; ".join(sanitize(f.get("fact"), 60) for f in half)
    prev = str((mem["summaries"].get("world") or {}).get("summary") or "")
    merged = (prev + " " + summary_text).strip()
    if len(merged) > 1200:
        merged = merged[-1200:]
    mem["summaries"]["world"] = {"summary": merged, "updated_at": time.time()}

    kept = [f for f in facts if f["id"] not in half_ids]
    summary_fact = {
        "id": _next_fact_id(mem),
        "ts": time.time(),
        "scope": anchor.get("scope"),
        "scene_id": anchor.get("scene_id"),
        "object_id": anchor.get("object_id"),
        "kind": "summary",
        "fact": sanitize(f"(summary of {len(half)} earlier facts) {summary_text}", 200),
        "confidence": "server_state",
        "supersedes": sorted(half_ids),
    }
    kept.append(summary_fact)
    mem["facts"] = kept

def memory_context(mem: Dict[str, Any]) -> str:
    """Render the summary and the newest facts, oldest first, under the hard char cap."""
    summary = (mem["summaries"].get("world") or {}).get("summary")
    lines: List[str] = [f"- (summary) {sanitize(summary, 200)}"] if summary else []
    used = sum(len(line) + 1 for line in lines)
    chosen: List[str] = []
    for f in reversed(mem["facts"]):
        line = f"- {sanitize(f.get('fact'), 160)}"
        if used + len(line) + 1 > NAVIGATE_CAP:
            break
        chosen.append(line)
        used += len(line) + 1
    return "\n".join(lines + chosen[::-1])[:NAVIGATE_CAP]

FACT_KEYS = {"subject_id", "label", "kind", "key", "value", "evidence", "frame_indices"}
_STOP = set("the a an is are was were and of to in on with as at from by this that it its same camera video frame scene view begins continues segment now then again previous original left right up down pan panning return returning reappears finally slowly".split())

def _text(value, name, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(name + ' must be a nonempty bounded string')
    return value.strip()

def validate_observation(value, available_indices=(0, 1)):
    if not isinstance(value, dict) or set(value) != {'summary', 'facts'}:
        raise ValueError('Observation must contain summary and facts only')
    summary = _text(value['summary'], 'summary', 400)
    if not isinstance(value['facts'], list) or not 1 <= len(value['facts']) <= 12:
        raise ValueError('One to twelve visible facts are required')
    facts, seen = [], set()
    for fact in value['facts']:
        if not isinstance(fact, dict) or set(fact) != FACT_KEYS:
            raise ValueError('Invalid observation fact fields')
        f = dict(fact)
        for key, limit in [('subject_id', 80), ('label', 80), ('key', 80), ('value', 100), ('evidence', 200)]:
            f[key] = _text(f[key], key, limit)
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', f['subject_id']):
            raise ValueError('subject_id must be a stable lowercase identifier')
        if f['kind'] not in {'appearance', 'state', 'location', 'world'}:
            raise ValueError('Unsupported observed fact kind')
        indices = f['frame_indices']
        if (not isinstance(indices, list) or not indices or len(indices) != len(set(indices))
                or any(type(i) is not int or i not in available_indices for i in indices)):
            raise ValueError('Fact must cite actual supplied frame indices')
        identity = (f['subject_id'], f['kind'], f['key'].casefold())
        if identity in seen:
            raise ValueError('One latest observed value per subject/kind/key is required')
        seen.add(identity)
        f['frame_indices'] = sorted(indices)
        facts.append(f)
    return {'summary': summary, 'facts': facts}

def _tokens(value):
    return {x for x in re.findall(r'[\w]+', value.casefold()) if len(x) > 2 and x not in _STOP}

def asks_initial_view(caption):
    return bool(re.search(
        r'\b(?:initial|starting|original)\s+(?:viewpoint|view|camera\s+(?:position|view|angle))\b',
        caption, re.IGNORECASE))

def select_initial_reference(entries, state_changes, source):
    eligible = []
    for entry in entries:
        if source and entry['image']['sha256'] == source['sha256']:
            continue
        subjects = {fact['subject_id'] for fact in entry['facts']}
        if any(entry['segment'] < state_changes.get(subject, 0) for subject in subjects):
            continue
        eligible.append(entry)
    return min(eligible, key=lambda entry: (entry['segment'], entry['frame_index'])) if eligible else None
