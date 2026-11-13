"""Observed fact storage, bounded compaction and accepted-image retrieval primitives."""
from __future__ import annotations
import json
import re
import time
from typing import Any, Dict, List, Optional

FACTS_PER_OBJECT = 80
FACTS_PER_SCENE = 250
FACTS_PER_WORLD = 1000
PURPOSE_CAPS = {"analyze":800,"dialogue":1200,"edit":240,"navigate":1100,"custom":700}
FACT_KINDS = ("appearance","state","dialogue","navigation","relationship","world","location","summary")

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

def ensure_world_memory(world) -> Dict[str, Any]:
    mem = getattr(world, "memory", None)
    if not isinstance(mem, dict):
        mem = {}
        world.memory = mem
    mem.setdefault("facts", [])
    mem.setdefault("summaries", {})
    mem.setdefault("fact_seq", 0)
    return mem

def _next_fact_id(mem: Dict[str, Any]) -> str:
    mem["fact_seq"] = int(mem.get("fact_seq", 0)) + 1
    return f"fact_{mem['fact_seq']:06d}"

def _scope_for(scene_id: Optional[str], object_id: Optional[str]) -> str:
    if object_id:
        return "object"
    if scene_id:
        return "scene"
    return "world"

def _scope_key(fact: Dict[str, Any]) -> str:
    scope = fact.get("scope") or _scope_for(fact.get("scene_id"), fact.get("object_id"))
    if scope == "object":
        return f"object:{fact.get('scene_id')}:{fact.get('object_id')}"
    if scope == "scene":
        return f"scene:{fact.get('scene_id')}"
    return "world"

def append_fact(
    world,
    *,
    kind: str,
    fact: str,
    scene_id: Optional[str] = None,
    object_id: Optional[str] = None,
    scope: Optional[str] = None,
    slots: Optional[Dict[str, Any]] = None,
    source: Optional[Dict[str, Any]] = None,
    confidence: Any = 1.0,
    replaces: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Append one durable fact to world memory (bounded growth, deduped).
    Returns the fact dict, or None when skipped (duplicate / empty)."""
    text = sanitize(fact, 200)
    if not text:
        return None
    kind = str(kind or "world").strip().lower()
    if kind not in FACT_KINDS:
        kind = "world"
    mem = ensure_world_memory(world)
    facts: List[Dict[str, Any]] = mem["facts"]

    # Dedup: identical (scene, object, kind, text) is a no-op (bounded growth).
    for f in facts:
        if (
            f.get("scene_id") == scene_id
            and f.get("object_id") == object_id
            and f.get("kind") == kind
            and f.get("fact") == text
        ):
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
        "scope": scope or _scope_for(scene_id, object_id),
        "scene_id": scene_id,
        "object_id": object_id,
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
    _attach_fact_id(world, entry)
    _enforce_caps(world)
    return entry

def _attach_fact_id(world, fact: Dict[str, Any]) -> None:
    """Best-effort fact_id bookkeeping on the scene/object memory dicts."""
    store = getattr(world, "store", None)
    if store is None:
        return
    try:
        scene = store.get_scene(fact.get("scene_id"))
        if scene is None:
            return
        smem = getattr(scene, "memory", None)
        if isinstance(smem, dict):
            smem.setdefault("fact_ids", []).append(fact["id"])
        obj = scene.objects.get(fact.get("object_id")) if fact.get("object_id") else None
        if obj is not None:
            omem = getattr(obj, "memory", None)
            if isinstance(omem, dict):
                omem.setdefault("fact_ids", []).append(fact["id"])
    except Exception:
        pass

def _enforce_caps(world) -> None:
    """§3.7: per-object, per-scene, then world caps; oldest-half compaction."""
    mem = ensure_world_memory(world)
    facts = mem["facts"]

    # object caps
    by_obj: Dict[tuple, int] = {}
    for f in facts:
        if f.get("object_id"):
            key = (f.get("scene_id"), f.get("object_id"))
            by_obj[key] = by_obj.get(key, 0) + 1
    for (sid, oid), n in list(by_obj.items()):
        if n > FACTS_PER_OBJECT:
            _compact(world, lambda f, s=sid, o=oid: (
                f.get("scene_id") == s and f.get("object_id") == o))

    # scene caps
    by_scene: Dict[Any, int] = {}
    for f in mem["facts"]:
        if f.get("scene_id"):
            by_scene[f.get("scene_id")] = by_scene.get(f.get("scene_id"), 0) + 1
    for sid, n in list(by_scene.items()):
        if n > FACTS_PER_SCENE:
            _compact(world, lambda f, s=sid: f.get("scene_id") == s)

    # world cap
    if len(mem["facts"]) > FACTS_PER_WORLD:
        _compact(world, lambda f: True)

def _compact(world, predicate) -> None:
    """Summarize the OLDEST HALF of the predicate-matched facts into
    memory.summaries[scope_key] + one kind="summary" fact; recent facts stay
    verbatim. Deterministic (template strings) — never blocks (§3.7)."""
    mem = ensure_world_memory(world)
    facts = mem["facts"]
    subset = [f for f in facts if predicate(f)]
    if len(subset) < 4:
        return
    half = subset[: len(subset) // 2]
    half_ids = {f["id"] for f in half}
    anchor = half[-1]
    key = _scope_key(anchor)

    summary_text = "; ".join(sanitize(f.get("fact"), 60) for f in half)
    prev = ""
    entry = mem["summaries"].get(key)
    if isinstance(entry, dict):
        prev = str(entry.get("summary") or "")
    merged = (prev + " " + summary_text).strip()
    if len(merged) > 1200:
        merged = merged[-1200:]
    mem["summaries"][key] = {"summary": merged, "updated_at": time.time()}

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
    _prune_fact_ids(world, half_ids)

def _prune_fact_ids(world, removed_ids) -> None:
    store = getattr(world, "store", None)
    if store is None:
        return
    try:
        for scene in store.scenes.values():
            smem = getattr(scene, "memory", None)
            if isinstance(smem, dict) and smem.get("fact_ids"):
                smem["fact_ids"] = [i for i in smem["fact_ids"] if i not in removed_ids]
            for obj in scene.objects.values():
                omem = getattr(obj, "memory", None)
                if isinstance(omem, dict) and omem.get("fact_ids"):
                    omem["fact_ids"] = [i for i in omem["fact_ids"] if i not in removed_ids]
    except Exception:
        pass

def memory_context(world, scene_id, object_id, purpose: str) -> str:
    """Render the memory block for one prompt purpose under its hard char cap.
    Scope matrix per MEMORY_DESIGN.md §5. Returns "" when nothing relevant.
    NEVER raises (a memory failure must not fail the main action)."""
    try:
        return _memory_context_inner(world, scene_id, object_id, purpose)
    except Exception:
        return ""

def _memory_context_inner(world, scene_id, object_id, purpose: str) -> str:
    cap = PURPOSE_CAPS.get(purpose, 400)
    mem = getattr(world, "memory", None)
    if not isinstance(mem, dict):
        return ""
    facts = [f for f in mem.get("facts") or [] if isinstance(f, dict)]
    summaries = mem.get("summaries") or {}

    obj_facts = [f for f in facts if object_id and f.get("object_id") == object_id]
    scene_other = [
        f for f in facts
        if f.get("scene_id") == scene_id and f.get("object_id") != object_id
    ]
    world_facts = [
        f for f in facts
        if f.get("scope") == "world" or f.get("kind") == "world"
    ]

    if purpose == "edit":
        # Same-object visual state ONLY — never world facts / raw dialogue.
        sel = [f for f in obj_facts if f.get("kind") in ("state", "appearance")]
        summary_keys = []
    elif purpose == "analyze":
        sel = (
            [f for f in obj_facts if f.get("kind") in ("state", "appearance", "summary")]
            + [f for f in scene_other if f.get("kind") in
               ("appearance", "navigation", "location")]
        )
        summary_keys = [f"object:{scene_id}:{object_id}", f"scene:{scene_id}"]
    elif purpose == "dialogue":
        sel = (
            [f for f in obj_facts if f.get("kind") != "summary"]
            + [f for f in scene_other if f.get("kind") in
               ("appearance", "state", "navigation", "relationship", "world")]
        )
        summary_keys = [f"object:{scene_id}:{object_id}", f"scene:{scene_id}"]
    elif purpose == "navigate":
        sel = (
            [f for f in obj_facts if f.get("kind") != "dialogue"]
            + [f for f in scene_other if f.get("kind") in
               ("appearance", "navigation", "location")]
            + [f for f in world_facts if f.get("kind") != "dialogue"]
        )
        summary_keys = [f"scene:{scene_id}", "world"]
    elif purpose == "custom":
        sel = [f for f in obj_facts if f.get("kind") != "summary"]
        summary_keys = [f"object:{scene_id}:{object_id}"]
    else:
        sel = []
        summary_keys = []

    # scope match first (object > scene > world is already the list order),
    # then recency: keep the MOST RECENT facts when trimming.
    seen = set()
    ordered: List[Dict[str, Any]] = []
    for f in sel:
        fid = f.get("id")
        if fid in seen:
            continue
        seen.add(fid)
        ordered.append(f)

    lines: List[str] = []
    used = 0
    for key in summary_keys:
        entry = summaries.get(key)
        text = entry.get("summary") if isinstance(entry, dict) else None
        if text:
            line = f"- (summary) {sanitize(text, 200)}"
            if used + len(line) + 1 > cap:
                break
            lines.append(line)
            used += len(line) + 1

    # newest-first selection under the cap, rendered oldest-first.
    chosen: List[str] = []
    for f in reversed(ordered):
        line = f"- {sanitize(f.get('fact'), 160)}"
        if used + len(line) + 1 > cap:
            break
        chosen.append(line)
        used += len(line) + 1
    lines.extend(reversed(chosen))
    out = "\n".join(lines)
    return out[:cap]

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
