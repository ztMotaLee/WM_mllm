"""Frozen endpoint protocol and validators; pure action selection only."""
from __future__ import annotations
import json, re
from . import read_text, read_json_data

_MULTI_SHOT = re.compile('\\b(?:jump[- ]?cuts?|match[- ]?cuts?|cross[- ]?(?:cuts?|dissolv(?:e|es|ing)|fades?)|montages?|split[- ]screens?|shot[- ]reverse[- ]shot)\\b|\\b(?:cuts?|cutting|dissolv(?:e|es|ing)|fades?|fading|wipes?|wiping)\\s+(?:in\\b|out\\b|(?:away\\s+)?to\\b|into\\b|between\\b)|\\b(?:two|three|four|several|multiple|different|\\d+)\\s+(?:separate\\s+|distinct\\s+)?(?:shots?|scenes?)\\b|\\b(?:next|second|third|another|new)\\s+shot\\b|\\b(?:the\\s+)?(?:shot|scene)\\s+(?:changes?|switch(?:es)?|transitions?)\\b|\\b(?:switch(?:es|ing)?|transition(?:s|ing)?|cut(?:s|ting)?|jump(?:s|ing)?)\\s+(?:to\\s+|between\\s+)(?:(?:a|the|another|different|new|next)\\s+){0,2}(?:shots?|scenes?)\\b|多镜头|多个镜头|两[个段]?镜头|三[个段]?镜头|蒙太奇|交叉剪辑|跳切|硬切|叠化|淡入|淡出|转场|下一个镜头|第二个?镜头|(?:切换|转换|切到|切至)[^。！？；\\n]{0,12}(?:镜头|场景)', re.IGNORECASE)
_NEGATED_CHANGE = re.compile("(?:\\b(?:no|without|avoid|avoiding|never)|\\bdo\\s+not|\\bdon['’]t|\\bmust\\s+not|\\bshould\\s+not|不要|不得|禁止|避免|没有|无需)(?:\\s+\\w+){0,3}\\s*$", re.IGNORECASE)

def explicit_multiple_shots(action):
    patterns = [_MULTI_SHOT, re.compile('\\b(?:wide|close[- ]up|medium|tracking)\\s+shots\\b', re.I)]
    for pattern in patterns:
        for match in pattern.finditer(action):
            prefix = re.split('[.!?;。！？；]', action[:match.start()])[-1]
            if not _NEGATED_CHANGE.search(prefix):
                return True
    return False
PLAN_FIELDS = ('current_state', 'remaining_action', 'endpoint_instruction')
SEMANTIC_FIELDS = ('state_matches_source', 'action_preserves_request', 'endpoint_matches_action', 'scene_identity_preserved')
PLANNER_SYSTEM = read_text('policy_001.txt')
VERIFIER_COMMON = read_text('policy_002.txt')
VERIFIER_SYSTEM = VERIFIER_COMMON + read_text('policy_003.txt')
MULTISHOT_VERIFIER_SYSTEM = VERIFIER_COMMON + read_text('policy_004.txt')

def verifier_system(current_caption):
    return MULTISHOT_VERIFIER_SYSTEM if explicit_multiple_shots(current_caption) else VERIFIER_SYSTEM

def planner_system(current_caption, duration):
    if not explicit_multiple_shots(current_caption):
        return PLANNER_SYSTEM
    return PLANNER_SYSTEM + f' The current caption explicitly specifies multiple shots; fit them within {duration:.6f} seconds. Return h3_shots in order, each with action and cut_time_seconds; first time is 0, later times strictly increase before the end. Later actions begin "The camera cuts to" or the explicitly requested transition. remaining_action must exactly join those action strings with single spaces. Cut timing is your implementation schedule, not a timestamp supplied by the benchmark.'

def validate_requested_shots(plan, current_caption):
    expected = explicit_multiple_shots(current_caption)
    if expected != ('h3_shots' in plan):
        raise ValueError('A shot schedule is required only for explicitly requested multiple shots')

VERIFIER_RESPONSE_FORMAT = read_json_data('verifier_response_format.json')

def parse_response(text):
    """Pure equivalent of the pinned native provider's _parse_json parser."""
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    fence = re.search('```(?:json)?\\s*(\\{.*?\\})\\s*```', text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        match = re.search('\\{.*\\}', text, re.DOTALL)
        candidate = match.group(0) if match else None
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _paragraph(value, name):
    if not isinstance(value, str) or not value.strip() or (not re.search('[A-Za-z]{2,}', value)) or any((ord(c) < 32 or c in '\x85\u2028\u2029' for c in value)):
        raise ValueError(name + ' must be a nonempty English paragraph without control characters')
    return value.strip()

def planner_input(preceding_captions, current_caption, memory_context=None):
    if not isinstance(preceding_captions, list) or not all((isinstance(x, str) for x in preceding_captions)):
        raise ValueError('preceding_captions must be a list of already supplied caption strings')
    if not isinstance(current_caption, str) or not current_caption.strip():
        raise ValueError('current_caption must be the nonempty original caption')
    result = {'preceding_captions': list(preceding_captions), 'current_caption': current_caption}
    if memory_context is not None:
        result['memory_context'] = memory_context
    return result
_IMAGE_REFERENCE = re.compile(read_text('policy_007.txt'), re.I)

def validate_standalone_description(value, name='action'):
    """Reject unresolved cross-stage image references; do not rewrite model text."""
    if _IMAGE_REFERENCE.search(value):
        raise ValueError(name + read_text('policy_008.txt'))
    return value

def validate_plan(value):
    if not isinstance(value, dict) or set(value) not in (set(PLAN_FIELDS), set(PLAN_FIELDS) | {'h3_shots'}):
        raise ValueError(read_text('policy_009.txt'))
    plan = {key: _paragraph(value[key], key) for key in PLAN_FIELDS}
    plan['remaining_action'] = value['remaining_action']
    if re.search('\\b(?:the|this|that)\\s+(?:subject|object)\\b', plan['endpoint_instruction'], re.I):
        raise ValueError(read_text('policy_010.txt'))
    if 'h3_shots' in value:
        if not isinstance(value['h3_shots'], list):
            raise ValueError('h3_shots must be an ordered list')
        plan['h3_shots'] = value['h3_shots']
    for key in ('remaining_action', 'endpoint_instruction'):
        validate_standalone_description(plan[key], key)
        if re.search('\\b(?:requested|specified|given) (?:amount|angle|distance|direction|action)\\b', plan[key], re.I):
            raise ValueError(key + ' contains a reference that cannot be resolved from its images alone')
    return plan

def validate_semantic_verification(value):
    if not isinstance(value, dict) or set(value) != set(SEMANTIC_FIELDS) | {'reason'}:
        raise ValueError('Semantic verifier returned an invalid schema')
    if any((type(value[key]) is not str or value[key] not in {'pass', 'fail', 'uncertain'} for key in SEMANTIC_FIELDS)):
        raise ValueError(read_text('policy_011.txt'))
    if not isinstance(value['reason'], str):
        raise ValueError('Semantic explanation must be a string')
    return {**{key: value[key] for key in SEMANTIC_FIELDS}, 'reason': value['reason']}
def semantic_decision(checked):
    """Three-state verdict of a validated response: any FAIL fails; otherwise any UNCERTAIN is labeled."""
    failed = [key for key in SEMANTIC_FIELDS if checked[key] == 'fail']
    uncertain = [key for key in SEMANTIC_FIELDS if checked[key] == 'uncertain']
    return {'semantic_status': 'FAIL' if failed else 'UNCERTAIN' if uncertain else 'PASS',
            'uncertain_criteria': uncertain, 'failed_criteria': failed}

def image_prompt(plan):
    """Render one final frame of an accepted plan; the motion/shot timeline belongs to H3."""
    return ('Edit Image 1 into the ending frame: ' + plan['endpoint_instruction']
            + ' Keep identities consistent where visible. Output one image.')

def verification_input(context, plan):
    plan = validate_plan(plan)
    validate_requested_shots(plan, context['current_caption'])
    value = planner_input(context['preceding_captions'], context['current_caption'], context.get('memory_context'))
    value['proposed_current_state'] = plan['current_state']
    # Deterministic route; single-shot planner prose cannot override the caption.
    value['actual_h3_action'] = plan['remaining_action'] if 'h3_shots' in plan else context['current_caption']
    if 'h3_shots' in plan:
        value['proposed_shot_schedule_to_check_against_current_caption'] = plan['h3_shots']
        value['verification_scope'] = {'action_check': read_text('policy_015.txt'), 'endpoint_target_shot_number': len(plan['h3_shots']), 'earlier_shots_need_not_appear_in_endpoint': True, 'actual_video_execution_verified': False}
    return value
