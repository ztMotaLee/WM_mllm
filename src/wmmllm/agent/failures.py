"""Pure method failure classification; provider/transport errors cannot become holds."""
from __future__ import annotations
import re

_TRANSPORT_CLASSES = frozenset((
    'TransportError', 'TransportExhausted', 'TimeoutError', 'ConnectionError',
    'APIConnectionError', 'APITimeoutError', 'ConnectTimeout', 'ReadTimeout',
    'InternalServerError', 'RateLimitError', 'CallInterrupted',
))

_CONFIG_CLASSES = frozenset((
    'BadRequestError', 'APIStatusError', 'AuthenticationError', 'PermissionDeniedError',
    'NotFoundError', 'ConflictError', 'UnprocessableEntityError', 'RouteMismatch',
    'LocalCallError', 'JournalMismatch', 'InfrastructureHoldRejected',
    'TokenLimitError', 'MalformedResponse', 'SchemaError', 'JSONDecodeError',
))

_TECHNICAL_IMAGE_CLASSES = frozenset((
    'ImagePayloadInvalid', 'ReturnedQualityMismatch', 'AspectMismatch',
))

_PROVIDER_CLASSES = frozenset((
    'ProviderError', 'ProviderRefusal', 'RouteMismatch', 'LocalCallError', 'TokenLimitError',
    'BadRequestError', 'APIStatusError', 'AuthenticationError', 'PermissionDeniedError',
    'NotFoundError', 'ConflictError', 'UnprocessableEntityError',
))

_POLICY_CODES = frozenset((
    'contentpolicyviolation', 'contentpolicyrefusal', 'contentpolicyblocked',
    'moderationblocked', 'moderationrefusal', 'contentfilter', 'safetyviolation',
))

_CONTENT_POLICY = re.compile(
    r'(?:request was rejected by the safety system'
    r'|(?:blocked|rejected|refused) by (?:the )?safety (?:system|filters?)'
    r'|(?:request|image|content) (?:violates|violated) (?:the )?content policy)', re.I)

_POLICY_CODE_TEXT = re.compile(
    r'(?:code|error_class|finish_reason)[\"\']?\s*[:=]\s*[\"\']'
    r'(content_policy_violation|content_policy_refusal|content_policy_blocked'
    r'|moderation_blocked|moderation_refusal|content_filter|safety_violation)[\"\']', re.I)

_NEGATED_POLICY = re.compile(r'\b(?:no|not|never|without)\b', re.I)

def _policy_code(value):
    return isinstance(value, str) and re.sub(r'[_ -]', '', value).lower() in _POLICY_CODES

def _policy_message(value):
    if not isinstance(value, str):
        return False
    if _policy_code(value.strip()) or _POLICY_CODE_TEXT.search(value):
        return True
    return any(not _NEGATED_POLICY.search(re.split(r'[.!?;\n]',
               value[max(0, match.start() - 40):match.start()])[-1])
               for match in _CONTENT_POLICY.finditer(value))

def content_policy_refusal(value):
    """Require recorded policy evidence, rather than the ProviderRefusal class alone."""
    if isinstance(value, (list, tuple)):
        return any(content_policy_refusal(child) for child in value)
    if not isinstance(value, dict):
        return False
    codes = ('error_type', 'error_class', 'class', 'code', 'type', 'finish_reason')
    messages = ('error', 'error_message', 'message', 'refusal')
    if any(_policy_code(value.get(field)) for field in codes) or any(
            _policy_message(value.get(field)) for field in messages):
        return True
    # Inspect provider error bodies only; ordinary request prompts are not policy evidence.
    return any(content_policy_refusal(value[field]) for field in
               ('raw', 'body', 'error', 'record', 'causes', 'original_error', 'original_error_record',
                'detail', 'errors', 'choices', 'data', 'response', 'message')
               if isinstance(value.get(field), (dict, list, tuple)))

def _image_contract_failure(value):
    if isinstance(value, (list, tuple)):
        return any(_image_contract_failure(child) for child in value)
    if not isinstance(value, dict):
        return False
    names = {str(value.get(field, '')).rsplit('.', 1)[-1]
             for field in ('error_type', 'error_class', 'class')}
    return bool(names & _TECHNICAL_IMAGE_CLASSES) or any(
        _image_contract_failure(child) for child in value.values()
        if isinstance(child, (dict, list, tuple)))

def provider_technical_failure(value):
    """No usable provider result is a transport failure unless policy refusal is explicit."""
    if isinstance(value, (list, tuple)):
        return any(provider_technical_failure(child) for child in value)
    if not isinstance(value, dict):
        return False
    if _image_contract_failure(value):
        return True
    names = {str(value.get(field, '')).rsplit('.', 1)[-1]
             for field in ('error_type', 'error_class', 'class')}
    # A real policy refusal may be wrapped in a generic HTTP/provider cause.
    # Explicit image-contract errors still override that policy evidence.
    if content_policy_refusal(value):
        return False
    kinds = {str(value.get(field, '')).lower()
             for field in ('failure_kind', 'error_kind', 'kind')}
    provider = bool(names & _PROVIDER_CLASSES or
                    kinds & {'provider', 'refusal', 'local', 'token_limit'})
    if provider and not content_policy_refusal(value):
        return True
    return any(provider_technical_failure(child) for child in value.values()
               if isinstance(child, (dict, list, tuple)))

_HTTP = re.compile(r'(?:HTTP(?:\s+status)?|Error\s+code|status(?:_code)?)\s*[:=]?\s*[\"\']?([45]\d\d)\b', re.I)

_CONFIG = re.compile(
    r'(?:AE\s+(?:ultra|max)\s+configuration\s+(?:was\s+)?rejected'
    r'|Invalid\s+value\s*:\s*[\"\']?(?:ultra|max)\b'
    r'|(?:reasoning[._ ]?effort|reasoning\.effort).*(?:invalid|unsupported|rejected|must\s+send)'
    r'|AE\s+chat\s+client\s+must\s+send'
    r'|AE\s+chat\s+result\s+has\s+no\s+(?:ultra|max)\s+request\s+evidence'
    r'|prohibited\s+model\s+route'
    r'|(?:ultra|max)\s+gateway\s+acceptance\s+is\s+required'
    r'|Persisted\s+(?:ultra|max)\s+rejection)', re.I | re.S)

def exception_record(error):
    """Keep status and nested causes that a generic stage error would otherwise hide."""
    seen = set()

    def record(current):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        result = {'error_type': type(current).__name__, 'error': str(current)}
        for name in ('http_status', 'status_code', 'error_class', 'kind', 'failure_kind'):
            value = getattr(current, name, None)
            if isinstance(value, (str, int, float, bool)):
                result[name] = value
        for name in ('raw', 'body'):
            value = getattr(current, name, None)
            if isinstance(value, (dict, list)):
                result[name] = value
        for name in ('hold_contract', 'attempts'):
            value = getattr(current, name, None)
            if isinstance(value, (dict, list)):
                result[name] = value
        causes = [record(getattr(current, name, None)) for name in
                  ('__cause__', '__context__', 'original_error')]
        if any(causes):
            result['causes'] = [cause for cause in causes if cause]
        return result

    return record(error)

def infrastructure_failure(value, _policy_context=False):
    """Return the first blocking class; the method allowlist remains separate."""
    if isinstance(value, (list, tuple)):
        return next((blocked for child in value if (blocked := infrastructure_failure(child, _policy_context))), None)
    if not isinstance(value, dict):
        return None
    if provider_technical_failure(value) and (not _policy_context or _image_contract_failure(value)):
        return {'classification': 'transport'}
    policy_refusal = _policy_context or content_policy_refusal(value)
    for field in ('http_status', 'status_code'):
        status = value.get(field)
        if str(status).isdigit() and 400 <= int(status) <= 599:
            if policy_refusal and 400 <= int(status) < 500 and int(status) != 429:
                continue
            return {'classification': 'http_failure', 'http_status': int(status)}
    kinds = {str(value.get(field, '')).lower() for field in ('failure_kind', 'error_kind', 'kind')}
    if 'transport' in kinds or value.get('transport_error') is True:
        return {'classification': 'transport'}
    if kinds & {'configuration', 'config', 'http', 'infrastructure', 'local', 'token_limit', 'schema', 'parse'}:
        return {'classification': 'configuration_failure'}
    names = {str(value.get(field, '')).rsplit('.', 1)[-1]
             for field in ('error_type', 'error_class', 'class')}
    if names & _TRANSPORT_CLASSES:
        return {'classification': 'transport'}
    if names & _CONFIG_CLASSES and not policy_refusal:
        return {'classification': 'configuration_failure'}
    for field in ('error', 'error_message', 'message', 'repair_error'):
        message = value.get(field)
        if isinstance(message, str):
            match = _HTTP.search(message)
            if match:
                if policy_refusal and 400 <= int(match.group(1)) < 500 and int(match.group(1)) != 429:
                    continue
                return {'classification': 'http_failure', 'http_status': int(match.group(1))}
            if _CONFIG.search(message):
                return {'classification': 'configuration_failure'}
    # Only mappings/lists recurse; ordinary prompts and semantic reasons are not error messages.
    return next((blocked for child in value.values() if isinstance(child, (dict, list, tuple))
                 and (blocked := infrastructure_failure(child, policy_refusal))), None)

def classify_failure(error_type, error, error_dict=None):
    """Only approved rejection classes become holds; transport/schema errors do not."""
    record = error_dict or {}
    (name, text) = (str(error_type or record.get('error_type') or ''), str(error or record.get('error') or ''))
    blocked = infrastructure_failure({'record': record, 'error_type': name, 'error': text})
    if blocked:
        return {'classification': blocked['classification'], 'hold_eligible': False, 'bootstrap_hold': False}
    kind = None
    if name.rsplit('.', 1)[-1] == 'IdentityFirstProposalRejected':
        kind = 'unclassified'
    elif record.get('failure_kind') == 'transport' or name in ('TransportExhausted', 'TimeoutError', 'ConnectionError'):
        kind = 'transport'
    elif name.rsplit('.', 1)[-1] == 'RegistrationFailure' or text in ('source registration fails v5 distributed support/orientation gates', 'registered context fails v5 support/orientation/complete coverage gates'):
        kind = 'bootstrap_registration'
    elif name.rsplit('.', 1)[-1] == 'ProviderRefusal' or any((token in text.lower() for token in ('moderation_blocked', 'request was rejected by the safety system', 'content_policy_violation'))):
        kind = 'provider_refusal'
    elif text == 'endpoint rejected after exactly one semantic retry':
        kind = 'semantic_retry_rejected'
    elif text.startswith('Geometric endpoint semantic verification failed'):
        kind = 'geometric_semantic_rejected'
    elif text == 'Geometric observer failed':
        kind = 'geometric_observer_rejected'
    elif text == 'Saved return semantic check failed':
        kind = 'saved_return_semantic_rejected'
    elif text == 'Saved return observer failed':
        kind = 'saved_return_observer_rejected'
    elif text == 'accepted endpoint observer failed':
        kind = 'endpoint_observer_rejected'
    elif text == 'empty actual generated support':
        kind = 'generated_support_rejected'
    elif text == 'empty actual permitted support':
        kind = 'planner_noop_missing_premise'
    elif text in ('Current editor source differs from the failed proposal transition', 'planner scope_bbox must be a nonempty normalized rectangle', 'No unique metadata binding for unresolved subject'):
        kind = 'technical_schema_or_source'
    else:
        kind = 'unclassified'
    eligible = kind not in ('transport', 'technical_schema_or_source', 'unclassified')
    result = {'classification': kind, 'hold_eligible': eligible, 'bootstrap_hold': kind == 'bootstrap_registration'}
    if kind == 'planner_noop_missing_premise':
        result['classification_label'] = 'planner no-op / missing premise'
    return result
