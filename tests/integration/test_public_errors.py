"""Public errors describe the endpoint check using documented terminology."""
import pytest

from wmmllm.agent.navigation import CRITERIA, navigation_decision


def test_navigation_semantic_error_uses_public_terminology():
    invalid = {'criteria': {key: 'PASS' for key in CRITERIA}, 'semantic': {},
               'reason': '', 'retry_instruction': ''}
    with pytest.raises(ValueError, match=r'^Invalid endpoint semantic fields\.$'):
        navigation_decision(invalid)
