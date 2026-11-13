import pytest
from wmmllm.cli import main


@pytest.mark.parametrize('argv',[
    ['check-config'], ['init','--image','seed.png','--run','unused'],
    ['step','--run','unused','Turn right.'], ['session','--run','unused','--script','turns.json']])
def test_live_cli_configuration_fail_fast(argv,monkeypatch,capsys):
    for name in ('WMM_LLM_API_KEY','WMM_IMAGE_API_KEY','WMM_LLM_MODEL','WMM_IMAGE_MODEL'):
        monkeypatch.delenv(name,raising=False)
    from wmmllm.providers import factory
    monkeypatch.setattr(factory,'create_providers',lambda *a,**k:pytest.fail('client construction before preflight'))
    assert main(argv)==2
    assert 'Missing WMM_LLM_API_KEY' in capsys.readouterr().err


def test_check_config_presence_only(monkeypatch,capsys):
    monkeypatch.setenv('WMM_LLM_API_KEY','offline-config-presence')
    monkeypatch.setenv('WMM_LLM_MODEL','offline-selected-model')
    assert main(['check-config'])==0
    output=capsys.readouterr().out
    assert 'true' in output
    assert 'offline-config-presence' not in output
    assert 'offline-selected-model' not in output


def test_runtime_error_is_sanitized_without_secondary_exception(tmp_path,capsys):
    assert main(['view','--run',str(tmp_path),'--yaw','0','--pitch','0',
                 '--out',str(tmp_path/'out.png')]) == 1
    output=capsys.readouterr().err
    assert '"status": "error"' in output
    assert 'Traceback' not in output
