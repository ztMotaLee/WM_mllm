import pytest
from wmmllm.cli import main


@pytest.mark.parametrize('argv',[
    ['check-config'], ['init','--image','seed.png','--run','unused'],
    ['step','--run','unused','Turn right.'], ['session','--run','unused','--script','turns.json']])
def test_live_cli_configuration_fail_fast(argv,monkeypatch,capsys):
    for name in ('WMM_LLM_API_KEY','WMM_IMAGE_API_KEY','WMM_LLM_MODEL','WMM_IMAGE_MODEL','WMM_LLM_BASE_URL','WMM_IMAGE_BASE_URL'):
        monkeypatch.delenv(name,raising=False)
    from wmmllm.providers import factory
    monkeypatch.setattr(factory,'create_providers',lambda *a,**k:pytest.fail('client construction before preflight'))
    assert main(argv)==2; assert 'Missing WMM_LLM_API_KEY' in capsys.readouterr().err


def test_check_config_presence_only(monkeypatch,capsys):
    monkeypatch.setenv('WMM_LLM_API_KEY','offline-config-presence')
    monkeypatch.setenv('WMM_LLM_MODEL','offline-selected-model')
    monkeypatch.setenv('WMM_LLM_BASE_URL','https://provider.invalid/v1')
    assert main(['check-config'])==0
    output=capsys.readouterr().out
    assert 'true' in output; assert 'offline-config-presence' not in output
    assert 'offline-selected-model' not in output


def test_runtime_error_is_sanitized_without_secondary_exception(tmp_path,capsys):
    assert main(['view','--run',str(tmp_path),'--yaw','0','--pitch','0',
                 '--out',str(tmp_path/'out.png')]) == 1
    output=capsys.readouterr().err
    assert '"status": "error"' in output; assert 'Traceback' not in output


def test_cli_rejects_inline_secret_without_echo(capsys):
    key='canary-argument-value-3461'
    with pytest.raises(SystemExit) as captured:main(['--api-key',key])
    assert captured.value.code==2
    output=capsys.readouterr()
    assert key not in output.out+output.err and 'WMM_*' in output.err


def test_cli_version_requires_no_provider(capsys):
    with pytest.raises(SystemExit) as captured:main(['--version'])
    assert captured.value.code==0 and capsys.readouterr().out.strip()=='0.1.0'
