"""The temporary command boundary cannot echo secret command-line values."""
import pytest

from wmmllm.cli import main


def test_cli_rejects_inline_secret_without_echo(capsys):
    key = "canary-argument-value-3461"
    with pytest.raises(SystemExit) as captured:
        main(["--api-key", key])
    assert captured.value.code == 2
    output = capsys.readouterr()
    assert key not in output.out + output.err
    assert "WMM_*" in output.err


def test_cli_version_requires_no_provider(capsys):
    with pytest.raises(SystemExit) as captured:
        main(["--version"])
    assert captured.value.code == 0
    assert capsys.readouterr().out.strip() == "0.1.0"
