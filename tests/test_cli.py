import json

from llmgw.cli import main
from llmgw.config import GatewayConfig, hash_key


def test_init_and_key_creation_store_only_a_hash(tmp_path, capsys):
    config = tmp_path / "gateway.json"
    assert main(["init", "--config", str(config)]) == 0
    assert main(["init", "--config", str(config)]) == 1  # never overwrites

    capsys.readouterr()
    assert main(["keys", "add", "research", "--config", str(config)]) == 0
    key = capsys.readouterr().out.strip()
    assert key.startswith("gw_research_")

    saved = config.read_text(encoding="utf-8")
    assert key not in saved
    loaded = GatewayConfig.load(config)
    assert loaded.teams["research"].key_hash == hash_key(key)
    assert json.loads(saved)["teams"][0]["key_hash"] == hash_key(key)


def test_report_on_an_empty_ledger(tmp_path, capsys):
    config = tmp_path / "gateway.json"
    main(["init", "--config", str(config)])
    assert main(["report", "--config", str(config), "--month", "2026-10"]) == 0
    assert "No usage recorded" in capsys.readouterr().out
