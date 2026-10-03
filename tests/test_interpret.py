"""Tests for the Claude interpretation layer, using a fake client (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.interpret import narrator as N
from src.interpret.snapshot import build_snapshot, snapshot_hash


class FakeClient:
    """Records requests; answers with a fixed text (or a refusal)."""

    def __init__(self, text="**Calm, put-skewed surface.**\n\n#### Surface\n...", refuse=False):
        self.calls = []
        self.text, self.refuse = text, refuse
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.refuse:
            return SimpleNamespace(stop_reason="refusal", content=[],
                                   stop_details=SimpleNamespace(category="general_harms"))
        return SimpleNamespace(
            stop_reason="end_turn", model=kw["model"],
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=2500, output_tokens=400,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


@pytest.fixture
def snapshot(stores):
    from src.features.table import build_feature_table

    feats = build_feature_table(surfaces_dir=stores["surfaces_dir"],
                                underlying_dir=stores["underlying_dir"], vix_dir=stores["vix_dir"])
    return build_snapshot(feats, pd.DataFrame(), "TEST", stores["dates"][-1]), feats


# ── Snapshot ─────────────────────────────────────────────────────────────


def test_snapshot_uses_explicit_units(snapshot):
    snap, feats = snapshot
    row = feats.iloc[-1]
    assert snap["implied_vol_pct"]["atm_30d"] == pytest.approx(100 * row["atm_30d"], abs=0.01)
    assert snap["smile_30d"]["risk_reversal_25d_pts"] == pytest.approx(100 * row["rr25_30d"], abs=0.01)
    assert snap["history_available_days"] == 5
    assert snap["volatility_forecast"] is None        # no forecasts supplied


def test_snapshot_has_no_nan(snapshot):
    import json

    snap, _ = snapshot
    assert "NaN" not in json.dumps(snap)


def test_snapshot_hash_ignores_key_order():
    assert snapshot_hash({"a": 1, "b": 2}, "v") == snapshot_hash({"b": 2, "a": 1}, "v")
    assert snapshot_hash({"a": 1}, "v1") != snapshot_hash({"a": 1}, "v2")


# ── Request and caching ──────────────────────────────────────────────────


def test_request_shape(snapshot, tmp_path):
    snap, _ = snapshot
    client = FakeClient()
    it = N.interpret(snap, client=client, cache_dir=str(tmp_path))
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["system"] == N.SYSTEM_PROMPT
    assert '"ticker": "TEST"' in kw["messages"][0]["content"]
    assert it.text.startswith("**Calm") and it.usage["output_tokens"] == 400


def test_cached_result_is_reused(snapshot, tmp_path):
    snap, _ = snapshot
    client = FakeClient()
    N.interpret(snap, client=client, cache_dir=str(tmp_path))
    N.interpret(snap, client=client, cache_dir=str(tmp_path))
    assert len(client.calls) == 1
    cached, current = N.load_cached(snap, str(tmp_path))
    assert cached is not None and current


def test_changed_data_marks_cache_stale_and_regenerates(snapshot, tmp_path):
    snap, _ = snapshot
    client = FakeClient()
    N.interpret(snap, client=client, cache_dir=str(tmp_path))
    changed = {**snap, "spot": 123.0}
    assert N.load_cached(changed, str(tmp_path)) == (N.load_cached(changed, str(tmp_path))[0], False)
    N.interpret(changed, client=client, cache_dir=str(tmp_path))
    assert len(client.calls) == 2


def test_force_regenerates(snapshot, tmp_path):
    snap, _ = snapshot
    client = FakeClient()
    N.interpret(snap, client=client, cache_dir=str(tmp_path))
    N.interpret(snap, client=client, cache_dir=str(tmp_path), force=True)
    assert len(client.calls) == 2


def test_refusal_raises_and_is_not_cached(snapshot, tmp_path):
    snap, _ = snapshot
    with pytest.raises(RuntimeError, match="declined"):
        N.interpret(snap, client=FakeClient(refuse=True), cache_dir=str(tmp_path))
    assert N.load_cached(snap, str(tmp_path))[0] is None


def test_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(N.MissingCredentials, match="ANTHROPIC_API_KEY"):
        N._client()


def test_workspace_id_is_sent_as_a_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    client = N._client()
    assert client.default_headers.get("anthropic-workspace-id") == "wrkspc_123"


def test_no_workspace_header_without_the_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert "anthropic-workspace-id" not in N._client().default_headers


@pytest.mark.parametrize("api_message, hint", [
    ("This API key is not scoped to a workspace, so this request must include the "
     "anthropic-workspace-id header", "ANTHROPIC_WORKSPACE_ID"),
    ("Your credit balance is too low to access the Anthropic API.", "credit"),
])
def test_account_setup_errors_become_instructions(snapshot, tmp_path, api_message, hint):
    snap, _ = snapshot

    class Failing(FakeClient):
        def _create(self, **kw):
            raise RuntimeError(api_message)

    with pytest.raises(N.MissingCredentials, match=hint):
        N.interpret(snap, client=Failing(), cache_dir=str(tmp_path))
