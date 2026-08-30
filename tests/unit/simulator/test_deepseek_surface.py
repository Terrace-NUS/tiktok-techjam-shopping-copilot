from __future__ import annotations

import json
from pathlib import Path

from shopping_copilot.providers import HttpResponse
from shopping_copilot.simulator import DeepSeekSurfaceRealizer


class _FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    def post_json(
        self,
        *,
        url: str,
        headers: object,
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        del url, headers, timeout_seconds
        self.calls += 1
        request = json.loads(body)
        canonical = json.loads(request["messages"][-1]["content"])["canonical_message"]
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"message": f"natural: {canonical}"},
                            ensure_ascii=False,
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
        return HttpResponse(status=200, body=json.dumps(response).encode())


def _realizer(path: Path, transport: _FakeTransport) -> DeepSeekSurfaceRealizer:
    return DeepSeekSurfaceRealizer(
        api_key="test-key",
        cache_path=path,
        transport=transport,
    )


def test_surface_realizer_caches_and_persists(tmp_path: Path) -> None:
    cache = tmp_path / "surface.json"
    transport = _FakeTransport()
    realizer = _realizer(cache, transport)

    assert realizer.rewrite("canonical", "initial message") == "natural: canonical"
    assert realizer.rewrite("canonical", "initial message") == "natural: canonical"
    assert transport.calls == 1
    assert realizer.usage.as_payload() == {
        "api_calls": 1,
        "cache_hits": 1,
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }

    replay_transport = _FakeTransport()
    replay = _realizer(cache, replay_transport)
    assert replay.rewrite("canonical", "initial message") == "natural: canonical"
    assert replay_transport.calls == 0


def test_prewarm_deduplicates_requests(tmp_path: Path) -> None:
    transport = _FakeTransport()
    realizer = _realizer(tmp_path / "surface.json", transport)
    requests = [
        ("one", "initial message"),
        ("one", "initial message"),
        ("two", "follow-up customer reply"),
    ]

    realizer.prewarm(requests, max_workers=2)

    assert transport.calls == 2
    assert realizer.rewrite("one", "initial message") == "natural: one"
