"""Consumer contracts through the real Gray Swan HTTP and failure-policy paths."""

from unittest.mock import Mock

import httpx
import pytest

from litellm.proxy.guardrails.guardrail_hooks.grayswan.grayswan import (
    GraySwanGuardrail,
    GraySwanGuardrailAPIError,
)
from litellm.types.guardrails import GuardrailEventHooks


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
@pytest.mark.parametrize("action", ["block", "monitor", "passthrough"])
@pytest.mark.parametrize("fail_open", [False, True])
@pytest.mark.parametrize(
    "payload", [{}, {"violation": None}, {"error": True, "violation": 0.0}]
)
async def test_application_errors_respect_failure_policy(
    monkeypatch, input_type, action, fail_open, payload
):
    guardrail = GraySwanGuardrail(
        guardrail_name="response-contract",
        api_key="test-key",
        policy_id="test-policy",
        fail_open=fail_open,
        on_flagged_action=action,
        event_hook=GuardrailEventHooks.pre_call,
    )
    log_failure = Mock()
    monkeypatch.setattr(guardrail, "_log_guardrail_failure", log_failure)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        guardrail.async_handler = client
        inputs = {"texts": ["synthetic input"]}
        if fail_open:
            assert await guardrail.apply_guardrail(inputs, {}, input_type) is inputs
        else:
            with pytest.raises(GraySwanGuardrailAPIError):
                await guardrail.apply_guardrail(inputs, {}, input_type)
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer test-key"
    log_failure.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"violation": "0"},
        {"violation": True},
        {"violation": -0.1},
        {"violation": 1.1},
        {"violation": float("nan")},
        {"violation": float("inf")},
        {"error": True, "violation": 1.0},
    ],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_invalid_decisions_are_errors(payload, legacy):
    guardrail = GraySwanGuardrail(guardrail_name="invalid-decision", api_key="test-key")
    with pytest.raises(GraySwanGuardrailAPIError):
        if legacy:
            guardrail._process_grayswan_response(payload)
        else:
            guardrail._process_response_internal(
                payload, {}, {"texts": ["hello"]}, False
            )


@pytest.mark.parametrize("score", [0, 0.49, 0.5, 1])
def test_valid_decisions_keep_threshold_semantics(score):
    from fastapi import HTTPException

    guardrail = GraySwanGuardrail(
        guardrail_name="valid-decision",
        api_key="test-key",
        violation_threshold=0.5,
        on_flagged_action="block",
    )
    inputs = {"texts": ["hello"]}
    payload = {
        "violation": score,
        "error": False,
        "observe_violation": 1.0,
        "block_message": "Informational only below threshold",
    }
    if score >= 0.5:
        with pytest.raises(HTTPException) as error:
            guardrail._process_response_internal(payload, {}, inputs, False)
        assert error.value.status_code == 400
    else:
        assert (
            guardrail._process_response_internal(payload, {}, inputs, False) is inputs
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "malformed_json", "http_error"])
@pytest.mark.parametrize("fail_open", [False, True])
async def test_transport_errors_respect_failure_policy(monkeypatch, failure, fail_open):
    guardrail = GraySwanGuardrail(
        guardrail_name="transport-contract",
        api_key="test-key",
        policy_id="test-policy",
        fail_open=fail_open,
        on_flagged_action="block",
    )
    log_failure = Mock()
    monkeypatch.setattr(guardrail, "_log_guardrail_failure", log_failure)

    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("controlled timeout", request=request)
        if failure == "malformed_json":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(503, json={"error": "controlled failure"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        guardrail.async_handler = client
        inputs = {"texts": ["synthetic input"]}
        if fail_open:
            assert await guardrail.apply_guardrail(inputs, {}, "request") is inputs
        else:
            with pytest.raises(GraySwanGuardrailAPIError):
                await guardrail.apply_guardrail(inputs, {}, "request")
    log_failure.assert_called_once()
