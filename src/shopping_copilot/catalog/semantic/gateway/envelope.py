"""Strict outer session envelope pinned to one catalog-semantic release."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal, NoReturn, TypeAlias, cast

from shopping_copilot.session_context import (
    IntentState,
    SessionContext,
    decode_snapshot,
    encode_snapshot,
)

from ..canonical import canonical_json_bytes
from .equality import exact_domain_equal
from .errors import CatalogGatewayError, CatalogGatewayErrorCode
from .gateway import CatalogSemanticGateway

CatalogBoundSessionSchema: TypeAlias = Literal["shopping-copilot/catalog-bound-session/v0"]

CATALOG_BOUND_SESSION_SCHEMA: CatalogBoundSessionSchema = (
    "shopping-copilot/catalog-bound-session/v0"
)

_RELEASE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SNAPSHOT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BASE64URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogBoundSessionEnvelope:
    """Canonical wire wrapper for one unchanged session-context v1 snapshot."""

    schema: CatalogBoundSessionSchema
    session_id: str
    catalog_semantic_release_id: str
    session_snapshot_sha256: str
    session_snapshot_base64url: str

    def __post_init__(self) -> None:
        if self.schema != CATALOG_BOUND_SESSION_SCHEMA:
            raise ValueError("CatalogBoundSessionEnvelope.schema is invalid")
        if type(self.session_id) is not str or not self.session_id.strip():
            raise ValueError("CatalogBoundSessionEnvelope.session_id is invalid")
        if self.session_id != self.session_id.strip():
            raise ValueError("CatalogBoundSessionEnvelope.session_id is not canonical")
        if (
            type(self.catalog_semantic_release_id) is not str
            or _RELEASE_ID_PATTERN.fullmatch(self.catalog_semantic_release_id) is None
        ):
            raise ValueError("CatalogBoundSessionEnvelope release ID is invalid")
        if (
            type(self.session_snapshot_sha256) is not str
            or _SNAPSHOT_HASH_PATTERN.fullmatch(self.session_snapshot_sha256) is None
        ):
            raise ValueError("CatalogBoundSessionEnvelope snapshot hash is invalid")
        encoded = self.session_snapshot_base64url
        if (
            type(encoded) is not str
            or _BASE64URL_PATTERN.fullmatch(encoded) is None
            or len(encoded) % 4 == 1
        ):
            raise ValueError("CatalogBoundSessionEnvelope base64url is invalid")


def encode_catalog_bound_session(
    context: SessionContext,
    gateway: CatalogSemanticGateway,
) -> bytes:
    """Encode a replay-verified session snapshot under the gateway's release ID."""

    _validate_gateway_context(context, gateway)
    inner = encode_snapshot(context, gateway.registry)
    envelope = CatalogBoundSessionEnvelope(
        schema=CATALOG_BOUND_SESSION_SCHEMA,
        session_id=context.session_id,
        catalog_semantic_release_id=gateway.release_id,
        session_snapshot_sha256=hashlib.sha256(inner).hexdigest(),
        session_snapshot_base64url=base64.urlsafe_b64encode(inner).rstrip(b"=").decode("ascii"),
    )
    return canonical_json_bytes(envelope)


def decode_catalog_bound_session(
    data: bytes,
    gateway: CatalogSemanticGateway,
) -> SessionContext:
    """Decode in deterministic envelope, release, hash, snapshot, and replay order."""

    document = _load_canonical_envelope(data)
    envelope = _materialize_envelope(document)
    gateway.require_release(envelope.catalog_semantic_release_id)
    inner = _strict_base64url_decode(envelope.session_snapshot_base64url)
    if hashlib.sha256(inner).hexdigest() != envelope.session_snapshot_sha256:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.SESSION_SNAPSHOT_HASH_MISMATCH,
            path=("session_snapshot_sha256",),
        )
    context = decode_snapshot(inner, gateway.registry)
    if encode_snapshot(context, gateway.registry) != inner:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.INVALID_SESSION_ENVELOPE,
            path=("session_snapshot_base64url",),
        )
    if context.session_id != envelope.session_id:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH,
            path=("session_id",),
        )
    _validate_gateway_context(context, gateway)
    return context


def replay_catalog_intent(
    context: SessionContext,
    gateway: CatalogSemanticGateway,
) -> IntentState:
    """Replay accepted batches through the catalog boundary, not only the raw reducer."""

    current = IntentState(
        goal=None,
        preferences=(),
        dont_care_facets=frozenset(),
        version=0,
    )
    for record in context.state.interaction.turns:
        if record.accepted_update is not None:
            current = gateway.preview(
                current,
                record.accepted_update,
                catalog_semantic_release_id=gateway.release_id,
            )
    return current


def _validate_gateway_context(
    context: SessionContext,
    gateway: CatalogSemanticGateway,
) -> None:
    replayed = replay_catalog_intent(context, gateway)
    if not exact_domain_equal(replayed, context.state.intent):
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.SESSION_REPLAY_MISMATCH,
            path=("state", "intent"),
        )
    gateway.validate_intent(
        context.state.intent,
        catalog_semantic_release_id=gateway.release_id,
    )
    belief = context.state.search_belief
    if belief is not None:
        gateway.validate_search_belief(
            belief,
            intent=context.state.intent,
            catalog_semantic_release_id=gateway.release_id,
        )


def _load_canonical_envelope(data: bytes) -> dict[str, object]:
    if type(data) is not bytes or data.startswith(b"\xef\xbb\xbf"):
        _invalid_envelope()
    try:
        parsed: object = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_object_without_duplicates,
        )
        if type(parsed) is not dict or data != canonical_json_bytes(parsed):
            _invalid_envelope()
    except CatalogGatewayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        _invalid_envelope()
    return cast(dict[str, object], parsed)


def _materialize_envelope(document: dict[str, object]) -> CatalogBoundSessionEnvelope:
    expected = {
        "schema",
        "session_id",
        "catalog_semantic_release_id",
        "session_snapshot_sha256",
        "session_snapshot_base64url",
    }
    if set(document) != expected or any(type(value) is not str for value in document.values()):
        _invalid_envelope()
    try:
        return CatalogBoundSessionEnvelope(
            schema=cast(CatalogBoundSessionSchema, document["schema"]),
            session_id=cast(str, document["session_id"]),
            catalog_semantic_release_id=cast(str, document["catalog_semantic_release_id"]),
            session_snapshot_sha256=cast(str, document["session_snapshot_sha256"]),
            session_snapshot_base64url=cast(str, document["session_snapshot_base64url"]),
        )
    except (TypeError, ValueError) as error:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.INVALID_SESSION_ENVELOPE,
        ) from error


def _strict_base64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.INVALID_SESSION_ENVELOPE,
            path=("session_snapshot_base64url",),
        ) from error
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != value:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.INVALID_SESSION_ENVELOPE,
            path=("session_snapshot_base64url",),
        )
    return decoded


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object member")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON token: {value}")


def _invalid_envelope() -> NoReturn:
    raise CatalogGatewayError(code=CatalogGatewayErrorCode.INVALID_SESSION_ENVELOPE)
