"""CS7 catalog-bound session gateway, store, and envelope."""

from .envelope import (
    CATALOG_BOUND_SESSION_SCHEMA,
    CatalogBoundSessionEnvelope,
    decode_catalog_bound_session,
    encode_catalog_bound_session,
    replay_catalog_intent,
)
from .errors import CatalogGatewayError, CatalogGatewayErrorCode
from .gateway import CatalogSemanticGateway
from .store import (
    CatalogBoundSessionStore,
    CatalogBoundSessionTransaction,
    CatalogProbeToken,
)

__all__ = (
    "CATALOG_BOUND_SESSION_SCHEMA",
    "CatalogBoundSessionEnvelope",
    "CatalogBoundSessionStore",
    "CatalogBoundSessionTransaction",
    "CatalogGatewayError",
    "CatalogGatewayErrorCode",
    "CatalogProbeToken",
    "CatalogSemanticGateway",
    "decode_catalog_bound_session",
    "encode_catalog_bound_session",
    "replay_catalog_intent",
)
