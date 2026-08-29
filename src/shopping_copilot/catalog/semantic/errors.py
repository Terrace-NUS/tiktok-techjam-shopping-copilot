"""Internal, deterministic failures for catalog-semantic build boundaries."""

from __future__ import annotations


class CatalogSemanticError(ValueError):
    """Base class for catalog-semantic validation failures."""


class CanonicalJsonError(CatalogSemanticError):
    """Raised when a value cannot be represented by the pinned JCS contract."""


class RawCatalogValidationError(CatalogSemanticError):
    """Raised when exact raw catalog bytes fail the release input gate."""

    def __init__(self, reason: str, *, line_number: int | None = None) -> None:
        self.reason = reason
        self.line_number = line_number
        location = "" if line_number is None else f" at physical line {line_number}"
        super().__init__(f"{reason}{location}")


class CatalogChangedError(CatalogSemanticError):
    """Raised when the raw catalog changes between identity and parse passes."""


class CategoryBuildError(CatalogSemanticError):
    """Raised when category graph or assignment materialization is invalid."""


class CategorySelectionError(CatalogSemanticError):
    """Raised when source-controlled scope selection is malformed or stale."""


class CategoryCodecError(CatalogSemanticError):
    """Raised when an artifact cannot be decoded exactly and canonically."""


class CategoryBundleIntegrityError(CatalogSemanticError):
    """Raised when a generated category candidate bundle is incoherent."""


class CategoryBundleBusyError(RuntimeError):
    """Raised when another writer owns a category bundle publication lock."""


class FacetProfileBuildError(CatalogSemanticError):
    """Raised when a CS2 source-profile proposal cannot be built safely."""


class FacetProfileSelectionError(CatalogSemanticError):
    """Raised when the source-controlled CS2 profiling selection is invalid or stale."""


class FacetProfileCodecError(CatalogSemanticError):
    """Raised when a CS2 profiling input cannot be decoded exactly."""


class FacetProfileBundleIntegrityError(CatalogSemanticError):
    """Raised when a generated CS2 source-profile bundle is incoherent."""


class FacetProfileBundleBusyError(RuntimeError):
    """Raised when another writer owns a CS2 source-profile publication lock."""


class GateABuildError(CatalogSemanticError):
    """Raised when reviewed Gate-A artifacts cannot be built safely."""


class GateASelectionError(CatalogSemanticError):
    """Raised when a reviewed Gate-A decision is malformed or stale."""


class GateACodecError(CatalogSemanticError):
    """Raised when a Gate-A input cannot be decoded exactly."""


class GateABundleIntegrityError(CatalogSemanticError):
    """Raised when a generated Gate-A candidate bundle is incoherent."""


class GateABundleBusyError(RuntimeError):
    """Raised when another writer owns a Gate-A candidate publication lock."""


class ResolutionBuildError(CatalogSemanticError):
    """Raised when CS3 evidence, resolution, or statistics cannot be built safely."""


class ResolutionCodecError(CatalogSemanticError):
    """Raised when a CS3 contract artifact cannot be decoded exactly."""


class ResolutionBundleIntegrityError(CatalogSemanticError):
    """Raised when a generated CS3 candidate bundle is incoherent."""


class ResolutionBundleBusyError(RuntimeError):
    """Raised when another writer owns a CS3 candidate publication lock."""


class GateBReviewBuildError(CatalogSemanticError):
    """Raised when a Gate-B review packet cannot be derived safely."""


class GateBReviewCodecError(CatalogSemanticError):
    """Raised when a Gate-B review artifact cannot be decoded exactly."""


class GateBReviewBundleIntegrityError(CatalogSemanticError):
    """Raised when a generated Gate-B review bundle is incoherent."""


class GateBReviewBundleBusyError(RuntimeError):
    """Raised when another writer owns a Gate-B review packet publication lock."""


class GateBSelectionError(CatalogSemanticError):
    """Raised when an owner-approved Gate-B selection is malformed or stale."""


class GateBBuildError(CatalogSemanticError):
    """Raised when approved Gate-B capabilities cannot be built safely."""


class GateBCodecError(CatalogSemanticError):
    """Raised when an approved Gate-B artifact cannot be decoded exactly."""


class GateBBundleIntegrityError(CatalogSemanticError):
    """Raised when an approved Gate-B candidate bundle is incoherent."""


class GateBBundleBusyError(RuntimeError):
    """Raised when another writer owns approved Gate-B bundle publication."""


class RuntimeProjectionBuildError(CatalogSemanticError):
    """Raised when approved semantics cannot project into the runtime contract."""


class RuntimeProjectionCodecError(CatalogSemanticError):
    """Raised when a runtime-projection artifact cannot be decoded exactly."""


class RuntimeProjectionBundleIntegrityError(CatalogSemanticError):
    """Raised when a generated runtime-projection bundle is incoherent."""


class RuntimeProjectionBundleBusyError(RuntimeError):
    """Raised when another writer owns runtime-projection publication."""


class ReleaseBuildError(CatalogSemanticError):
    """Raised when verified candidates cannot form one coherent release."""


class ReleaseCodecError(CatalogSemanticError):
    """Raised when a release manifest cannot be decoded exactly."""


class ReleaseBundleIntegrityError(CatalogSemanticError):
    """Raised when a release directory is incomplete, stale, or incoherent."""


class ReleaseBundleBusyError(RuntimeError):
    """Raised when another writer owns release publication."""
