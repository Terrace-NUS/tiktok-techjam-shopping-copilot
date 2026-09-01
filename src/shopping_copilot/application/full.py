"""Construction of the full APERTURE execution profile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from .contracts import AgentDelegate

DEFAULT_BGE_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@dataclass(frozen=True, slots=True, kw_only=True)
class FullApertureConfig:
    """Explicit dependencies for the full QU-to-ranking system."""

    api_key: str
    semantic_release: Path = Path("artifacts/catalog-semantic/release-v0")
    dense_index: Path = Path("artifacts/retrieval/dense-v0")
    density_cache: Path = Path("artifacts/retrieval/intent-volume-density-v0.npz")
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 45.0
    max_tokens: int = 2048
    ranking_timeout_seconds: float = 90.0
    ranking_max_tokens: int = 8192
    device: str = "cuda"
    cross_encoder: bool = True
    quality_ranking: bool = True
    product_card_sidecar: Path | None = None
    product_card_mode: str = "replace"
    qu_retry_count: int = 3
    repeat_noop_cache: bool = True

    def __post_init__(self) -> None:
        if type(self.api_key) is not str or not self.api_key.strip():
            raise ValueError("full mode requires a non-empty DeepSeek API key")
        if type(self.model) is not str or not self.model.strip():
            raise ValueError("model must be non-empty")
        if type(self.base_url) is not str or not self.base_url.startswith("https://"):
            raise ValueError("base_url must be HTTPS")
        if type(self.timeout_seconds) not in (int, float) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if (
            type(self.ranking_timeout_seconds) not in (int, float)
            or self.ranking_timeout_seconds <= 0
        ):
            raise ValueError("ranking_timeout_seconds must be positive")
        if type(self.ranking_max_tokens) is not int or self.ranking_max_tokens < 1:
            raise ValueError("ranking_max_tokens must be positive")
        if type(self.device) is not str or not self.device.strip():
            raise ValueError("device must be non-empty")
        if type(self.cross_encoder) is not bool:
            raise TypeError("cross_encoder must be a bool")
        if type(self.quality_ranking) is not bool:
            raise TypeError("quality_ranking must be a bool")
        if self.product_card_mode not in {"augment", "replace"}:
            raise ValueError("product_card_mode must be 'augment' or 'replace'")
        if type(self.qu_retry_count) is not int or self.qu_retry_count < 1:
            raise ValueError("qu_retry_count must be positive")
        if type(self.repeat_noop_cache) is not bool:
            raise TypeError("repeat_noop_cache must be a bool")


def build_full_aperture_agent(
    catalog_path: str | Path,
    config: FullApertureConfig,
) -> AgentDelegate:
    """Build the already-tested full pipeline only when explicitly requested.

    Imports are intentionally lazy: the default offline profile must not
    import Torch, sentence-transformers, CUDA runtimes, or API transports.
    """

    from shopping_copilot.catalog.product_facts import load_product_fact_sidecar
    from shopping_copilot.catalog.semantic import CatalogSemanticGateway
    from shopping_copilot.catalog.semantic.release import load_catalog_semantic_release
    from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
    from shopping_copilot.query_compiler import QueryCompiler
    from shopping_copilot.query_understanding import (
        IntentMaterializer,
        QueryUnderstandingService,
        category_options_from_registry,
    )
    from shopping_copilot.query_understanding.deepseek import DeepSeekConfig, DeepSeekProvider
    from shopping_copilot.retrieval import (
        CrossEncoderRelevanceReranker,
        GreedyDPPSelector,
        IntentVolumeEstimator,
        IntentVolumePolicy,
        ProductCardMode,
        SentenceTransformerCrossEncoderScorer,
        create_retrieval_controller,
        load_catalog_density,
        load_product_documents,
        project_product_documents,
    )
    from shopping_copilot.retrieval.deepseek_ranking import (
        DeepSeekQualityPipeline,
        DeepSeekQualityRanker,
        DeepSeekRankingConfig,
        DeepSeekRankingProvider,
        TransparencyAwareDPPFinalizer,
    )

    from .quality_ranking import ApertureRankingCoordinator
    from .response_generation import DeterministicResponseComposer

    catalog = Path(catalog_path).resolve()
    semantic_release = config.semantic_release.resolve()
    dense_index = config.dense_index.resolve()
    density_cache = config.density_cache.resolve()
    _require_file(catalog, name="catalog")
    _require_directory(semantic_release, name="semantic release")
    _require_directory(dense_index, name="dense index")
    _require_file(density_cache, name="intent-volume density cache")

    release = load_catalog_semantic_release(semantic_release)
    gateway = CatalogSemanticGateway(release)
    service = QueryUnderstandingService(
        provider=DeepSeekProvider(
            api_key=config.api_key,
            config=DeepSeekConfig(
                model=config.model,
                base_url=config.base_url,
                timeout_seconds=config.timeout_seconds,
                max_tokens=config.max_tokens,
                temperature=0.0,
                strict_tools=False,
                disable_thinking=True,
            ),
        ),
        materializer=IntentMaterializer(gateway=gateway, grounder=release.grounder),
    )
    compiler = QueryCompiler(
        catalog_semantic_release_id=release.release_id,
        category_registry=release.category_registry,
    )

    product_fact_cards = None
    if config.product_card_sidecar is not None:
        sidecar = config.product_card_sidecar.resolve()
        _require_file(sidecar, name="product-card sidecar")
        product_fact_cards = load_product_fact_sidecar(sidecar, catalog_path=catalog)
    product_card_mode = ProductCardMode(config.product_card_mode)
    controller = create_retrieval_controller(
        index_path=dense_index,
        release_dir=semantic_release,
        catalog_path=catalog,
        device=config.device,
        local_files_only=True,
        product_fact_cards=product_fact_cards,
        product_card_mode=product_card_mode,
    )

    policy = IntentVolumePolicy()
    density = load_catalog_density(
        density_cache,
        dense_index=controller.retriever.index,
        temperature=policy.density_temperature,
    )
    estimator = IntentVolumeEstimator(
        dense_index=controller.retriever.index,
        embedder=controller.retriever.embedder,
        hard_mask_resolver=controller.hard_mask_resolver,
        density=density,
        policy=policy,
    )

    loaded_documents = load_product_documents(
        catalog,
        expected_parent_asins=set(controller.retriever.index.parent_asins),
    )
    if product_fact_cards is not None:
        loaded_documents = project_product_documents(
            loaded_documents,
            product_fact_cards,
            mode=product_card_mode,
        )
    documents = {item.parent_asin: _compact_document(item.text) for item in loaded_documents}
    metadata = _load_product_metadata(catalog)

    reranker = None
    if config.cross_encoder:
        reranker = CrossEncoderRelevanceReranker(
            scorer=SentenceTransformerCrossEncoderScorer(
                DEFAULT_BGE_MODEL,
                revision=DEFAULT_BGE_REVISION,
                device=config.device,
                local_files_only=True,
                max_length=384,
            )
        )

    quality_pipeline = None
    quality_finalizer = None
    if config.quality_ranking and reranker is not None:
        quality_pipeline = DeepSeekQualityPipeline(
            index=controller.retriever.index,
            bge_reranker=reranker,
            deepseek_ranker=DeepSeekQualityRanker(
                provider=DeepSeekRankingProvider(
                    api_key=config.api_key,
                    config=DeepSeekRankingConfig(
                        model=config.model,
                        base_url=config.base_url,
                        timeout_seconds=config.ranking_timeout_seconds,
                        max_tokens=config.ranking_max_tokens,
                        temperature=0.0,
                        strict_tools=False,
                        disable_thinking=True,
                    ),
                ),
                deepseek_weight=0.8,
                repair_once=True,
            ),
            shortlist_k=48,
            protected_per_direction=6,
        )
        quality_finalizer = TransparencyAwareDPPFinalizer(
            index=controller.retriever.index,
        )
    ranking_coordinator = ApertureRankingCoordinator(
        documents=documents,
        quality_pipeline=quality_pipeline,
        quality_finalizer=quality_finalizer,
        fallback_reranker=reranker,
        fallback_selector=GreedyDPPSelector(index=controller.retriever.index),
    )

    runtime_module = import_module("scripts.simulator.evaluate_full_pipeline_other")
    full_pipeline_agent = runtime_module.FullPipelineOtherAgent
    return cast(
        AgentDelegate,
        full_pipeline_agent(
            service=service,
            compiler=compiler,
            estimator=estimator,
            controller=controller,
            reranker=reranker,
            selector=GreedyDPPSelector(index=controller.retriever.index),
            ranking_coordinator=ranking_coordinator,
            response_composer=DeterministicResponseComposer(),
            documents=documents,
            product_metadata=metadata,
            category_options=category_options_from_registry(release.category_registry),
            allowed_dont_care_facets=tuple(
                spec.id for spec in gateway.registry if spec.id != SYSTEM_PRODUCT_CATEGORY_FACET_ID
            ),
            facet_registry=gateway.registry,
            qu_retry_count=config.qu_retry_count,
            repeat_noop_cache=config.repeat_noop_cache,
        ),
    )


def _load_product_metadata(catalog_path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    with catalog_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if type(value) is not dict:
                raise ValueError(f"catalog row {line_number} must be an object")
            product = cast(dict[str, Any], value)
            parent_asin = product.get("parent_asin")
            if type(parent_asin) is not str or not parent_asin:
                raise ValueError(f"catalog row {line_number} has no parent_asin")
            metadata[parent_asin] = {
                "title": product.get("title"),
                "categories": product.get("categories"),
                "price": product.get("price"),
                "store": product.get("store"),
            }
    return metadata


def _compact_document(text: str) -> str:
    kept = []
    for line in text.splitlines():
        label = line.partition(":")[0]
        if label in {"title", "categories", "store", "features", "details"}:
            kept.append(line)
    return "\n".join(kept)[:2400]


def _require_file(path: Path, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def _require_directory(path: Path, *, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")
