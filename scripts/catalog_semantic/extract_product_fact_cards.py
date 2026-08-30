#!/usr/bin/env python3
"""Generate resumable DeepSeek product-fact sidecars without changing catalog bytes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_copilot.catalog.product_facts import (  # noqa: E402
    PRODUCT_FACT_PROMPT_VERSION,
    DeepSeekProductFactConfig,
    DeepSeekProductFactProvider,
    ProductFactError,
    ProductFactRequest,
    ProductFactResult,
    product_fact_request_from_raw_line,
)
from shopping_copilot.facet_language import FACET_LANGUAGE_VERSION  # noqa: E402

SCHEMA = "shopping-copilot/product-fact-sidecar/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorkItem:
    line_number: int
    request: ProductFactRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class _Success:
    parent_asin: str
    skipped: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Failure:
    parent_asin: str
    line_number: int
    error: str


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.workers < 1 or args.progress_every < 1 or args.retries < 0:
        parser.error("workers must be positive and retries must be non-negative")
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        parser.error("offset must be non-negative and limit must be positive")
    catalog = Path(args.catalog).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cards_dir = output / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    api_key = _api_key(args.api_key_file)
    config = DeepSeekProductFactConfig(
        model=args.model,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
    )
    provider = DeepSeekProductFactProvider(api_key=api_key, config=config)
    requested_asins = _requested_asins(args.asins)
    work = _load_work(
        catalog,
        offset=args.offset,
        limit=args.limit,
        requested_asins=requested_asins,
    )

    started = time.perf_counter()
    successes, failures, selected_count = _execute(
        work,
        provider=provider,
        cards_dir=cards_dir,
        model=args.model,
        retries=args.retries,
        resume=args.resume,
        workers=args.workers,
        progress_every=args.progress_every,
        expected_count=(len(requested_asins) if requested_asins is not None else args.limit),
    )
    if selected_count == 0:
        parser.error("selection produced no catalog products")

    _write_jsonl(
        output / "product-facts.jsonl",
        _card_records(cards_dir, successes),
    )
    _write_json(
        output / "failures.json",
        [
            {
                "parent_asin": item.parent_asin,
                "line_number": item.line_number,
                "error": item.error,
            }
            for item in sorted(failures, key=lambda value: value.parent_asin)
        ],
    )
    run = {
        "schema": "shopping-copilot/product-fact-run/v1",
        "catalog": str(catalog),
        "output": str(output),
        "model": args.model,
        "prompt_version": PRODUCT_FACT_PROMPT_VERSION,
        "facet_language_version": FACET_LANGUAGE_VERSION,
        "quality_policy": "full_source_no_token_saving",
        "selected_count": selected_count,
        "success_count": len(successes),
        "failed_count": len(failures),
        "skipped_count": sum(item.skipped for item in successes),
        "workers": args.workers,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "reported_token_usage": {
            "prompt_tokens": sum(item.prompt_tokens or 0 for item in successes),
            "completion_tokens": sum(item.completion_tokens or 0 for item in successes),
            "total_tokens": sum(item.total_tokens or 0 for item in successes),
        },
    }
    _write_json(output / "run.json", run)
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def _execute(
    work: Iterable[_WorkItem],
    *,
    provider: DeepSeekProductFactProvider,
    cards_dir: Path,
    model: str,
    retries: int,
    resume: bool,
    workers: int,
    progress_every: int,
    expected_count: int | None,
) -> tuple[list[_Success], list[_Failure], int]:
    iterator = iter(work)
    pending: dict[Future[_Success | _Failure], _WorkItem] = {}
    successes: list[_Success] = []
    failures: list[_Failure] = []
    selected_count = 0
    completed = 0
    buffer_size = max(workers * 2, 1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(pending) < buffer_size:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending[
                executor.submit(
                    _extract_one,
                    item,
                    provider=provider,
                    cards_dir=cards_dir,
                    model=model,
                    retries=retries,
                    resume=resume,
                )
            ] = item
            selected_count += 1

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                completed += 1
                if type(result) is _Success:
                    successes.append(result)
                else:
                    failure = cast(_Failure, result)
                    failures.append(failure)
                    print(
                        f"product-facts failure: {failure.parent_asin} {failure.error}",
                        flush=True,
                    )
                if completed % progress_every == 0 or completed == expected_count:
                    denominator = f"/{expected_count}" if expected_count is not None else ""
                    print(
                        f"product-facts: {completed}{denominator} "
                        f"ok={len(successes)} failed={len(failures)}",
                        flush=True,
                    )

            while len(pending) < buffer_size:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending[
                    executor.submit(
                        _extract_one,
                        item,
                        provider=provider,
                        cards_dir=cards_dir,
                        model=model,
                        retries=retries,
                        resume=resume,
                    )
                ] = item
                selected_count += 1
    return successes, failures, selected_count


def _extract_one(
    item: _WorkItem,
    *,
    provider: DeepSeekProductFactProvider,
    cards_dir: Path,
    model: str,
    retries: int,
    resume: bool,
) -> _Success | _Failure:
    target = cards_dir / f"{item.request.parent_asin}.json"
    if resume:
        existing = _matching_existing(target, request=item.request, model=model)
        if existing is not None:
            prompt_tokens, completion_tokens, total_tokens = _trace(existing)
            return _Success(
                parent_asin=item.request.parent_asin,
                skipped=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    repair: str | None = None
    last_error = "unknown extraction error"
    for attempt in range(retries + 1):
        try:
            result = provider.extract(item.request, repair_instruction=repair)
            record = _record(item.request, result=result, model=model)
            _write_json(target, record)
            return _Success(
                parent_asin=item.request.parent_asin,
                skipped=False,
                prompt_tokens=result.trace.prompt_tokens,
                completion_tokens=result.trace.completion_tokens,
                total_tokens=result.trace.total_tokens,
            )
        except (ProductFactError, OSError, TypeError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            repair = last_error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return _Failure(
        parent_asin=item.request.parent_asin,
        line_number=item.line_number,
        error=last_error,
    )


def _record(
    request: ProductFactRequest,
    *,
    result: ProductFactResult,
    model: str,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "parent_asin": request.parent_asin,
        "source_id": request.source_id,
        "extractor": {
            "model": model,
            "prompt_version": PRODUCT_FACT_PROMPT_VERSION,
            "facet_language_version": FACET_LANGUAGE_VERSION,
        },
        "facts": [
            {
                "facet": fact.facet,
                "value": fact.value,
                "aliases": list(fact.aliases),
                "polarity": fact.polarity.value,
                "component": fact.component,
                "meaning": fact.meaning,
                "evidence": fact.evidence,
                "source_ref": fact.source_ref,
                "confidence": fact.confidence,
            }
            for fact in result.card.facts
        ],
        "summary": result.card.summary,
        "warnings": list(result.card.warnings),
        "trace": {
            "response_id": result.trace.response_id,
            "model": result.trace.model,
            "prompt_tokens": result.trace.prompt_tokens,
            "completion_tokens": result.trace.completion_tokens,
            "total_tokens": result.trace.total_tokens,
        },
    }


def _load_work(
    catalog: Path,
    *,
    offset: int,
    limit: int | None,
    requested_asins: frozenset[str] | None,
) -> Iterable[_WorkItem]:
    if requested_asins is None:
        return _iter_work(catalog, offset=offset, limit=limit)

    selected: list[_WorkItem] = []
    with catalog.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            request = product_fact_request_from_raw_line(raw_line)
            if request.parent_asin not in requested_asins:
                continue
            selected.append(_WorkItem(line_number=line_number, request=request))
            if len(selected) == len(requested_asins):
                break
    found = {item.request.parent_asin for item in selected}
    missing = requested_asins - found
    if missing:
        raise ValueError(f"catalog is missing requested ASINs: {sorted(missing)}")
    return tuple(selected)


def _iter_work(catalog: Path, *, offset: int, limit: int | None) -> Iterator[_WorkItem]:
    selected = 0
    with catalog.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if line_number <= offset:
                continue
            request = product_fact_request_from_raw_line(raw_line)
            yield _WorkItem(line_number=line_number, request=request)
            selected += 1
            if limit is not None and selected >= limit:
                break


def _matching_existing(
    path: Path,
    *,
    request: ProductFactRequest,
    model: str,
) -> dict[str, object] | None:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if type(decoded) is not dict:
        return None
    record = cast(dict[str, object], decoded)
    extractor = record.get("extractor")
    if type(extractor) is not dict:
        return None
    metadata = cast(dict[str, object], extractor)
    if (
        record.get("schema") != SCHEMA
        or record.get("parent_asin") != request.parent_asin
        or record.get("source_id") != request.source_id
        or metadata.get("model") != model
        or metadata.get("prompt_version") != PRODUCT_FACT_PROMPT_VERSION
        or metadata.get("facet_language_version") != FACET_LANGUAGE_VERSION
    ):
        return None
    if "warnings" not in record:
        record["warnings"] = []
        _write_json(path, record)
    return record


def _trace(record: dict[str, object]) -> tuple[int | None, int | None, int | None]:
    value = record.get("trace")
    trace = cast(dict[str, object], value) if type(value) is dict else {}
    return (
        _optional_nonnegative_int(trace.get("prompt_tokens")),
        _optional_nonnegative_int(trace.get("completion_tokens")),
        _optional_nonnegative_int(trace.get("total_tokens")),
    )


def _optional_nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _card_records(
    cards_dir: Path,
    successes: list[_Success],
) -> Iterator[dict[str, object]]:
    for success in sorted(successes, key=lambda value: value.parent_asin):
        path = cards_dir / f"{success.parent_asin}.json"
        try:
            decoded: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot assemble product fact card {path.name}") from error
        if type(decoded) is not dict:
            raise ValueError(f"product fact card {path.name} is not an object")
        yield cast(dict[str, object], decoded)


def _api_key(path_value: str) -> str:
    environment = os.environ.get("DEEPSEEK_API_KEY")
    if environment and environment.strip():
        return environment.strip()
    path = Path(path_value)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError("set DEEPSEEK_API_KEY or provide --api-key-file") from error
    if not value:
        raise ValueError("DeepSeek API key file is empty")
    return value


def _requested_asins(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    result = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("--asins must contain at least one ASIN")
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--output",
        default=ROOT / "artifacts/catalog-semantic/product-facts-v1",
    )
    parser.add_argument("--api-key-file", default=ROOT / "dpskapi")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--asins", help="comma-separated exact parent_asin selection")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
