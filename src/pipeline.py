"""Batch generation pipeline: reference analysis → copy → complete creative.

The pipeline has up to three phases:
  0. (Optional) Analyze reference creatives (vision model, cached)
  1. Generate ad copy variations (text model)
  2. Generate complete ad creatives with references + product baked in (image model)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from pathlib import Path
from typing import Callable, Optional

from google import genai

from .copy_generator import generate_copy
from .image_generator import COST_PER_IMAGE_USD, generate_complete_creative
from .models import (
    BrandConfig,
    CopyVariation,
    CreativeSpec,
    GenerationResult,
    GenerationStatus,
    ProductInfo,
    PromptPattern,
    ReferenceCreative,
    StylePreset,
    TemplateConfig,
)
from .reference_analyzer import analyze_and_cache, select_references

logger = logging.getLogger(__name__)


@dataclass
class PatternSelection:
    """A selected prompt pattern with its desired image count."""
    pattern: PromptPattern
    image_count: int = 5


@dataclass
class PipelineConfig:
    """Configuration for a batch generation run."""
    brand: BrandConfig
    products: list[ProductInfo]
    templates: list[TemplateConfig]
    styles: list[StylePreset]
    references: list[ReferenceCreative] = field(default_factory=list)
    copy_variations_per_combo: int = 5  # legacy fallback
    pattern_selections: list[PatternSelection] = field(default_factory=list)
    max_concurrent_api_calls: int = 10
    output_dir: str = "output"
    text_model: str = "gemini-2.5-flash"
    image_model: str = "gemini-2.5-flash-image"


@dataclass
class PipelineProgress:
    """Track pipeline progress for UI updates."""
    phase: str = "idle"
    total_steps: int = 0
    completed_steps: int = 0
    current_item: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps


class Pipeline:
    """Orchestrate the batch generation pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        api_key: str,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ):
        self.config = config
        self.client = genai.Client(api_key=api_key)
        self.progress = PipelineProgress()
        self._progress_callback = progress_callback
        self.run_id = uuid.uuid4().hex[:8]
        self.run_dir = Path(config.output_dir) / self.run_id
        self.images_dir = self.run_dir / "images"

    def _update_progress(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self.progress, k, v)
        if self._progress_callback:
            self._progress_callback(self.progress)

    def estimate(self) -> dict:
        """Estimate the number of creatives and cost."""
        n_products = len(self.config.products)
        n_templates = len(self.config.templates)
        n_styles = len(self.config.styles)

        if self.config.pattern_selections:
            total_copies = sum(ps.image_count for ps in self.config.pattern_selections)
            n_patterns = len(self.config.pattern_selections)
        else:
            total_copies = self.config.copy_variations_per_combo
            n_patterns = 1

        n_creatives = n_products * n_templates * n_styles * total_copies
        cost = n_creatives * COST_PER_IMAGE_USD

        n_unanalyzed = sum(1 for r in self.config.references if not r.analyzed)

        return {
            "total_creatives": n_creatives,
            "image_generation_calls": n_creatives,
            "copy_requests": n_products * n_templates * n_patterns,
            "reference_analyses": n_unanalyzed,
            "estimated_cost_usd": round(cost, 2),
        }

    async def run(self) -> list[GenerationResult]:
        """Execute the pipeline.

        Phase 0: Analyze reference creatives (if any, cached)
        Phase 1: Generate copy variations (text model, fast)
        Phase 2: Generate complete creatives (image model)

        Returns list of GenerationResult for every creative.
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(self.config.max_concurrent_api_calls)

        # ---------------------------------------------------------------
        # Phase 0: Analyze reference creatives (if any unanalyzed)
        # ---------------------------------------------------------------
        references = self.config.references
        unanalyzed = [r for r in references if not r.analyzed]

        if unanalyzed:
            self._update_progress(
                phase="reference_analysis",
                completed_steps=0,
                total_steps=len(unanalyzed),
            )
            for ref in unanalyzed:
                self._update_progress(
                    current_item=f"参考クリエイティブ分析: {Path(ref.image_path).name}"
                )
                try:
                    analyze_and_cache(self.client, ref, self.config.text_model)
                except Exception as exc:
                    logger.error("Reference analysis failed: %s", exc)
                    self.progress.errors.append(
                        f"Reference: {ref.image_path}: {exc}"
                    )
                self._update_progress(
                    completed_steps=self.progress.completed_steps + 1
                )

        # ---------------------------------------------------------------
        # Build pattern list (fall back to no-pattern mode for compat)
        # ---------------------------------------------------------------
        if self.config.pattern_selections:
            pattern_sels = self.config.pattern_selections
        else:
            pattern_sels = [PatternSelection(
                pattern=PromptPattern(name="default"),
                image_count=self.config.copy_variations_per_combo,
            )]

        # ---------------------------------------------------------------
        # Phase 1: Generate copy for each product × template × pattern
        # ---------------------------------------------------------------
        self._update_progress(phase="copy_generation", completed_steps=0)

        copy_combos = list(cartesian_product(
            self.config.products, self.config.templates, pattern_sels
        ))
        self._update_progress(total_steps=len(copy_combos))

        # { (product_name, template_name, pattern_name): [CopyVariation, ...] }
        copy_map: dict[tuple[str, str, str], list[CopyVariation]] = {}

        async def _gen_copy(prod: ProductInfo, tmpl: TemplateConfig, ps: PatternSelection):
            async with sem:
                self._update_progress(
                    current_item=f"コピー生成: {prod.name} × {tmpl.name} × {ps.pattern.name}"
                )
                try:
                    pattern_arg = ps.pattern if ps.pattern.copy_style_prompts else None
                    copies = await generate_copy(
                        self.client,
                        prod,
                        tmpl,
                        ps.image_count,
                        self.config.text_model,
                        prompt_pattern=pattern_arg,
                    )
                    copy_map[(prod.name, tmpl.name, ps.pattern.name)] = copies
                except Exception as exc:
                    logger.error("Copy generation failed: %s", exc)
                    self.progress.errors.append(
                        f"Copy: {prod.name}×{tmpl.name}×{ps.pattern.name}: {exc}"
                    )
                    copy_map[(prod.name, tmpl.name, ps.pattern.name)] = []
                self._update_progress(completed_steps=self.progress.completed_steps + 1)

        await asyncio.gather(*[_gen_copy(p, t, ps) for p, t, ps in copy_combos])

        # ---------------------------------------------------------------
        # Phase 2: Generate complete creatives for every combination
        # Each (product × template × style × pattern × copy_variation) = 1 API call
        # ---------------------------------------------------------------
        self._update_progress(phase="creative_generation", completed_steps=0)

        creative_tasks: list[tuple[CreativeSpec, str, list[ReferenceCreative], PromptPattern | None]] = []

        for prod, tmpl, sty in cartesian_product(
            self.config.products, self.config.templates, self.config.styles
        ):
            selected_refs = select_references(
                references,
                category=prod.category,
                style_name=sty.name,
                max_refs=3,
            )

            for ps in pattern_sels:
                copies = copy_map.get((prod.name, tmpl.name, ps.pattern.name), [])
                if not copies:
                    continue

                pattern_arg = ps.pattern if ps.pattern.layout_prompts else None

                for copy_var in copies:
                    slug_p = prod.name.replace(" ", "-").lower()[:30]
                    slug_t = tmpl.copy_style
                    slug_s = sty.name.replace(" ", "-").lower()[:20]
                    slug_pat = ps.pattern.name.replace(" ", "-").lower()[:20]
                    filename = f"{slug_p}_{slug_t}_{slug_s}_{slug_pat}_{copy_var.variant_id}.jpg"
                    output_path = str(self.images_dir / filename)

                    spec = CreativeSpec(
                        product=prod,
                        brand=self.config.brand,
                        template=tmpl,
                        style=sty,
                        copy=copy_var,
                        spec_id=f"{self.run_id}_{filename}",
                    )
                    creative_tasks.append((spec, output_path, selected_refs, pattern_arg))

        self._update_progress(total_steps=len(creative_tasks))

        results: list[GenerationResult] = []

        async def _gen_creative(
            spec: CreativeSpec,
            output_path: str,
            refs: list[ReferenceCreative],
            pat: PromptPattern | None,
        ):
            async with sem:
                self._update_progress(
                    current_item=f"クリエイティブ生成: {spec.copy.headline} ({spec.style.name})"
                )
                try:
                    img, prompt = generate_complete_creative(
                        self.client,
                        spec.product,
                        spec.copy,
                        spec.style,
                        spec.template,
                        spec.brand,
                        self.config.image_model,
                        references=refs,
                        prompt_pattern=pat,
                    )
                    img.save(output_path, "JPEG", quality=95)
                    results.append(GenerationResult(
                        spec=spec,
                        status=GenerationStatus.SUCCESS,
                        image_path=output_path,
                        image_prompt=prompt[:500],
                        cost_usd=COST_PER_IMAGE_USD,
                    ))
                except Exception as exc:
                    logger.error("Creative generation failed: %s", exc)
                    results.append(GenerationResult(
                        spec=spec,
                        status=GenerationStatus.FAILED,
                        error_message=str(exc),
                    ))
                    self.progress.errors.append(
                        f"Creative: {spec.copy.headline} ({spec.style.name}): {exc}"
                    )
                self._update_progress(completed_steps=self.progress.completed_steps + 1)

        await asyncio.gather(*[_gen_creative(s, p, r, pat) for s, p, r, pat in creative_tasks])

        self._update_progress(phase="done", current_item="完了")
        return results
