"""Reference creative analysis and management.

Analyzes uploaded reference ad creatives using Gemini vision to extract
design elements (layout, colors, typography, composition). The analysis
results are cached as JSON and used to enrich generation prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import ReferenceCreative

logger = logging.getLogger(__name__)

# Directory for persisting analysis cache
REFERENCE_DIR = Path("assets/references")
ANALYSIS_CACHE_FILE = REFERENCE_DIR / "_analysis_cache.json"

# Analysis prompt for Gemini vision
ANALYSIS_PROMPT = """\
あなたはMeta広告（Facebook/Instagram）のデザイン分析の専門家です。
この広告クリエイティブ画像を詳細に分析し、以下の要素をJSON形式で出力してください。

分析項目:
1. **layout_type**: レイアウトパターン（例: "product_center", "split_left_right", "full_bleed", "text_overlay", "minimal_center"）
2. **color_palette**: 使用されている主要カラー（HEXコード3-5色）
3. **color_mood**: カラーの雰囲気（例: "warm_premium", "cool_corporate", "vibrant_energetic"）
4. **typography_style**: タイポグラフィの特徴（例: "bold_sans", "elegant_serif", "handwritten", "mixed"）
5. **text_position**: テキストの配置（例: "top_left", "center", "bottom_overlay", "left_half"）
6. **text_size_ratio**: テキストが画像面積に占める割合（例: "small_10pct", "medium_20pct", "large_30pct"）
7. **cta_style**: CTAボタンのデザイン（例: "rounded_solid", "outline", "text_only", "pill_shape"）
8. **composition_notes**: 構図の特徴（自由記述、日本語50文字以内）
9. **visual_technique**: 使用されているビジュアル技法（例: "gradient_overlay", "shadow_depth", "blur_background", "geometric_shapes"）
10. **overall_quality_score**: 広告としての品質スコア（1-10）
11. **style_tags**: このクリエイティブを表すタグ（3-5個）
12. **design_summary**: このクリエイティブのデザインを再現するための詳細な指示（日本語100文字以内）

JSONのみを返してください。
"""


def _load_cache() -> dict:
    """Load the analysis cache from disk."""
    if ANALYSIS_CACHE_FILE.exists():
        try:
            return json.loads(ANALYSIS_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    """Save the analysis cache to disk."""
    ANALYSIS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def analyze_reference(
    client: genai.Client,
    image_path: str,
    model: str = "gemini-2.5-flash",
) -> dict:
    """Analyze a single reference creative using Gemini vision.

    Args:
        client: Gemini API client.
        image_path: Path to the reference image.
        model: Gemini model for vision analysis.

    Returns:
        Dictionary with analysis results.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    # Read image bytes
    image_data = path.read_bytes()
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/jpeg")

    image_part = types.Part(inline_data=types.Blob(mime_type=mime, data=image_data))

    response = client.models.generate_content(
        model=model,
        contents=[image_part, ANALYSIS_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.2,  # Low temperature for consistent analysis
            response_mime_type="application/json",
        ),
    )

    text = response.text
    if not text:
        raise RuntimeError("Empty analysis response from Gemini")

    # Parse JSON response
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    return json.loads(cleaned)


def analyze_and_cache(
    client: genai.Client,
    ref: ReferenceCreative,
    model: str = "gemini-2.5-flash",
    force: bool = False,
) -> ReferenceCreative:
    """Analyze a reference creative and cache the result.

    Skips analysis if already cached (unless force=True).

    Returns:
        Updated ReferenceCreative with analysis populated.
    """
    cache = _load_cache()
    cache_key = ref.image_path

    # Use cache if available
    if not force and cache_key in cache:
        cached = cache[cache_key]
        ref.analysis = cached.get("design_summary", "")
        ref.style_tags = cached.get("style_tags", [])
        ref.analyzed = True
        logger.info("Using cached analysis for %s", ref.image_path)
        return ref

    # Run fresh analysis
    logger.info("Analyzing reference creative: %s", ref.image_path)
    result = analyze_reference(client, ref.image_path, model)

    # Update the reference object
    ref.analysis = result.get("design_summary", "")
    ref.style_tags = result.get("style_tags", [])
    ref.analyzed = True

    # Save to cache
    cache[cache_key] = result
    _save_cache(cache)

    return ref


def get_full_analysis(image_path: str) -> dict | None:
    """Retrieve the full cached analysis for a reference image."""
    cache = _load_cache()
    return cache.get(image_path)


def select_references(
    references: list[ReferenceCreative],
    category: str = "",
    style_name: str = "",
    max_refs: int = 3,
) -> list[ReferenceCreative]:
    """Select the best matching reference creatives for a generation task.

    Prioritizes references that match the product category and style.
    Falls back to highest-quality references if no matches.

    Args:
        references: All available reference creatives.
        category: Product category to match.
        style_name: Style preset name to match.
        max_refs: Maximum number of references to return.

    Returns:
        List of selected ReferenceCreative objects.
    """
    if not references:
        return []

    scored: list[tuple[float, ReferenceCreative]] = []

    for ref in references:
        if not ref.analyzed:
            continue

        score = 0.0

        # Category match
        if category and ref.category:
            if ref.category.lower() == category.lower():
                score += 3.0
            elif category.lower() in ref.category.lower():
                score += 1.5

        # Style tag match
        if style_name:
            style_lower = style_name.lower()
            for tag in ref.style_tags:
                if style_lower in tag.lower() or tag.lower() in style_lower:
                    score += 2.0
                    break

        # Quality score from analysis
        full_analysis = get_full_analysis(ref.image_path)
        if full_analysis:
            quality = full_analysis.get("overall_quality_score", 5)
            score += quality * 0.5  # Weight quality

        scored.append((score, ref))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ref for _, ref in scored[:max_refs]]


def build_reference_prompt_section(references: list[ReferenceCreative]) -> str:
    """Build the prompt section describing reference creative analysis results.

    This text is included in the generation prompt to guide the AI
    based on patterns learned from reference creatives.
    """
    if not references:
        return ""

    sections: list[str] = []
    sections.append(
        "\n=== REFERENCE CREATIVE ANALYSIS ===\n"
        "I am providing reference ad creatives as style guides. "
        "Study their design language carefully and create a new creative that "
        "captures the same level of professional quality, while being original.\n"
        "Key patterns to replicate from the references:"
    )

    for i, ref in enumerate(references, 1):
        full = get_full_analysis(ref.image_path)
        if not full:
            continue

        sections.append(
            f"\nReference #{i}:\n"
            f"- Layout: {full.get('layout_type', 'unknown')}\n"
            f"- Colors: {', '.join(full.get('color_palette', []))}\n"
            f"- Color mood: {full.get('color_mood', '')}\n"
            f"- Typography: {full.get('typography_style', '')}\n"
            f"- Text position: {full.get('text_position', '')}\n"
            f"- CTA style: {full.get('cta_style', '')}\n"
            f"- Visual technique: {full.get('visual_technique', '')}\n"
            f"- Design summary: {full.get('design_summary', '')}"
        )

    sections.append(
        "\nIMPORTANT: Match the professional quality level of these references. "
        "Use similar layout principles, typography weight, color harmony, "
        "and CTA styling. Do NOT copy them—create a new design inspired by "
        "their best qualities."
    )

    return "\n".join(sections)
