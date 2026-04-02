"""Complete ad creative generation using Gemini Nano Banana.

Instead of generating a background and overlaying text with Pillow,
this module asks Gemini to produce a fully finished 1:1 ad creative
with text, layout, product image, and design elements integrated.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    BrandConfig,
    CopyVariation,
    ProductInfo,
    PromptPattern,
    ReferenceCreative,
    StylePreset,
    TemplateConfig,
)
from .reference_analyzer import build_reference_prompt_section

logger = logging.getLogger(__name__)

# Cost per image (Gemini 2.5 Flash Image standard pricing)
COST_PER_IMAGE_USD = 0.039


def _image_to_part(img_path: str) -> types.Part | None:
    """Load an image file and return a Gemini Part for inline_data."""
    path = Path(img_path)
    if not path.exists():
        logger.warning("Product image not found: %s", img_path)
        return None

    # Read raw bytes and detect MIME type
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/jpeg")

    return types.Part(inline_data=types.Blob(mime_type=mime, data=data))


def _build_creative_prompt(
    product: ProductInfo,
    copy: CopyVariation,
    style: StylePreset,
    template: TemplateConfig,
    brand: BrandConfig,
    has_product_image: bool,
    references: list[ReferenceCreative] | None = None,
    prompt_pattern: PromptPattern | None = None,
) -> str:
    """Build a prompt that instructs Gemini to generate a complete ad creative.

    The prompt describes the exact visual design wanted—typography,
    layout, colors, product placement—so the AI outputs a ready-to-use
    Meta ad image with all elements baked in.
    """

    # ----- Layout direction based on template -----
    layout_map = {
        "product_showcase": (
            "Product-centered hero layout. "
            "Place the product prominently in the center or center-right area, "
            "occupying about 40-50% of the canvas. "
            "Headline text top-left or top-center. "
            "CTA button bottom-center."
        ),
        "benefit_highlight": (
            "Split layout: left side for bold text message, "
            "right side for the product/visual. "
            "Headline large and dominant on the left half. "
            "CTA button at the bottom-left area."
        ),
        "social_proof": (
            "Trust-building layout. "
            "Product at center with a review/testimonial quote overlay. "
            "Star rating or number badge prominently displayed. "
            "Headline at top, CTA at bottom-center."
        ),
        "urgency_offer": (
            "Urgency-driven layout. "
            "Big bold discount/offer number in the center. "
            "Product slightly smaller, positioned to the side. "
            "Headline at top with urgency cue (timer/badge). "
            "CTA button large and prominent at bottom."
        ),
        "testimonial": (
            "UGC/testimonial layout. "
            "Large quotation-style text in the center. "
            "Product image smaller, in a corner or bottom area. "
            "Natural, editorial design—like a magazine review. "
            "CTA subtle at the bottom."
        ),
        "minimal_premium": (
            "Ultra-minimal premium layout. "
            "Maximum negative space (60%+ empty/clean). "
            "Product centered with one line of text below it. "
            "CTA as a simple text link, not a heavy button. "
            "Apple-style elegance."
        ),
    }
    # Use pattern's layout prompts if provided, fall back to built-in defaults
    if prompt_pattern and prompt_pattern.layout_prompts:
        layout_desc = prompt_pattern.layout_prompts.get(
            template.copy_style,
            layout_map.get(template.copy_style, layout_map["product_showcase"]),
        )
    else:
        layout_desc = layout_map.get(
            template.copy_style,
            layout_map["product_showcase"],
        )

    # ----- Headline / CTA text -----
    headline_text = copy.headline
    sub_text = copy.subheadline or ""
    cta_text = copy.cta_text

    # ----- Brand colors -----
    colors = brand.colors

    # ----- Assemble prompt -----
    parts: list[str] = []

    # Role & task
    parts.append(
        "You are a world-class Meta (Facebook/Instagram) ad designer. "
        "Generate a COMPLETE, ready-to-publish 1080x1080 square ad creative image. "
        "The image must include ALL the following elements rendered directly into the image:"
    )

    # Required text elements
    parts.append(
        f"\n=== TEXT ELEMENTS (render these as typography IN the image) ===\n"
        f"Headline: 「{headline_text}」\n"
        f"{'Subheadline: 「' + sub_text + '」' if sub_text else ''}\n"
        f"CTA button text: 「{cta_text}」"
    )

    # Typography rules
    parts.append(
        "\n=== TYPOGRAPHY RULES ===\n"
        "- Headline: large, bold, high contrast against background. "
        f"Approximate size: {style.font_size_headline}pt equivalent.\n"
        "- Text must be SHARP, perfectly rendered, and 100% readable.\n"
        "- Use clean sans-serif Japanese-compatible typeface (Noto Sans CJK, "
        "Hiragino, or similar professional Japanese font style).\n"
        "- Text should look like it was set by a professional graphic designer, "
        "NOT like AI-generated text. Kerning and line-spacing must be perfect.\n"
        "- Maximum text coverage: 20% of image area (Meta ad rule)."
    )

    # CTA button
    parts.append(
        "\n=== CTA BUTTON ===\n"
        f"- Background color: {colors.accent}\n"
        "- Shape: rounded rectangle, high contrast, clearly clickable.\n"
        "- Position: near the bottom of the image, within safe zone (50px from edges).\n"
        "- CTA button must look like a real UI button people want to tap."
    )

    # Layout
    parts.append(f"\n=== LAYOUT ===\n{layout_desc}")

    # Product image reference
    if has_product_image:
        parts.append(
            "\n=== PRODUCT IMAGE ===\n"
            "I am providing a reference photo of the actual product. "
            "Incorporate this product into the ad creative. "
            "The product must be clearly visible and recognizable, "
            "maintaining its real appearance, colors, and proportions. "
            "Do NOT alter the product design. "
            "Place it according to the layout instructions above."
        )
    else:
        parts.append(
            "\n=== PRODUCT VISUAL ===\n"
            f"No product photo provided. Create an abstract/lifestyle visual "
            f"that represents the product category: {product.category or product.name}. "
            f"Use evocative imagery that matches the product: {product.description[:100]}."
        )

    # Style & mood
    if style.background_style:
        parts.append(f"\n=== VISUAL STYLE ===\n{style.background_style}")
    if style.mood_keywords:
        parts.append(f"\n=== MOOD ===\n{style.mood_keywords}")
    if style.color_guidance:
        parts.append(f"\n=== COLOR PALETTE ===\n{style.color_guidance}")

    # Brand colors
    parts.append(
        f"\n=== BRAND COLORS ===\n"
        f"Primary: {colors.primary}\n"
        f"Secondary: {colors.secondary}\n"
        f"Accent (CTA): {colors.accent}\n"
        f"Text on dark: {colors.text_light}\n"
        f"Text on light: {colors.text_dark}"
    )

    # Technical requirements
    parts.append(
        "\n=== TECHNICAL REQUIREMENTS ===\n"
        "- Exact square format: 1080x1080 pixels, 1:1 aspect ratio.\n"
        "- Safe zone: keep all important elements 50px from any edge.\n"
        "- Professional commercial quality: crisp, high-resolution, agency-grade.\n"
        "- Text must be perfectly legible—use contrast, shadows, or background "
        "panels to ensure readability over any background area.\n"
        "- The overall design should look like a real Meta ad created by "
        "a top advertising agency, NOT an AI experiment.\n"
        "- Do NOT include any placeholder text, lorem ipsum, or watermarks."
    )

    # Reference creative analysis (if available)
    if references:
        ref_section = build_reference_prompt_section(references)
        if ref_section:
            parts.append(ref_section)

    return "\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def generate_complete_creative(
    client: genai.Client,
    product: ProductInfo,
    copy: CopyVariation,
    style: StylePreset,
    template: TemplateConfig,
    brand: BrandConfig,
    model: str = "gemini-2.5-flash-image",
    references: list[ReferenceCreative] | None = None,
    prompt_pattern: PromptPattern | None = None,
) -> tuple[Image.Image, str]:
    """Generate a complete ad creative using Gemini Nano Banana.

    Sends the product image and reference creatives as reference images,
    along with detailed instructions for text, layout, colors, and style.
    Gemini returns a fully finished ad creative with all elements
    integrated.

    Args:
        client: Gemini API client.
        product: Product information (may include image_path).
        copy: Ad copy (headline, subheadline, CTA).
        style: Visual style preset.
        template: Layout template.
        brand: Brand configuration (colors, logo).
        model: Gemini image model ID.
        references: Reference creatives for style guidance (max ~3).
        prompt_pattern: Optional prompt pattern override for layout prompts.

    Returns:
        Tuple of (PIL Image, prompt used).
    """
    has_product_image = bool(product.image_path)
    prompt_text = _build_creative_prompt(
        product, copy, style, template, brand, has_product_image,
        references=references,
        prompt_pattern=prompt_pattern,
    )
    logger.info("Generating complete creative: %s | %s", copy.headline, style.name)

    # Build content parts: [reference_images, product_image, text_prompt]
    # Gemini supports up to 14 reference images per request
    content_parts: list[types.Part | str] = []

    # Add reference creative images first (style guides)
    if references:
        ref_count = 0
        for ref in references:
            ref_part = _image_to_part(ref.image_path)
            if ref_part:
                content_parts.append(ref_part)
                ref_count += 1
        if ref_count > 0:
            content_parts.append(
                f"Above {ref_count} image(s) are REFERENCE AD CREATIVES. "
                "Study their design style, layout, typography, and color usage. "
                "Create a NEW creative inspired by their quality and style, "
                "but with the specific content and branding described below.\n\n"
            )

    # Add product image as reference if available
    if product.image_path:
        img_part = _image_to_part(product.image_path)
        if img_part:
            content_parts.append(img_part)
            content_parts.append(
                "Above is the actual product photo. "
                "Use this exact product in the ad creative design below.\n\n"
            )

    content_parts.append(prompt_text)

    response = client.models.generate_content(
        model=model,
        contents=content_parts,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="1K",
            ),
        ),
    )

    # Extract image data from response
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            img = Image.open(io.BytesIO(image_bytes))
            # Ensure 1080x1080 for Meta ad spec
            if img.size != (1080, 1080):
                img = img.resize((1080, 1080), Image.LANCZOS)
            return img, prompt_text

    raise RuntimeError("No image data in Gemini response")
