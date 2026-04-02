"""Image composition: overlay text, CTA buttons, and logos onto background images."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .models import BrandConfig, CopyVariation, CreativeSpec, StylePreset, TemplateConfig, TextArea

logger = logging.getLogger(__name__)

# Meta ad safe zone (pixels from edge on 1080x1080 canvas)
SAFE_ZONE = 50
CANVAS_SIZE = (1080, 1080)


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back to default if not found."""
    try:
        return ImageFont.truetype(font_path, size)
    except (OSError, IOError):
        logger.warning("Font not found: %s, using default", font_path)
        for fallback in [
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]:
            try:
                return ImageFont.truetype(fallback, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    lines: list[str] = []
    avg_char_width = font.getlength("あ") or 20
    chars_per_line = max(1, int(max_width / avg_char_width))

    for raw_line in text.split("\n"):
        wrapped = textwrap.wrap(raw_line, width=chars_per_line) or [""]
        for line in wrapped:
            while font.getlength(line) > max_width and chars_per_line > 1:
                chars_per_line -= 1
                wrapped = textwrap.wrap(raw_line, width=chars_per_line) or [""]
                break
            lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Professional text rendering with multi-layer shadow
# ---------------------------------------------------------------------------

def _draw_text_with_professional_shadow(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    shadow_color: str = "#000000",
    glow_radius: int = 3,
) -> None:
    """Draw text with professional multi-layer shadow for readability.

    Creates: outer glow (4-directional) + drop shadow + main text.
    This ensures readability over any background - bright, dark, or complex.
    """
    x, y = position

    # Layer 1: Outer glow (draws text at multiple offsets for outline/glow effect)
    for offset in range(1, glow_radius + 1):
        alpha_color = shadow_color  # Could be made semi-transparent with RGBA
        draw.text((x + offset, y), text, font=font, fill=alpha_color)
        draw.text((x - offset, y), text, font=font, fill=alpha_color)
        draw.text((x, y + offset), text, font=font, fill=alpha_color)
        draw.text((x, y - offset), text, font=font, fill=alpha_color)
        # Diagonal offsets for smoother glow
        draw.text((x + offset, y + offset), text, font=font, fill=alpha_color)
        draw.text((x - offset, y - offset), text, font=font, fill=alpha_color)
        draw.text((x + offset, y - offset), text, font=font, fill=alpha_color)
        draw.text((x - offset, y + offset), text, font=font, fill=alpha_color)

    # Layer 2: Drop shadow (offset down-right for depth)
    draw.text((x + 3, y + 3), text, font=font, fill=shadow_color)

    # Layer 3: Main text on top
    draw.text((x, y), text, font=font, fill=fill)


# ---------------------------------------------------------------------------
# Text background overlay for readability
# ---------------------------------------------------------------------------

def _add_text_background_overlay(
    img: Image.Image,
    area: TextArea,
    opacity: float = 0.4,
    padding: int = 40,
    gradient: bool = True,
) -> Image.Image:
    """Add a semi-transparent dark gradient overlay behind text area.

    This ensures text is readable regardless of the background image content.
    Uses a gradient that fades from transparent to dark, looking professional
    rather than a harsh solid rectangle.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Expand area with padding
    x1 = max(0, area.x - padding)
    y1 = max(0, area.y - padding)
    x2 = min(CANVAS_SIZE[0], area.x + area.width + padding)
    y2 = min(CANVAS_SIZE[1], area.y + area.height + padding + 60)  # Extra for CTA

    if gradient:
        # Draw gradient: transparent at top → dark at bottom
        total_h = y2 - y1
        for i in range(total_h):
            progress = i / total_h
            # Ease-in curve for smoother gradient
            alpha = int(255 * opacity * (progress ** 0.7))
            overlay_draw.rectangle(
                [(x1, y1 + i), (x2, y1 + i + 1)],
                fill=(0, 0, 0, alpha),
            )
    else:
        # Solid overlay with rounded corners
        alpha = int(255 * opacity)
        overlay_draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=20,
            fill=(0, 0, 0, alpha),
        )

    return Image.alpha_composite(img, overlay)


# ---------------------------------------------------------------------------
# Logo positioning
# ---------------------------------------------------------------------------

def _get_logo_position(
    logo_size: tuple[int, int],
    position_name: str,
) -> tuple[int, int]:
    """Calculate logo position on the canvas."""
    w, h = logo_size
    margin = SAFE_ZONE + 10
    positions = {
        "top_right": (CANVAS_SIZE[0] - w - margin, margin),
        "top_left": (margin, margin),
        "bottom_right": (CANVAS_SIZE[0] - w - margin, CANVAS_SIZE[1] - h - margin),
        "bottom_left": (margin, CANVAS_SIZE[1] - h - margin),
    }
    return positions.get(position_name, positions["top_right"])


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_creative(
    background: Image.Image,
    spec: CreativeSpec,
) -> Image.Image:
    """Compose the final ad creative by overlaying text, CTA, and logo.

    Args:
        background: AI-generated background image (1080x1080).
        spec: Full creative specification.

    Returns:
        Composed PIL Image ready for export.
    """
    # Work on a copy
    img = background.copy()
    if img.size != CANVAS_SIZE:
        img = img.resize(CANVAS_SIZE, Image.LANCZOS)
    img = img.convert("RGBA")

    brand = spec.brand
    template = spec.template
    style = spec.style
    copy = spec.copy

    # ---------------------------------------------------------------
    # 0. Text background overlay (gradient scrim for readability)
    # ---------------------------------------------------------------
    img = _add_text_background_overlay(
        img,
        template.headline_area,
        opacity=0.45,
        gradient=True,
    )

    draw = ImageDraw.Draw(img)

    # ---------------------------------------------------------------
    # 1. Headline (professional multi-layer shadow)
    # ---------------------------------------------------------------
    headline_font = _load_font(brand.fonts.headline, style.font_size_headline)
    headline_area = template.headline_area

    lines = _wrap_text(copy.headline, headline_font, headline_area.width)
    line_height = style.font_size_headline + 12  # Increased line spacing for readability
    y_offset = headline_area.y

    for line in lines:
        _draw_text_with_professional_shadow(
            draw,
            (headline_area.x, y_offset),
            line,
            headline_font,
            fill=brand.colors.text_light,
            glow_radius=3,
        )
        y_offset += line_height

    # ---------------------------------------------------------------
    # 2. Subheadline (if present)
    # ---------------------------------------------------------------
    if copy.subheadline:
        sub_font = _load_font(brand.fonts.body, style.font_size_subheadline)
        sub_y = y_offset + 10
        _draw_text_with_professional_shadow(
            draw,
            (headline_area.x, sub_y),
            copy.subheadline,
            sub_font,
            fill=brand.colors.text_light,
            glow_radius=2,
        )
        y_offset = sub_y + style.font_size_subheadline + 12

    # ---------------------------------------------------------------
    # 3. CTA button (3D effect with shadow and highlight)
    # ---------------------------------------------------------------
    cta_font = _load_font(brand.fonts.cta, style.font_size_cta)
    cta_area = template.cta_area

    # Calculate CTA text size for button sizing
    cta_bbox = draw.textbbox((0, 0), copy.cta_text, font=cta_font)
    cta_text_w = cta_bbox[2] - cta_bbox[0]
    cta_text_h = cta_bbox[3] - cta_bbox[1]
    padding_x = 36
    padding_y = 18

    button_w = max(cta_text_w + 2 * padding_x, cta_area.width)
    button_h = cta_text_h + 2 * padding_y

    button_x = cta_area.x
    button_y = cta_area.y

    # 3D Layer 1: Bottom shadow (depth effect)
    shadow_offset = 4
    draw.rounded_rectangle(
        [button_x + 2, button_y + shadow_offset,
         button_x + button_w + 2, button_y + button_h + shadow_offset],
        radius=16,
        fill=(0, 0, 0, 100),
    )

    # 3D Layer 2: Main button body
    draw.rounded_rectangle(
        [button_x, button_y, button_x + button_w, button_y + button_h],
        radius=16,
        fill=brand.colors.accent,
    )

    # 3D Layer 3: Top highlight line (simulates light from above)
    draw.rounded_rectangle(
        [button_x + 2, button_y + 2,
         button_x + button_w - 2, button_y + button_h // 3],
        radius=14,
        fill=(255, 255, 255, 35),
    )

    # 3D Layer 4: Subtle border for definition
    draw.rounded_rectangle(
        [button_x, button_y, button_x + button_w, button_y + button_h],
        radius=16,
        outline=(255, 255, 255, 60),
        width=1,
    )

    # Center CTA text in button
    text_x = button_x + (button_w - cta_text_w) // 2
    text_y = button_y + (button_h - cta_text_h) // 2
    # Subtle text shadow on button
    draw.text((text_x + 1, text_y + 1), copy.cta_text, font=cta_font, fill=(0, 0, 0, 80))
    draw.text((text_x, text_y), copy.cta_text, font=cta_font, fill=brand.colors.text_light)

    # ---------------------------------------------------------------
    # 4. Logo (if available)
    # ---------------------------------------------------------------
    if brand.logo_path:
        logo_path = Path(brand.logo_path)
        if logo_path.exists():
            try:
                logo = Image.open(logo_path).convert("RGBA")
                max_logo_h = 120
                if logo.height > max_logo_h:
                    ratio = max_logo_h / logo.height
                    logo = logo.resize(
                        (int(logo.width * ratio), max_logo_h),
                        Image.LANCZOS,
                    )
                pos = _get_logo_position(logo.size, template.logo_position)
                img.paste(logo, pos, logo)
            except Exception as exc:
                logger.warning("Failed to load logo: %s", exc)

    # Convert back to RGB for JPEG output
    return img.convert("RGB")
