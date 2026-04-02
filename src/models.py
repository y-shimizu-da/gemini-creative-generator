"""Pydantic data models for Meta ad creative generation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Settings (from environment / .env)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    google_api_key: str = Field(..., description="Gemini API key")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# ---------------------------------------------------------------------------
# Brand configuration
# ---------------------------------------------------------------------------

class ColorPalette(BaseModel):
    primary: str = "#1A1A2E"
    secondary: str = "#16213E"
    accent: str = "#E94560"
    text_light: str = "#FFFFFF"
    text_dark: str = "#1A1A2E"


class FontConfig(BaseModel):
    headline: str = "assets/fonts/NotoSansJP-Bold.ttf"
    body: str = "assets/fonts/NotoSansJP-Regular.ttf"
    cta: str = "assets/fonts/NotoSansJP-Bold.ttf"


class BrandConfig(BaseModel):
    name: str
    colors: ColorPalette = ColorPalette()
    logo_path: Optional[str] = None
    fonts: FontConfig = FontConfig()


# ---------------------------------------------------------------------------
# Product information
# ---------------------------------------------------------------------------

class ProductInfo(BaseModel):
    name: str
    description: str
    category: str = ""
    target_audience: str = ""
    usp: str = ""  # Unique Selling Proposition
    image_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Template & style
# ---------------------------------------------------------------------------

class TextPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    OVERLAY = "overlay"


class TextArea(BaseModel):
    """Defines a rectangular area for text placement (pixels, 1080x1080 canvas)."""
    x: int = 50
    y: int = 700
    width: int = 980
    height: int = 330


class TemplateConfig(BaseModel):
    name: str
    description: str = ""
    text_position: TextPosition = TextPosition.BOTTOM
    headline_area: TextArea = TextArea()
    cta_area: TextArea = TextArea(x=50, y=950, width=300, height=80)
    logo_position: str = "top_right"  # top_right, top_left, bottom_right, bottom_left
    copy_style: str = "product_showcase"  # determines copy generation prompt


class StylePreset(BaseModel):
    name: str
    description: str = ""
    background_style: str = ""  # prompt keywords for background generation
    mood_keywords: str = ""     # e.g. "warm, inviting, premium"
    color_guidance: str = ""    # e.g. "warm tones with coral accents"
    font_size_headline: int = 64
    font_size_subheadline: int = 36
    font_size_cta: int = 32


class PromptPattern(BaseModel):
    """A named set of copy-style and layout prompts.

    Each pattern provides alternative prompt text for every copy_style key
    (product_showcase, benefit_highlight, etc.).
    """
    name: str
    description: str = ""
    copy_style_prompts: dict[str, str] = {}   # copy_style key -> copy prompt
    layout_prompts: dict[str, str] = {}        # copy_style key -> layout description


# ---------------------------------------------------------------------------
# Reference creative (for style guidance)
# ---------------------------------------------------------------------------

class ReferenceCreative(BaseModel):
    """A reference ad creative used as style guidance for generation."""
    image_path: str
    category: str = ""  # e.g. "beauty", "food", "tech", "fashion"
    style_tags: list[str] = []  # e.g. ["minimal", "bold", "ugc"]
    analysis: str = ""  # AI-generated analysis of design elements
    analyzed: bool = False


# ---------------------------------------------------------------------------
# Copy / creative spec
# ---------------------------------------------------------------------------

class CopyVariation(BaseModel):
    headline: str
    subheadline: str = ""
    cta_text: str = "詳しくはこちら"
    description: str = ""
    variant_id: str = ""


class CreativeSpec(BaseModel):
    """Full specification for a single creative to be generated."""
    product: ProductInfo
    brand: BrandConfig
    template: TemplateConfig
    style: StylePreset
    copy: CopyVariation
    spec_id: str = ""


# ---------------------------------------------------------------------------
# Generation results
# ---------------------------------------------------------------------------

class GenerationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class GenerationResult(BaseModel):
    spec: CreativeSpec
    status: GenerationStatus = GenerationStatus.SUCCESS
    image_path: Optional[str] = None
    bg_image_path: Optional[str] = None
    error_message: str = ""
    image_prompt: str = ""
    cost_usd: float = 0.0
