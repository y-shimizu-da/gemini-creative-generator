"""Output module: manifest files (JSON/CSV) and HTML preview generation."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from .models import GenerationResult, GenerationStatus

logger = logging.getLogger(__name__)


def _result_to_dict(r: GenerationResult) -> dict[str, Any]:
    """Convert a GenerationResult to a flat dict for manifest output."""
    return {
        "spec_id": r.spec.spec_id,
        "status": r.status.value,
        "image_path": r.image_path or "",
        "bg_image_path": r.bg_image_path or "",
        "product_name": r.spec.product.name,
        "product_category": r.spec.product.category,
        "template_name": r.spec.template.name,
        "template_style": r.spec.template.copy_style,
        "style_name": r.spec.style.name,
        "headline": r.spec.copy.headline,
        "subheadline": r.spec.copy.subheadline,
        "cta_text": r.spec.copy.cta_text,
        "description": r.spec.copy.description,
        "copy_variant_id": r.spec.copy.variant_id,
        "image_prompt": r.image_prompt,
        "cost_usd": r.cost_usd,
        "error_message": r.error_message,
    }


def write_manifest_json(results: list[GenerationResult], output_path: Path) -> None:
    """Write full manifest as JSON."""
    data = [_result_to_dict(r) for r in results]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifest JSON written to %s", output_path)


def write_manifest_csv(results: list[GenerationResult], output_path: Path) -> None:
    """Write manifest as CSV for spreadsheet analysis."""
    if not results:
        return

    rows = [_result_to_dict(r) for r in results]
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Manifest CSV written to %s", output_path)


# ---------------------------------------------------------------------------
# HTML preview
# ---------------------------------------------------------------------------

PREVIEW_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ad Creative Preview - {{ run_id }}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', 'Hiragino Sans', sans-serif; background: #1a1a2e; color: #fff; padding: 20px; }
h1 { text-align: center; margin-bottom: 10px; font-size: 24px; }
.stats { text-align: center; margin-bottom: 20px; color: #aaa; font-size: 14px; }
.filters { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 24px; }
.filters select { padding: 8px 16px; border-radius: 8px; border: 1px solid #333; background: #16213e; color: #fff; font-size: 14px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.card { background: #16213e; border-radius: 12px; overflow: hidden; transition: transform 0.2s; }
.card:hover { transform: scale(1.02); }
.card img { width: 100%; aspect-ratio: 1; object-fit: cover; cursor: pointer; }
.card-info { padding: 12px; }
.card-info .headline { font-weight: 700; font-size: 15px; margin-bottom: 4px; }
.card-info .meta { font-size: 12px; color: #aaa; }
.card-info .meta span { margin-right: 8px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 4px; }
.tag-template { background: #e94560; }
.tag-style { background: #0f3460; }
.modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; }
.modal.active { display: flex; }
.modal img { max-width: 90vw; max-height: 90vh; border-radius: 8px; }
.modal-close { position: fixed; top: 20px; right: 30px; font-size: 32px; color: #fff; cursor: pointer; z-index: 1001; }
</style>
</head>
<body>
<h1>Ad Creative Preview</h1>
<div class="stats">
  Run: {{ run_id }} | Total: {{ total }} | Success: {{ success }} | Failed: {{ failed }} | Cost: ${{ cost }}
</div>
<div class="filters">
  <select id="filterProduct" onchange="applyFilters()">
    <option value="">All Products</option>
    {% for p in products %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
  </select>
  <select id="filterTemplate" onchange="applyFilters()">
    <option value="">All Templates</option>
    {% for t in templates %}<option value="{{ t }}">{{ t }}</option>{% endfor %}
  </select>
  <select id="filterStyle" onchange="applyFilters()">
    <option value="">All Styles</option>
    {% for s in styles %}<option value="{{ s }}">{{ s }}</option>{% endfor %}
  </select>
</div>
<div class="grid" id="grid">
{% for item in items %}
  <div class="card" data-product="{{ item.product_name }}" data-template="{{ item.template_name }}" data-style="{{ item.style_name }}">
    <img src="{{ item.image_rel }}" alt="{{ item.headline }}" loading="lazy" onclick="openModal(this.src)">
    <div class="card-info">
      <div class="headline">{{ item.headline }}</div>
      <div class="meta">
        <span class="tag tag-template">{{ item.template_name }}</span>
        <span class="tag tag-style">{{ item.style_name }}</span>
      </div>
      <div class="meta" style="margin-top:4px;">
        <span>CTA: {{ item.cta_text }}</span>
      </div>
    </div>
  </div>
{% endfor %}
</div>
<div class="modal" id="modal" onclick="closeModal()">
  <span class="modal-close" onclick="closeModal()">&times;</span>
  <img id="modalImg" src="" alt="">
</div>
<script>
function applyFilters() {
  const p = document.getElementById('filterProduct').value;
  const t = document.getElementById('filterTemplate').value;
  const s = document.getElementById('filterStyle').value;
  document.querySelectorAll('.card').forEach(c => {
    const show = (!p || c.dataset.product === p) && (!t || c.dataset.template === t) && (!s || c.dataset.style === s);
    c.style.display = show ? '' : 'none';
  });
}
function openModal(src) {
  document.getElementById('modalImg').src = src;
  document.getElementById('modal').classList.add('active');
}
function closeModal() {
  document.getElementById('modal').classList.remove('active');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
</script>
</body>
</html>
"""


def write_preview_html(
    results: list[GenerationResult],
    output_path: Path,
    run_id: str,
) -> None:
    """Generate an HTML preview page with filterable grid."""
    successful = [r for r in results if r.status == GenerationStatus.SUCCESS and r.image_path]

    items = []
    for r in successful:
        img_rel = str(Path(r.image_path).relative_to(output_path.parent))
        items.append({
            "image_rel": img_rel.replace("\\", "/"),
            "headline": r.spec.copy.headline,
            "cta_text": r.spec.copy.cta_text,
            "product_name": r.spec.product.name,
            "template_name": r.spec.template.name,
            "style_name": r.spec.style.name,
        })

    products = sorted(set(i["product_name"] for i in items))
    templates = sorted(set(i["template_name"] for i in items))
    styles = sorted(set(i["style_name"] for i in items))

    total_cost = sum(r.cost_usd for r in results)

    template = Template(PREVIEW_TEMPLATE)
    html = template.render(
        run_id=run_id,
        total=len(results),
        success=len(successful),
        failed=len(results) - len(successful),
        cost=f"{total_cost:.2f}",
        products=products,
        templates=templates,
        styles=styles,
        items=items,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Preview HTML written to %s", output_path)


def write_all_outputs(
    results: list[GenerationResult],
    run_dir: Path,
    run_id: str,
) -> None:
    """Write all output files (JSON, CSV, HTML preview)."""
    write_manifest_json(results, run_dir / "manifest.json")
    write_manifest_csv(results, run_dir / "manifest.csv")
    write_preview_html(results, run_dir / "preview.html", run_id)
