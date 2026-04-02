"""Streamlit Web UI for Meta Ad Creative Generator."""

from __future__ import annotations

import asyncio
import io
import os
import zipfile
from pathlib import Path

import streamlit as st
import yaml

from src.models import (
    BrandConfig,
    ColorPalette,
    CopyVariation,
    FontConfig,
    ProductInfo,
    PromptPattern,
    ReferenceCreative,
    StylePreset,
    TemplateConfig,
    TextArea,
    TextPosition,
)
from src.output import write_all_outputs
from src.pipeline import PatternSelection, Pipeline, PipelineConfig
from src.reference_analyzer import get_full_analysis

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Meta Ad Creative Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "products" not in st.session_state:
    st.session_state.products = []
if "references" not in st.session_state:
    st.session_state.references = []
if "generation_results" not in st.session_state:
    st.session_state.generation_results = None
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

# 参考クリエイティブ保存ディレクトリ
Path("assets/references").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: load YAML configs
# ---------------------------------------------------------------------------

def _load_yaml_configs(config_dir: str, model_cls):
    """Load all YAML files from a directory and parse into model instances."""
    configs = []
    path = Path(config_dir)
    if not path.exists():
        return configs
    for f in sorted(path.glob("*.yaml")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            configs.append(model_cls(**data))
        except Exception as e:
            st.warning(f"設定ファイル読み込みエラー: {f.name}: {e}")
    return configs


def _load_templates() -> list[TemplateConfig]:
    return _load_yaml_configs("config/templates", TemplateConfig)


def _load_styles() -> list[StylePreset]:
    return _load_yaml_configs("config/styles", StylePreset)


def _load_prompt_patterns() -> list[PromptPattern]:
    return _load_yaml_configs("config/prompt_patterns", PromptPattern)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_brand, tab_product, tab_settings, tab_generate, tab_preview = st.tabs([
    "1. ブランド設定",
    "2. 商品入力",
    "3. 生成設定",
    "4. 生成実行",
    "5. 結果プレビュー",
])

# ======================= TAB 1: BRAND =======================

with tab_brand:
    st.header("ブランド設定")

    col1, col2 = st.columns(2)

    with col1:
        brand_name = st.text_input("ブランド名", value="マイブランド", key="brand_name")

        st.subheader("カラーパレット")
        c1, c2, c3 = st.columns(3)
        color_primary = c1.color_picker("プライマリ", value="#1A1A2E", key="color_primary")
        color_secondary = c2.color_picker("セカンダリ", value="#16213E", key="color_secondary")
        color_accent = c3.color_picker("アクセント", value="#E94560", key="color_accent")

        c4, c5 = st.columns(2)
        color_text_light = c4.color_picker("テキスト（明）", value="#FFFFFF", key="color_text_light")
        color_text_dark = c5.color_picker("テキスト（暗）", value="#1A1A2E", key="color_text_dark")

    with col2:
        st.subheader("ロゴ")
        logo_file = st.file_uploader("ロゴ画像をアップロード（PNG推奨）", type=["png", "jpg", "jpeg"], key="logo_upload")
        logo_path = None
        if logo_file:
            logo_dir = Path("assets/logos")
            logo_dir.mkdir(parents=True, exist_ok=True)
            logo_path = str(logo_dir / logo_file.name)
            with open(logo_path, "wb") as f:
                f.write(logo_file.getvalue())
            st.image(logo_file, width=150)
            st.success(f"ロゴ保存: {logo_path}")

        st.subheader("フォント")
        st.info("デフォルト: システムフォント（Meiryo等）を使用。カスタムフォントはassets/fonts/に配置してください。")


# ======================= TAB 2: PRODUCTS =======================

with tab_product:
    st.header("商品入力")
    st.markdown(
        "生成する広告の対象商品を追加してください。"
        "**商品画像をアップロードすると、AIがその実際の商品をクリエイティブに組み込みます。**"
    )

    with st.form("add_product_form"):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            p_name = st.text_input("商品名 *", key="p_name")
            p_desc = st.text_area("商品説明 *", key="p_desc", height=100)
            p_category = st.text_input("カテゴリ", placeholder="例: 美容, 食品, テック", key="p_category")
        with p_col2:
            p_target = st.text_input("ターゲット層", placeholder="例: 20-30代女性, 健康志向", key="p_target")
            p_usp = st.text_input("USP（独自の強み）", placeholder="例: 業界初の○○技術", key="p_usp")
            p_image = st.file_uploader(
                "商品画像（推奨）",
                type=["png", "jpg", "jpeg", "webp"],
                key="p_image",
                help="商品の写真をアップロードすると、AIが実際の商品をクリエイティブに組み込みます。",
            )

        submitted = st.form_submit_button("商品を追加", type="primary")

    if submitted and p_name and p_desc:
        # Save product image if uploaded
        p_image_path = None
        if p_image is not None:
            img_dir = Path("assets/product_images")
            img_dir.mkdir(parents=True, exist_ok=True)
            # Use product name in filename for clarity
            safe_name = p_name.replace(" ", "_").replace("/", "_")[:30]
            ext = Path(p_image.name).suffix
            p_image_path = str(img_dir / f"{safe_name}{ext}")
            with open(p_image_path, "wb") as f:
                f.write(p_image.getvalue())

        product = ProductInfo(
            name=p_name,
            description=p_desc,
            category=p_category,
            target_audience=p_target,
            usp=p_usp,
            image_path=p_image_path,
        )
        st.session_state.products.append(product)
        st.success(f"「{p_name}」を追加しました" + (" (画像付き)" if p_image_path else ""))

    # Display registered products
    if st.session_state.products:
        st.subheader(f"登録済み商品 ({len(st.session_state.products)}件)")
        for i, prod in enumerate(st.session_state.products):
            with st.expander(f"{i+1}. {prod.name}", expanded=False):
                cols_info = st.columns([1, 2]) if prod.image_path else [st.container()]
                if prod.image_path and Path(prod.image_path).exists():
                    with cols_info[0]:
                        st.image(prod.image_path, width=200, caption="商品画像")
                info_col = cols_info[1] if prod.image_path else cols_info[0]
                with info_col:
                    st.markdown(f"**説明:** {prod.description}")
                    if prod.category:
                        st.markdown(f"**カテゴリ:** {prod.category}")
                    if prod.target_audience:
                        st.markdown(f"**ターゲット:** {prod.target_audience}")
                    if prod.usp:
                        st.markdown(f"**USP:** {prod.usp}")
                    if not prod.image_path:
                        st.caption("商品画像なし（抽象ビジュアルで生成されます）")
                if st.button(f"削除", key=f"del_prod_{i}"):
                    st.session_state.products.pop(i)
                    st.rerun()
    else:
        st.info("まだ商品が登録されていません。上のフォームから追加してください。")


# ======================= TAB 3: SETTINGS =======================

with tab_settings:
    st.header("生成設定")

    # ----- 参考クリエイティブ -----
    st.subheader("参考クリエイティブ")
    st.markdown(
        "実際の広告クリエイティブ画像をアップロードすると、AIがデザイン要素を解析し、"
        "そのスタイルを参考にして生成品質を向上させます。"
    )

    ref_col1, ref_col2 = st.columns([1, 2])

    with ref_col1:
        ref_files = st.file_uploader(
            "参考画像をアップロード",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="ref_upload",
        )
        ref_category = st.selectbox(
            "カテゴリ",
            ["美容", "食品", "テック", "ファッション", "ヘルスケア", "ライフスタイル", "その他"],
            key="ref_category",
        )
        if st.button("参考クリエイティブを追加", type="secondary") and ref_files:
            ref_dir = Path("assets/references")
            added = 0
            existing_paths = {r.image_path for r in st.session_state.references}
            for rf in ref_files:
                save_path = str(ref_dir / rf.name)
                if save_path not in existing_paths:
                    with open(save_path, "wb") as f:
                        f.write(rf.getvalue())
                    st.session_state.references.append(
                        ReferenceCreative(
                            image_path=save_path,
                            category=ref_category,
                        )
                    )
                    added += 1
            if added > 0:
                st.success(f"{added}件の参考クリエイティブを追加しました")
                st.rerun()

    with ref_col2:
        if st.session_state.references:
            st.caption(f"登録済み: {len(st.session_state.references)}件")
            ref_cols_per_row = 3
            refs = st.session_state.references
            for row_start in range(0, len(refs), ref_cols_per_row):
                ref_cols = st.columns(ref_cols_per_row)
                for col_idx, ref in enumerate(refs[row_start:row_start + ref_cols_per_row]):
                    with ref_cols[col_idx]:
                        ref_path = Path(ref.image_path)
                        if ref_path.exists():
                            st.image(ref.image_path, use_container_width=True)
                        st.caption(f"{ref_path.name}\nカテゴリ: {ref.category}")
                        # 解析済みの場合はサマリー表示
                        full = get_full_analysis(ref.image_path)
                        if full:
                            st.caption(
                                f"解析済 | 品質: {full.get('overall_quality_score', '-')}/10\n"
                                f"{full.get('design_summary', '')}"
                            )
                        global_idx = row_start + col_idx
                        if st.button("削除", key=f"del_ref_{global_idx}"):
                            st.session_state.references.pop(global_idx)
                            st.rerun()
        else:
            st.info("参考クリエイティブが未登録です。左のフォームから追加すると生成品質が向上します。")

    st.divider()

    # ----- プロンプトパターン選択 -----
    prompt_patterns = _load_prompt_patterns()

    st.subheader("プロンプトパターン選択")
    st.markdown("テンプレプロンプトのパターンを選択してください。複数選択するとそれぞれのパターンで生成されます。")

    selected_pattern_selections: list[PatternSelection] = []

    if not prompt_patterns:
        st.warning("config/prompt_patterns/ にパターンYAMLがありません")
    else:
        for pat in prompt_patterns:
            pat_selected = st.checkbox(
                f"{pat.name}",
                value=(pat.name == "スタンダード"),
                key=f"pat_{pat.name}",
                help=pat.description,
            )
            if pat_selected:
                # 画像枚数スライダー
                img_count = st.slider(
                    f"「{pat.name}」の生成枚数（テンプレート×商品ごと）",
                    min_value=1, max_value=10, value=5,
                    key=f"pat_count_{pat.name}",
                )
                selected_pattern_selections.append(PatternSelection(pattern=pat, image_count=img_count))

                # プロンプト記載欄
                with st.expander(f"「{pat.name}」プロンプト内容", expanded=False):
                    if pat.copy_style_prompts:
                        st.markdown("**コピースタイルプロンプト:**")
                        for style_key, prompt_text in pat.copy_style_prompts.items():
                            st.markdown(f"`{style_key}`")
                            st.text(prompt_text.strip())
                            st.markdown("---")
                    if pat.layout_prompts:
                        st.markdown("**レイアウトプロンプト:**")
                        for style_key, layout_text in pat.layout_prompts.items():
                            st.markdown(f"`{style_key}`")
                            st.text(layout_text.strip())
                            st.markdown("---")

    st.divider()

    # ----- テンプレート・スタイル選択 -----
    templates = _load_templates()
    styles = _load_styles()

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("テンプレート選択")
        selected_templates = []
        for tmpl in templates:
            if st.checkbox(f"{tmpl.name}", value=True, key=f"tmpl_{tmpl.name}", help=tmpl.description):
                selected_templates.append(tmpl)

        if not templates:
            st.warning("config/templates/ にテンプレートYAMLがありません")

    with col_s2:
        st.subheader("スタイル選択")
        selected_styles = []
        for sty in styles:
            if st.checkbox(f"{sty.name}", value=True, key=f"style_{sty.name}", help=sty.description):
                selected_styles.append(sty)

        if not styles:
            st.warning("config/styles/ にスタイルYAMLがありません")

    # Estimate
    st.divider()
    st.subheader("生成見積もり")

    n_products = len(st.session_state.products)
    n_templates = len(selected_templates)
    n_styles = len(selected_styles)
    n_total_copies = sum(ps.image_count for ps in selected_pattern_selections) if selected_pattern_selections else 0
    n_patterns = len(selected_pattern_selections)

    n_total = n_products * n_templates * n_styles * n_total_copies
    cost = n_total * 0.039

    est_col1, est_col2, est_col3, est_col4 = st.columns(4)
    est_col1.metric("合計クリエイティブ数", f"{n_total}枚")
    est_col2.metric("AI画像生成数", f"{n_total}回")
    est_col3.metric("推定コスト", f"${cost:.2f}")
    est_col4.metric("コピー生成リクエスト", f"{n_products * n_templates * n_patterns}回")

    if n_total > 0:
        pattern_detail = " + ".join(f"{ps.pattern.name}:{ps.image_count}枚" for ps in selected_pattern_selections)
        st.caption(
            f"計算式: {n_products}商品 x {n_templates}テンプレート x {n_styles}スタイル x "
            f"パターン合計({pattern_detail}) = {n_total}枚"
        )
        st.caption("各クリエイティブはAIが完成形を一発生成します（テキスト・レイアウト・商品画像統合済み）")


# ======================= TAB 4: GENERATE =======================

with tab_generate:
    st.header("生成実行")

    api_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        key="api_key",
        help="Gemini APIキーを入力してください。環境変数GOOGLE_API_KEYからも読み取れます。",
        value=os.environ.get("GOOGLE_API_KEY", ""),
    )

    # Validation
    can_generate = True
    issues = []
    if not api_key:
        issues.append("APIキーが未設定です")
        can_generate = False
    if not st.session_state.products:
        issues.append("商品が登録されていません")
        can_generate = False
    if not selected_templates:
        issues.append("テンプレートが選択されていません")
        can_generate = False
    if not selected_styles:
        issues.append("スタイルが選択されていません")
        can_generate = False
    if not selected_pattern_selections:
        issues.append("プロンプトパターンが選択されていません")
        can_generate = False

    if issues:
        for issue in issues:
            st.warning(issue)

    # Generate button
    if st.button("生成開始", type="primary", disabled=not can_generate, use_container_width=True):
        brand = BrandConfig(
            name=brand_name,
            colors=ColorPalette(
                primary=color_primary,
                secondary=color_secondary,
                accent=color_accent,
                text_light=color_text_light,
                text_dark=color_text_dark,
            ),
            logo_path=logo_path,
        )

        config = PipelineConfig(
            brand=brand,
            products=st.session_state.products,
            templates=selected_templates,
            styles=selected_styles,
            references=st.session_state.references,
            pattern_selections=selected_pattern_selections,
        )

        # Progress display
        progress_bar = st.progress(0, text="初期化中...")
        status_text = st.empty()
        preview_area = st.container()

        def update_progress(progress):
            phase_labels = {
                "reference_analysis": "Phase 0: 参考クリエイティブ分析中",
                "copy_generation": "Phase 1/2: コピー生成中",
                "creative_generation": "Phase 2/2: クリエイティブ生成中（AI一発生成）",
                "done": "完了",
            }
            label = phase_labels.get(progress.phase, progress.phase)
            progress_bar.progress(
                min(progress.progress_pct, 1.0),
                text=f"{label}: {progress.current_item}",
            )
            status_text.caption(
                f"Phase: {label} | "
                f"進捗: {progress.completed_steps}/{progress.total_steps} | "
                f"エラー: {len(progress.errors)}"
            )

        pipeline = Pipeline(config, api_key, progress_callback=update_progress)

        with st.spinner("クリエイティブを生成中..."):
            results = asyncio.run(pipeline.run())

        # Write output files
        write_all_outputs(results, pipeline.run_dir, pipeline.run_id)

        st.session_state.generation_results = results
        st.session_state.run_dir = str(pipeline.run_dir)

        # Summary
        success_count = sum(1 for r in results if r.status.value == "success")
        fail_count = len(results) - success_count
        total_cost = sum(r.cost_usd for r in results)

        progress_bar.progress(1.0, text="生成完了!")
        st.success(f"生成完了: {success_count}枚成功 / {fail_count}枚失敗 | 推定コスト: ${total_cost:.2f}")

        if pipeline.progress.errors:
            with st.expander(f"エラー一覧 ({len(pipeline.progress.errors)}件)"):
                for err in pipeline.progress.errors:
                    st.error(err)


# ======================= TAB 5: PREVIEW =======================

with tab_preview:
    st.header("結果プレビュー")

    results = st.session_state.generation_results
    run_dir = st.session_state.run_dir

    if results is None:
        st.info("まだ生成が実行されていません。「生成実行」タブから開始してください。")
    else:
        successful = [r for r in results if r.status.value == "success" and r.image_path]

        if not successful:
            st.warning("成功したクリエイティブがありません。")
        else:
            # Filters
            all_products = sorted(set(r.spec.product.name for r in successful))
            all_templates = sorted(set(r.spec.template.name for r in successful))
            all_styles = sorted(set(r.spec.style.name for r in successful))

            fc1, fc2, fc3 = st.columns(3)
            filter_product = fc1.selectbox("商品", ["すべて"] + all_products, key="filter_product")
            filter_template = fc2.selectbox("テンプレート", ["すべて"] + all_templates, key="filter_template")
            filter_style = fc3.selectbox("スタイル", ["すべて"] + all_styles, key="filter_style")

            filtered = successful
            if filter_product != "すべて":
                filtered = [r for r in filtered if r.spec.product.name == filter_product]
            if filter_template != "すべて":
                filtered = [r for r in filtered if r.spec.template.name == filter_template]
            if filter_style != "すべて":
                filtered = [r for r in filtered if r.spec.style.name == filter_style]

            st.caption(f"{len(filtered)}枚表示中 / 全{len(successful)}枚")

            # Grid display
            cols_per_row = 4
            for row_start in range(0, len(filtered), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, r in enumerate(filtered[row_start:row_start + cols_per_row]):
                    with cols[col_idx]:
                        if r.image_path and Path(r.image_path).exists():
                            st.image(r.image_path, use_container_width=True)
                        st.markdown(f"**{r.spec.copy.headline}**")
                        st.caption(
                            f"{r.spec.template.name} | {r.spec.style.name}\n\n"
                            f"CTA: {r.spec.copy.cta_text}"
                        )

            # Download section
            st.divider()
            dc1, dc2, dc3 = st.columns(3)

            # ZIP download
            with dc1:
                if st.button("画像を一括ダウンロード (ZIP)", use_container_width=True):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for r in filtered:
                            if r.image_path and Path(r.image_path).exists():
                                zf.write(r.image_path, Path(r.image_path).name)
                    st.download_button(
                        "ZIPをダウンロード",
                        data=zip_buffer.getvalue(),
                        file_name="ad_creatives.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            # Manifest downloads
            with dc2:
                manifest_json = Path(run_dir) / "manifest.json"
                if manifest_json.exists():
                    st.download_button(
                        "manifest.json",
                        data=manifest_json.read_text(encoding="utf-8"),
                        file_name="manifest.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            with dc3:
                manifest_csv = Path(run_dir) / "manifest.csv"
                if manifest_csv.exists():
                    st.download_button(
                        "manifest.csv",
                        data=manifest_csv.read_bytes(),
                        file_name="manifest.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            # Preview HTML link
            preview_html = Path(run_dir) / "preview.html"
            if preview_html.exists():
                st.markdown(f"フルスクリーンプレビュー: `{preview_html}`")
