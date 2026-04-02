"""Ad copy generation using Gemini text model."""

from __future__ import annotations

import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from .models import CopyVariation, ProductInfo, PromptPattern, TemplateConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt with 2026 Meta ad best practices
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
あなたはMeta広告（Facebook/Instagram）で年間数億円規模の運用実績を持つ\
トップクラスのコピーライターです。CTRとCVRを最大化する広告コピーを生成してください。

## コピーライティング・フレームワーク:

### AIDA（各バリエーションで必ず意識）:
- **Attention**: 最初の3語で注目を奪う。常識の逆、疑問形、衝撃的な数字
- **Interest**: ターゲットの「自分ごと化」を促す。悩みへの共感、具体的シーン描写
- **Desire**: 商品を使った後の理想の未来を見せる。社会的証明、限定性
- **Action**: 今すぐ行動する明確な理由とCTA

### PAS（問題→扇動→解決）:
- **Problem**: ターゲットの痛点を具体的に指摘（「まだ○○で悩んでいませんか？」）
- **Agitate**: その問題の深刻さを増幅（「放置すると○○になる可能性も」）
- **Solution**: 商品がどう解決するか端的に示す

## ヘッドラインの鉄則:

1. **8文字以内**が最強。長くても12文字まで
2. **パターン割り込み**: フィードの単調さを壊す意外な言葉を使う
3. **好奇心ギャップ**: 答えを知りたくなる「開いたループ」を作る
4. **数字は具体的に**: 「多くの」→「12,847人の」、「高い」→「96.3%の」

## 感情トリガーワード（日本語）:

| カテゴリ | ワード例 |
|---|---|
| 緊急性 | 今だけ、本日限定、ラストチャンス、なくなり次第終了、残りわずか |
| 独占性 | 秘密の、限定公開、会員だけの、特別招待、先行公開 |
| 好奇心 | 驚きの、知られざる、意外な真実、なぜ？、～の秘密 |
| 信頼 | 実証済み、医師推奨、専門家が認めた、○○受賞 |
| 価値 | 無料、特典付き、送料無料、全額返金保証 |

## 日本語コピーの極意:
- **感情先行、理由後追い**: 日本の広告は感情→論理の順が効く
- **オノマトペ活用**: ふわふわ、もちもち、サラサラ、ぐんぐん → 五感に訴える
- **体言止め**: 「美肌の秘密。」「究極の一杯。」→ 余韻と力強さ
- **「あなた」を使わず「自分ごと」にする**: シーン描写で共感を誘う

## CTA（行動喚起）ルール:
- 必ずアクション動詞で始める: 「購入する」「試す」「確認する」「申し込む」
- 「詳しくはこちら」は最弱。具体的ベネフィット付きCTAが2倍効果的
- 所有感: 「無料で手に入れる」>「無料で試す」>「詳しく見る」
- 緊急性付加: 「今すぐ50%OFFで購入」のように価値+行動+期限を1文に

## 出力ルール:
- 必ず指定された数のバリエーションをJSON配列で返す
- **各バリエーションは異なる心理的トリガーを使う**（感情/理性/社会的証明/緊急性/好奇心）
- 日本語で生成する
- ヘッドラインは改行なしの1行、8文字以内推奨（最大12文字）
- CTAはアクション動詞で始めること
"""

# Per-template copy style instructions
COPY_STYLE_PROMPTS = {
    "product_showcase": (
        "【ダイレクト商品訴求】スタイルで生成してください。\n"
        "構成: 商品名 + 主要ベネフィット + 裏付け（数字/実績）\n"
        "例ヘッドライン: 「美肌革命」「至福の一杯」「神コスパ」\n"
        "例サブヘッドライン: 「医学誌掲載の独自成分配合」「プロが認めた本格派」\n"
        "トーン: 確信的、プレミアム感、一目で価値が伝わる"
    ),
    "benefit_highlight": (
        "【PAS（問題→扇動→解決）】フレームワークで生成してください。\n"
        "構成: ヘッドラインで痛点指摘 → サブヘッドラインで解決策提示\n"
        "例ヘッドライン: 「まだ悩んでる？」「時間の無駄です」「それ、間違いかも」\n"
        "例サブヘッドライン: 「たった10分で解決する方法」\n"
        "トーン: 共感的→希望を与える。ターゲットの日常シーンを想起させる"
    ),
    "social_proof": (
        "【社会的証明・権威性】重視スタイルで生成してください。\n"
        "構成: 具体的な数字（実績・ユーザー数・満足度） + 権威の裏付け\n"
        "例ヘッドライン: 「12,847人が選んだ」「満足度96.3%」「3冠達成」\n"
        "例サブヘッドライン: 「専門家も推薦する実力派」\n"
        "数字は端数まで具体的に（「約1万人」より「12,847人」が信頼性3倍）\n"
        "トーン: 客観的、データに基づく信頼感"
    ),
    "urgency_offer": (
        "【緊急性＆希少性】訴求スタイルで生成してください。\n"
        "構成: 割引/特典 + 期限/残数 + 強いアクションCTA\n"
        "例ヘッドライン: 「本日限定50%OFF」「残り3個」「今だけ無料」\n"
        "例サブヘッドライン: 「このページを閉じたら二度とこの価格では買えません」\n"
        "トーン: エネルギッシュ、見逃す恐怖（FOMO）を刺激"
    ),
    "testimonial": (
        "【顧客証言・UGC風】スタイルで生成してください。\n"
        "構成: リアルな使用者の声（年代・属性付き） + Before/After + 推薦\n"
        "例ヘッドライン: 「人生変わった」「もう手放せない」「もっと早く知りたかった」\n"
        "例サブヘッドライン: 「30代営業職・売上3倍達成」\n"
        "トーン: 素朴で等身大、リアルな体験談風。広告っぽさを排除"
    ),
}


def _build_user_prompt(
    product: ProductInfo,
    template: TemplateConfig,
    num_variations: int,
    prompt_pattern: PromptPattern | None = None,
) -> str:
    """Build the user prompt for copy generation."""
    # Use pattern's prompts if provided, fall back to built-in defaults
    source = prompt_pattern.copy_style_prompts if prompt_pattern else COPY_STYLE_PROMPTS
    style_instruction = source.get(
        template.copy_style,
        COPY_STYLE_PROMPTS.get(template.copy_style, COPY_STYLE_PROMPTS["product_showcase"]),
    )

    return f"""\
以下の商品情報に基づいて、{num_variations}パターンの広告コピーを生成してください。

## 商品情報:
- 商品名: {product.name}
- 説明: {product.description}
- カテゴリ: {product.category or "指定なし"}
- ターゲット: {product.target_audience or "指定なし"}
- USP（強み）: {product.usp or "指定なし"}

## コピースタイル:
{style_instruction}

## テンプレート: {template.name}

## 出力形式（JSON配列）:
```json
[
  {{
    "headline": "ヘッドライン（8文字以内推奨）",
    "subheadline": "サブヘッドライン（15文字以内推奨）",
    "cta_text": "CTAボタンのテキスト",
    "description": "説明文（省略可）"
  }}
]
```

{num_variations}パターンのバリエーションをJSON配列で返してください。
それぞれ異なる訴求軸（切り口）を使ってください。
JSONのみを返し、他のテキストは含めないでください。
"""


def _parse_copy_response(text: str, num_variations: int) -> list[CopyVariation]:
    """Parse LLM response into CopyVariation list."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        data = [data]

    variations = []
    for i, item in enumerate(data[:num_variations]):
        variations.append(
            CopyVariation(
                headline=item.get("headline", ""),
                subheadline=item.get("subheadline", ""),
                cta_text=item.get("cta_text", "詳しくはこちら"),
                description=item.get("description", ""),
                variant_id=f"v{i + 1:02d}",
            )
        )
    return variations


async def generate_copy(
    client: genai.Client,
    product: ProductInfo,
    template: TemplateConfig,
    num_variations: int = 5,
    model: str = "gemini-2.5-flash",
    prompt_pattern: PromptPattern | None = None,
) -> list[CopyVariation]:
    """Generate ad copy variations using Gemini.

    Args:
        client: Gemini API client.
        product: Product information.
        template: Template configuration (determines copy style).
        num_variations: Number of copy variations to generate.
        model: Gemini text model to use.
        prompt_pattern: Optional prompt pattern override for copy style prompts.

    Returns:
        List of CopyVariation objects.
    """
    user_prompt = _build_user_prompt(product, template, num_variations, prompt_pattern)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.9,  # Higher creativity for diverse variations
            response_mime_type="application/json",
        ),
    )

    text = response.text
    if not text:
        logger.error("Empty response from Gemini copy generation")
        return []

    try:
        return _parse_copy_response(text, num_variations)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Failed to parse copy response: %s\nRaw: %s", exc, text)
        return []
