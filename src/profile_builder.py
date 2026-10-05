"""
Soul Flight Recorder — バックグラウンドプロファイルビルダー
5ターンごとに会話履歴からユーザープロファイルJSONを自動更新する
"""

from __future__ import annotations

import json
import os
from typing import Generator

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

EMPTY_PROFILE = {
    "excitement_triggers": [],   # ワクワクする状況・行動
    "strengths": [],             # 経験から読み取れる強み・資質
    "core_values": [],           # 大切にしていること・価値観
    "avoidance_patterns": [],    # 避けている・モヤモヤする状況
    "future_keywords": [],       # 「やりたいこと」に関連するキーワード
    "confidence_score": 0.0,     # プロファイルの確信度 0.0〜1.0
}


def _build_prompt(conv_text: str, existing: dict, lang: str) -> str:
    existing_json = json.dumps(existing, ensure_ascii=False, indent=2)

    if lang == "ja":
        return f"""\
以下はユーザーとのインタビュー記録です。

【会話記録】
{conv_text}

【現在のプロファイル】
{existing_json}

---
上記の会話を分析し、プロファイルを更新してください。
- 既存の項目と矛盾する場合は新しい情報を優先してください
- 会話から読み取れない項目は既存値をそのまま維持してください
- confidence_score は会話量・一貫性に応じて 0.0〜1.0 で更新してください
- 各リスト項目は具体的な言葉で（「楽しかった」ではなく「ゼロから何かを作る瞬間」など）

必ず以下のJSONのみを返してください（説明文・マークダウン不要）：
{{
  "excitement_triggers": [],
  "strengths": [],
  "core_values": [],
  "avoidance_patterns": [],
  "future_keywords": [],
  "confidence_score": 0.0
}}"""
    else:
        return f"""\
Below is an interview record with the user.

【Conversation】
{conv_text}

【Current Profile】
{existing_json}

---
Analyze the conversation and update the profile.
- Prioritize new information over existing if there's a conflict
- Keep existing values for items not mentioned in the conversation
- Update confidence_score (0.0-1.0) based on conversation volume and consistency
- Be specific in list items (not "enjoyed it" but "building something from scratch")

Return ONLY the following JSON (no explanation, no markdown):
{{
  "excitement_triggers": [],
  "strengths": [],
  "core_values": [],
  "avoidance_patterns": [],
  "future_keywords": [],
  "confidence_score": 0.0
}}"""


class ProfileBuilder:
    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def update(
        self,
        history: list[dict],
        existing_profile: dict,
        lang: str = "ja",
    ) -> dict:
        """会話履歴から新しいプロファイルを生成して返す（同期）"""
        if not history:
            return existing_profile

        conv_text = "\n".join(
            f"{'AI' if m['role'] == 'assistant' else 'User'}: {m['content']}"
            for m in history
        )
        prompt = _build_prompt(conv_text, existing_profile or EMPTY_PROFILE, lang)

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": "You are a JSON-only responder. Output valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()

        # コードブロックがある場合は除去
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            new_profile = json.loads(raw)
            # 必須キーが欠けていたら既存値で補完
            for key, default in EMPTY_PROFILE.items():
                if key not in new_profile:
                    new_profile[key] = existing_profile.get(key, default)
            return new_profile
        except json.JSONDecodeError:
            # パース失敗時は既存プロファイルをそのまま返す
            return existing_profile


def should_update(user_turn_count: int, interval: int = 5) -> bool:
    """プロファイル更新タイミングかどうかを判定する"""
    return user_turn_count > 0 and user_turn_count % interval == 0
