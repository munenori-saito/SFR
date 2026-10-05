"""
Soul Flight Recorder — AIインタビューエンジン
フェーズ管理 + Groq によるストリーミング応答
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Generator

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


# ------------------------------------------------------------------ #
# フェーズ定義
# ------------------------------------------------------------------ #

class Phase(str, Enum):
    ICEBREAK = "icebreak"
    EXPLORE  = "explore"
    PATTERN  = "pattern"
    FUTURE   = "future"


def get_phase(user_turn_count: int) -> Phase:
    """ユーザーの発言回数からフェーズを判定する"""
    if user_turn_count < 3:
        return Phase.ICEBREAK
    elif user_turn_count < 10:
        return Phase.EXPLORE
    elif user_turn_count < 15:
        return Phase.PATTERN
    else:
        return Phase.FUTURE


PHASE_LABELS: dict[str, dict[Phase, str]] = {
    "ja": {
        Phase.ICEBREAK: "① アイスブレイク",
        Phase.EXPLORE:  "② 経験の掘り下げ",
        Phase.PATTERN:  "③ パターンの発見",
        Phase.FUTURE:   "④ 未来への接続",
    },
    "en": {
        Phase.ICEBREAK: "① Ice Break",
        Phase.EXPLORE:  "② Deep Dive",
        Phase.PATTERN:  "③ Pattern Discovery",
        Phase.FUTURE:   "④ Future Connection",
    },
}

PHASE_GOALS: dict[str, dict[Phase, str]] = {
    "ja": {
        Phase.ICEBREAK: "緊張をほぐし、自己開示を促すウォームアップ",
        Phase.EXPLORE:  "具体的なエピソードと感情を引き出す",
        Phase.PATTERN:  "複数の経験に共通するパターンを見つける",
        Phase.FUTURE:   "過去のパターンを未来の可能性に結びつける",
    },
    "en": {
        Phase.ICEBREAK: "Ease tension and warm up self-disclosure",
        Phase.EXPLORE:  "Draw out specific episodes and emotions",
        Phase.PATTERN:  "Find common patterns across experiences",
        Phase.FUTURE:   "Connect past patterns to future possibilities",
    },
}


# ------------------------------------------------------------------ #
# システムプロンプト
# ------------------------------------------------------------------ #

def _build_system_prompt(phase: Phase, lang: str, profile: dict | None = None) -> str:
    goal = PHASE_GOALS[lang][phase]
    phase_label = PHASE_LABELS[lang][phase]

    # プロファイルが存在する場合のインジェクション文字列を生成
    profile_section = ""
    if profile and any(profile.get(k) for k in ["excitement_triggers", "strengths", "core_values", "future_keywords"]):
        if lang == "ja":
            lines = ["【これまでに分かっているユーザーの情報】（積極的に活用して質問を深めること）"]
            if profile.get("excitement_triggers"):
                lines.append(f"ワクワクのトリガー: {', '.join(profile['excitement_triggers'])}")
            if profile.get("strengths"):
                lines.append(f"強み: {', '.join(profile['strengths'])}")
            if profile.get("core_values"):
                lines.append(f"価値観: {', '.join(profile['core_values'])}")
            if profile.get("avoidance_patterns"):
                lines.append(f"モヤモヤのパターン: {', '.join(profile['avoidance_patterns'])}")
            if profile.get("future_keywords"):
                lines.append(f"将来のキーワード: {', '.join(profile['future_keywords'])}")
            profile_section = "\n".join(lines) + "\n\n"
        else:
            lines = ["【Known User Profile】(Actively use this to deepen your questions)"]
            if profile.get("excitement_triggers"):
                lines.append(f"Excitement triggers: {', '.join(profile['excitement_triggers'])}")
            if profile.get("strengths"):
                lines.append(f"Strengths: {', '.join(profile['strengths'])}")
            if profile.get("core_values"):
                lines.append(f"Core values: {', '.join(profile['core_values'])}")
            if profile.get("avoidance_patterns"):
                lines.append(f"Avoidance patterns: {', '.join(profile['avoidance_patterns'])}")
            if profile.get("future_keywords"):
                lines.append(f"Future keywords: {', '.join(profile['future_keywords'])}")
            profile_section = "\n".join(lines) + "\n\n"

    if lang == "ja":
        return f"""\
あなたは「Soul Flight Recorder」のAIインタビュアーです。
ユーザーの過去の経験・感情を通じて、本人も気づいていない
「本当の関心・強み・行動パターン」を自然に引き出してください。

{profile_section}【現在のフェーズ】{phase_label}
【このフェーズの目的】{goal}

【インタビューの原則】
1. 1ターンに質問は必ず1つだけ。複数聞かない。
2. ユーザーの言葉をそのまま使って次の質問を組み立てる（受け売りではなく接続）。
3. 「なぜ？」は直接使わない。代わりに：
   「どんな気持ちがありましたか？」
   「そこに引き寄せられた理由、心当たりはありますか？」
4. 抽象的な答えが来たら、具体的な場面に落とす：
   「例えば最近、どんな場面でそう感じましたか？」
5. 評価・アドバイスはしない。このフェーズは「引き出す」だけ。
6. ネガティブな経験も大切な情報として受け止める：
   「しんどかったんですね。それでも続けていたのは何かあったんでしょうか？」
7. 返答は短く（共感 1〜2文 ＋ 質問 1文）にまとめる。

最初のターンでは、温かく自然な挨拶と共にアイスブレイクの質問をしてください。\
"""
    else:
        return f"""\
You are an AI interviewer for "Soul Flight Recorder."
Your role is to naturally draw out the user's hidden interests, strengths,
and behavioral patterns through exploring their past experiences and emotions.

{profile_section}【Current Phase】{phase_label}
【Phase Goal】{goal}

【Interview Principles】
1. Ask only ONE question per turn — never multiple.
2. Build your next question using the user's own words (connect, don't parrot).
3. Avoid direct "Why?" — use instead:
   "How did that feel?"
   "What do you think drew you to it?"
4. For abstract answers, ask for a concrete moment:
   "Can you think of a specific situation where you felt that?"
5. No evaluation or advice — focus purely on drawing out.
6. Treat negative experiences as valuable data:
   "That sounds tough. What kept you going through it?"
7. Keep responses brief: empathy (1-2 sentences) + one question.

On the first turn, greet the user warmly and open with an icebreak question.\
"""


# ------------------------------------------------------------------ #
# インタビューエンジン
# ------------------------------------------------------------------ #

class InterviewEngine:
    """Groq を使ったフェーズ管理付きインタビューエンジン"""

    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def stream_response(
        self,
        history: list[dict],
        lang: str = "ja",
        phase: Phase | None = None,
        profile: dict | None = None,
        context_memo: str | None = None,
    ) -> Generator[str, None, None]:
        """
        history: [{"role": "user"|"assistant", "content": "..."}] の順序付きリスト
        lang: "ja" | "en"
        phase: 明示的に指定する場合。None なら history の turn 数から自動判定
        profile: プロファイルJSONを注入する場合
        context_memo: Eco モード時に注入する過去会話の要約メモ
        """
        if phase is None:
            user_turns = sum(1 for m in history if m["role"] == "user")
            phase = get_phase(user_turns)
        system_prompt = _build_system_prompt(phase, lang, profile)

        messages = [{"role": "system", "content": system_prompt}]
        if context_memo:
            memo_label = (
                "【これまでの会話の要約（Ecoモードによる省略）】"
                if lang == "ja"
                else "[Summary of earlier conversation (Eco mode)]"
            )
            messages.append({
                "role": "system",
                "content": f"{memo_label}\n{context_memo}",
            })
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)

        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort="low",
            messages=messages,
            stream=True,
            temperature=0.75,
            max_tokens=512,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text

    def current_phase(self, history: list[dict]) -> Phase:
        user_turns = sum(1 for m in history if m["role"] == "user")
        return get_phase(user_turns)

    def stream_summary(
        self,
        history: list[dict],
        lang: str = "ja",
    ) -> Generator[str, None, None]:
        """会話履歴からこれまでの気づき・まとめを生成する"""
        if not history:
            return

        conv_text = "\n".join(
            f"{'Q' if m['role'] == 'assistant' else 'A'}: {m['content']}"
            for m in history
        )

        if lang == "ja":
            prompt = f"""\
以下はSoul Flight Recorder のインタビュー記録です。

{conv_text}

---
この会話から読み取れることを、以下の形式で簡潔にまとめてください：

## ✨ 見えてきたこと
（エネルギーが上がった活動・状況・感情を箇条書きで3〜5個）

## 💡 キーワード
（会話に繰り返し出てきた言葉・テーマを5〜8個、カンマ区切り）

## 🔍 まだ深掘りできそうなこと
（次に聞くと面白そうなポイントを1〜2個）
"""
        else:
            prompt = f"""\
Below is an interview record from Soul Flight Recorder.

{conv_text}

---
Summarize what can be read from this conversation in the following format:

## ✨ What's Emerging
(3-5 bullet points: activities, situations, or emotions that energized the user)

## 💡 Keywords
(5-8 recurring words/themes from the conversation, comma-separated)

## 🔍 Areas to Explore Further
(1-2 points that seem worth digging into next)
"""

        messages = [
            {"role": "system", "content": "You are a concise summarizer. Output only the requested format."},
            {"role": "user", "content": prompt},
        ]
        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort="low",
            messages=messages,
            stream=True,
            temperature=0.5,
            max_tokens=600,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
