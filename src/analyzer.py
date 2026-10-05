"""
Soul Flight Recorder — 自己分析モジュール
RAG（テキストファイル）＋インタビュープロファイルの両方を使って分析する
store が None のときはプロファイルのみで動作（インタビュー専用モード）
"""

from __future__ import annotations

import os
from typing import Generator

from groq import Groq
from dotenv import load_dotenv

from src.vectorstore import MemoryStore

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

# ------------------------------------------------------------------ #
# ペルソナ定義
# ------------------------------------------------------------------ #

PERSONA_OPTIONS = {
    "ja": {"general": "一般", "job_seeker": "就活生", "freshman": "大学新入生"},
    "en": {"general": "General", "job_seeker": "Job Seeker", "freshman": "University Freshman"},
}

PERSONA_INSTRUCTIONS = {
    "ja": {
        "general": "",
        "job_seeker": """\
【就活生モード】
- 強み・経験を「自己PR」「志望動機」に使える言葉で整理する
- 「この経験を面接でどう話すか」の視点を加える
- 就活の軸（大切にしたい働き方・環境・価値観）を明確にする
""",
        "freshman": """\
【大学新入生モード】
- 大学4年間で何に力を入れるべきかという視点で整理する
- サークル・研究室・インターン・資格など具体的な選択肢に言及する
- 「今学期中にできる最初の一歩」を特に具体的に提案する
""",
    },
    "en": {
        "general": "",
        "job_seeker": """\
【Job Seeker Mode】
- Frame strengths and experiences in terms of self-PR and career motivation
- Add the perspective of "how to articulate this experience in an interview"
- Clarify the job-search axis: what kind of work environment and values matter most
""",
        "freshman": """\
【University Freshman Mode】
- Frame insights in terms of what to focus on across 4 years of university
- Mention concrete options: clubs, labs, internships, certifications
- Be especially specific about "the first step you can take this semester"
""",
    },
}


# ------------------------------------------------------------------ #
# ヘルパー関数
# ------------------------------------------------------------------ #

def _format_rag_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[記録 {i}] (関連度: {h['score']})\n{h['content']}")
    return "\n\n".join(lines)


def _format_profile(profile: dict, lang: str) -> str:
    if not profile:
        return ""
    if lang == "ja":
        field_labels = {
            "excitement_triggers": "ワクワクのトリガー",
            "strengths": "強み",
            "core_values": "価値観",
            "avoidance_patterns": "モヤモヤのパターン",
            "future_keywords": "将来のキーワード",
        }
    else:
        field_labels = {
            "excitement_triggers": "Excitement Triggers",
            "strengths": "Strengths",
            "core_values": "Core Values",
            "avoidance_patterns": "Avoidance Patterns",
            "future_keywords": "Future Keywords",
        }
    lines = []
    for key, label in field_labels.items():
        val = profile.get(key)
        if val:
            lines.append(f"{label}: {', '.join(val)}")
    return "\n".join(lines)


def _build_system_prompt(lang: str, persona: str) -> str:
    persona_instruction = PERSONA_INSTRUCTIONS[lang].get(persona, "")
    if lang == "ja":
        return f"""\
あなたは「Soul Flight Recorder (SFR)」というAIコーチです。
ユーザーの過去の経験・日記・メモ、およびインタビューで蓄積されたプロファイルをもとに、
客観的かつ温かみのある視点で「自己理解」を深めるサポートをします。

{persona_instruction}
分析する際は以下を意識してください：
1. 事実の列挙ではなく、**感情の振れ幅（ワクワク・モヤモヤ・没頭感）** に注目する
2. ユーザーが気づいていないパターンや強みを引き出す
3. 断定ではなく「問いかけ」を交えて、本人が内省できるよう促す
4. インタビュープロファイルとテキスト記録の両方を統合して分析する

出力は日本語で、読みやすいマークダウン形式で。\
"""
    else:
        return f"""\
You are an AI coach for "Soul Flight Recorder (SFR)."
Using both the user's past experience records and the profile built through interviews,
you support deeper self-understanding with an objective yet warm perspective.

{persona_instruction}
When analyzing, keep in mind:
1. Focus on **emotional range (excitement, discomfort, deep focus)** rather than listing facts
2. Draw out patterns and strengths the user hasn't noticed themselves
3. Mix in questions rather than declarations, to encourage self-reflection
4. Integrate both the interview profile and text records in your analysis

Output in English, in readable markdown format.\
"""


def _build_context_block(
    rag_context: str,
    profile: dict | None,
    lang: str,
) -> str:
    """RAGコンテキストとプロファイルを1つのブロックにまとめる"""
    parts = []
    if rag_context:
        header = "【テキスト記録】" if lang == "ja" else "【Text Records】"
        parts.append(f"{header}\n{rag_context}")
    if profile:
        profile_str = _format_profile(profile, lang)
        if profile_str:
            header = "【インタビューで判明したプロファイル】" if lang == "ja" else "【Interview Profile】"
            parts.append(f"{header}\n{profile_str}")
    if not parts:
        return "（利用可能な情報がありません）" if lang == "ja" else "(No information available)"
    return "\n\n".join(parts)


# ------------------------------------------------------------------ #
# アナライザー本体
# ------------------------------------------------------------------ #

class SFRAnalyzer:
    """RAG + プロファイル統合の自己分析クラス"""

    def __init__(self, store: MemoryStore | None = None, api_key: str | None = None):
        self.store = store
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client: Groq | None = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self._api_key)
        return self._client

    def _search(self, query: str, n: int) -> list[dict]:
        if self.store is None or self.store.count() == 0:
            return []
        return self.store.search(query, n_results=n)

    def _stream(self, prompt: str, lang: str = "ja", persona: str = "general") -> Generator[str, None, None]:
        system_prompt = _build_system_prompt(lang, persona)
        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text

    # ------------------------------------------------------------------ #
    # 強み・関心の抽出
    # ------------------------------------------------------------------ #

    def extract_strengths_stream(
        self,
        profile: dict | None = None,
        lang: str = "ja",
        persona: str = "general",
    ) -> Generator[str, None, None]:
        hits = self._search("強み ワクワク 没頭 好き 達成感 楽しい strength excitement", 6)
        context = _build_context_block(_format_rag_context(hits), profile, lang)

        if lang == "ja":
            prompt = f"""\
以下の情報をもとに分析してください。

{context}

## 🔥 ワクワクのパターン（Excitement Map）
どんな状況・行動で感情が最も高まっているか、具体的に説明。

## 💪 潜在的な強み（Strength Profile）
経験・プロファイルから読み取れる能力・資質を3〜5個、根拠とともに。

## 🎯 関心の核心（Core Interest）
様々な活動の奥にある「共通のテーマ」は何か？

## 💭 問いかけ（Reflection Questions）
自己理解をさらに深めるための問いを2〜3個。\
"""
        else:
            prompt = f"""\
Analyze based on the following information.

{context}

## 🔥 Excitement Map
In what situations and actions does the user feel most energized? Be specific.

## 💪 Strength Profile
3-5 abilities and qualities from their experiences and profile, each with supporting evidence.

## 🎯 Core Interest
What is the common theme underlying their various activities?

## 💭 Reflection Questions
2-3 questions to deepen their self-understanding.\
"""
        yield from self._stream(prompt, lang, persona)

    # ------------------------------------------------------------------ #
    # 感情マップ
    # ------------------------------------------------------------------ #

    def analyze_emotion_timeline_stream(
        self,
        profile: dict | None = None,
        lang: str = "ja",
        persona: str = "general",
    ) -> Generator[str, None, None]:
        hits = self._search("感じたこと 嬉しい しんどい 達成感 モヤモヤ 刺激 emotion feeling", 8)
        context = _build_context_block(_format_rag_context(hits), profile, lang)

        if lang == "ja":
            prompt = f"""\
以下の情報をもとに、各活動における感情の強度を分析してください。

{context}

## 📊 感情マッピング（Emotion Mapping）
各活動について以下を評価してください（表形式で）：
- **没頭度** (1-5): どれだけ時間を忘れて集中できたか
- **ワクワク度** (1-5): 取り組む前・最中の高揚感
- **達成感** (1-5): 完了後の充実感
- **モヤモヤ度** (1-5): 違和感・葛藤の大きさ

表のあと、「最もエネルギーが湧いた活動」と「そうでない活動」の違いを考察。\
"""
        else:
            prompt = f"""\
Analyze the emotional intensity across activities based on the following information.

{context}

## 📊 Emotion Mapping
Rate each activity in a table format:
- **Immersion** (1-5): How much did time disappear?
- **Excitement** (1-5): Energy before and during the activity
- **Fulfillment** (1-5): Satisfaction after completion
- **Discomfort** (1-5): Sense of friction or inner conflict

After the table, analyze what differs between the highest and lowest energy activities.\
"""
        yield from self._stream(prompt, lang, persona)

    # ------------------------------------------------------------------ #
    # 未来ビジョン
    # ------------------------------------------------------------------ #

    def generate_future_vision_stream(
        self,
        profile: dict | None = None,
        lang: str = "ja",
        persona: str = "general",
    ) -> Generator[str, None, None]:
        hits = self._search("やりたいこと 将来 可能性 未来 vision future goal", 6)
        context = _build_context_block(_format_rag_context(hits), profile, lang)

        if lang == "ja":
            prompt = f"""\
以下の情報をもとに、「等身大の5年後ビジョン」を描いてください。

{context}

## 🚀 等身大の未来予想図（Future Vision Map）

このユーザーが「やりたくてたまらない」と感じるような5年後のシナリオを描いてください。

条件：
- 現在の強み・関心・価値観を踏まえる
- 「研究者 / エンジニア / 起業家」などの固定ラベルに縛られない
- 具体的な活動内容（何を作っているか、誰と関わっているか）を描く
- 「その実現に向けた最初の一歩」を具体的に提案する\
"""
        else:
            prompt = f"""\
Based on the following information, paint a "realistic 5-year vision."

{context}

## 🚀 Future Vision Map

Describe a 5-year scenario that would make this user feel "I can't wait to live this."

Conditions:
- Ground it in their current strengths, interests, and values
- Don't confine it to fixed labels like "researcher / engineer / entrepreneur"
- Describe concrete activities (what they're building, who they're working with)
- Propose a specific "first step toward making this happen"\
"""
        yield from self._stream(prompt, lang, persona)

    # ------------------------------------------------------------------ #
    # RAG チャット
    # ------------------------------------------------------------------ #

    def chat_stream(
        self,
        user_input: str,
        history: list[dict] | None = None,
        profile: dict | None = None,
        lang: str = "ja",
        persona: str = "general",
    ) -> Generator[str, None, None]:
        hits = self._search(user_input, 4)
        context = _build_context_block(_format_rag_context(hits), profile, lang)

        ref_label = "【参照できる情報】" if lang == "ja" else "【Available Information】"
        q_label = "【ユーザーの質問・相談】" if lang == "ja" else "【User Question】"

        system_prompt = _build_system_prompt(lang, persona)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or []):
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"{ref_label}\n{context}\n\n{q_label}\n{user_input}"})

        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort="low",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
