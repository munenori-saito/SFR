"""
Soul Flight Recorder (SFR) — Streamlit UI
実行: streamlit run app.py
"""

from __future__ import annotations

import os
import uuid

import groq as groq_module
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud の Secrets にも対応（ローカルでは secrets.toml がなくてもOK）
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# ------------------------------------------------------------------ #
# DB 初期化（起動時1回）
# ------------------------------------------------------------------ #

from src.db import init_db
init_db()

# ------------------------------------------------------------------ #
# ページ設定
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Soul Flight Recorder",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛸 Soul Flight Recorder")
st.caption("自分の経験・感情を記録し、内側にある「本当にやりたいこと」を見つけるAI")

# ------------------------------------------------------------------ #
# セッション状態の初期化
# ------------------------------------------------------------------ #

if "store" not in st.session_state:
    st.session_state.store = None
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
# インタビュー用セッションID（ブラウザセッション中は固定）
if "interview_session_id" not in st.session_state:
    st.session_state.interview_session_id = str(uuid.uuid4())
# インタビュー履歴キャッシュ（DB再読み込みを減らすため）
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
# 手動フェーズ管理（0=ICEBREAK, 1=EXPLORE, 2=PATTERN, 3=FUTURE）
if "interview_phase_idx" not in st.session_state:
    st.session_state.interview_phase_idx = 0
# まとめキャッシュ
if "interview_summary" not in st.session_state:
    st.session_state.interview_summary = ""
# APIキー（BYOKモード）
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
# 言語変更検出
if "lang_prev" not in st.session_state:
    st.session_state.lang_prev = None


# ------------------------------------------------------------------ #
# サイドバー
# ------------------------------------------------------------------ #

with st.sidebar:
    # ── 言語スイッチ ──
    lang_display = st.radio(
        "Language / 言語",
        options=["日本語", "English"],
        horizontal=True,
        key="lang_display",
    )
    lang = "ja" if lang_display == "日本語" else "en"

    st.divider()

    # ── ペルソナ選択 ──
    from src.analyzer import PERSONA_OPTIONS
    persona_opts = PERSONA_OPTIONS[lang]
    persona_display = st.selectbox(
        "モード / Mode" if lang == "ja" else "Mode",
        options=list(persona_opts.values()),
        key="persona_display",
    )
    # display名 → キーに逆変換
    persona = next(k for k, v in persona_opts.items() if v == persona_display)

    st.divider()

    # ── API Key (BYOK) ──
    st.markdown("🔑 **Groq API Key**")
    api_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        help=(
            "console.groq.com で無料取得できます"
            if lang == "ja"
            else "Get your free key at console.groq.com"
        ),
    )
    api_key = api_key.strip()
    st.session_state.groq_api_key = api_key
    # 全角文字や途中の空白・改行が入るとHTTPヘッダに載せられず "Connection error." になる
    if api_key and (not api_key.isascii() or any(c.isspace() for c in api_key)):
        st.error(
            "APIキーに全角文字・空白・改行が含まれています。半角で貼り直してください。"
            if lang == "ja"
            else "API key contains spaces, line breaks, or non-ASCII characters. Please paste it again."
        )
        api_key = ""
    if not api_key:
        st.caption(
            "⚠️ APIキーを入力するとインタビューが開始されます"
            if lang == "ja"
            else "⚠️ Enter API key to start the interview"
        )

    st.divider()

    # ── トークンモード ──
    eco_options = ["通常", "節約 (Eco)"] if lang == "ja" else ["Normal", "Eco"]
    eco_selection = st.radio(
        "トークンモード" if lang == "ja" else "Token Mode",
        options=eco_options,
        horizontal=True,
        key="eco_mode_radio",
    )
    eco_mode = eco_selection in ("節約 (Eco)", "Eco")
    if eco_mode:
        st.caption(
            "直近3往復 ＋ 要約メモのみ送信（低消費）"
            if lang == "ja"
            else "Sends last 3 exchanges + summary memo (saves tokens)"
        )

    st.divider()

    # ── データ管理（既存機能） ──
    st.header("📁 データ管理" if lang == "ja" else "📁 Data")

    data_dir = st.text_input(
        "データディレクトリ" if lang == "ja" else "Data directory",
        value="data/raw",
    )
    force_rebuild = st.checkbox(
        "インデックスを再構築" if lang == "ja" else "Rebuild index",
        value=False,
    )

    if st.button(
        "🔄 インデックス構築 / 更新" if lang == "ja" else "🔄 Build / Update Index",
        use_container_width=True,
    ):
        if not api_key:
            st.error("Groq API Key を入力してください" if lang == "ja" else "Please enter your Groq API Key")
        else:
            with st.spinner("インデックスを構築中..." if lang == "ja" else "Building index..."):
                from src.vectorstore import build_index
                from src.analyzer import SFRAnalyzer

                st.session_state.store = build_index(
                    data_dir=data_dir, force=force_rebuild
                )
                st.session_state.analyzer = SFRAnalyzer(st.session_state.store)
                st.session_state.indexed = True
            st.success(
                f"完了！ {st.session_state.store.count()} チャンクをインデックス済み"
                if lang == "ja"
                else f"Done! {st.session_state.store.count()} chunks indexed"
            )

    if st.session_state.indexed:
        st.info(
            f"インデックス済み: {st.session_state.store.count()} チャンク"
            if lang == "ja"
            else f"Indexed: {st.session_state.store.count()} chunks"
        )

    st.divider()
    st.caption(
        "📝 `data/raw/` に .txt / .md ファイルを追加して\n「インデックス構築」を押すと記憶が増えます"
        if lang == "ja"
        else "📝 Add .txt / .md files to `data/raw/`\nthen press Build Index."
    )


# ── 言語変更でインタビューセッションをリセット ──
if st.session_state.lang_prev is not None and st.session_state.lang_prev != lang:
    st.session_state.interview_session_id = str(uuid.uuid4())
    st.session_state.interview_history = []
    st.session_state.interview_phase_idx = 0
    st.session_state.interview_summary = ""
st.session_state.lang_prev = lang


# ------------------------------------------------------------------ #
# エラーハンドリングヘルパー
# ------------------------------------------------------------------ #

def _api_error_warning(e: Exception, lang: str) -> None:
    if isinstance(e, groq_module.AuthenticationError):
        st.warning(
            "APIキーが無効です。正しいGroq APIキーを入力してください。"
            if lang == "ja"
            else "Invalid API key. Please enter a valid Groq API key."
        )
    elif isinstance(e, groq_module.RateLimitError):
        st.warning(
            "レート制限に達しました。しばらく待ってから再試行してください。"
            if lang == "ja"
            else "Rate limit reached. Please wait a moment and try again."
        )
    else:
        detail = f"{type(e).__name__}: {e}"
        if isinstance(e, groq_module.APIStatusError):
            body = e.body if isinstance(e.body, dict) else {}
            err = body.get("error", body) if isinstance(body.get("error", body), dict) else {}
            detail = f"HTTP {e.status_code} - {err.get('message') or e}"
        print(f"[SFR] API error: {e!r}")
        st.warning(
            f"APIエラーが発生しました: {detail}"
            if lang == "ja"
            else f"API error occurred: {detail}"
        )


# ------------------------------------------------------------------ #
# メインエリア：タブ構成
# ------------------------------------------------------------------ #

if lang == "ja":
    tab_labels = ["🎤 インタビュー", "🔥 強み分析", "📊 感情マップ", "🚀 未来ビジョン", "💬 対話コーチング"]
else:
    tab_labels = ["🎤 Interview", "🔥 Strengths", "📊 Emotion Map", "🚀 Future Vision", "💬 Coaching"]

tab_interview, tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

def _require_index():
    if not st.session_state.indexed:
        msg = (
            "サイドバーで **インデックス構築** を実行してください。\n\n"
            "`data/raw/` にテキストファイルを置き、API Key を入力後にボタンを押します。"
            if lang == "ja"
            else "Please run **Build Index** in the sidebar.\n\n"
            "Add text files to `data/raw/` and click the button after entering your API Key."
        )
        st.warning(msg)
        return False
    return True


# ------------------------------------------------------------------ #
# Tab 0: インタビュー
# ------------------------------------------------------------------ #

with tab_interview:
    from src.db import (
        create_session, save_message, get_messages,
        user_turn_count, clear_session_messages,
    )
    from src.interview import InterviewEngine, PHASE_LABELS, Phase

    if lang == "ja":
        st.subheader("🎤 AIインタビュー")
        st.write("質問に答えていくだけで、あなたの「本当にやりたいこと」が見えてきます。")
    else:
        st.subheader("🎤 AI Interview")
        st.write("Just answer the questions — your true passions and strengths will emerge naturally.")

    session_id = st.session_state.interview_session_id
    create_session(session_id, lang)

    # DB から履歴取得
    db_history = get_messages(session_id)
    st.session_state.interview_history = db_history

    # ── フェーズ（手動管理） ──
    phases_list = list(Phase)
    phase_idx = st.session_state.interview_phase_idx
    current_phase = phases_list[phase_idx]
    phase_labels = PHASE_LABELS[lang]

    # フェーズインジケーター + 「次のフェーズへ」ボタン
    indicator_cols = st.columns(len(phases_list) + 1)
    for i, (col, ph) in enumerate(zip(indicator_cols[:len(phases_list)], phases_list)):
        label = phase_labels[ph]
        if i < phase_idx:
            col.success(label)
        elif i == phase_idx:
            col.info(f"**{label}**")
        else:
            col.markdown(f"<span style='color:gray'>{label}</span>", unsafe_allow_html=True)

    with indicator_cols[-1]:
        next_label = "次のフェーズへ →" if lang == "ja" else "Next Phase →"
        next_disabled = phase_idx >= len(phases_list) - 1
        if st.button(next_label, disabled=next_disabled, key="btn_next_phase"):
            st.session_state.interview_phase_idx += 1
            st.rerun()

    st.divider()

    # ── レイアウト: チャット | まとめ ──
    chat_col, summary_col = st.columns([3, 2])

    with summary_col:
        summary_title = "📋 これまでのまとめ" if lang == "ja" else "📋 Summary So Far"
        st.markdown(f"**{summary_title}**")

        update_label = "まとめを更新" if lang == "ja" else "Update Summary"
        if st.button(update_label, key="btn_summary", use_container_width=True):
            if not api_key:
                st.error("API Key が必要です" if lang == "ja" else "API Key required")
            elif not st.session_state.interview_history:
                st.info("まだ会話がありません" if lang == "ja" else "No conversation yet")
            else:
                engine = InterviewEngine(api_key=api_key)
                summary_area = st.empty()
                full_summary = ""
                try:
                    with st.spinner("まとめを生成中..." if lang == "ja" else "Generating summary..."):
                        for chunk in engine.stream_summary(
                            history=[{"role": m["role"], "content": m["content"]}
                                     for m in st.session_state.interview_history],
                            lang=lang,
                        ):
                            full_summary += chunk
                            summary_area.markdown(full_summary + "▌")
                    summary_area.markdown(full_summary)
                    st.session_state.interview_summary = full_summary
                except Exception as e:
                    summary_area.empty()
                    _api_error_warning(e, lang)

        if st.session_state.interview_summary:
            st.markdown(st.session_state.interview_summary)
        else:
            placeholder_text = (
                "「まとめを更新」ボタンを押すと、\nここに会話のまとめが表示されます。"
                if lang == "ja"
                else "Press 'Update Summary' to see\na summary of your conversation here."
            )
            st.caption(placeholder_text)

        # プロファイルJSON表示
        from src.db import get_profile
        current_profile = get_profile(session_id)
        if current_profile and any(current_profile.get(k) for k in ["excitement_triggers", "strengths"]):
            profile_label = "🧬 自動生成プロファイル" if lang == "ja" else "🧬 Auto Profile"
            with st.expander(profile_label, expanded=False):
                if lang == "ja":
                    field_labels = {
                        "excitement_triggers": "ワクワクのトリガー",
                        "strengths": "強み",
                        "core_values": "価値観",
                        "avoidance_patterns": "モヤモヤのパターン",
                        "future_keywords": "将来のキーワード",
                        "confidence_score": "確信度",
                    }
                else:
                    field_labels = {
                        "excitement_triggers": "Excitement Triggers",
                        "strengths": "Strengths",
                        "core_values": "Core Values",
                        "avoidance_patterns": "Avoidance Patterns",
                        "future_keywords": "Future Keywords",
                        "confidence_score": "Confidence Score",
                    }
                for key, label in field_labels.items():
                    val = current_profile.get(key)
                    if key == "confidence_score":
                        st.progress(float(val or 0), text=f"{label}: {val}")
                    elif val:
                        st.markdown(f"**{label}**")
                        for item in val:
                            st.markdown(f"- {item}")

    with chat_col:
        # チャット履歴の表示
        for msg in st.session_state.interview_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 初回: AIから先に挨拶を出す
        if len(st.session_state.interview_history) == 0:
            if not api_key:
                st.info(
                    "サイドバーに Groq API Key を入力するとインタビューが始まります。"
                    if lang == "ja"
                    else "Enter your Groq API Key in the sidebar to start the interview."
                )
            else:
                engine = InterviewEngine(api_key=api_key)
                try:
                    with st.chat_message("assistant"):
                        response_area = st.empty()
                        full_response = ""
                        for chunk in engine.stream_response(history=[], lang=lang, phase=current_phase):
                            full_response += chunk
                            response_area.markdown(full_response + "▌")
                        response_area.markdown(full_response)
                    save_message(session_id, "assistant", full_response, current_phase.value)
                    st.rerun()
                except Exception as e:
                    _api_error_warning(e, lang)

        # ユーザー入力
        placeholder = "ここに答えを入力してください..." if lang == "ja" else "Type your answer here..."
        if user_input := st.chat_input(placeholder):
            if not api_key:
                st.error("Groq API Key を入力してください" if lang == "ja" else "Please enter your Groq API Key")
            else:
                save_message(session_id, "user", user_input, current_phase.value)
                with st.chat_message("user"):
                    st.markdown(user_input)

                updated_history = get_messages(session_id)
                current_user_turns = user_turn_count(session_id)

                # ── バックグラウンドプロファイル更新（5ターンごと）──
                from src.profile_builder import ProfileBuilder, should_update
                from src.db import get_profile, update_profile
                if should_update(current_user_turns):
                    profile_builder = ProfileBuilder(api_key=api_key)
                    existing = get_profile(session_id)
                    new_profile = profile_builder.update(
                        history=[{"role": m["role"], "content": m["content"]} for m in updated_history],
                        existing_profile=existing,
                        lang=lang,
                    )
                    update_profile(session_id, new_profile)
                    current_profile = new_profile
                else:
                    from src.db import get_profile
                    current_profile = get_profile(session_id)

                # ── Eco モード: 送信履歴を削減 ──
                raw_history = [{"role": m["role"], "content": m["content"]} for m in updated_history]
                if eco_mode and len(raw_history) > 6:
                    send_history = raw_history[-6:]
                    context_memo = st.session_state.interview_summary or None
                else:
                    send_history = raw_history
                    context_memo = None

                engine = InterviewEngine(api_key=api_key)
                full_response = ""
                try:
                    with st.chat_message("assistant"):
                        response_area = st.empty()
                        for chunk in engine.stream_response(
                            history=send_history,
                            lang=lang,
                            phase=current_phase,
                            profile=current_profile if current_profile else None,
                            context_memo=context_memo,
                        ):
                            full_response += chunk
                            response_area.markdown(full_response + "▌")
                        response_area.markdown(full_response)
                except Exception as e:
                    _api_error_warning(e, lang)

                if full_response:
                    save_message(session_id, "assistant", full_response, current_phase.value)
                    st.rerun()

        # リセットボタン
        if st.session_state.interview_history:
            st.divider()
            reset_label = "🗑️ インタビューをリセット" if lang == "ja" else "🗑️ Reset Interview"
            if st.button(reset_label, key="btn_interview_reset"):
                clear_session_messages(session_id)
                st.session_state.interview_history = []
                st.session_state.interview_phase_idx = 0
                st.session_state.interview_summary = ""
                st.rerun()


# ------------------------------------------------------------------ #
# Tab 1〜4 共通: プロファイル取得 + アナライザー準備
# ------------------------------------------------------------------ #

from src.db import get_profile as _get_profile
from src.analyzer import SFRAnalyzer

_session_profile = _get_profile(st.session_state.interview_session_id)

# インデックスがなくてもプロファイルがあれば動作可能
def _get_analyzer() -> SFRAnalyzer:
    if st.session_state.indexed and st.session_state.analyzer:
        # 既存analyzerにapi_keyを補完
        st.session_state.analyzer._api_key = api_key
        st.session_state.analyzer._client = None
        return st.session_state.analyzer
    return SFRAnalyzer(store=None, api_key=api_key)

def _has_data() -> bool:
    """RAGインデックスまたはプロファイルのどちらかがあれば True"""
    has_index = st.session_state.indexed
    has_profile = bool(_session_profile and any(
        _session_profile.get(k) for k in ["strengths", "excitement_triggers"]
    ))
    return has_index or has_profile

def _require_data():
    if not _has_data():
        if lang == "ja":
            st.warning(
                "まず **🎤 インタビュー** タブで会話するか、\n"
                "サイドバーで **インデックス構築** を実行してください。"
            )
        else:
            st.warning(
                "Please complete some **🎤 Interview** conversation first,\n"
                "or run **Build Index** in the sidebar."
            )
        return False
    return True


# ------------------------------------------------------------------ #
# Tab 1: 強み・関心の抽出
# ------------------------------------------------------------------ #

with tab1:
    if lang == "ja":
        st.subheader("🔥 強み・関心の抽出")
        st.write("インタビューの記録とテキストファイルを統合して、強みとワクワクのパターンを抽出します。")
        btn_label, spinner_label = "分析スタート", "分析中..."
    else:
        st.subheader("🔥 Strengths & Interests")
        st.write("Combines interview profile and text records to extract strengths and excitement patterns.")
        btn_label, spinner_label = "Start Analysis", "Analyzing..."

    if st.button(btn_label, key="btn_strength", use_container_width=True):
        if _require_data():
            analyzer = _get_analyzer()
            result_area = st.empty()
            full_text = ""
            try:
                with st.spinner(spinner_label):
                    for chunk in analyzer.extract_strengths_stream(
                        profile=_session_profile, lang=lang, persona=persona
                    ):
                        full_text += chunk
                        result_area.markdown(full_text + "▌")
                result_area.markdown(full_text)
            except Exception as e:
                result_area.empty()
                _api_error_warning(e, lang)


# ------------------------------------------------------------------ #
# Tab 2: 感情マップ
# ------------------------------------------------------------------ #

with tab2:
    if lang == "ja":
        st.subheader("📊 感情マッピング")
        st.write("各活動での没頭度・ワクワク度・達成感・モヤモヤ度を分析します。")
        btn_label, spinner_label = "感情マップを生成", "感情パターンを解析中..."
    else:
        st.subheader("📊 Emotion Map")
        st.write("Analyzes immersion, excitement, fulfillment, and discomfort across activities.")
        btn_label, spinner_label = "Generate Emotion Map", "Analyzing emotion patterns..."

    if st.button(btn_label, key="btn_emotion", use_container_width=True):
        if _require_data():
            analyzer = _get_analyzer()
            result_area = st.empty()
            full_text = ""
            try:
                with st.spinner(spinner_label):
                    for chunk in analyzer.analyze_emotion_timeline_stream(
                        profile=_session_profile, lang=lang, persona=persona
                    ):
                        full_text += chunk
                        result_area.markdown(full_text + "▌")
                result_area.markdown(full_text)
            except Exception as e:
                result_area.empty()
                _api_error_warning(e, lang)


# ------------------------------------------------------------------ #
# Tab 3: 未来ビジョン
# ------------------------------------------------------------------ #

with tab3:
    if lang == "ja":
        st.subheader("🚀 等身大の未来予想図")
        st.write("強みと関心を統合した「5年後のビジョン」と「最初の一歩」を描きます。")
        btn_label, spinner_label = "ビジョンを生成", "未来像を描いています..."
    else:
        st.subheader("🚀 Future Vision")
        st.write("Paints a realistic 5-year vision and concrete first steps based on your strengths and interests.")
        btn_label, spinner_label = "Generate Vision", "Painting your future..."

    if st.button(btn_label, key="btn_vision", use_container_width=True):
        if _require_data():
            analyzer = _get_analyzer()
            result_area = st.empty()
            full_text = ""
            try:
                with st.spinner(spinner_label):
                    for chunk in analyzer.generate_future_vision_stream(
                        profile=_session_profile, lang=lang, persona=persona
                    ):
                        full_text += chunk
                        result_area.markdown(full_text + "▌")
                result_area.markdown(full_text)
            except Exception as e:
                result_area.empty()
                _api_error_warning(e, lang)


# ------------------------------------------------------------------ #
# Tab 4: 対話コーチング
# ------------------------------------------------------------------ #

with tab4:
    if lang == "ja":
        st.subheader("💬 自己理解コーチング")
        st.write("インタビュー記録とテキストファイルを参照しながら、あなたの問いに答えます。")
        chat_placeholder = "何でも聞いてみてください..."
        reset_label = "🗑️ 会話をリセット"
    else:
        st.subheader("💬 Coaching Chat")
        st.write("Answers your questions using interview records and text files as reference.")
        chat_placeholder = "Ask me anything..."
        reset_label = "🗑️ Reset Chat"

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(chat_placeholder):
        if not api_key:
            st.error("Groq API Key を入力してください" if lang == "ja" else "Please enter your Groq API Key")
        elif not _has_data():
            _require_data()
        else:
            with st.chat_message("user"):
                st.markdown(prompt)

            analyzer = _get_analyzer()
            full_response = ""
            try:
                with st.chat_message("assistant"):
                    response_area = st.empty()
                    for chunk in analyzer.chat_stream(
                        user_input=prompt,
                        history=[{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat_history],
                        profile=_session_profile,
                        lang=lang,
                        persona=persona,
                    ):
                        full_response += chunk
                        response_area.markdown(full_response + "▌")
                    response_area.markdown(full_response)
            except Exception as e:
                _api_error_warning(e, lang)

            if full_response:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.session_state.chat_history:
        if st.button(reset_label, key="btn_reset"):
            st.session_state.chat_history = []
            st.rerun()
