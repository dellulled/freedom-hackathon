"""
Freedom Insight Engine — Groq + Streamlit
Читает реальные CSV, считает живую статистику, передаёт в Groq LLM.

Положи рядом со скриптом:
  - posts_with_journey.csv
  - comments_with_journey.csv
  - master_features.csv

Запуск:
  $env:GROQ_API_KEY = "gsk_..."
  streamlit run insight_engine.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Freedom Insight Engine",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 820px; }

.hero-title { font-size: 2rem; font-weight: 800; color: #1A2340; text-align: center; margin-bottom: 6px; }
.hero-sub   { font-size: 0.93rem; color: #7A8BA6; text-align: center; margin-bottom: 2rem; line-height: 1.6; }

.cards-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1.5rem; }
.stat-card { background: #fff; border: 1px solid #E8ECF2; border-radius: 14px; padding: 16px 18px; }
.stat-card h4 { font-size: 0.74rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.07em; color: #7A8BA6; margin-bottom: 10px; }
.stat-row { display: flex; justify-content: space-between; align-items: center;
            font-size: 0.84rem; padding: 6px 0; border-bottom: 1px solid #F5F7FA; }
.stat-row:last-child { border-bottom: none; }
.sl  { color: #7A8BA6; }
.sv  { font-weight: 600; color: #1A2340; }
.pos { color: #10B981; font-weight: 700; }
.neg { color: #EF4444; font-weight: 700; }

div[data-testid="stButton"] > button {
    border-radius: 999px !important; font-weight: 600 !important;
    font-size: 0.84rem !important; padding: 8px 16px !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #1A2340 !important; color: white !important;
    border: 2px solid #1A2340 !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #F0F4FA !important; color: #7A8BA6 !important;
    border: 2px solid #E8ECF2 !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #E2E8F4 !important; color: #1A2340 !important;
}
.run-wrap div[data-testid="stButton"] > button {
    border-radius: 12px !important; font-size: 1rem !important;
    padding: 14px 32px !important; width: 100% !important;
    background: #1A2340 !important; color: white !important;
}
.divider { border-top: 1px solid #E8ECF2; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(show_spinner=False)
def load_data():
    posts    = pd.read_csv(os.path.join(BASE, "posts_with_journey.csv"),    low_memory=False)
    comments = pd.read_csv(os.path.join(BASE, "comments_with_journey.csv"), low_memory=False)
    master   = pd.read_csv(os.path.join(BASE, "master_features.csv"),       low_memory=False)

    # Fix types
    for df in [posts, comments]:
        df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0)
    master["txn_total_spend"] = pd.to_numeric(master["txn_total_spend"], errors="coerce")
    master["txn_count"]       = pd.to_numeric(master["txn_count"],       errors="coerce")
    master["txn_recency_days"]= pd.to_numeric(master["txn_recency_days"], errors="coerce")
    master["pp_unique_apps"]  = pd.to_numeric(master["pp_unique_apps"],   errors="coerce")
    master["engagement_score"]= pd.to_numeric(master["engagement_score"], errors="coerce")

    return posts, comments, master


@st.cache_data(show_spinner=False)
def compute_stats(_posts, _comments, _master):
    posts, comments, master = _posts, _comments, _master

    # ── Freedom Score + segments ─────────────────────────────────────
    def minmax(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    rec      = 1 - minmax(master["txn_recency_days"].fillna(master["txn_recency_days"].max() + 1))
    freq     = minmax(master["txn_count"].fillna(0))
    mon      = minmax(master["txn_total_spend"].fillna(0))
    eng      = master["engagement_score"].fillna(0) / 3.0
    prod_b   = minmax(master["pp_unique_apps"].fillna(0) + master["txn_unique_mcc"].fillna(0) if "txn_unique_mcc" in master.columns else master["pp_unique_apps"].fillna(0))
    master   = master.copy()
    master["freedom_score"] = (0.20*rec + 0.30*freq + 0.30*mon + 0.10*eng + 0.10*prod_b) * 100

    p33 = master["freedom_score"].quantile(0.33)
    p66 = master["freedom_score"].quantile(0.66)
    master["segment"] = pd.cut(
        master["freedom_score"],
        bins=[-0.001, p33, p66, 101],
        labels=["Low", "Medium", "High"]
    )

    seg_stats = (
        master.groupby("segment", observed=True)
        .agg(
            n               =("customer_id",      "count"),
            avg_score       =("freedom_score",    "mean"),
            avg_spend       =("txn_total_spend",  "mean"),
            avg_txn         =("txn_count",        "mean"),
            avg_recency     =("txn_recency_days", "mean"),
            avg_pp_apps     =("pp_unique_apps",   "mean"),
            avg_engagement  =("engagement_score", "mean"),
        )
        .round(1)
    )

    # ── Social NPS per product ────────────────────────────────────────
    social = pd.concat([
        posts[["product", "sentiment", "engagement"]],
        comments[["product", "sentiment", "engagement"]],
    ], ignore_index=True).dropna(subset=["product"])

    def nps(grp):
        total = len(grp)
        if total == 0: return 0.0
        pos = (grp["sentiment"] == "Positive").sum()
        neg = (grp["sentiment"] == "Negative").sum()
        return round((pos - neg) / total * 100, 1)

    nps_series   = social.groupby("product").apply(nps).sort_values(ascending=False)
    total_counts = social.groupby("product").size()

    # ── Complaints ───────────────────────────────────────────────────
    complaints = (
        posts[posts["sentiment"] == "Negative"]
        .groupby("product").size()
        .sort_values(ascending=False)
    )

    # ── Advocacy ────────────────────────────────────────────────────
    advocacy = (
        posts[posts["customer_journey"] == "Advocacy"]
        .groupby("product").size()
        .sort_values(ascending=False)
    )

    # ── Journey breakdown ────────────────────────────────────────────
    journey_tbl = (
        posts.dropna(subset=["product", "customer_journey"])
        .groupby(["product", "customer_journey"]).size()
        .unstack(fill_value=0)
    )

    # ── Top quoted posts (real text for LLM) ─────────────────────────
    top_posts = posts.nlargest(10, "engagement")[["product", "sentiment", "engagement", "text"]]

    # ── Top complaints per product (sample texts) ────────────────────
    complaint_quotes = {}
    for prod in complaints.head(5).index:
        rows = (
            posts[(posts["product"] == prod) & (posts["sentiment"] == "Negative")]
            .nlargest(3, "engagement")[["text", "engagement"]]
        )
        complaint_quotes[prod] = [
            {"text": r["text"][:200], "eng": int(r["engagement"])}
            for _, r in rows.iterrows()
        ]

    # ── Top advocacy quotes ───────────────────────────────────────────
    advocacy_quotes = (
        posts[posts["customer_journey"] == "Advocacy"]
        .nlargest(5, "engagement")[["product", "text", "engagement"]]
    )

    return {
        "master":           master,
        "seg_stats":        seg_stats,
        "nps":              nps_series,
        "total_counts":     total_counts,
        "complaints":       complaints,
        "advocacy":         advocacy,
        "journey_tbl":      journey_tbl,
        "top_posts":        top_posts,
        "complaint_quotes": complaint_quotes,
        "advocacy_quotes":  advocacy_quotes,
        "n_posts":          len(posts),
        "n_comments":       len(comments),
        "n_users":          len(master),
    }


def build_llm_context(s: dict, mode_extra: str = "") -> str:
    """Строит реальный контекст из живых данных для Groq."""

    seg = s["seg_stats"]
    lines = [
        f"=== ТРАНЗАКЦИОННЫЕ ДАННЫЕ (Freedom Score ML, {s['n_users']:,} пользователей) ===\n",
        "Сегменты (по перцентилям 33/66 Freedom Score 0–100):",
    ]
    for seg_name in ["High", "Medium", "Low"]:
        if seg_name in seg.index:
            r = seg.loc[seg_name]
            spend = f"{r['avg_spend']:,.0f} KZT" if pd.notna(r["avg_spend"]) else "нет данных"
            txn   = f"{r['avg_txn']:.1f}/мес"    if pd.notna(r["avg_txn"])   else "нет данных"
            rec   = f"{r['avg_recency']:.0f} дней" if pd.notna(r["avg_recency"]) else "нет данных"
            lines.append(
                f"• {seg_name} ({r['n']:,} чел, {r['n']/s['n_users']*100:.0f}%): "
                f"avg_score={r['avg_score']:.1f}, avg_spend={spend}, "
                f"avg_txn={txn}, avg_recency={rec}, avg_engagement={r['avg_engagement']:.1f}/3"
            )

    lines += [
        f"\n=== СОЦИАЛЬНЫЕ ДАННЫЕ (Threads, {s['n_posts']:,} постов + {s['n_comments']:,} комментариев) ===\n",
        "Social NPS по продуктам (% позитив − % негатив):",
    ]
    for prod, val in s["nps"].items():
        cnt = s["total_counts"].get(prod, 0)
        sign = "+" if val >= 0 else ""
        lines.append(f"  {prod}: {sign}{val}  ({cnt} упоминаний)")

    lines += ["\nКоличество жалоб (негативные посты):"]
    for prod, cnt in s["complaints"].head(8).items():
        lines.append(f"  {prod}: {cnt}")

    lines += ["\nAdvocacy (посты с journey=Advocacy):"]
    for prod, cnt in s["advocacy"].head(6).items():
        lines.append(f"  {prod}: {cnt}")

    lines += ["\nРеальные цитаты из жалоб (топ по вовлечённости):"]
    for prod, quotes in s["complaint_quotes"].items():
        lines.append(f"\n  [{prod}]")
        for q in quotes:
            lines.append(f'    engagement={q["eng"]}: "{q["text"]}"')

    lines += ["\nРеальные цитаты адвокатов бренда:"]
    for _, row in s["advocacy_quotes"].iterrows():
        lines.append(f'  [{row["product"]}, eng={int(row["engagement"])}]: "{str(row["text"])[:200]}"')

    if mode_extra:
        lines += [f"\n=== ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ===\n{mode_extra}"]

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "Ты старший аналитик данных финтех-суперприложения Freedom (Казахстан). "
    "Тебе дают два реальных источника данных:\n"
    "1. Транзакционные данные — ML-сегментация 899K пользователей (High/Medium/Low по Freedom Score).\n"
    "2. Социальные данные — реальные посты и комментарии из Threads с NPS, жалобами, цитатами.\n\n"
    "Прямого customer_id в Threads нет — связь концептуальная через продукты и поведение.\n\n"
    "Правила:\n"
    "- Отвечай ТОЛЬКО на русском\n"
    "- Нумерованный список: 4–5 инсайтов\n"
    "- Каждый инсайт: **жирный заголовок** → объяснение с реальными числами из данных → 💡 рекомендация\n"
    "- Используй реальные цитаты из постов где уместно\n"
    "- Actionable выводы для бизнеса"
)

MODES = {
    "⚠️ Риски оттока":   "Проанализируй РИСКИ ОТТОКА используя реальные данные. Какие продукты имеют высокие жалобы и низкий NPS? Какой сегмент (High/Medium/Low) под наибольшим риском? Используй реальные цитаты для иллюстрации проблем. Дай 4–5 инсайтов с числами и рекомендациями.",
    "📈 Точки роста":     "Найди ТОЧКИ РОСТА в реальных данных. Где разрыв между высоким NPS и потенциально низкой транзакционной активностью? Какие продукты могут конвертировать Medium → High сегмент? Дай 4–5 инсайтов.",
    "🗂️ По продуктам":   "Построй МАТРИЦУ ПРОДУКТОВ по реальным данным: транзакционная ценность × социальная репутация. Кто в каком квадранте (Звезда/Потенциал/Риск/Кризис)? Дай 4–5 инсайтов с числами.",
    "❤️ Адвокаты":        "Проанализируй АДВОКАТОВ БРЕНДА по реальным данным. Кто органически защищает бренд в Threads? Используй реальные цитаты. Как бизнес может усилить эффект? Дай 4–5 инсайтов.",
    "🔁 Retention":       "Сфокусируйся на RETENTION. Low-сегмент (неактивные пользователи) — что говорит Threads об их причинах ухода? Реальные жалобы как сигнал оттока. Как Freedom Score помогает предсказать отток? Дай 4–5 инсайтов.",
    "✍️ Свой вопрос":     None,
}


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "⚠️ Риски оттока"


# ─────────────────────────────────────────────
# CLUSTER CHART (live data)
# ─────────────────────────────────────────────
def cluster_chart(nps_series, complaints):
    products = []
    for prod, nps_val in nps_series.items():
        if prod in ("Other", "Event", None) or str(prod) == "nan":
            continue
        c = complaints.get(prod, 0)
        if nps_val >= 10:           cluster = "Звезда"
        elif nps_val >= 0:          cluster = "Потенциал"
        elif nps_val >= -20:        cluster = "Риск"
        else:                       cluster = "Кризис"
        products.append({"Продукт": prod, "NPS": nps_val, "Жалобы": c, "Кластер": cluster})

    df = pd.DataFrame(products)
    colors = {"Звезда": "#10B981", "Потенциал": "#F59E0B", "Риск": "#F97316", "Кризис": "#EF4444"}

    fig = px.scatter(
        df, x="NPS", y="Жалобы", text="Продукт",
        color="Кластер", color_discrete_map=colors,
        labels={"NPS": "Social NPS (Threads) →", "Жалобы": "Количество жалоб →"},
    )
    fig.update_traces(
        marker=dict(size=18, opacity=0.9, line=dict(width=1.5, color="white")),
        textposition="top center",
        textfont=dict(size=11, family="Plus Jakarta Sans"),
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#CBD5E1", line_width=1.5)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font_family="Plus Jakarta Sans",
        margin=dict(t=10, b=50, l=40, r=20), height=340,
        legend=dict(title="Кластер", orientation="h", y=-0.3),
        xaxis=dict(showgrid=True, gridcolor="#F5F7FA", zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#F5F7FA", zeroline=False, tickfont=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Freedom Insight Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">ИИ читает реальные данные: транзакции 899K пользователей '
    'и посты из Threads — и выдаёт живые бизнес-инсайты</div>',
    unsafe_allow_html=True,
)

# Load
missing = []
for fname in ["posts_with_journey.csv", "comments_with_journey.csv", "master_features.csv"]:
    if not os.path.exists(os.path.join(BASE, fname)):
        missing.append(fname)

if missing:
    st.error(f"Не найдены файлы: {', '.join(missing)}\n\nПоложи их рядом со скриптом `insight_engine.py`")
    st.stop()

with st.spinner("Загружаю данные..."):
    posts, comments, master = load_data()

with st.spinner("Считаю статистику..."):
    s = compute_stats(posts, comments, master)

# Stat cards (live numbers)
seg = s["seg_stats"]
high   = seg.loc["High"]   if "High"   in seg.index else None
medium = seg.loc["Medium"] if "Medium" in seg.index else None
low    = seg.loc["Low"]    if "Low"    in seg.index else None

top_nps_prod = s["nps"].head(2)
bot_nps_prod = s["nps"].tail(2)
top_complaint = list(s["complaints"].head(1).items())[0]

st.markdown(f"""
<div class="cards-row">
  <div class="stat-card">
    <h4>📊 Сегменты · {s['n_users']:,} пользователей</h4>
    <div class="stat-row">
      <span class="sl">🟢 High ({high['n']:,})</span>
      <span class="sv">₸{high['avg_spend']:,.0f} · {high['avg_txn']:.0f} транз/мес</span>
    </div>
    <div class="stat-row">
      <span class="sl">🟡 Medium ({medium['n']:,})</span>
      <span class="sv">₸{medium['avg_spend']:,.0f} · {medium['avg_txn']:.0f} транз/мес</span>
    </div>
    <div class="stat-row">
      <span class="sl">🔴 Low ({low['n']:,})</span>
      <span class="sv">Неактивные пользователи</span>
    </div>
  </div>
  <div class="stat-card">
    <h4>💬 Social NPS · {s['n_posts']:,} постов</h4>
    {''.join(
        f'<div class="stat-row"><span class="sl">{p}</span>'
        f'<span class="pos">{v:+.1f}</span></div>'
        for p, v in top_nps_prod.items()
    )}
    {''.join(
        f'<div class="stat-row"><span class="sl">{p}</span>'
        f'<span class="neg">{v:+.1f} · {s["complaints"].get(p, 0)} жалоб</span></div>'
        for p, v in bot_nps_prod.items()
    )}
  </div>
</div>
""", unsafe_allow_html=True)

# Chart
st.markdown("**Карта продуктов: репутация vs жалобы**")
st.caption("Правый нижний угол — идеал (высокий NPS, мало жалоб).")
st.plotly_chart(cluster_chart(s["nps"], s["complaints"]), use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Mode buttons
st.markdown("**Выбери тип анализа:**")
cols = st.columns(len(MODES))
for i, mode_name in enumerate(MODES.keys()):
    with cols[i]:
        is_active = st.session_state.selected_mode == mode_name
        if st.button(
            mode_name, key=f"mode_{i}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_mode = mode_name
            st.rerun()

custom_q = ""
if st.session_state.selected_mode == "✍️ Свой вопрос":
    st.markdown("<br>", unsafe_allow_html=True)
    custom_q = st.text_area(
        "Задай любой вопрос об этих данных:",
        placeholder="Например: почему Freedom Bank имеет и больший advocacy и больше жалоб одновременно?",
        height=90,
    )

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🚀 Сгенерировать инсайты", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────
if run:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "Groq API Key не найден.\n\n"
            'Запусти в PowerShell: `$env:GROQ_API_KEY = "gsk_..."`\n\n'
            "Затем снова: `streamlit run insight_engine.py`"
        )
        st.stop()

    user_prompt = MODES[st.session_state.selected_mode]
    if st.session_state.selected_mode == "✍️ Свой вопрос":
        if not custom_q.strip():
            st.warning("Введи свой вопрос выше.")
            st.stop()
        user_prompt = custom_q.strip()

    # Build context from REAL data
    llm_context = build_llm_context(s)
    full_prompt  = llm_context + "\n\n=== ЗАДАЧА ===\n" + user_prompt

    client = Groq(api_key=api_key)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**💡 Инсайты на основе реальных данных:**")
    output    = st.empty()
    collected = ""

    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": full_prompt},
            ],
            temperature=0.3,
            max_tokens=1400,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            collected += delta
            output.markdown(collected + "▌")
        output.markdown(collected)

    except Exception as e:
        st.error(f"Ошибка Groq API: {e}")