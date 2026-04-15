import streamlit as st
import requests
import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

# Plotly（可选依赖）：未安装时自动降级到 matplotlib，避免应用直接崩溃
try:
    import plotly.graph_objects as go  # type: ignore
    _PLOTLY_AVAILABLE = True
except Exception:
    go = None  # type: ignore
    _PLOTLY_AVAILABLE = False

import os # <-- 确保引入了 os 模块
from zoneinfo import ZoneInfo

from openai import OpenAI


# --- 配置区 (安全修改版) ---
# 下面这些变量全部通过 Streamlit 的加密环境变量读取，绝证明文写在代码里
APP_ID = st.secrets["APP_ID"]
APP_SECRET = st.secrets["APP_SECRET"]
APP_TOKEN = st.secrets["APP_TOKEN"]
TABLE_ID = st.secrets["TABLE_ID"]
INSIGHT_TABLE_ID = st.secrets["INSIGHT_TABLE_ID"]
MATERIAL_TABLE_ID = st.secrets["MATERIAL_TABLE_ID"]
AI_KNOWLEDGE_TABLE_ID = st.secrets["AI_KNOWLEDGE_TABLE_ID"]
TIMEZONE = "Asia/Shanghai"

# DeepSeek 密钥也通过环境变量读取
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_API_KEY,
)

DEFAULT_AI_SYSTEM_PROMPT = """# Role: 深度个人成长教练与认知心理学专家
# Profile:
擅长通过精神分析、CBT认知行为疗法和ORID模型，对无序的文本进行深度解构。
拥有极强的逻辑归纳能力，能够从繁杂的日常琐事中精准捕捉用户的底层价值观、情绪锚点和行为模式。
# Task:
接收我提供的“日常碎碎念”（无逻辑、碎片化的想法/抱怨/记录），将其结构化，并输出深度的自我洞察与成长轨迹报告。
""".strip()

def deepseek_chat(system_prompt: str, user_prompt: str, model: str = "deepseek-chat") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": (user_prompt or "").strip()},
        ],
    )
    return ((resp.choices[0].message.content or "").strip()) or ""


def _prepare_df_with_local_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "记录日期" not in df.columns:
        return df
    out = df.copy()
    # 飞书时间戳通常为 Unix epoch ms；这里先按 UTC 解析，再转换到配置时区，确保按“本地日期”过滤一致
    out["时间"] = pd.to_datetime(out["记录日期"], unit="ms", utc=True).dt.tz_convert(ZoneInfo(TIMEZONE))
    return out


def generate_daily_review(daily_logs, system_prompt: str):
    """
    聚合分析模式：接收某天全部日志文本，交给 DeepSeek 做深度复盘。
    daily_logs: list[str] | pd.Series | str
    return: str (markdown)
    """
    if daily_logs is None:
        return "今天没有可用的日志内容，暂时无法复盘。"

    if isinstance(daily_logs, str):
        logs_text = daily_logs.strip()
    else:
        try:
            logs_text = "\n\n".join([str(x).strip() for x in list(daily_logs) if str(x).strip()])
        except Exception:
            logs_text = str(daily_logs).strip()

    if not logs_text:
        return "今天没有可用的日志内容，暂时无法复盘。"

    if not DEEPSEEK_API_KEY:
        return (
            "未检测到 `DEEPSEEK_API_KEY`。\n\n"
            "- 请在脚本顶部“核心配置区”直接填写 `DEEPSEEK_API_KEY = \"...\"`。\n"
            "- 填好后刷新页面再试。"
        )

    sys_prompt = (system_prompt or DEFAULT_AI_SYSTEM_PROMPT).strip()
    user_prompt = f"""
下面是用户所选日期的全部日志内容（可能包含情绪、事件、碎片想法与待办）。
请严格按照以下四个维度对我提供的文本进行解构和分析。

事实与情绪重塑：剥离主观情绪，还原客观事实，并识别隐藏的复合情绪。
认知与信念洞察：挖掘导致这些情绪和想法的“底层逻辑”（例如：完美主义、讨好型人格、特定恐惧、核心价值观等）。
能量与模式评估：分析这件事对我的“心理能量”是消耗还是滋养，总结我近期的注意力分布模式。
觉醒与微小行动：提供直接落地的微小行动建议（Micro-habits），将日常琐事转化为成长经验值。

# Constraint (约束条件):
保持客观、克制但充满同理心。
语言风格凝练、直击痛点，像一面清晰的镜子。
拒绝空洞的安慰，提供锐利且有建设性的剖析。

# Output Format (输出格式):
请以Markdown格式输出，严格使用以下结构展示分析结果：
🔬 原始切片还原
客观事实：用绝对客观、不带评判的一句话总结发生了什么。
情绪标签：精准提取3个描述当前状态的核心情绪词汇。
👁️ 深度自我洞察
认知偏差：指出我在碎碎念中表现出的思维局限（如非黑即白、灾难化思维等）。
核心需求：剖析我潜意识里真正在渴求什么（如被认可、安全感、掌控感、边界感）。
价值闪光点：即使是消极的碎碎念，也要挖掘出我积极的特质（如：抱怨说明我还在乎标准）。

🌱 科学成长轨迹
能量评估：判断该事件属于“高耗能”还是“高充能”，并简述原因。
模式追踪：评估这是否属于我反复出现的行为/情绪模式，如果是，其触发机制是什么。

🚀 觉醒与行动指南
认知翻转：用一句通透且有哲理的话，帮我打破当下的思维局限。
微小行动：提供一个5分钟内就可以立即执行的极简行动建议，帮助我夺回生活掌控感。

输出要求：
- 使用中文
- 使用 Markdown 排版
- 允许使用小标题、项目符号

【日志全文开始】
{logs_text}
【日志全文结束】
""".strip()

    content = deepseek_chat(sys_prompt, user_prompt, model="deepseek-chat")
    return content or "AI 未返回有效内容，请稍后重试。"


# ==========================================
# 2. 飞书 API 交互逻辑
# ==========================================
def get_tenant_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    res = requests.post(url, json={"app_id": APP_ID, "app_secret": APP_SECRET})
    return res.json().get("tenant_access_token")


def add_record(timestamp_ms, content, energy):
    token = get_tenant_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{TABLE_ID}/records"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # 严格匹配你的飞书列名
    payload = {
        "fields": {
            "记录日期": timestamp_ms,
            "日记内容": content,
            "能量分值": int(energy)
        }
    }
    return requests.post(url, headers=headers, json=payload).json()


def fetch_data():
    token = get_tenant_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{TABLE_ID}/records"
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(url, headers=headers).json()
    items = [r['fields'] for r in res.get('data', {}).get('items', [])]
    return pd.DataFrame(items)


def fetch_material_data(page_size: int = 200, max_pages: int = 10) -> pd.DataFrame:
    """
    拉取《原始素材库》数据，返回 DataFrame（含 record_id 与 fields）。
    """
    items = _fetch_records(APP_TOKEN, MATERIAL_TABLE_ID, page_size=page_size, max_pages=max_pages)
    rows = []
    for it in items or []:
        fields = it.get("fields", {}) or {}
        fields["_record_id"] = it.get("record_id")
        rows.append(fields)
    return pd.DataFrame(rows)


def fetch_ai_knowledge_data(page_size: int = 200, max_pages: int = 10) -> pd.DataFrame:
    """
    拉取《AI知识库》数据，返回 DataFrame（含 record_id 与 fields）。
    """
    items = _fetch_records(APP_TOKEN, AI_KNOWLEDGE_TABLE_ID, page_size=page_size, max_pages=max_pages)
    rows = []
    for it in items or []:
        fields = it.get("fields", {}) or {}
        fields["_record_id"] = it.get("record_id")
        rows.append(fields)
    return pd.DataFrame(rows)


def _fetch_records(app_token: str, table_id: str, page_size: int = 200, max_pages: int = 3):
    token = get_tenant_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
    headers = {"Authorization": f"Bearer {token}"}
    items_all = []
    page_token = None
    for _ in range(max_pages):
        params = {"page_size": page_size}
        if page_token:
            params["page_token"] = page_token
        res = requests.get(url, headers=headers, params=params).json()
        items = res.get("data", {}).get("items", []) or []
        items_all.extend(items)
        page_token = res.get("data", {}).get("page_token")
        if not page_token:
            break
    return items_all


def add_material_record(entry_date, desc: str, raw_content: str, related_status: str = "待关联"):
    token = get_tenant_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{MATERIAL_TABLE_ID}/records"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    date_ms = _date_to_ms_in_tz(entry_date)

    payload = {
        "fields": {
            "录入时间": date_ms,
            "描述": (desc or "").strip(),
            "内容": (raw_content or "").strip(),
            "关联状态": related_status,
        }
    }
    return requests.post(url, headers=headers, json=payload).json()


@st.cache_data(ttl=300, show_spinner=False)
def get_material_list(limit: int = 20):
    """
    从《原始素材库》拉取最近的若干条记录（用于知识库的单向关联下拉）。
    返回：list[dict]，包含 record_id、label、date_str
    """
    if not MATERIAL_TABLE_ID or "请替换" in MATERIAL_TABLE_ID:
        return [{"record_id": "NONE", "label": "无（未配置素材库 TableID）", "date_str": ""}]

    items = _fetch_records(APP_TOKEN, MATERIAL_TABLE_ID, page_size=200, max_pages=3)
    out = [{"record_id": "NONE", "label": "无", "date_str": ""}]
    for it in items[:limit]:
        rid = it.get("record_id") or ""
        fields = it.get("fields", {}) or {}
        desc = str(fields.get("描述") or "").strip()
        ts = fields.get("录入时间")
        if ts:
            try:
                dt = datetime.fromtimestamp(int(ts) / 1000, tz=ZoneInfo(TIMEZONE))
                dstr = dt.strftime("%Y-%m-%d")
            except Exception:
                dstr = "未知日期"
        else:
            dstr = "未知日期"
        label = f"{dstr} | {desc[:40] if desc else '（无描述）'}"
        if rid:
            out.append({"record_id": rid, "label": label, "date_str": dstr})
    return out


def add_ai_knowledge_record(gen_date, core_points: str, structured_text: str, link_record_ids, tags_text: str):
    token = get_tenant_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{AI_KNOWLEDGE_TABLE_ID}/records"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    date_ms = _date_to_ms_in_tz(gen_date)
    if not isinstance(link_record_ids, list):
        link_record_ids = []

    payload = {
        "fields": {
            "生成时间": date_ms,
            "核心观点": (core_points or "").strip(),
            "AI 结构化文本": (structured_text or "").strip(),
            "单向关联": link_record_ids,
            "标签": (tags_text or "").strip(),
        }
    }
    return requests.post(url, headers=headers, json=payload).json()


def _date_to_ms_in_tz(log_date) -> int:
    """
    将 date/datetime 转为“所配时区的当天 00:00:00”对应的 Unix epoch 毫秒。
    用于飞书表字段“日期”的唯一键匹配。
    """
    if isinstance(log_date, datetime):
        d = log_date.date()
    else:
        d = log_date
    dt_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=ZoneInfo(TIMEZONE))
    return int(dt_local.timestamp() * 1000)


def _find_insight_record_id_by_date_ms(date_ms: int):
    token = get_tenant_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{INSIGHT_TABLE_ID}/records"
    headers = {"Authorization": f"Bearer {token}"}

    page_token = None
    while True:
        params = {"page_size": 200}
        if page_token:
            params["page_token"] = page_token
        res = requests.get(url, headers=headers, params=params).json()
        items = res.get("data", {}).get("items", []) or []
        for it in items:
            fields = it.get("fields", {}) or {}
            if fields.get("日期") == date_ms:
                return it.get("record_id")
        page_token = res.get("data", {}).get("page_token")
        if not page_token:
            return None


def save_insight_to_feishu(log_date, report_text: str):
    """
    写入/更新 INSIGHT_TABLE_ID（每日深思）：
    - 若已存在“日期”=当天00:00(毫秒)的记录：更新
    - 否则：新增
    字段映射：
    - 日期 -> log_date(毫秒时间戳)
    - AI洞察报告 -> report_text
    """
    date_ms = _date_to_ms_in_tz(log_date)
    record_id = _find_insight_record_id_by_date_ms(date_ms)

    token = get_tenant_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {"fields": {"日期": date_ms, "AI洞察报告": report_text}}

    if record_id:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{INSIGHT_TABLE_ID}/records/{record_id}"
        return requests.put(url, headers=headers, json=payload).json()
    else:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{APP_TOKEN}/tables/{INSIGHT_TABLE_ID}/records"
        return requests.post(url, headers=headers, json=payload).json()


# ==========================================
# 3. Streamlit 前端界面
# ==========================================
st.set_page_config(page_title="能量大脑", layout="wide", page_icon="🔋")

# --- 全局样式注入（简约高级风） ---
st.markdown(
    """
<style>
  /* Global font */
  html, body, [class*="css"]  {
    font-family: 'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'Microsoft YaHei', serif !important;
  }

  /* App background */
  .stApp {
    background: #F8F9FA;
  }

  /* Main container breathing space + width constraint */
  .main .block-container {
    max-width: 1100px;
    padding-top: 3rem;
  }

  /* Native border container styling (st.container(border=True)) */
  .stElementContainer:has(div[data-testid="stVerticalBlockBorderWrapper"]) {
    border: none !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 12px !important;
    border: 1px solid #F0F2F6 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    padding: 20px;
    margin-bottom: 20px;
  }

  /* 强制统计区两张图的卡片高度一致（仅作用于含图表的卡片） */
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.js-plotly-plot),
  div[data-testid="stVerticalBlockBorderWrapper"]:has(canvas),
  div[data-testid="stVerticalBlockBorderWrapper"]:has(svg) {
    min-height: 420px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  /* Vertical spacing helper */
  .vspace-sm { height: 0.75rem; }
  .vspace-md { height: 1.25rem; }

  /* Make subheaders less cramped */
  div[data-testid="stMarkdownContainer"] h2,
  div[data-testid="stMarkdownContainer"] h3 {
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
  }

  /* Unified heading styles */
  div[data-testid="stMarkdownContainer"] h1 {
    font-size: 2.0rem;
    font-weight: 750;
    letter-spacing: 0.2px;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
  }
  div[data-testid="stMarkdownContainer"] h2 {
    font-size: 1.35rem;
    font-weight: 700;
  }
  div[data-testid="stMarkdownContainer"] h3 {
    font-size: 1.15rem;
    font-weight: 650;
  }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.expander("🤖 AI 助手设定", expanded=False):
        st.text_area(
            "System Prompt（可自定义）",
            value=DEFAULT_AI_SYSTEM_PROMPT,
            key="ai_system_prompt",
            height=220,
            help="这里的内容会作为 AI 的最高优先级指令，用于控制输出风格与边界。",
        )

# --- 全局 session_state 初始化 ---
if "df_cache" not in st.session_state:
    st.session_state["df_cache"] = pd.DataFrame()
if "last_review_md" not in st.session_state:
    st.session_state["last_review_md"] = ""
if "last_review_range" not in st.session_state:
    st.session_state["last_review_range"] = None
if "material_df_cache" not in st.session_state:
    st.session_state["material_df_cache"] = pd.DataFrame()
if "knowledge_df_cache" not in st.session_state:
    st.session_state["knowledge_df_cache"] = pd.DataFrame()
if "instant_entry" not in st.session_state:
    st.session_state["instant_entry"] = None
if "stats_history_mode" not in st.session_state:
    st.session_state["stats_history_mode"] = False

def _columns(*args, **kwargs):
    """
    Streamlit 版本兼容：部分版本不支持 st.columns 的 vertical_alignment 参数。
    支持则使用；不支持则自动降级忽略该参数，避免运行时报 TypeError。
    """
    try:
        return st.columns(*args, **kwargs)
    except TypeError:
        kwargs.pop("vertical_alignment", None)
        return st.columns(*args, **kwargs)

def _card_container():
    """
    使用原生 st.container(border=True) 做卡片化；旧版本不支持 border 参数时自动降级。
    """
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()

tabs = st.tabs(["⚡ 能量工作台", "📽️ 素材库", "🧠 知识库", "🔍 全局搜索"])
tab1, tab2, tab3, tab4 = tabs

# ==========================================
# Tab 1: 能量工作台（输入-统计同屏）
# ==========================================
with tab1:
    st.title("⚡ 能量工作台")

    col_input, col_stats = st.columns([0.4, 0.6], gap="large")

    # --- 左侧：记一笔（卡片） ---
    with col_input:
        with _card_container():
            st.subheader("📝 记一笔")
            c1, c2 = _columns(2, vertical_alignment="bottom")
            with c1:
                log_date = st.date_input("记录日期", datetime.today(), key="log_date")
            with c2:
                log_time = st.time_input("记录时间", datetime.now().time(), key="log_time")

            selected_dt = datetime.combine(log_date, log_time)
            timestamp_ms = int(selected_dt.timestamp() * 1000)

            st.markdown('<div class="vspace-sm"></div>', unsafe_allow_html=True)
            content = st.text_area("内容", placeholder="输入你的心流或琐事...", height=150, key="log_content")
            energy = st.slider("能量状态 (1-10)", min_value=1, max_value=10, value=5, key="log_energy")

            st.markdown('<div class="vspace-sm"></div>', unsafe_allow_html=True)
            submit = st.button("🚀 提交并同步至飞书", use_container_width=True, key="submit_log")
            if submit:
                if not content.strip():
                    st.warning("请填写内容！")
                else:
                    with st.spinner("正在同步至数字大脑..."):
                        res = add_record(timestamp_ms, content, energy)
                        if res.get("code") == 0:
                            st.toast("记录已同步", icon="✅")
                            st.success("✅ 已成功记录")
                            # 即时反馈：不做全量拉取，直接用当前输入临时更新右侧视图
                            st.session_state["instant_entry"] = {
                                "dt": selected_dt,
                                "timestamp_ms": timestamp_ms,
                                "content": (content or "").strip(),
                                "energy": int(energy),
                            }
                            st.session_state["stats_history_mode"] = False
                        else:
                            st.error(f"❌ 同步失败！报错信息：{res.get('msg')}")

    # --- 右侧：统计（日期区间 + 图表 + 词云，卡片） ---
    with col_stats:
        with st.container():
            # 统计控制区：原生卡片（border=True）
            with _card_container():
                st.subheader("📊 统计面板")

            # 显著开关：即时反馈 / 历史统计
            st.caption("默认展示：刚刚记录的这一条（即时反馈）")
            mode_l, mode_m, mode_r = _columns([0.5, 0.22, 0.28], vertical_alignment="bottom")
            with mode_l:
                st.empty()
            with mode_m:
                # 刷新仅更新缓存，不改变日期选择器/视图状态
                refresh_cache = st.button("🔄 刷新缓存", use_container_width=True, key="refresh_cache_btn")
            with mode_r:
                history_mode = st.toggle("查看历史时段", key="stats_history_mode")

            if refresh_cache:
                with st.spinner("正在刷新后台缓存..."):
                    try:
                        st.session_state["df_cache"] = fetch_data()
                    except Exception as e:
                        st.warning(f"刷新缓存失败：{e}")

            def _render_square_energy_chart_plotly(df_one: pd.DataFrame):
                if not _PLOTLY_AVAILABLE:
                    # 降级：matplotlib 正方形图（无交互，但保证可用）
                    if "plotly_missing_warned" not in st.session_state:
                        st.session_state["plotly_missing_warned"] = True
                        st.warning("未安装 plotly，已自动降级为基础图表。可运行 `pip install plotly` 启用高级交互图表。")

                    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
                    try:
                        dfp = df_one.copy()
                        xs = pd.to_datetime(dfp["时间"])
                        ys = dfp["能量分值"].astype(float)
                        ax.plot(xs, ys, marker="o", linewidth=2, color="#5A8CE6")
                        ax.fill_between(xs, ys, [0] * len(ys), color="#8AB4F8", alpha=0.22)
                        fig.autofmt_xdate(rotation=20)
                        ax.grid(False)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        for spine in ["top", "right", "left", "bottom"]:
                            ax.spines[spine].set_visible(False)
                    except Exception:
                        ax.text(0.5, 0.5, "暂无可绘制数据", ha="center", va="center")
                    st.pyplot(fig, use_container_width=True)
                    return

                dfp = df_one.copy()
                dfp["时间"] = pd.to_datetime(dfp["时间"])
                dfp = dfp.sort_values("时间")

                line_color = "rgba(163, 142, 255, 0.95)"  # 淡紫
                fill_color = "rgba(163, 142, 255, 0.2)"

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=dfp["时间"],
                        y=dfp["能量分值"],
                        mode="lines+markers",
                        line=dict(color=line_color, width=3, shape="spline", smoothing=1.3),
                        marker=dict(size=7, color="rgba(163, 142, 255, 1.0)"),
                        fill="tozeroy",
                        fillcolor=fill_color,
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>能量：%{y}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title=None,
                    title_text=None,
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    hovermode="x unified",
                    font=dict(family="'PingFang SC','Microsoft YaHei',serif"),
                )
                # 极简：移除网格线/轴杂项
                fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", title=None, title_text=None)
                fig.update_yaxes(showgrid=False, zeroline=False, showline=False, title=None, title_text=None)
                st.plotly_chart(fig, use_container_width=True)

            def _render_square_wordcloud(text: str):
                text = (text or "").strip()
                if not text:
                    st.info("暂无内容可生成词云。")
                    return
                words = jieba.cut(text)
                stop_words = ["我们", "可以", "这个", "但是", "就是", "觉得", "还是", "确实","然后","没有","一些","一个","时候","因为","所以","感觉","不是","对于","什么","其实","东西","完全","可能","好像","已经",
                              "现在","这样","那么","那个","还有","非常","如果","最后","只是","并且","以及","不会","他们","一直","事情","怎样","一点","知道","看到","怎么","一下","里面","今天","虽然","但是","晚上",
                              "突然","进行","真的","不过","一条","为什么","反正","一天","一段","有点","后面","开始","刚开始","大家","一股","这次","一次","一种","不少","一定","一样","一会儿","一起","实际","终于",
                              "很多","特别","最近","下午","需要","一位","这里","那里","地方","我会","之前","结果","应该","不论","出于","一是","二是","立马","这么","的话"]
                filtered_words = [w for w in words if len(w) > 1 and w not in stop_words]
                if not filtered_words:
                    st.info("内容太少，暂时无法提取有效关键词。")
                    return
                try:
                    wc = WordCloud(
                        font_path="simhei.ttf",
                        width=650,
                        height=650,
                        background_color="white",
                        colormap="viridis",
                    ).generate(" ".join(filtered_words))
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                except ValueError:
                    st.error("⚠️ 词云生成失败：找不到中文字体。请在代码中修改 font_path 为你电脑上存在的中文字体路径。")

            # --- 即时反馈模式：只展示刚刚记录的这一条 ---
            if not history_mode:
                inst = st.session_state.get("instant_entry")
                if not inst:
                    st.info("先在左侧提交一条记录，即可在这里即时看到这一条的图表与关键词。")
                else:
                    sub_col1, sub_col2 = st.columns([1, 1], gap="medium")
                    dt = inst.get("dt")
                    df_one = pd.DataFrame(
                        [
                            {
                                "时间": pd.to_datetime(dt),
                                "能量分值": inst.get("energy"),
                                "日记内容": inst.get("content", ""),
                            }
                        ]
                    )
                    with sub_col1:
                        with _card_container():
                            _render_square_energy_chart_plotly(df_one)
                    with sub_col2:
                        with _card_container():
                            _render_square_wordcloud(inst.get("content", ""))

                # 不再额外包一层外卡片，避免出现“幽灵空白框”

            # --- 历史统计模式：14 天区间（按需刷新/拉取） ---
            else:
                top_a, top_b = _columns([0.75, 0.25], vertical_alignment="bottom")
                with top_a:
                    date_range = st.date_input(
                        "统计时间段（默认单日，最多 14 天）",
                        value=(datetime.today().date(), datetime.today().date()),
                        key="stats_date_range",
                    )
                with top_b:
                    refresh = st.button("🔄 刷新数据", use_container_width=True, key="refresh_stats")

                if isinstance(date_range, (tuple, list)):
                    if len(date_range) == 1:
                        start_date = date_range[0]
                        end_date = date_range[0]
                    elif len(date_range) >= 2:
                        start_date = date_range[0]
                        end_date = date_range[1]
                    else:
                        start_date = datetime.today().date()
                        end_date = start_date
                else:
                    start_date = date_range
                    end_date = date_range

                if start_date and end_date and start_date > end_date:
                    start_date, end_date = end_date, start_date

                span_days = (end_date - start_date).days if (start_date and end_date) else 0
                over_limit = span_days > 14
                if over_limit:
                    st.warning("为保证 AI 分析质量和防止 Token 超限，最多仅支持选择 14 天")

                if refresh or st.session_state.get("df_cache", pd.DataFrame()).empty:
                    with st.spinner("正在拉取飞书数据..."):
                        st.session_state["df_cache"] = fetch_data()

                df = st.session_state.get("df_cache", pd.DataFrame())
                if df.empty:
                    st.info("历史数据为空或尚未拉取。")
                elif "记录日期" not in df.columns:
                    st.warning("数据列名不匹配：未找到「记录日期」。")
                else:
                    df_local = _prepare_df_with_local_time(df)
                    df_day = df_local[
                        (df_local["时间"].dt.date >= start_date) & (df_local["时间"].dt.date <= end_date)
                    ].sort_values("时间")

                    if df_day.empty:
                        st.info("所选日期区间没有数据。你可以换个区间或刷新数据。")
                        # 不再额外包一层外卡片，避免出现“幽灵空白框”
                    else:
                        sub_col1, sub_col2 = st.columns([1, 1], gap="medium")

                        df_chart = df_day.copy()
                        df_chart["时间"] = df_chart["时间"].dt.tz_localize(None)

                        with sub_col1:
                            with _card_container():
                                _render_square_energy_chart_plotly(df_chart)

                        with sub_col2:
                            with _card_container():
                                if "日记内容" in df_chart.columns:
                                    all_text = " ".join(df_chart["日记内容"].dropna().astype(str))
                                    _render_square_wordcloud(all_text)
                                else:
                                    st.info("所选日期没有可用于词云分析的日志内容。")

    # --- 分栏下方：深度复盘报告（全宽卡片） ---
    # 仅在“历史时段”模式下展示，避免即时模式出现空阴影长条
    if st.session_state.get("stats_history_mode") or st.session_state.get("last_review_md"):
        st.markdown('<div class="vspace-sm"></div>', unsafe_allow_html=True)

        # 从 session_state 读取区间（若尚未生成则默认今日）
        _dr = st.session_state.get("stats_date_range", (datetime.today().date(), datetime.today().date()))
        if isinstance(_dr, (tuple, list)) and len(_dr) >= 1:
            _start = _dr[0]
            _end = _dr[0] if len(_dr) == 1 else _dr[1]
        else:
            _start = datetime.today().date()
            _end = _start
        if _start and _end and _start > _end:
            _start, _end = _end, _start

        _span = (_end - _start).days if (_start and _end) else 0
        _over_limit = _span > 14

        with _card_container():

            st.subheader("🧠 深度复盘报告（AI）")

            if not st.session_state.get("stats_history_mode"):
                st.info("开启右侧「查看历史时段」后，才可以进行区间复盘。")
            else:
                btn_a, _ = _columns([1, 4], vertical_alignment="bottom")
                with btn_a:
                    deep_review = st.button("🧠 生成复盘", use_container_width=True, key="deep_review_btn", disabled=_over_limit)
                st.caption(f"区间最多 14 天；生成后将展示在下方。当前区间：{_start} ~ {_end}")

                if deep_review:
                    if _over_limit:
                        st.info("已禁用：所选日期区间超过 14 天。")
                        st.stop()
                    with st.spinner("正在拉取并分析所选日期记录..."):
                        if st.session_state["df_cache"].empty:
                            st.session_state["df_cache"] = fetch_data()

                        df = st.session_state["df_cache"]
                        if not df.empty and "记录日期" in df.columns:
                            df_tmp = _prepare_df_with_local_time(df)
                            df_day = df_tmp[
                                (df_tmp["时间"].dt.date >= _start) & (df_tmp["时间"].dt.date <= _end)
                            ].sort_values("时间")

                            if "日记内容" not in df_day.columns or df_day["日记内容"].dropna().empty:
                                st.info("所选日期还没有可用于复盘的日志内容。")
                            else:
                                daily_logs = df_day["日记内容"].dropna().astype(str).tolist()
                                try:
                                    result_md = generate_daily_review(
                                        daily_logs=daily_logs,
                                        system_prompt=st.session_state.get("ai_system_prompt", DEFAULT_AI_SYSTEM_PROMPT),
                                    )
                                    st.session_state["last_review_md"] = result_md
                                    st.session_state["last_review_range"] = (_start, _end)
                                except Exception as e:
                                    st.error(f"大模型调用失败，详细错误：{e}")
                        else:
                            st.info("飞书里好像还没有记录，或者列名不匹配哦！")

                # 展示报告（仅在区间匹配时显示，避免误用旧缓存）
                if st.session_state.get("last_review_md") and st.session_state.get("last_review_range") == (_start, _end):
                    st.markdown(st.session_state["last_review_md"])

                    st.markdown('<div class="vspace-sm"></div>', unsafe_allow_html=True)
                    col_a, col_b = _columns(2, vertical_alignment="bottom")
                    with col_a:
                        if st.button("💾 保存报告到飞书", use_container_width=True, key="save_report_btn", disabled=_over_limit):
                            try:
                                save_res = save_insight_to_feishu(_end, st.session_state["last_review_md"])
                                if isinstance(save_res, dict) and save_res.get("code") == 0:
                                    st.toast("报告已保存到飞书", icon="✅")
                                    st.success("✅ 已保存到飞书‘每日深思’表")
                                else:
                                    st.error(f"飞书写入失败：{save_res}")
                            except Exception as e:
                                st.error(f"飞书写入失败：{e}")
                        st.caption("建议单日复盘使用，保持飞书表格规范")

                    with col_b:
                        filename = f"复盘报告_{_start}_至_{_end}.md"
                        st.download_button(
                            label="📥 下载为 MD",
                            data=(st.session_state["last_review_md"] or "").encode("utf-8"),
                            file_name=filename,
                            mime="text/markdown; charset=utf-8",
                            use_container_width=True,
                        )
                        st.caption("建议多日周期复盘使用")
                else:
                    st.caption("点击「🧠 生成复盘」后，这里会展示 AI 报告。")

# ==========================================
# Tab 2: 素材库（对齐修复 + 卡片）
# ==========================================
with tab2:
    st.title("📽️ 素材库")
    with _card_container():
        st.subheader("🎬 素材录入")

        # 标题行（用于统一对齐的“标签”）
        h_a, h_b = _columns([1, 3], vertical_alignment="bottom")
        with h_a:
            st.caption("录入时间")
        with h_b:
            st.caption("素材描述")

        col_a, col_b = _columns([1, 3], vertical_alignment="bottom")
        with col_a:
            material_date = st.date_input(
                "录入时间",
                value=datetime.today().date(),
                key="material_date",
                label_visibility="collapsed",
            )
        with col_b:
            material_desc = st.text_area(
                "素材描述",
                placeholder="例如：B站某某视频：如何用结构化表达吸引人",
                height=90,
                key="material_desc",
                label_visibility="collapsed",
            )

        material_raw = st.text_area(
            "素材原始文本内容（粘贴视频转文字/文章全文）",
            placeholder="把转录稿/全文粘贴到这里...",
            height=260,
            key="material_raw",
        )

        if st.button("入库", use_container_width=True, key="material_submit"):
            if not material_raw.strip():
                st.warning("请先粘贴「素材原始文本内容」。")
            elif not MATERIAL_TABLE_ID or "请替换" in MATERIAL_TABLE_ID:
                st.error("你还没配置 `MATERIAL_TABLE_ID`，请先在代码顶部替换为真实 TableID。")
            else:
                with st.spinner("正在写入飞书《原始素材库》..."):
                    try:
                        res = add_material_record(
                            entry_date=material_date,
                            desc=material_desc,
                            raw_content=material_raw,
                            related_status="待关联",
                        )
                        if isinstance(res, dict) and res.get("code") == 0:
                            st.toast("素材已入库", icon="✅")
                            st.success("✅ 已入库到飞书素材库。")
                            # 可选：刷新缓存
                            st.session_state["material_df_cache"] = pd.DataFrame()
                        else:
                            st.error(f"❌ 飞书写入失败：{res}")
                    except Exception as e:
                        st.error(f"❌ 飞书写入异常：{e}")


# ==========================================
# Tab 3: 知识库（对齐修复 + 卡片）
# ==========================================
with tab3:
    st.title("🧠 知识库")

    with _card_container():

        st.subheader("🧠 知识入库")

        # 标题行（用于统一对齐的“标签”）
        kh1, kh2 = _columns([1, 2], vertical_alignment="bottom")
        with kh1:
            st.caption("生成时间")
        with kh2:
            st.caption("关联素材（选填）")

        col_k1, col_k2 = _columns([1, 2], vertical_alignment="bottom")
        with col_k1:
            knowledge_date = st.date_input(
                "生成时间",
                value=datetime.today().date(),
                key="knowledge_date",
                label_visibility="collapsed",
            )

        with col_k2:
            # 同行对齐：关联素材下拉 + 刷新按钮
            sel_col, btn_col = _columns([0.78, 0.22], vertical_alignment="bottom")
            with btn_col:
                refresh_materials = st.button("🔄 刷新", use_container_width=True, key="refresh_material_list")

            if refresh_materials:
                st.cache_data.clear()
                with st.spinner("正在刷新素材列表..."):
                    material_list = get_material_list(limit=20)
            else:
                material_list = get_material_list(limit=20)

            option_labels = [x["label"] for x in material_list]
            with sel_col:
                selected_label = st.selectbox(
                    "关联素材（选填）",
                    options=option_labels,
                    index=0,
                    key="knowledge_related_material",
                    label_visibility="collapsed",
                )

            selected_idx = option_labels.index(selected_label) if selected_label in option_labels else 0
            selected_record_id = material_list[selected_idx].get("record_id", "NONE")

        core_points_input = st.text_area(
            "核心观点（多行文本）",
            placeholder="1-3 条核心观点，每行一条也可以…",
            height=120,
            key="knowledge_core_points",
        )
        structured_text_input = st.text_area(
            "AI 结构化文本（多行文本 / Markdown）",
            placeholder="把结构化后的 markdown 文本粘贴到这里…",
            height=220,
            key="knowledge_structured_text",
        )
        tags_input = st.text_area(
            "标签（多行文本/多选字段也可先用逗号分隔）",
            placeholder="例如：沟通,表达,结构化",
            height=80,
            key="knowledge_tags",
        )

        if st.button("入库", use_container_width=True, key="knowledge_submit"):
            if not (core_points_input.strip() or structured_text_input.strip()):
                st.warning("请至少填写「核心观点」或「AI 结构化文本」。")
            elif not AI_KNOWLEDGE_TABLE_ID or "请替换" in AI_KNOWLEDGE_TABLE_ID:
                st.error("你还没配置 `AI_KNOWLEDGE_TABLE_ID`，请先在代码顶部替换为真实 TableID。")
            else:
                link_ids = [] if (not selected_record_id or selected_record_id == "NONE") else [selected_record_id]

                with st.spinner("正在写入飞书《AI知识库》..."):
                    try:
                        res = add_ai_knowledge_record(
                            gen_date=knowledge_date,
                            core_points=core_points_input,
                            structured_text=structured_text_input,
                            link_record_ids=link_ids,
                            tags_text=tags_input,
                        )
                        if isinstance(res, dict) and res.get("code") == 0:
                            st.toast("知识已入库", icon="✅")
                            st.success("✅ 已入库到飞书 AI 知识库。")
                            st.session_state["knowledge_df_cache"] = pd.DataFrame()
                        else:
                            st.error(f"❌ 飞书写入失败：{res}")
                    except Exception as e:
                        st.error(f"❌ 飞书写入异常：{e}")


# ==========================================
# Tab 4: 全局搜索（卡片 + 分类结果）
# ==========================================
with tab4:
    st.title("🔍 全局搜索")

    def _norm_text(x) -> str:
        if x is None:
            return ""
        try:
            return str(x)
        except Exception:
            return ""

    def _contains_kw(val, kw: str) -> bool:
        if not kw:
            return False
        return kw in _norm_text(val).lower()

    def _format_ts_ms(ts_ms) -> str:
        if not ts_ms:
            return "未知日期"
        try:
            dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=ZoneInfo(TIMEZONE))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "未知日期"

    def _snippet(text: str, n: int = 40) -> str:
        s = (text or "").strip().replace("\n", " ")
        return (s[:n] + "…") if len(s) > n else s

    with _card_container():
        keyword = st.text_input("关键词", value="", placeholder="例如：焦虑 / 沟通 / 项目 / 复盘 ...", key="global_search_kw")
        scopes = st.multiselect(
            "范围",
            options=["碎碎念", "素材", "知识"],
            default=["碎碎念", "素材", "知识"],
            key="global_search_scopes",
        )
        do_search = st.button("立即检索", use_container_width=True, key="do_global_search")


    if do_search:
        kw = (keyword or "").strip().lower()
        if not kw:
            st.warning("请输入关键词再检索。")
        elif not scopes:
            st.warning("请至少选择一个搜索范围。")
        else:
            any_hit = False

            with _card_container():
                st.subheader("📌 搜索结果")

                with st.spinner("正在检索中，请稍候..."):
                    # 1) 碎碎念
                    if "碎碎念" in scopes:
                        if st.session_state.get("df_cache", pd.DataFrame()).empty:
                            st.session_state["df_cache"] = fetch_data()

                        df = st.session_state.get("df_cache", pd.DataFrame())
                        if not df.empty and "日记内容" in df.columns:
                            df_local = _prepare_df_with_local_time(df)
                            mask = df_local["日记内容"].apply(lambda x: _contains_kw(x, kw))
                            hits = df_local[mask].sort_values(
                                "时间" if "时间" in df_local.columns else "记录日期",
                                ascending=False,
                            )
                            if not hits.empty:
                                any_hit = True
                                st.markdown("### 🧩 碎碎念")
                                for _, row in hits.iterrows():
                                    dstr = "未知日期"
                                    if "时间" in hits.columns and pd.notna(row.get("时间")):
                                        try:
                                            dstr = row["时间"].strftime("%Y-%m-%d %H:%M")
                                        except Exception:
                                            dstr = "未知日期"
                                    title = f"[{dstr}] {_snippet(_norm_text(row.get('日记内容')), 48)}"
                                    with st.expander(title):
                                        st.markdown(
                                            f"**时间**：{dstr}\n\n"
                                            f"**能量分值**：{_norm_text(row.get('能量分值'))}\n\n"
                                            f"**日记内容**：\n\n{_norm_text(row.get('日记内容'))}"
                                        )

                    # 2) 素材
                    if "素材" in scopes:
                        if st.session_state.get("material_df_cache", pd.DataFrame()).empty:
                            st.session_state["material_df_cache"] = fetch_material_data()
                        mdf = st.session_state.get("material_df_cache", pd.DataFrame())
                        if not mdf.empty:
                            m_mask = (
                                mdf.get("描述", pd.Series([""] * len(mdf))).apply(lambda x: _contains_kw(x, kw))
                                | mdf.get("内容", pd.Series([""] * len(mdf))).apply(lambda x: _contains_kw(x, kw))
                            )
                            mhits = mdf[m_mask].copy()
                            if not mhits.empty:
                                any_hit = True
                                st.markdown("### 🎬 素材")
                                if "录入时间" in mhits.columns:
                                    mhits["_d"] = mhits["录入时间"].apply(_format_ts_ms)
                                else:
                                    mhits["_d"] = "未知日期"
                                for _, row in mhits.iterrows():
                                    dstr = row.get("_d", "未知日期")
                                    desc = _norm_text(row.get("描述"))
                                    content_txt = _norm_text(row.get("内容"))
                                    title = f"[{dstr}] {_snippet(desc or content_txt, 48)}"
                                    with st.expander(title):
                                        st.markdown(
                                            f"**录入时间**：{dstr}\n\n"
                                            f"**描述**：{desc or '（空）'}\n\n"
                                            f"**内容**：\n\n{content_txt or '（空）'}"
                                        )

                    # 3) 知识
                    if "知识" in scopes:
                        if st.session_state.get("knowledge_df_cache", pd.DataFrame()).empty:
                            st.session_state["knowledge_df_cache"] = fetch_ai_knowledge_data()
                        kdf = st.session_state.get("knowledge_df_cache", pd.DataFrame())
                        if not kdf.empty:
                            k_mask = (
                                kdf.get("核心观点", pd.Series([""] * len(kdf))).apply(lambda x: _contains_kw(x, kw))
                                | kdf.get("AI 结构化文本", pd.Series([""] * len(kdf))).apply(lambda x: _contains_kw(x, kw))
                                | kdf.get("标签", pd.Series([""] * len(kdf))).apply(lambda x: _contains_kw(x, kw))
                            )
                            khits = kdf[k_mask].copy()
                            if not khits.empty:
                                any_hit = True
                                st.markdown("### 📚 知识")
                                if "生成时间" in khits.columns:
                                    khits["_d"] = khits["生成时间"].apply(_format_ts_ms)
                                else:
                                    khits["_d"] = "未知日期"
                                for _, row in khits.iterrows():
                                    dstr = row.get("_d", "未知日期")
                                    core = _norm_text(row.get("核心观点"))
                                    structured = _norm_text(row.get("AI 结构化文本"))
                                    tags = _norm_text(row.get("标签"))
                                    title = f"[{dstr}] {_snippet(core or structured or tags, 48)}"
                                    with st.expander(title):
                                        st.markdown(
                                            f"**生成时间**：{dstr}\n\n"
                                            f"**标签**：{tags or '（空）'}\n\n"
                                            f"**核心观点**：\n\n{core or '（空）'}\n\n"
                                            f"**AI 结构化文本**：\n\n{structured or '（空）'}"
                                        )

                if not any_hit:
                    st.info("未找到相关内容，换个词试试？")
