"""Look and feel for the demo, kept out of the app so the app reads as logic.

Streamlit widgets inherit the operating system's dark mode unless you pin
them, which turns chat bubbles and audio players into dark boxes on a light
page. The theme below pins everything to the DataCamp palette.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "datacamp-logo.png"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Spectral:wght@500;600;700&display=swap');

:root {
  --bg: #FFFFFF;
  --paper: #F5F7F9;
  --accent: #03EF62;
  --accent-ink: #05192D;
  --ink: #05192D;
  --muted: #5A6872;
  --border: #E1E5E9;
  --warn: #C4820E;
  --danger: #D64550;
  --live: #00A67E;
}

.stApp { background: var(--bg); }
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3, h4 { font-family: 'Spectral', Georgia, serif; color: var(--ink); letter-spacing: -0.015em; }
/* Hide the chrome, but never the header itself: Streamlit puts the arrow
   that reopens a collapsed sidebar inside it, so hiding the header locks
   the sidebar shut with no way back. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] { visibility: visible; }
.block-container { padding-top: 1.5rem; max-width: 1240px; }

.hero { padding: 0 0 6px 0; }
.hero h1 { font-size: 1.7rem; margin: 0 0 .15rem 0; }
.hero p { color: var(--muted); font-size: .92rem; margin: 0; }

[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h3 { font-size: 1rem; margin: .1rem 0; }
.logo-link { display:block; margin: 0 0 2px 0; }
.logo-link img { width: 100%; display: block; transition: opacity .15s ease; }
.logo-link:hover img { opacity: .82; }
.logo-sub { color: var(--muted); font-size: .78rem; margin: 2px 0 8px 2px; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF; border: 1px solid var(--border) !important;
  border-radius: 16px; box-shadow: 0 1px 3px rgba(5,25,45,.05);
}
[data-testid="stMetric"] {
  background: #fff; border: 1px solid var(--border); border-radius: 14px; padding: 10px 14px;
}
[data-testid="stMetricValue"] { font-family: 'Spectral', serif; }

.stButton > button {
  background: #fff; color: var(--ink); border: 1px solid var(--border);
  border-radius: 11px; padding: .5rem 1rem; font-weight: 600; font-size: .9rem;
  transition: all .15s ease;
}
.stButton > button:hover { border-color: var(--accent-ink); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: var(--accent); color: var(--accent-ink); border: none; }

.label { font-weight: 700; color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; }
.pill {
  display: inline-block; background: var(--paper); border: 1px solid var(--border);
  border-radius: 8px; padding: 4px 10px; margin: 3px 4px 3px 0; font-size: .84rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.hint { background:#fff; border:1px dashed var(--border); border-radius:14px; padding:16px 18px; color:var(--muted); }

/* Call state. The one thing that must be readable at a glance: whose turn
   it is. A level meter next to it answers "is it hearing me". */
.state {
  display:flex; align-items:center; gap:12px; padding:11px 15px;
  border:1px solid var(--border); border-radius:13px; background:#fff; margin-bottom:10px;
}
.state-label { font-weight:700; font-size:.92rem; white-space:nowrap; }
.state-sub { color:var(--muted); font-size:.8rem; margin-left:auto; white-space:nowrap; }
.livedot { width:10px; height:10px; border-radius:50%; flex:none; }
.live-off { background: var(--border); }
.live-listen { background: var(--live); animation: pulse 1.6s infinite; }
.live-speak { background: var(--accent); animation: pulse .8s infinite; }
.live-think { background: var(--warn); animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.3 } }

/* Input level. Bars fill from the left with the caller's volume. */
.meter { display:flex; gap:3px; align-items:flex-end; height:20px; flex:none; }
.meter i { width:4px; border-radius:2px; background: var(--border); display:block; }
.meter i.on { background: var(--live); }
.meter i.hot { background: var(--accent); }

/* Call-state banner, for a handover to a person. */
.banner {
  border-radius:12px; padding:10px 15px; margin:0 0 10px 0;
  font-weight:600; font-size:.9rem; border:1px solid;
}
.banner-transfer { background: rgba(196,130,14,.12); border-color: rgba(196,130,14,.4); color:#8A5A06; }

/* Transcript scrolls instead of pushing the page down forever. */
.script { max-height: 46vh; overflow-y:auto; padding-right:4px; }

/* Cost split bar. */
.costbar { display:flex; height:14px; border-radius:7px; overflow:hidden; border:1px solid var(--border); }
.costbar-audio { background: var(--accent); }
.costbar-text { background: #AAB8C2; }
.costbar-legend { display:flex; gap:16px; margin-top:8px; font-size:.82rem; color:var(--muted); flex-wrap:wrap; }
.dot { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; }
.dot-audio { background: var(--accent); } .dot-text { background:#AAB8C2; }

/* Tabs. Streamlit's default active tab is a thin underline that is easy to
   miss, so the selected one gets a filled pill and real weight. */
[data-baseweb="tab-list"] {
  gap: 4px; background: var(--paper); padding: 4px; border-radius: 12px;
  border: 1px solid var(--border);
}
[data-baseweb="tab-list"] button[data-baseweb="tab"] {
  border-radius: 9px; padding: 6px 14px; color: var(--muted);
  font-weight: 600; font-size: .85rem; transition: all .15s ease;
}
[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
  background: rgba(5,25,45,.05); color: var(--ink);
}
[data-baseweb="tab-list"] button[aria-selected="true"] {
  background: #fff; color: var(--ink);
  box-shadow: 0 1px 3px rgba(5,25,45,.12);
  border: 1px solid var(--border);
}
/* The default sliding underline duplicates the pill, so hide it. */
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] [data-testid="stVerticalBlock"] { gap: .4rem; }

/* Event flow. */
.evt { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.82rem; padding:1px 0; }
.evt-client { color:#04603A; }
.evt-server { color: var(--muted); }

/* Order record, with a highlight on whatever the agent just changed. */
.rec { border:1px solid var(--border); border-radius:12px; padding:10px 13px; margin-bottom:8px; background:#fff; }
.rec-id { font-family: ui-monospace, monospace; font-weight:700; font-size:.9rem; }
.rec-row { display:flex; justify-content:space-between; gap:12px; font-size:.86rem; padding:2px 0; }
.rec-key { color: var(--muted); }
.rec-val { font-family: ui-monospace, monospace; text-align:right; }
.rec-changed {
  background: rgba(3,239,98,.18); border-radius:5px; padding:0 5px;
  animation: flash 1.2s ease-out;
}
@keyframes flash { from { background: rgba(3,239,98,.65); } to { background: rgba(3,239,98,.18); } }

/* Transcript. */
.turn { border-radius:13px; padding:9px 14px; margin-bottom:8px; border:1px solid var(--border); font-size:.93rem; }
.turn-caller { background: var(--paper); }
.turn-agent { background:#fff; }
.turn-system { background:transparent; border:none; text-align:center; color:var(--muted); font-size:.8rem; padding:3px 0; }
.turn-error { background: rgba(214,69,80,.08); border-color: rgba(214,69,80,.35); color:#8A2731; }
.who { font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:700; margin-bottom:2px; }

hr { border-color: var(--border); margin: .7rem 0; }
a { color: #04603A; }
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def _logo_data_uri(path: str) -> str:
    with open(path, "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode()


def sidebar_logo():
    if not LOGO_PATH.exists():
        return
    st.markdown(
        f'<a class="logo-link" href="https://www.datacamp.com/blog" target="_blank" '
        f'rel="noopener"><img src="{_logo_data_uri(str(LOGO_PATH))}" alt="DataCamp"/></a>'
        f'<div class="logo-sub">Built for the DataCamp blog</div>',
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="hero">
          <h1>Order Support Voice Agent</h1>
          <p>Grok Voice Think Fast 2.0 on a live call. Talk over the agent to
          interrupt it, and watch the order record while it speaks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_state(playing: bool, speaking: bool, thinking: bool, level: float) -> str:
    """The call's current state, plus a meter so you can see it hearing you."""
    if not playing:
        label, cls, sub = "Not connected", "live-off", "Press START to call"
    elif speaking:
        label, cls, sub = "Agent speaking", "live-speak", "Talk over it to interrupt"
    elif thinking:
        label, cls, sub = "Thinking", "live-think", "Working on the last turn"
    else:
        label, cls, sub = "Listening", "live-listen", "Say something"

    bars = []
    for i in range(14):
        # Speech sits low in the range, so a square root spreads the quiet end
        # out instead of leaving every bar dark until someone shouts.
        lit = playing and (level ** 0.5) > (i + 1) / 14
        height = 5 + i
        # Only color a bar "hot" if the mic level actually lit it. Marking
        # every bar hot just because the agent is speaking made the meter
        # look maxed out even with a silent room, which hid whether a
        # barge-in was actually reaching the callback.
        klass = "hot" if (speaking and lit) else ("on" if lit else "")
        bars.append(f'<i class="{klass}" style="height:{height}px"></i>')

    return (
        f'<div class="state"><span class="livedot {cls}"></span>'
        f'<span class="state-label">{label}</span>'
        f'<span class="meter">{"".join(bars)}</span>'
        f'<span class="state-sub">{sub}</span></div>'
    )


def render_orders(orders: dict, changed: set[tuple[str, str]]) -> str:
    """Render the order store, flagging fields the agent changed this call."""
    blocks = []
    for number, order in orders.items():
        rows = []
        for key, value in order.items():
            mark = ' class="rec-val rec-changed"' if (number, key) in changed else ' class="rec-val"'
            shown = "not set" if value is None else value
            rows.append(
                f'<div class="rec-row"><span class="rec-key">{key}</span>'
                f'<span{mark}>{shown}</span></div>'
            )
        blocks.append(
            f'<div class="rec"><div class="rec-id">{number}</div>{"".join(rows)}</div>'
        )
    return "".join(blocks)


def render_transcript(turns: list[dict]) -> str:
    if not turns:
        return ('<div class="hint">Start the call and say something like '
                "&ldquo;what is the status of order ORD-1042?&rdquo;</div>")
    out = ['<div class="script">']
    for turn in turns:
        role = turn["role"]
        text = turn["text"]
        if role == "system":
            out.append(f'<div class="turn turn-system">{text}</div>')
            continue
        who = {"caller": "Caller", "agent": "Agent", "error": "Error"}.get(role, role)
        out.append(
            f'<div class="turn turn-{role}"><div class="who">{who}</div>{text}</div>'
        )
    out.append("</div>")
    return "".join(out)


def render_events(events: list[list]) -> str:
    if not events:
        return '<div class="evt evt-server">(nothing yet)</div>'
    out = []
    for direction, etype, count in events[-40:]:
        arrow = "&rarr;" if direction == "client" else "&larr;"
        cls = "evt-client" if direction == "client" else "evt-server"
        suffix = f" &times;{count}" if count > 1 else ""
        out.append(f'<div class="evt {cls}">{arrow} {etype}{suffix}</div>')
    return "".join(out)


def render_cost_bar(audio_usd: float, text_usd: float) -> str:
    total = (audio_usd + text_usd) or 1e-9
    return (
        f'<div class="costbar">'
        f'<div class="costbar-audio" style="width:{audio_usd / total * 100:.1f}%"></div>'
        f'<div class="costbar-text" style="width:{text_usd / total * 100:.1f}%"></div></div>'
        f'<div class="costbar-legend">'
        f'<span><i class="dot dot-audio"></i>Audio ${audio_usd:.5f}</span>'
        f'<span><i class="dot dot-text"></i>Text events ${text_usd:.5f}</span></div>'
    )
