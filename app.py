import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Pass Map Dashboard")

st.markdown("""
<style>
.small-metric{padding:6px 8px;}
.small-metric .label{font-size:12px;color:#ffffff;margin-bottom:3px;opacity:.95;}
.small-metric .value{font-size:18px;font-weight:600;color:#ffffff;}
.small-metric .delta{font-size:11px;color:#e6e6e6;margin-top:4px;}
.stats-section-title{font-size:14px;font-weight:600;margin-bottom:6px;color:#ffffff;}
.streamlit-expanderHeader{color:#ffffff!important;}
.streamlit-expander{background:rgba(255,255,255,.02);}
.filter-panel{
  background:linear-gradient(168deg,rgba(30,39,56,.92) 0%,rgba(22,28,40,.97) 100%);
  border:1px solid rgba(255,255,255,.08);border-radius:14px;
  padding:24px 18px 20px 18px;
  box-shadow:0 4px 24px rgba(0,0,0,.25),0 1px 4px rgba(0,0,0,.12);
  backdrop-filter:blur(6px);}
.filter-panel h3{font-size:15px;color:#c8d6e5;letter-spacing:.5px;margin-bottom:8px;}
.filter-panel .filter-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:14px 0;}
.stSubheader{color:#ffffff!important;}
.match-title{
  font-size:15px;font-weight:700;color:#c8d6e5;
  letter-spacing:.4px;margin-bottom:6px;margin-top:14px;
  padding:6px 10px;
  background:rgba(255,255,255,.04);
  border-left:3px solid #2F80ED;
  border-radius:4px;}
</style>
""", unsafe_allow_html=True)


def small_metric(label: str, value: str, delta: str | None = None):
    html = f'<div class="small-metric"><div class="label">{label}</div><div class="value">{value}</div>'
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


st.title("Pass Map Dashboard")

# ── Constants ────────────────────────────────────────────────────────────
FIELD_X, FIELD_Y   = 120.0, 80.0
HALF_LINE_X        = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN      = 53.33
LANE_RIGHT_MAX     = 26.67
LATERAL_MIN_DIST   = 12.0

COLOR_SUCCESS     = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL        = "#E07070"
ALPHA_SUCCESS     = 0.07

COLOR_LBP_WON  = "#F59E0B"
COLOR_LBP_LOST = "#E07070"
COLOR_BPP      = "#8B5CF6"

FIG_W, FIG_H       = 6.8, 4.6
FIG_W_HEAT, FIG_H_HEAT = 6.8, 4.6
FIG_DPI = 110

MATCH_SAC = "Vs Sacramento United (25/04/2026)"
MATCH_NYC = "Vs New York City FC (25/11/2025)"

POSITION_BY_MATCH: dict[str, str] = {
    MATCH_SAC: "LCB",
    MATCH_NYC: "LCB",
}

# ── Pass Map data ─────────────────────────────────────────────────────────
matches_data = {
    MATCH_SAC: [
        ("PASS WON",  6.14,26.04,15.28,25.70,"strong"),("PASS WON",28.08,22.05,28.42,58.78,"strong"),
        ("PASS WON",42.05,20.22,49.19, 4.59,"strong"),("PASS WON",46.54,17.89,66.98, 2.60,"strong"),
        ("PASS WON",49.36,11.41,63.66,12.07,"strong"),("PASS WON",63.99,26.87,75.63,10.08,"strong"),
        ("PASS WON",71.14,44.99,52.35,44.49,"strong"),("PASS WON",39.65, 4.70,22.43,30.09,"strong"),
        ("PASS WON",34.96,21.74,32.52,56.35,"strong"),("PASS WON",41.74,23.65,56.52,50.96,"strong"),
        ("PASS WON",48.17,56.17,69.91,54.61,"strong"),("PASS WON",55.48,65.74,41.04,74.09,"strong"),
        ("PASS WON",58.61,36.35,56.87,65.39,"strong"),("PASS WON",55.83,22.09,55.30,36.52,"strong"),
        ("PASS WON",62.43,37.57,78.61, 7.48,"strong"),("PASS WON",52.67,10.30,46.43,33.22,"strong"),
        ("PASS WON",77.60, 5.72,83.83,15.62,"strong"),("PASS WON",73.75,12.87,91.17, 7.37,"strong"),
        ("PASS WON",68.25,19.28,72.28, 7.00,"strong"),("PASS WON",64.95,18.73,82.18,35.05,"strong"),
        ("PASS WON",69.72,26.98,83.10,40.92,"strong"),("PASS WON",80.17,18.92,65.68,41.28,"strong"),
        ("PASS WON",77.42,22.58,71.00,42.02,"strong"),("PASS WON",73.20,19.65,75.22,39.27,"strong"),
        ("PASS WON",34.40,19.89,43.71, 6.92,"strong"),("PASS WON",37.06,19.05,46.87, 7.92,"strong"),
        ("PASS WON",38.06,23.21,38.72,53.46,"strong"),("PASS WON",54.18,32.35,54.68,55.46,"strong"),
        ("PASS WON",48.53,19.22,72.63,56.29,"strong"),("PASS WON",52.02,18.39,83.77,11.08,"strong"),
        ("PASS WON",75.13,23.21,78.12,39.34,"strong"),("PASS WON",78.78,19.72,81.44,47.98,"strong"),
        ("PASS WON",81.94,22.88,88.09,43.66,"strong"),("PASS WON",40.05,17.23,47.03,11.57,"strong"),
        ("PASS WON",46.20,16.06,53.52, 5.09,"strong"),("PASS WON",37.39,18.39,52.35,27.37,"strong"),
        ("PASS WON",84.60,23.88,94.91, 7.92,"strong"),("PASS WON",76.12,20.38,81.44,41.00,"strong"),
        ("PASS WON",66.48,31.69,84.94,40.50,"strong"),("PASS WON",58.17,38.17,75.29,15.56,"strong"),
        ("PASS WON",67.31,27.70,88.43,23.54,"strong"),("PASS WON",83.77,21.38,81.11,41.16,"strong"),
        ("PASS WON", 2.48, 4.26,12.29,16.56,"strong"),("PASS WON",14.62, 2.76, 2.82,22.21,"strong"),
        ("PASS WON",45.37,15.06,43.88,53.96,"strong"),("PASS WON",59.83,36.18,30.41,38.01,"strong"),
        ("PASS WON",45.04,18.22,54.68,32.35,"strong"),("PASS WON",55.18,32.02,64.65,15.23,"strong"),
        ("PASS WON",51.02,19.89,65.82, 2.76,"strong"),("PASS WON",76.12,19.72,81.61,30.36,"strong"),
        ("PASS WON",90.59,10.74,88.59,27.70,"strong"),("PASS WON",82.61,22.55,97.74, 5.76,"strong"),
        ("PASS WON",62.33,36.84,69.81,46.15,"strong"),
        ("PASS LOST",82.94,31.19,95.74,35.01,"strong"),("PASS LOST",84.27,35.68,75.63,34.18,"strong"),
        ("PASS LOST",51.36,12.07,66.65,19.05,"strong"),("PASS LOST",13.95,22.21,56.84,38.84,"strong"),
        ("PASS WON", 0.65,11.57,12.12, 2.26,"weak"),("PASS WON",11.13,22.71, 2.65,40.17,"weak"),
        ("PASS WON",51.85,13.07,66.48,12.41,"weak"),("PASS WON",36.39,21.71,51.52,17.56,"weak"),
        ("PASS WON",46.87,17.23,53.68,25.54,"weak"),("PASS WON",39.05,17.39,38.56,49.64,"weak"),
        ("PASS WON",59.34,17.39,53.35,37.01,"weak"),("PASS WON",54.68,18.39,56.18,38.84,"weak"),
        ("PASS WON",72.97,20.38,63.82,47.15,"weak"),("PASS WON",73.80,23.21,78.62,50.81,"weak"),
        ("PASS LOST",50.52,22.05,70.31,16.23,"weak"),
    ],
    MATCH_NYC: [
        ("PASS WON",10.13,27.70,27.92,1.93,"strong"),
        ("PASS WON",16.95,28.36,16.78,51.64,"strong"),
        ("PASS WON",31.08,36.68,10.79,35.18,"strong"),
        ("PASS WON",53.52,12.57,66.82,17.72,"strong"),
        ("PASS WON",66.82,7.75,55.68,39.83,"strong"),
        ("PASS WON",74.79,44.99,82.94,7.42,"strong"),
        ("PASS WON",78.45,8.75,63.16,26.87,"strong"),
        ("PASS WON",67.15,21.22,74.30,21.55,"strong"),
        ("PASS WON",69.64,11.24,55.51,36.68,"strong"),
        ("PASS WON",62.16,53.63,64.49,34.18,"strong"),
        ("PASS WON",28.58,39.17,28.75,52.80,"strong"),
        ("PASS WON",31.24,22.05,10.79,36.01,"strong"),
        ("PASS WON",36.73,15.56,42.71,0.77,"strong"),
        ("PASS WON",27.09,23.54,37.39,3.10,"strong"),
        ("PASS WON",31.57,12.41,22.93,39.50,"strong"),
        ("PASS WON",27.75,18.06,12.12,39.50,"strong"),
        ("PASS WON",35.56,4.09,14.12,31.02,"strong"),
        ("PASS WON",10.30,25.37,25.59,2.93,"strong"),
        ("PASS WON",11.13,28.36,28.75,5.26,"strong"),
        ("PASS WON",13.95,28.36,25.09,4.43,"strong"),
        ("PASS WON",36.23,12.74,11.96,29.19,"strong"),
        ("PASS WON",40.72,22.71,13.79,36.68,"strong"),
        ("PASS WON",32.41,26.54,43.88,26.37,"strong"),
        ("PASS WON",27.75,40.50,27.58,54.46,"strong"),
        ("PASS WON",34.40,46.82,48.20,62.94,"strong"),
        ("PASS WON",53.35,53.46,53.52,62.77,"strong"),
        ("PASS WON",49.19,10.58,40.88,40.33,"strong"),
        ("PASS WON",60.00,7.25,54.51,38.01,"strong"),
        ("PASS WON",58.67,26.87,80.78,24.54,"strong"),
        ("PASS WON",59.17,30.19,74.96,8.75,"strong"),
        ("PASS WON",76.29,24.54,85.10,3.76,"strong"),
        ("PASS LOST",41.43,53.55,59.07,48.54,"strong"),
        ("PASS LOST",10.24,29.97,49.79,23.85,"strong"),
        ("PASS LOST",33.08,25.15,65.76,2.13,"strong"),
        ("PASS LOST",57.40,18.65,71.70,0.46,"strong"),
        ("PASS LOST",63.90,24.03,92.30,27.19,"strong"),
        ("PASS WON",18.61,23.38,3.15,43.16,"weak"),
        ("PASS WON",32.24,22.21,26.75,46.48,"weak"),
        ("PASS WON",46.37,16.06,39.72,44.49,"weak"),
        ("PASS WON",52.35,12.57,44.71,44.16,"weak"),
        ("PASS WON",55.18,12.07,42.71,40.83,"weak"),
        ("PASS WON",55.51,2.76,73.96,3.10,"weak"),
        ("PASS WON",56.18,19.89,55.35,44.65,"weak"),
        ("PASS WON",58.50,29.03,51.19,45.15,"weak"),
        ("PASS WON",56.01,24.87,66.48,40.33,"weak"),
        ("PASS WON",61.16,36.84,73.63,27.20,"weak"),
        ("PASS WON",66.65,16.73,57.34,40.00,"weak"),
        ("PASS WON",79.45,4.26,64.65,26.20,"weak"),
    ],
}

# ── Advanced Passes data ─────────────────────────────────────────────────────
special_data = {
    MATCH_SAC: [
        ("LBP WON",44.37,19.05,57.84,52.30),("LBP WON",48.53,56.12,71.30,56.29),
        ("LBP WON",66.98,33.35,86.43,43.49),("LBP WON",66.15,18.22,84.94,32.52),
        ("LBP WON",53.35,19.22,87.26,14.23),("LBP WON",71.80,19.22,90.59,15.23),
        ("LBP WON",65.32,27.53,86.26,26.37),
        ("LBP LOST",82.44,31.85,94.24,35.51),
        ("BPP WON",61.99,10.24,79.78, 4.09),("BPP WON",58.84,36.01,77.29,14.07),
        ("BPP WON",39.72,19.39,51.69,28.36),("BPP WON",63.82,36.68,72.97,44.82),
    ],
    MATCH_NYC: [
        ("BPP WON",66.82,7.75,55.68,39.83),   # #5
        ("LBP WON",58.67,26.87,80.78,24.54),  # #29
        ("BPP WON",59.17,30.19,74.96,8.75),   # #30
    ],
}

# ── Helpers ─────────────────────────────────────────────────────────────
def classify_pass_direction(x_start, y_start, x_end, y_end) -> str:
    dx   = x_end - x_start
    dy   = y_end - y_start
    dist = np.sqrt(dx**2 + dy**2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:  return "forward"
    if angle_deg >= 135.0: return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"


def progressive_pass(x_start: float, x_end: float) -> bool:
    dist_start = FIELD_X - x_start
    dist_end   = FIELD_X - x_end
    closer_by  = dist_start - dist_end
    start_own  = x_start < HALF_LINE_X
    end_own    = x_end   < HALF_LINE_X
    if start_own and end_own:  return closer_by >= 30.0
    if start_own != end_own:   return closer_by >= 15.0
    return closer_by >= 10.0


def _save_fig(fig) -> Image.Image:
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)


# ── Build Pass Map DataFrames ─────────────────────────────────────────────────
dfs_by_match: dict = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(events, columns=["type","x_start","y_start","x_end","y_end","foot"])
    dfm["match"]    = match_name
    dfm["position"] = POSITION_BY_MATCH[match_name]
    dfm["number"]   = np.arange(1, len(dfm)+1)
    dfm["is_won"]   = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"]  = np.where(dfm["is_won"], "completed", "incomplete")
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r.x_start, r.y_start, r.x_end, r.y_end), axis=1)
    dfm["is_forward"]       = dfm["direction"] == "forward"
    dfm["is_backward"]      = dfm["direction"] == "backward"
    dfm["is_lateral_left"]  = dfm["direction"] == "lateral_left"
    dfm["is_lateral_right"] = dfm["direction"] == "lateral_right"
    dfm["is_lateral"]       = dfm["is_lateral_left"] | dfm["is_lateral_right"]
    dfm["is_progressive"]   = dfm.apply(lambda r: progressive_pass(r.x_start, r.x_end), axis=1)
    dfm["pass_distance"]    = np.sqrt((dfm.x_end-dfm.x_start)**2+(dfm.y_end-dfm.y_start)**2)
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

# ── Build Advanced Passes DataFrames ──────────────────────────────────────────
sp_dfs_by_match: dict = {}
for match_name, events in special_data.items():
    dfm = pd.DataFrame(events, columns=["type","x_start","y_start","x_end","y_end"])
    dfm["match"]         = match_name
    dfm["position"]      = POSITION_BY_MATCH[match_name]
    dfm["number"]        = np.arange(1, len(dfm)+1)
    dfm["is_won"]        = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"]       = np.where(dfm["is_won"], "completed", "incomplete")
    dfm["pass_type"]     = np.where(dfm["type"].str.startswith("LBP"),
                                    "line_breaking", "ball_progression")
    dfm["pass_distance"] = np.sqrt((dfm.x_end-dfm.x_start)**2+(dfm.y_end-dfm.y_start)**2)
    sp_dfs_by_match[match_name] = dfm

sp_df_all = pd.concat(sp_dfs_by_match.values(), ignore_index=True)


# ── Stats ──────────────────────────────────────────────────────────────────────
def _dir_stats(sub: pd.DataFrame):
    n = max(len(sub), 1)
    fwd = int(sub["is_forward"].sum());      bwd = int(sub["is_backward"].sum())
    ll  = int(sub["is_lateral_left"].sum()); lr  = int(sub["is_lateral_right"].sum())
    return {"fwd":fwd,"fwd_pct":round(fwd/n*100,1),"bwd":bwd,"bwd_pct":round(bwd/n*100,1),
            "ll":ll,"ll_pct":round(ll/n*100,1),"lr":lr,"lr_pct":round(lr/n*100,1)}


def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {k:0 for k in [
            "total_passes","completed_passes","incomplete_passes","accuracy_pct",
            "strong_total","strong_completed","strong_incomplete","strong_accuracy_pct",
            "strong_avg_dist","strong_prog_total","strong_prog_completed",
            "strong_fwd","strong_fwd_pct","strong_bwd","strong_bwd_pct",
            "strong_ll","strong_ll_pct","strong_lr","strong_lr_pct",
            "weak_total","weak_completed","weak_incomplete","weak_accuracy_pct",
            "weak_avg_dist","weak_tendency_pct","weak_prog_total","weak_prog_completed",
            "weak_fwd","weak_fwd_pct","weak_bwd","weak_bwd_pct",
            "weak_ll","weak_ll_pct","weak_lr","weak_lr_pct",
            "prog_total","prog_completed","prog_accuracy_pct","prog_pct_of_total"]}
    completed = int(df["is_won"].sum())
    strong = df[df["foot"]=="strong"]; weak = df[df["foot"]=="weak"]
    st_t = len(strong); st_c = int(strong["is_won"].sum())
    wk_t = len(weak);   wk_c = int(weak["is_won"].sum())
    prog_t = int(df["is_progressive"].sum())
    prog_c = int((df["is_progressive"] & df["is_won"]).sum())
    sd = _dir_stats(strong); wd = _dir_stats(weak)
    return {
        "total_passes":total,"completed_passes":completed,"incomplete_passes":total-completed,
        "accuracy_pct":round(completed/total*100,2),
        "strong_total":st_t,"strong_completed":st_c,"strong_incomplete":st_t-st_c,
        "strong_accuracy_pct":round(st_c/st_t*100,2) if st_t else 0,
        "strong_avg_dist":round(float(strong["pass_distance"].mean()),2) if st_t else 0,
        "strong_prog_total":int(strong["is_progressive"].sum()),
        "strong_prog_completed":int((strong["is_progressive"]&strong["is_won"]).sum()),
        **{f"strong_{k}":v for k,v in sd.items()},
        "weak_total":wk_t,"weak_completed":wk_c,"weak_incomplete":wk_t-wk_c,
        "weak_accuracy_pct":round(wk_c/wk_t*100,2) if wk_t else 0,
        "weak_avg_dist":round(float(weak["pass_distance"].mean()),2) if wk_t else 0,
        "weak_tendency_pct":round(wk_t/total*100,2),
        "weak_prog_total":int(weak["is_progressive"].sum()),
        "weak_prog_completed":int((weak["is_progressive"]&weak["is_won"]).sum()),
        **{f"weak_{k}":v for k,v in wd.items()},
        "prog_total":prog_t,"prog_completed":prog_c,
        "prog_accuracy_pct":round(prog_c/prog_t*100,2) if prog_t else 0,
        "prog_pct_of_total":round(prog_t/total*100,2),
    }


def compute_advanced_stats(sp_df: pd.DataFrame, total_passes: int) -> dict:
    lbp  = sp_df[sp_df["pass_type"] == "line_breaking"]
    bpp  = sp_df[sp_df["pass_type"] == "ball_progression"]
    lbp_t = len(lbp); lbp_c = int(lbp["is_won"].sum())
    bpp_t = len(bpp)
    ref   = max(total_passes, 1)
    return {
        "lbp_total":     lbp_t,
        "lbp_completed": lbp_c,
        "lbp_incomplete":lbp_t - lbp_c,
        "lbp_accuracy":  round(lbp_c / lbp_t * 100, 2) if lbp_t else 0,
        "lbp_tendency":  round(lbp_t / ref * 100, 2),
        "bpp_total":     bpp_t,
        "bpp_tendency":  round(bpp_t / ref * 100, 2),
    }


# ── Draw helpers ──────────────────────────────────────────────────────────────
def _base_pitch(fw=FIG_W, fh=FIG_H):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(fw, fh))
    fig.set_facecolor("#1a1a2e"); fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", lw=1.0, alpha=0.18)
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.6, alpha=0.10, linestyle="--")
    return fig, ax, pitch


def _attack_arrow(fig):
    fig.patches.append(FancyArrowPatch(
        (0.45,0.05),(0.55,0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc"))
    fig.text(0.5,0.02,"Attacking Direction",ha="center",va="center",
             fontsize=9,color="#cccccc")


def draw_pass_map(df: pd.DataFrame, title: str):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_won = bool(row["is_won"]); is_prog = bool(row["is_progressive"])
        if not is_won:   color, alpha = COLOR_FAIL,         0.70
        elif is_prog:    color, alpha = COLOR_PROGRESSIVE,  0.86
        else:            color, alpha = COLOR_SUCCESS,       ALPHA_SUCCESS
        pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                     color=color,width=1.55,headwidth=2.25,headlength=2.25,
                     ax=ax,zorder=3,alpha=alpha)
        pitch.scatter(row.x_start,row.y_start,s=45,marker="o",color=color,
                      edgecolors="white",linewidths=0.8,ax=ax,zorder=6,alpha=alpha)
    ax.set_title(title, fontsize=11, color="#ffffff", pad=8)
    leg = ax.legend(handles=[
        Line2D([0],[0],color=COLOR_SUCCESS,    lw=2.5,label="Completed",  alpha=0.65),
        Line2D([0],[0],color=COLOR_PROGRESSIVE,lw=2.5,label="Progressive",alpha=0.90),
        Line2D([0],[0],color=COLOR_FAIL,       lw=2.5,label="Incomplete", alpha=0.90),
    ], loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small",
       labelspacing=0.5, borderpad=0.5)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.92)
    _attack_arrow(fig)
    return _save_fig(fig), ax, fig


def draw_advanced_pass_map(df: pd.DataFrame, title: str):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        ptype  = row["pass_type"]
        is_won = bool(row["is_won"])
        if ptype == "ball_progression":  color, alpha = COLOR_BPP,      0.82
        elif is_won:                     color, alpha = COLOR_LBP_WON,  0.82
        else:                            color, alpha = COLOR_LBP_LOST, 0.68
        pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                     color=color,width=1.65,headwidth=2.40,headlength=2.40,
                     ax=ax,zorder=3,alpha=alpha)
        pitch.scatter(row.x_start,row.y_start,s=50,marker="o",color=color,
                      edgecolors="white",linewidths=0.8,ax=ax,zorder=6,alpha=alpha)
    ax.set_title(title, fontsize=11, color="#ffffff", pad=8)
    leg = ax.legend(handles=[
        Line2D([0],[0],color=COLOR_LBP_WON, lw=2.5,label="Line Breaking – Completed",  alpha=0.85),
        Line2D([0],[0],color=COLOR_LBP_LOST,lw=2.5,label="Line Breaking – Incomplete", alpha=0.80),
        Line2D([0],[0],color=COLOR_BPP,     lw=2.5,label="Ball Progression Pass",       alpha=0.85),
    ], loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small",
       labelspacing=0.5, borderpad=0.5)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.92)
    _attack_arrow(fig)
    return _save_fig(fig), ax, fig


def draw_corridor_heatmap(df: pd.DataFrame, title: str = "Zone Heatmap — Completed Passes"):
    df_s   = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left":   (LANE_LEFT_MIN,  FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right":  (0.0,            LANE_RIGHT_MAX),
    }
    counts: dict = {}
    for cname,(y0,y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_,x1_ = x_bins[i],x_bins[i+1]
            mask = ((df_s["x_end"]>=x0_)&(df_s["x_end"]<x1_)
                    &(df_s["y_end"]>=y0)&(df_s["y_end"]<y1))
            arr[i] = int(mask.sum())
        counts[cname] = arr
    all_vals  = np.concatenate([counts[c] for c in counts])
    vmax      = max(1, int(all_vals.max()))
    cmap      = LinearSegmentedColormap.from_list(
        "wr",["#ffffff","#ffecec","#ffbfbf","#ff8080","#ff3b3b","#ff0000"])
    norm      = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax*0.35)
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W_HEAT, FIG_H_HEAT))
    fig.set_facecolor("#1a1a2e"); fig.set_dpi(FIG_DPI)
    for cname,(y0,y1) in corridors.items():
        for i in range(6):
            x0_,x1_ = x_bins[i],x_bins[i+1]
            value    = counts[cname][i]
            ax.add_patch(Rectangle((x0_,y0),x1_-x0_,y1-y0,
                                   facecolor=cmap(norm(value)),
                                   edgecolor=(1,1,1,0.12),lw=0.6,alpha=0.95,zorder=2))
            ax.text((x0_+x1_)/2,(y0+y1)/2,str(value),ha="center",va="center",
                    color="#000000" if value<=threshold else "#ffffff",
                    fontsize=11,fontweight="700" if value>=vmax*0.5 else "600",zorder=4)
    ax.set_title(title, fontsize=11, color="#ffffff", pad=8)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff",lw=0.5,alpha=0.15,linestyle="--",zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX,color="#ffffff",lw=0.5,alpha=0.15,linestyle="--",zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), ax, fig


def _top_zone_transitions(df_s: pd.DataFrame, top_k: int = 3):
    x_bins = np.linspace(0.0,FIELD_X,7)
    y_bins = np.array([0.0,LANE_RIGHT_MAX,LANE_LEFT_MIN,FIELD_Y])
    if df_s.empty: return [], x_bins, y_bins
    sx = np.clip(np.searchsorted(x_bins,df_s["x_start"].to_numpy(),side="right")-1,0,5)
    sy = np.clip(np.searchsorted(y_bins,df_s["y_start"].to_numpy(),side="right")-1,0,2)
    ex = np.clip(np.searchsorted(x_bins,df_s["x_end"].to_numpy(),  side="right")-1,0,5)
    ey = np.clip(np.searchsorted(y_bins,df_s["y_end"].to_numpy(),  side="right")-1,0,2)
    transitions: dict = defaultdict(int)
    for a,b,c,d in zip(sx,sy,ex,ey):
        if int(a)==int(c) and int(b)==int(d): continue
        transitions[(int(a),int(b),int(c),int(d))] += 1
    return sorted(transitions.items(),key=lambda kv:kv[1],reverse=True)[:top_k], x_bins, y_bins


def draw_top_connection_minimaps(df: pd.DataFrame, top_k: int = 3,
                                  title: str = "Top Zone Connections — Completed Passes"):
    df_s = df[df["is_won"]].copy()
    links, x_bins, y_bins = _top_zone_transitions(df_s, top_k=top_k)
    x_cent = (x_bins[:-1]+x_bins[1:])/2.0
    y_cent = (y_bins[:-1]+y_bins[1:])/2.0
    max_cnt = max([v for _,v in links],default=1) if links else 1
    fig, axes = plt.subplots(1,top_k,figsize=(FIG_W*1.65,FIG_H*0.82),dpi=FIG_DPI)
    if top_k == 1: axes = [axes]
    fig.set_facecolor("#1a1a2e")
    pitch = Pitch(pitch_type="statsbomb",pitch_color="#1a1a2e",
                  line_color="#ffffff",line_alpha=0.90)
    for idx, ax in enumerate(axes):
        pitch.draw(ax=ax)
        ax.axhline(y=LANE_LEFT_MIN, color="#ffffff",lw=0.4,alpha=0.12,linestyle="--")
        ax.axhline(y=LANE_RIGHT_MAX,color="#ffffff",lw=0.4,alpha=0.12,linestyle="--")
        if idx >= len(links):
            ax.set_title("—",fontsize=9,color="#dbeafe",pad=4); continue
        (ix0,iy0,ix1,iy1),cnt = links[idx]
        x0,y0 = float(x_cent[ix0]),float(y_cent[iy0])
        x1,y1 = float(x_cent[ix1]),float(y_cent[iy1])
        rel   = cnt/max_cnt; color = plt.cm.Blues(0.40+0.55*rel)
        ax.add_patch(Rectangle(
            (x_bins[ix0],y_bins[iy0]),x_bins[ix0+1]-x_bins[ix0],y_bins[iy0+1]-y_bins[iy0],
            facecolor=(0.20,0.45,0.95,0.18),edgecolor=(1,1,1,0.18),lw=0.6,zorder=2))
        ax.add_patch(Rectangle(
            (x_bins[ix1],y_bins[iy1]),x_bins[ix1+1]-x_bins[ix1],y_bins[iy1+1]-y_bins[iy1],
            facecolor=(0.02,0.70,0.55,0.18),edgecolor=(1,1,1,0.18),lw=0.6,zorder=2))
        if ix0==ix1 and iy0==iy1:
            ax.scatter([x0],[y0],s=40+80*rel,c=[color],marker="o",
                       edgecolors="white",linewidths=0.5,alpha=0.35+0.60*rel,zorder=5)
        else:
            rad = float(np.clip(0.10*np.sign((ix1-ix0)+0.4*(iy1-iy0)),-0.30,0.30))
            ax.add_patch(FancyArrowPatch(
                (x0,y0),(x1,y1),connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",mutation_scale=10+9*rel,
                lw=1.2+4.2*rel,color=color,alpha=0.35+0.60*rel,zorder=4))
        ax.text((x0+x1)/2,(y0+y1)/2,f"{cnt}",color="#e5efff",fontsize=9,
                ha="center",va="center",zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",fc=(0.06,0.09,0.14,0.80),ec="none"))
        ax.set_title(f"#{idx+1}  ·  {cnt}×",fontsize=9,color="#dbeafe",pad=4)
    fig.suptitle(title,fontsize=11,color="#ffffff",y=0.99)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf,format="png",dpi=FIG_DPI,facecolor=fig.get_facecolor(),bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf), axes, fig


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_passmap, tab_advanced = st.tabs(["📋 Pass Map", "🎯 Advanced Passes"])


# ── TAB 1: PASS MAP — comparação lado a lado ──────────────────────────────────
with tab_passmap:
    st.caption("Grey = Completed  ·  🔵 Blue = Progressive  ·  🔴 Red = Incomplete")

    col_maps, col_heats = st.columns(2, gap="large")

    # ── Coluna esquerda: Pass Maps ────────────────────────────────────────────
    with col_maps:
        DW = 680

        # Sacramento Pass Map
        st.markdown(f'<div class="match-title">Pass Map — Vs Sacramento United (25/04/2026)</div>',
                    unsafe_allow_html=True)
        df_sac = dfs_by_match[MATCH_SAC].copy()
        img_sac, ax_sac, fig_sac = draw_pass_map(df_sac, title="")
        st.image(img_sac, use_container_width=True)
        plt.close(fig_sac)

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

        # NYC Pass Map
        st.markdown(f'<div class="match-title">Pass Map — Vs New York City FC (25/11/2025)</div>',
                    unsafe_allow_html=True)
        df_nyc = dfs_by_match[MATCH_NYC].copy()
        img_nyc, ax_nyc, fig_nyc = draw_pass_map(df_nyc, title="")
        st.image(img_nyc, use_container_width=True)
        plt.close(fig_nyc)

    # ── Coluna direita: Heatmaps ──────────────────────────────────────────────
    with col_heats:

        # Sacramento Heatmap
        st.markdown(f'<div class="match-title">Heatmap — Vs Sacramento United (25/04/2026)</div>',
                    unsafe_allow_html=True)
        heat_sac, _, hfig_sac = draw_corridor_heatmap(df_sac, title="")
        st.image(heat_sac, use_container_width=True)
        plt.close(hfig_sac)

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

        # NYC Heatmap
        st.markdown(f'<div class="match-title">Heatmap — Vs New York City FC (25/11/2025)</div>',
                    unsafe_allow_html=True)
        heat_nyc, _, hfig_nyc = draw_corridor_heatmap(df_nyc, title="")
        st.image(heat_nyc, use_container_width=True)
        plt.close(hfig_nyc)


# ── TAB 2: ADVANCED PASSES ────────────────────────────────────────────────────
with tab_advanced:
    st.caption("Line Breaking Passes (🟡 yellow) and Ball Progression Passes (🟣 purple).")
    sp_col_f, sp_col_field, sp_col_stats = st.columns([0.9, 2, 1], gap="large")

    with sp_col_f:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown("### 📍 Position")
        sp_pos = st.radio("Filter by position",["All Positions","LCB","RCB"],
                          index=0,key="sp_pos")
        st.markdown("<div style='font-size:11px;color:#94a3b8;margin-top:-6px;margin-bottom:4px;'>"
                    "LCB: Sacramento United (25/04/2026), New York City FC (25/11/2025)</div>",
                    unsafe_allow_html=True)
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

        if sp_pos == "All Positions": sp_avail = list(sp_dfs_by_match.keys())
        else: sp_avail = [m for m,p in POSITION_BY_MATCH.items() if p==sp_pos]

        sp_pos_all = (pd.concat([sp_dfs_by_match[m] for m in sp_avail],ignore_index=True)
                      if sp_avail else sp_df_all.iloc[0:0])
        pm_pos_all = (pd.concat([dfs_by_match[m] for m in sp_avail],ignore_index=True)
                      if sp_avail else df_all.iloc[0:0])

        sp_pos_full: dict = {"All Matches": sp_pos_all}
        sp_pos_full.update({m: sp_dfs_by_match[m] for m in sp_avail})
        pm_pos_full: dict = {"All Matches": pm_pos_all}
        pm_pos_full.update({m: dfs_by_match[m] for m in sp_avail})

        st.markdown("### 🏟️ Match")
        sp_match = st.selectbox("Choose the match",list(sp_pos_full.keys()),
                                index=0,key="sp_match")
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
        st.markdown("### 🎯 Pass Type")
        sp_filter = st.radio("Show passes",
                             ["All","Line Breaking Only","Ball Progression Only"],
                             index=0,key="sp_filter")
        st.markdown('</div>', unsafe_allow_html=True)

    with sp_col_field:
        sp_df_base = sp_pos_full[sp_match].copy()
        if sp_filter == "Line Breaking Only":
            sp_df_base = sp_df_base[sp_df_base["pass_type"]=="line_breaking"].reset_index(drop=True)
        elif sp_filter == "Ball Progression Only":
            sp_df_base = sp_df_base[sp_df_base["pass_type"]=="ball_progression"].reset_index(drop=True)
        else:
            sp_df_base = sp_df_base.reset_index(drop=True)

        SP_DW = 780

        st.markdown('<h4 style="color:#ffffff;margin:0 0 6px 0;">Advanced Passes Map</h4>',
                    unsafe_allow_html=True)
        sp_img,sp_ax,sp_fig = draw_advanced_pass_map(
            sp_df_base, title=f"Advanced Passes — {sp_match}")
        sp_click = streamlit_image_coordinates(sp_img,width=SP_DW,key="sp_map")

        sp_selected = None
        if sp_click is not None:
            rw,rh = sp_img.size
            px = sp_click["x"]*(rw/sp_click["width"])
            py = sp_click["y"]*(rh/sp_click["height"])
            fx,fy = sp_ax.transData.inverted().transform((px,rh-py))
            df_sel2 = sp_df_base.copy()
            df_sel2["_dist"] = np.sqrt((df_sel2.x_start-fx)**2+(df_sel2.y_start-fy)**2)
            cands2 = df_sel2[df_sel2["_dist"]<5.0].sort_values("_dist")
            if not cands2.empty: sp_selected = cands2.iloc[0]
        plt.close(sp_fig)

        st.markdown('<h4 style="color:#ffffff;margin:14px 0 4px 0;">Zone Heatmap — Completed</h4>',
                    unsafe_allow_html=True)
        sp_heat_img,_,sp_hfig = draw_corridor_heatmap(
            sp_df_base,title="Zone Heatmap — Advanced Passes Completed")
        st.image(sp_heat_img,use_container_width=True); plt.close(sp_hfig)

        st.divider(); st.subheader("Selected Event")
        if sp_selected is None:
            st.info("Click an origin dot on the map to inspect an event.")
        else:
            ptype_label = ("Line Breaking Pass"
                           if sp_selected["pass_type"]=="line_breaking"
                           else "Ball Progression Pass")
            status = "✅ Completed" if sp_selected["is_won"] else "❌ Incomplete"
            st.success(f"Pass #{int(sp_selected['number'])} — {ptype_label} | {status}")
            c1,c2 = st.columns(2)
            c1.write(f"**Origin:** ({sp_selected.x_start:.2f}, {sp_selected.y_start:.2f})")
            c2.write(f"**Destination:** ({sp_selected.x_end:.2f}, {sp_selected.y_end:.2f})")
            st.metric("Pass Distance",f"{sp_selected.pass_distance:.1f} m")

        with st.expander("📊 Full Data Table"):
            dc2 = ["number","type","pass_type","outcome",
                   "x_start","y_start","x_end","y_end","pass_distance"]
            st.dataframe(sp_df_base[dc2].style.format(
                {"x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}",
                 "y_end":"{:.2f}","pass_distance":"{:.1f}"}),
                use_container_width=True,height=320)

    with sp_col_stats:
        total_pm = len(pm_pos_full[sp_match])
        ss = compute_advanced_stats(sp_df_base, total_pm)

        with st.expander("🟡 Line Breaking Passes", expanded=True):
            st.markdown('<div class="stats-section-title">Line Breaking Passes</div>',
                        unsafe_allow_html=True)
            b1,b2,b3 = st.columns(3)
            with b1: small_metric("Total",      f"{ss['lbp_total']}")
            with b2: small_metric("Completed",  f"{ss['lbp_completed']}")
            with b3: small_metric("Incomplete", f"{ss['lbp_incomplete']}")
            st.markdown("<hr style='margin:6px 0 8px 0;'>",unsafe_allow_html=True)
            ba1,ba2 = st.columns(2)
            with ba1: small_metric("Accuracy", f"{ss['lbp_accuracy']:.1f}%")
            with ba2: small_metric("Tendency", f"{ss['lbp_tendency']:.1f}%",
                                   delta=f"{ss['lbp_total']} of {total_pm} total passes")

        with st.expander("🟣 Ball Progression Passes", expanded=True):
            st.markdown('<div class="stats-section-title">Ball Progression Passes</div>',
                        unsafe_allow_html=True)
            p1,p2 = st.columns(2)
            with p1: small_metric("Total",    f"{ss['bpp_total']}")
            with p2: small_metric("Tendency", f"{ss['bpp_tendency']:.1f}%",
                                  delta=f"{ss['bpp_total']} of {total_pm} total passes")

        st.divider()
        st.caption("🟡 Yellow = Line Breaking  ·  🟣 Purple = Ball Progression  ·  🔴 Red = Incomplete")
