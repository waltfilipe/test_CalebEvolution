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
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import math

st.set_page_config(layout="wide", page_title="Caleb Simmons - Pass Evolution - Season 25/26")

st.markdown("""
<style>
.row-label{
  font-size:12px;font-weight:700;color:#c8d6e5;letter-spacing:.3px;
  margin-bottom:4px;margin-top:8px;padding:4px 8px;
  background:rgba(255,255,255,.04);border-radius:4px;}
.row-label-blue  {border-left:3px solid #2F80ED;}
.row-label-green {border-left:3px solid #10b981;}
.row-label-amber {border-left:3px solid #f59e0b;}
.section-header{
  font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;
  color:#94a3b8;margin:10px 0 6px 0;padding-bottom:4px;
  border-bottom:1px solid rgba(255,255,255,.07);}
.cmp-box{
  background:rgba(255,255,255,.04);border-radius:10px;
  padding:9px 11px;margin-bottom:7px;}
.cmp-label{font-size:9px;color:#94a3b8;text-transform:uppercase;
  letter-spacing:.6px;font-weight:600;margin-bottom:6px;}
.cmp-row{display:flex;justify-content:space-between;align-items:flex-end;gap:4px;}
.cmp-cell{flex:1;}
.cc-tag{font-size:8px;font-weight:700;margin-bottom:1px;}
.cc-val{font-size:18px;font-weight:700;color:#f1f5f9;line-height:1.1;}
.cc-sub{font-size:9px;color:#64748b;margin-top:2px;}
.cmp-sep{width:1px;background:rgba(255,255,255,.08);height:32px;flex-shrink:0;align-self:center;}
.row-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:10px 0 6px 0;}
.streamlit-expanderHeader{color:#ffffff!important;}
.filter-panel{
  background:linear-gradient(168deg,rgba(30,39,56,.92) 0%,rgba(22,28,40,.97) 100%);
  border:1px solid rgba(255,255,255,.08);border-radius:14px;
  padding:20px 14px 16px 14px;
  box-shadow:0 4px 24px rgba(0,0,0,.25);backdrop-filter:blur(6px);}
.filter-panel h3{font-size:14px;color:#c8d6e5;letter-spacing:.5px;margin-bottom:8px;}
.filter-panel .filter-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:12px 0;}
.small-metric{padding:5px 6px;}
.small-metric .label{font-size:11px;color:#ffffff;margin-bottom:2px;opacity:.9;}
.small-metric .value{font-size:17px;font-weight:600;color:#ffffff;}
.small-metric .delta{font-size:10px;color:#e6e6e6;margin-top:3px;}
.stats-section-title{font-size:13px;font-weight:600;margin-bottom:6px;color:#ffffff;}
</style>
""", unsafe_allow_html=True)

st.title("Caleb Simmons — Pass Evolution — Season 25/26")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FIELD_X, FIELD_Y   = 120.0, 80.0
HALF_LINE_X        = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN      = 53.33
LANE_RIGHT_MAX     = 26.67
LATERAL_MIN_DIST   = 12.0
NX_XT, NY_XT       = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60

FIG_W, FIG_H = 7.0, 4.7
FIG_DPI      = 180

COLOR_SUCCESS     = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL        = "#E07070"
ALPHA_SUCCESS     = 0.07

COLOR_LBP_WON  = "#F59E0B"
COLOR_LBP_LOST = "#E07070"
COLOR_BPP      = "#8B5CF6"

CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a","#f97316","#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)

CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "density",
    ["#0f172a","#1e1b4b","#312e81","#1d4ed8","#0ea5e9",
     "#2dd4bf","#ca8a04","#f97316","#fcd34d","#fef08a"]
)

MATCH_SAC = "Vs Sacramento United (25/04/2026)"
MATCH_NYC = "Vs New York City FC (25/11/2025)"
POSITION_BY_MATCH = {MATCH_SAC: "LCB", MATCH_NYC: "LCB"}

NYC_COLOR = "#f87171"
SAC_COLOR = "#60a5fa"

# ─────────────────────────────────────────────────────────────────────────────
# xT grid
# ─────────────────────────────────────────────────────────────────────────────
def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24,
    goal_width=11.0, penalty_depth=18.5, penalty_width=45.32,
    prox_w=0.50, central_w=0.50,
    internal_prox_power=2.8, internal_central_power=2.4, center_boost=0.20,
    FUNNEL_INFLUENCE_RANGE=35.0, FUNNEL_POWER=1.3, BASE_BOOST_WEIGHT=0.15,
    band_width_m=180.0, blur_window_m=60.0, final_blur_m=12.0,
    ANGLE_WEIGHT=0.50, ANGLE_POWER=1.4, BASE_ANGLE_WEIGHT=0.40):

    ncols_hr = NX*sub; nrows_hr = NY*sub
    xe = np.linspace(0,FIELD_X,ncols_hr+1); ye = np.linspace(0,FIELD_Y,nrows_hr+1)
    xc = (xe[:-1]+xe[1:])/2; yc_arr = (ye[:-1]+ye[1:])/2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01+(Xc/FIELD_X)*0.99; yc = 1.0-np.abs((Yc/FIELD_Y)-0.5)*2.0
    BASE = xp*(0.8+0.2*yc); BASE=(BASE-BASE.min())/(BASE.max()-BASE.min()+1e-12)
    cy = FIELD_Y/2.0
    fv = [(FIELD_X,cy-goal_width/2),(FIELD_X-penalty_depth,cy-penalty_width/2),
          (FIELD_X-penalty_depth,cy+penalty_width/2),(FIELD_X,cy+goal_width/2)]
    bpts = []
    for i in range(len(fv)):
        a,b=fv[i],fv[(i+1)%len(fv)]; dx,dy=b[0]-a[0],b[1]-a[1]
        n=max(2,int(round(math.hypot(dx,dy)/0.5)))
        for t in np.linspace(0,1,n,endpoint=False): bpts.append((a[0]+dx*t,a[1]+dy*t))
    bpts = np.array(bpts)
    fX=Xc.ravel(); fY=Yc.ravel(); md2=np.full(fX.size,np.inf)
    for bp in bpts: np.minimum(md2,(fX-bp[0])**2+(fY-bp[1])**2,out=md2)
    adist=np.sqrt(md2).reshape(Xc.shape)
    infl=np.clip((1-np.clip(adist/FUNNEL_INFLUENCE_RANGE,0,1))**FUNNEL_POWER,0,1)
    D=np.hypot(FIELD_X-Xc,cy-Yc)
    prox=1-np.clip(D/np.hypot(FIELD_X,FIELD_Y/2),0,1)
    cent=1-np.clip(np.abs((Yc-cy)/cy),0,1)
    ub=np.clip((prox_w*np.clip(prox**internal_prox_power,0,1)+
                central_w*np.clip(cent**internal_central_power,0,1))*(1+center_boost*prox),0,1)
    v1x=FIELD_X-Xc; v1y=(cy+goal_width/2)-Yc; v2x=FIELD_X-Xc; v2y=(cy-goal_width/2)-Yc
    ca=np.clip((v1x*v2x+v1y*v2y)/(np.hypot(v1x,v1y)*np.hypot(v2x,v2y)+1e-12),-1,1)
    ang=np.arccos(ca); af=np.clip((ang/(ang.max()+1e-12))**ANGLE_POWER,0,1)
    ub=np.clip(ub*((1-ANGLE_WEIGHT)+ANGLE_WEIGHT*af),0,1)
    Bc=BASE*((1-BASE_ANGLE_WEIGHT)+BASE_ANGLE_WEIGHT*af)
    Bc=(Bc-Bc.min())/(Bc.max()-Bc.min()+1e-12); XTB=Bc+infl*BASE_BOOST_WEIGHT*ub
    pw=FIELD_X/ncols_hr; ph=FIELD_Y/nrows_hr
    rx=max(1,int(round((blur_window_m/pw)/2))); ry=max(1,int(round((blur_window_m/ph)/2)))
    def blur(a,rx,ry):
        H,W=a.shape; p=np.pad(a,((ry,ry),(rx,rx)),mode='edge').astype(np.float64)
        ii=p.cumsum(0).cumsum(1); s=ii[2*ry:2*ry+H,2*rx:2*rx+W].copy()
        s+=ii[:H,:W]; s-=ii[:H,2*rx:2*rx+W]; s-=ii[2*ry:2*ry+H,:W]
        return s/((2*ry+1)*(2*rx+1))
    w=0.5*(1-np.cos(np.pi*np.clip(adist/band_width_m,0,1)))
    XTbl=w*XTB+(1-w)*blur(XTB,rx,ry)
    rf=max(1,int(round((final_blur_m/pw)/2))); rfy=max(1,int(round((final_blur_m/ph)/2)))
    XT=0.85*XTbl+0.15*blur(XTbl,rf,rfy); XT=(XT-XT.min())/(XT.max()-XT.min()+1e-12)
    XTc=np.zeros((NY,NX))
    for iy in range(NY):
        for ix in range(NX): XTc[iy,ix]=XT[iy*sub:(iy+1)*sub,ix*sub:(ix+1)*sub].mean()
    XTc=(XTc-XTc.min())/(XTc.max()-XTc.min()+1e-12)
    return XTc, XT

XT_GRID, _ = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x/FIELD_X)*NX_XT, 0, NX_XT-1))
    iy = int(np.clip((y/FIELD_Y)*NY_XT, 0, NY_XT-1))
    return float(XT_GRID[iy, ix])

# ─────────────────────────────────────────────────────────────────────────────
# Match data
# ─────────────────────────────────────────────────────────────────────────────
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
        ("PASS WON",10.13,27.70,27.92,1.93,"strong"),("PASS WON",16.95,28.36,16.78,51.64,"strong"),
        ("PASS WON",31.08,36.68,10.79,35.18,"strong"),("PASS WON",53.52,12.57,66.82,17.72,"strong"),
        ("PASS WON",66.82,7.75,55.68,39.83,"strong"),("PASS WON",74.79,44.99,82.94,7.42,"strong"),
        ("PASS WON",78.45,8.75,63.16,26.87,"strong"),("PASS WON",67.15,21.22,74.30,21.55,"strong"),
        ("PASS WON",69.64,11.24,55.51,36.68,"strong"),("PASS WON",62.16,53.63,64.49,34.18,"strong"),
        ("PASS WON",28.58,39.17,28.75,52.80,"strong"),("PASS WON",31.24,22.05,10.79,36.01,"strong"),
        ("PASS WON",36.73,15.56,42.71,0.77,"strong"),("PASS WON",27.09,23.54,37.39,3.10,"strong"),
        ("PASS WON",31.57,12.41,22.93,39.50,"strong"),("PASS WON",27.75,18.06,12.12,39.50,"strong"),
        ("PASS WON",35.56,4.09,14.12,31.02,"strong"),("PASS WON",10.30,25.37,25.59,2.93,"strong"),
        ("PASS WON",11.13,28.36,28.75,5.26,"strong"),("PASS WON",13.95,28.36,25.09,4.43,"strong"),
        ("PASS WON",36.23,12.74,11.96,29.19,"strong"),("PASS WON",40.72,22.71,13.79,36.68,"strong"),
        ("PASS WON",32.41,26.54,43.88,26.37,"strong"),("PASS WON",27.75,40.50,27.58,54.46,"strong"),
        ("PASS WON",34.40,46.82,48.20,62.94,"strong"),("PASS WON",53.35,53.46,53.52,62.77,"strong"),
        ("PASS WON",49.19,10.58,40.88,40.33,"strong"),("PASS WON",60.00,7.25,54.51,38.01,"strong"),
        ("PASS WON",58.67,26.87,80.78,24.54,"strong"),("PASS WON",59.17,30.19,74.96,8.75,"strong"),
        ("PASS WON",76.29,24.54,85.10,3.76,"strong"),
        ("PASS LOST",41.43,53.55,59.07,48.54,"strong"),("PASS LOST",10.24,29.97,49.79,23.85,"strong"),
        ("PASS LOST",33.08,25.15,65.76,2.13,"strong"),("PASS LOST",57.40,18.65,71.70,0.46,"strong"),
        ("PASS LOST",63.90,24.03,92.30,27.19,"strong"),
        ("PASS WON",18.61,23.38,3.15,43.16,"weak"),("PASS WON",32.24,22.21,26.75,46.48,"weak"),
        ("PASS WON",46.37,16.06,39.72,44.49,"weak"),("PASS WON",52.35,12.57,44.71,44.16,"weak"),
        ("PASS WON",55.18,12.07,42.71,40.83,"weak"),("PASS WON",55.51,2.76,73.96,3.10,"weak"),
        ("PASS WON",56.18,19.89,55.35,44.65,"weak"),("PASS WON",58.50,29.03,51.19,45.15,"weak"),
        ("PASS WON",56.01,24.87,66.48,40.33,"weak"),("PASS WON",61.16,36.84,73.63,27.20,"weak"),
        ("PASS WON",66.65,16.73,57.34,40.00,"weak"),("PASS WON",79.45,4.26,64.65,26.20,"weak"),
    ],
}

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
        ("BPP WON",66.82,7.75,55.68,39.83),
        ("LBP WON",58.67,26.87,80.78,24.54),
        ("BPP WON",59.17,30.19,74.96,8.75),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def classify_pass_direction(x_start, y_start, x_end, y_end) -> str:
    dx = x_end - x_start; dy = y_end - y_start
    dist = np.sqrt(dx**2 + dy**2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:  return "forward"
    if angle_deg >= 135.0: return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"

def progressive_pass(x_start: float, x_end: float) -> bool:
    dist_start = FIELD_X - x_start; dist_end = FIELD_X - x_end
    closer_by = dist_start - dist_end
    start_own = x_start < HALF_LINE_X; end_own = x_end < HALF_LINE_X
    if start_own and end_own:  return closer_by >= 30.0
    if start_own != end_own:   return closer_by >= 15.0
    return closer_by >= 10.0

# ─────────────────────────────────────────────────────────────────────────────
# Build DataFrames
# ─────────────────────────────────────────────────────────────────────────────
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
    dfm["xt_start"]     = dfm.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    dfm["xt_end"]       = dfm.apply(lambda r: xt_value(r.x_end,   r.y_end),   axis=1)
    dfm["delta_xt"]     = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"]   = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"],
                                   dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

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

df_all    = pd.concat(dfs_by_match.values(),    ignore_index=True)
sp_df_all = pd.concat(sp_dfs_by_match.values(), ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {k: 0 for k in [
            "total","completed","incomplete","accuracy",
            "prog_total","prog_completed","prog_pct",
            "fwd","fwd_pct","bwd","bwd_pct","lat","lat_pct",
            "pos_pct","high_xt_pct","sum_dxt"]}
    completed = int(df["is_won"].sum())
    prog_t    = int(df["is_progressive"].sum())
    prog_c    = int((df["is_progressive"] & df["is_won"]).sum())
    fwd       = int(df["is_forward"].sum())
    bwd       = int(df["is_backward"].sum())
    lat       = int(df["is_lateral"].sum())
    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    high_xt   = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt   = float(df.loc[df["is_won"], "delta_xt_adj"].sum())
    return {
        "total":      total,
        "completed":  completed,
        "incomplete": total - completed,
        "accuracy":   round(completed / total * 100, 1),
        "prog_total":     prog_t,
        "prog_completed": prog_c,
        "prog_pct":       round(prog_t / total * 100, 1),
        "fwd":  fwd, "fwd_pct": round(fwd / total * 100, 1),
        "bwd":  bwd, "bwd_pct": round(bwd / total * 100, 1),
        "lat":  lat, "lat_pct": round(lat / total * 100, 1),
        "pos_pct":     round(pos_count / total * 100, 1),
        "high_xt_pct": round(high_xt   / total * 100, 1),
        "sum_dxt":     round(sum_dxt, 3),
    }

def compute_advanced_stats(sp_df: pd.DataFrame, total_passes: int) -> dict:
    lbp   = sp_df[sp_df["pass_type"] == "line_breaking"]
    bpp   = sp_df[sp_df["pass_type"] == "ball_progression"]
    lbp_t = len(lbp); lbp_c = int(lbp["is_won"].sum()); bpp_t = len(bpp)
    ref   = max(total_passes, 1)
    return {
        "lbp_total": lbp_t, "lbp_completed": lbp_c,
        "lbp_incomplete": lbp_t - lbp_c,
        "lbp_accuracy": round(lbp_c / lbp_t * 100, 1) if lbp_t else 0,
        "lbp_tendency": round(lbp_t / ref * 100, 1),
        "bpp_total": bpp_t,
        "bpp_tendency": round(bpp_t / ref * 100, 1),
    }

def small_metric(label: str, value: str, delta: str | None = None):
    html = (f'<div class="small-metric"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>')
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def cmp_box(label: str,
            val_nyc, val_sac,
            sub_nyc: str = "", sub_sac: str = "",
            border: str = "#3b82f6"):
    """Render a side-by-side comparison card. NYC (left) vs SAC (right)."""
    sub_nyc_html = f'<div class="cc-sub">{sub_nyc}</div>' if sub_nyc else ""
    sub_sac_html = f'<div class="cc-sub">{sub_sac}</div>' if sub_sac else ""
    html = (
        f'<div class="cmp-box" style="border-left:3px solid {border};">'
        f'<div class="cmp-label">{label}</div>'
        f'<div class="cmp-row">'
        f'<div class="cmp-cell">'
        f'<div class="cc-tag" style="color:{NYC_COLOR};">NYC</div>'
        f'<div class="cc-val">{val_nyc}</div>'
        f'{sub_nyc_html}'
        f'</div>'
        f'<div class="cmp-sep"></div>'
        f'<div class="cmp-cell">'
        f'<div class="cc-tag" style="color:{SAC_COLOR};">SAC</div>'
        f'<div class="cc-val">{val_sac}</div>'
        f'{sub_sac_html}'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def sec_hdr(label: str):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def row_divider():
    st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

def row_label(text: str, cls: str = "row-label-blue"):
    st.markdown(f'<div class="row-label {cls}">{text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg,
                  line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg); fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X,        color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig, has_cbar=False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(FancyArrowPatch(
        (0.44+ox, 0.045), (0.56+ox, 0.045), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11, linewidth=1.6, color="#aaaaaa"))
    fig.text(0.50+ox, 0.012, "Attacking Direction", ha="center", va="bottom",
             transform=fig.transFigure, fontsize=7.5, color="#aaaaaa")

def _save_fig(fig) -> Image.Image:
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df: pd.DataFrame):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_won = bool(row["is_won"]); is_prog = bool(row["is_progressive"])
        if not is_won:   color, alpha = COLOR_FAIL,         0.72
        elif is_prog:    color, alpha = COLOR_PROGRESSIVE,  0.88
        else:            color, alpha = COLOR_SUCCESS,       ALPHA_SUCCESS
        pitch.arrows(row.x_start, row.y_start, row.x_end, row.y_end,
                     color=color, width=1.3, headwidth=2.0, headlength=2.0,
                     ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row.x_start, row.y_start, s=32, marker="o", color=color,
                      edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(handles=[
        Line2D([0],[0], color=COLOR_SUCCESS,     lw=2.0, label="Completed",   alpha=0.65),
        Line2D([0],[0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
        Line2D([0],[0], color=COLOR_FAIL,        lw=2.0, label="Incomplete",  alpha=0.90),
    ], loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5,
       labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_corridor_heatmap(df: pd.DataFrame):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left":   (LANE_LEFT_MIN,  FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right":  (0.0,            LANE_RIGHT_MAX),
    }
    counts: dict = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i+1]
            arr[i] = int(((df_s["x_end"]>=x0_)&(df_s["x_end"]<x1_)
                          &(df_s["y_end"]>=y0)&(df_s["y_end"]<y1)).sum())
        counts[cname] = arr
    all_vals  = np.concatenate([counts[c] for c in counts])
    vmax      = max(1, int(all_vals.max()))
    cmap      = LinearSegmentedColormap.from_list(
        "wr", ["#ffffff","#ffecec","#ffbfbf","#ff8080","#ff3b3b","#ff0000"])
    norm      = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i+1]
            value = counts[cname][i]
            ax.add_patch(Rectangle((x0_,y0), x1_-x0_, y1-y0,
                                   facecolor=cmap(norm(value)),
                                   edgecolor=(1,1,1,0.12), lw=0.5, alpha=0.95, zorder=2))
            ax.text((x0_+x1_)/2, (y0+y1)/2, str(value), ha="center", va="center",
                    color="#000000" if value<=threshold else "#ffffff",
                    fontsize=9, fontweight="700" if value>=vmax*0.5 else "600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN,  color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12; ts = np.linspace(0.0, 1.0, segs+1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i+1]
        xa = x0+(x1-x0)*t0; ya = y0+(y1-y0)*t0
        xb = x0+(x1-x0)*t1; yb = y0+(y1-y0)*t1
        alpha = 0.85 * (0.15 + 0.85*t1)
        lw    = 2.5  * (0.80 + 0.20*t1)
        ax.plot([xa,xb],[ya,yb], color=color, linewidth=lw, alpha=alpha,
                zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color,
               linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white",
               linewidths=0.9, zorder=6, alpha=0.85)

def draw_top10_xt_map(df: pd.DataFrame):
    fig, ax, pitch = _base_pitch()
    top10 = (
        df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
        .sort_values("delta_xt_adj", ascending=False)
        .head(10).copy().reset_index(drop=True)
    )
    if not top10.empty:
        for _, row in top10.iterrows():
            val   = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(ax,
                              float(row.x_start), float(row.y_start),
                              float(row.x_end),   float(row.y_end), color)
    sm   = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
    cbar.set_label("ΔxT", color="#ffffff", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

def draw_advanced_pass_map(df: pd.DataFrame, title: str):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        ptype  = row["pass_type"]; is_won = bool(row["is_won"])
        if ptype == "ball_progression": color, alpha = COLOR_BPP,      0.82
        elif is_won:                    color, alpha = COLOR_LBP_WON,  0.82
        else:                           color, alpha = COLOR_LBP_LOST, 0.68
        pitch.arrows(row.x_start, row.y_start, row.x_end, row.y_end,
                     color=color, width=1.4, headwidth=2.1, headlength=2.1,
                     ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row.x_start, row.y_start, s=36, marker="o", color=color,
                      edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=alpha)
    ax.set_title(title, fontsize=9, color="#ffffff", pad=5)
    leg = ax.legend(handles=[
        Line2D([0],[0], color=COLOR_LBP_WON,  lw=2.0, label="Line Breaking – Completed",  alpha=0.85),
        Line2D([0],[0], color=COLOR_LBP_LOST,  lw=2.0, label="Line Breaking – Incomplete", alpha=0.80),
        Line2D([0],[0], color=COLOR_BPP,       lw=2.0, label="Ball Progression Pass",      alpha=0.85),
    ], loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5,
       labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), ax, fig

# ─────────────────────────────────────────────────────────────────────────────
# Pre-render images
# ─────────────────────────────────────────────────────────────────────────────
df_sac = dfs_by_match[MATCH_SAC].copy()
df_nyc = dfs_by_match[MATCH_NYC].copy()

img_pm_nyc,  fig_pm_nyc  = draw_pass_map(df_nyc);          plt.close(fig_pm_nyc)
img_pm_sac,  fig_pm_sac  = draw_pass_map(df_sac);          plt.close(fig_pm_sac)
img_ht_nyc,  fig_ht_nyc  = draw_corridor_heatmap(df_nyc);  plt.close(fig_ht_nyc)
img_ht_sac,  fig_ht_sac  = draw_corridor_heatmap(df_sac);  plt.close(fig_ht_sac)
img_xt_nyc,  fig_xt_nyc  = draw_top10_xt_map(df_nyc);      plt.close(fig_xt_nyc)
img_xt_sac,  fig_xt_sac  = draw_top10_xt_map(df_sac);      plt.close(fig_xt_sac)

s_sac = compute_stats(df_sac)
s_nyc = compute_stats(df_nyc)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_passmap, tab_advanced = st.tabs(["📋 Pass Map", "🎯 Advanced Passes"])

# ────────────────────────────────────────────────���────────────────────────────
# TAB 1  —  NYC (left) | SAC (right) | Stats (right)
# ─────────────────────────────────────────────────────────────────────────────
with tab_passmap:
    col_nyc, col_sac, col_stats = st.columns([1, 1, 1], gap="medium")

    # ── ROW 1: Pass Maps ─────────────────────────────────────────────────────
    with col_nyc:
        row_label("🟦 Pass Map · New York City FC · 25/11/2025", "row-label-blue")
        st.image(img_pm_nyc, use_container_width=True)

    with col_sac:
        row_label("🟦 Pass Map · Sacramento United · 25/04/2026", "row-label-blue")
        st.image(img_pm_sac, use_container_width=True)

    with col_stats:
        sec_hdr("📋 Pass Overview")
        cmp_box("Total Passes",
                s_nyc["total"], s_sac["total"],
                border="#3b82f6")
        cmp_box("Completed",
                f"{s_nyc['completed']} ({s_nyc['accuracy']:.0f}%)",
                f"{s_sac['completed']} ({s_sac['accuracy']:.0f}%)",
                sub_nyc=f"{s_nyc['incomplete']} incomplete",
                sub_sac=f"{s_sac['incomplete']} incomplete",
                border="#10b981")
        cmp_box("Progressive Passes",
                f"{s_nyc['prog_total']} ({s_nyc['prog_pct']:.0f}%)",
                f"{s_sac['prog_total']} ({s_sac['prog_pct']:.0f}%)",
                sub_nyc=f"{s_nyc['prog_completed']} completed",
                sub_sac=f"{s_sac['prog_completed']} completed",
                border="#2F80ED")

    # ── Divider ───────────────────────────────────────────────────────────────
    with col_nyc:   row_divider()
    with col_sac:   row_divider()
    with col_stats: row_divider()

    # ── ROW 2: Zone Heatmaps ─────────────────────────────────────────────────
    with col_nyc:
        row_label("🟩 Zone Heatmap · New York City FC", "row-label-green")
        st.image(img_ht_nyc, use_container_width=True)

    with col_sac:
        row_label("🟩 Zone Heatmap · Sacramento United", "row-label-green")
        st.image(img_ht_sac, use_container_width=True)

    with col_stats:
        sec_hdr("🧭 Pass Direction")
        cmp_box("⬆️ Forward",
                f"{s_nyc['fwd']} ({s_nyc['fwd_pct']:.0f}%)",
                f"{s_sac['fwd']} ({s_sac['fwd_pct']:.0f}%)",
                border="#10b981")
        cmp_box("⬇️ Backward",
                f"{s_nyc['bwd']} ({s_nyc['bwd_pct']:.0f}%)",
                f"{s_sac['bwd']} ({s_sac['bwd_pct']:.0f}%)",
                border="#f59e0b")
        cmp_box("↔️ Lateral",
                f"{s_nyc['lat']} ({s_nyc['lat_pct']:.0f}%)",
                f"{s_sac['lat']} ({s_sac['lat_pct']:.0f}%)",
                border="#8b5cf6")

    # ── Divider ───────────────────────────────────────────────────────────────
    with col_nyc:   row_divider()
    with col_sac:   row_divider()
    with col_stats: row_divider()

    # ── ROW 3: Top-10 xT Maps ────────────────────────────────────────────────
    with col_nyc:
        row_label("🟡 Top 10 ΔxT · New York City FC", "row-label-amber")
        st.image(img_xt_nyc, use_container_width=True)

    with col_sac:
        row_label("🟡 Top 10 ΔxT · Sacramento United", "row-label-amber")
        st.image(img_xt_sac, use_container_width=True)

    with col_stats:
        sec_hdr("⚡ xT Analysis")
        cmp_box("% Positive ΔxT",
                f"{s_nyc['pos_pct']:.1f}%",
                f"{s_sac['pos_pct']:.1f}%",
                sub_nyc="passes that gained xT",
                sub_sac="passes that gained xT",
                border="#f59e0b")
        cmp_box("% ΔxT > 0.1",
                f"{s_nyc['high_xt_pct']:.1f}%",
                f"{s_sac['high_xt_pct']:.1f}%",
                sub_nyc="high-threat passes",
                sub_sac="high-threat passes",
                border="#f97316")
        cmp_box("Σ ΔxT",
                f"{s_nyc['sum_dxt']:.3f}",
                f"{s_sac['sum_dxt']:.3f}",
                sub_nyc="total xT generated",
                sub_sac="total xT generated",
                border="#b91c1c")

    st.caption(
        "Grey = Completed  ·  🔵 Progressive  ·  🔴 Incomplete  ·  "
        "White dashed = Final Third  ·  🟡 Comet = Top-10 ΔxT passes"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Advanced Passes
# ─────────────────────────────────────────────────────────────────────────────
with tab_advanced:
    st.caption("Line Breaking Passes (🟡 yellow) and Ball Progression Passes (🟣 purple).")
    sp_col_f, sp_col_field, sp_col_stats = st.columns([0.9, 2, 1], gap="large")

    with sp_col_f:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown("### 📍 Position")
        sp_pos = st.radio("Filter by position", ["All Positions","LCB","RCB"],
                          index=0, key="sp_pos")
        st.markdown("<div style='font-size:11px;color:#94a3b8;margin-top:-6px;margin-bottom:4px;'>"
                    "LCB: Sacramento (25/04/2026), NYC (25/11/2025)</div>",
                    unsafe_allow_html=True)
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

        if sp_pos == "All Positions": sp_avail = list(sp_dfs_by_match.keys())
        else: sp_avail = [m for m,p in POSITION_BY_MATCH.items() if p==sp_pos]

        sp_pos_all = (pd.concat([sp_dfs_by_match[m] for m in sp_avail], ignore_index=True)
                      if sp_avail else sp_df_all.iloc[0:0])
        pm_pos_all = (pd.concat([dfs_by_match[m]    for m in sp_avail], ignore_index=True)
                      if sp_avail else df_all.iloc[0:0])

        sp_pos_full: dict = {"All Matches": sp_pos_all}
        sp_pos_full.update({m: sp_dfs_by_match[m] for m in sp_avail})
        pm_pos_full: dict = {"All Matches": pm_pos_all}
        pm_pos_full.update({m: dfs_by_match[m] for m in sp_avail})

        st.markdown("### 🏟️ Match")
        sp_match = st.selectbox("Choose the match", list(sp_pos_full.keys()),
                                index=0, key="sp_match")
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
        st.markdown("### 🎯 Pass Type")
        sp_filter = st.radio("Show passes",
                             ["All","Line Breaking Only","Ball Progression Only"],
                             index=0, key="sp_filter")
        st.markdown('</div>', unsafe_allow_html=True)

    with sp_col_field:
        sp_df_base = sp_pos_full[sp_match].copy()
        if sp_filter == "Line Breaking Only":
            sp_df_base = sp_df_base[sp_df_base["pass_type"]=="line_breaking"].reset_index(drop=True)
        elif sp_filter == "Ball Progression Only":
            sp_df_base = sp_df_base[sp_df_base["pass_type"]=="ball_progression"].reset_index(drop=True)
        else:
            sp_df_base = sp_df_base.reset_index(drop=True)

        st.markdown('<h4 style="color:#ffffff;margin:0 0 6px 0;">Advanced Passes Map</h4>',
                    unsafe_allow_html=True)
        sp_img, sp_ax, sp_fig = draw_advanced_pass_map(
            sp_df_base, title=f"Advanced Passes — {sp_match}")
        st.image(sp_img, use_container_width=True)
        plt.close(sp_fig)

        st.markdown('<h4 style="color:#ffffff;margin:14px 0 4px 0;">Zone Heatmap — Completed</h4>',
                    unsafe_allow_html=True)
        sp_heat_img, sp_hfig = draw_corridor_heatmap(sp_df_base)
        st.image(sp_heat_img, use_container_width=True)
        plt.close(sp_hfig)

        with st.expander("📊 Full Data Table"):
            dc2 = ["number","type","pass_type","outcome",
                   "x_start","y_start","x_end","y_end","pass_distance"]
            st.dataframe(sp_df_base[dc2].style.format(
                {"x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}",
                 "y_end":"{:.2f}","pass_distance":"{:.1f}"}),
                use_container_width=True, height=320)

    with sp_col_stats:
        total_pm = len(pm_pos_full[sp_match])
        ss = compute_advanced_stats(sp_df_base, total_pm)

        with st.expander("🟡 Line Breaking Passes", expanded=True):
            st.markdown('<div class="stats-section-title">Line Breaking Passes</div>',
                        unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1: small_metric("Total",      f"{ss['lbp_total']}")
            with b2: small_metric("Completed",  f"{ss['lbp_completed']}")
            with b3: small_metric("Incomplete", f"{ss['lbp_incomplete']}")
            st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
            ba1, ba2 = st.columns(2)
            with ba1: small_metric("Accuracy", f"{ss['lbp_accuracy']:.1f}%")
            with ba2: small_metric("Tendency", f"{ss['lbp_tendency']:.1f}%",
                                   delta=f"{ss['lbp_total']} of {total_pm} passes")

        with st.expander("🟣 Ball Progression Passes", expanded=True):
            st.markdown('<div class="stats-section-title">Ball Progression Passes</div>',
                        unsafe_allow_html=True)
            p1, p2 = st.columns(2)
            with p1: small_metric("Total",    f"{ss['bpp_total']}")
            with p2: small_metric("Tendency", f"{ss['bpp_tendency']:.1f}%",
                                  delta=f"{ss['bpp_total']} of {total_pm} passes")

        st.divider()
        st.caption("🟡 Yellow = Line Breaking  ·  🟣 Purple = Ball Progression  ·  🔴 Red = Incomplete")
