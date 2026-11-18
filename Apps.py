"""
app.py  -  Streamlit dashboard (Chicago street-level + India city-level hotspots).
Run:  python -m streamlit run app.py      (after  python build.py)
Theme: sky blue + gray. Put config.toml inside a  .streamlit/  folder next to this file.
"""
import json
import math
from datetime import datetime, timedelta, timezone

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

from safety import LEVEL_CITY, find_amber_point, get_safety, haversine_km

IST = timezone(timedelta(hours=5, minutes=30))
try:
    from zoneinfo import ZoneInfo
    CHICAGO_TZ = ZoneInfo("America/Chicago")
except Exception:  # tzdata missing: fall back to a fixed offset
    CHICAGO_TZ = timezone(timedelta(hours=-6))

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REGIONS = ["Both", "Chicago, USA", "India"]

# Sky blue + gray palette
SKY = "#0ea5e9"
SKY_LIGHT = "#38bdf8"
SKY_PALE = "#e0f2fe"
GRAY_BG = "#f1f5f9"
GRAY_MID = "#cbd5e1"
GRAY_TEXT = "#475569"
INK = "#1e293b"

# Status colours (kept universal so RED / AMBER / GREEN stay instantly readable)
STATUS_COLOR = {"RED": "#ef4444", "AMBER": "#f59e0b", "GREEN": "#22c55e"}
STATUS_ICON = {"RED": "red", "AMBER": "orange", "GREEN": "green"}

st.set_page_config(page_title="SafeCity Hotspot Dashboard", page_icon="🛡️", layout="wide")

st.markdown(f"""
<style>
.stApp {{background: {GRAY_BG}; color: {INK};}}
[data-testid="stSidebar"] {{background: #e2e8f0; border-right: 1px solid {GRAY_MID};}}
[data-testid="stMetric"] {{background: #ffffff; border: 1px solid {GRAY_MID}; border-left: 4px solid {SKY};
                           border-radius: 12px; padding: 10px 14px;}}
[data-baseweb="tab-list"] {{gap: 6px;}}
[data-baseweb="tab"] {{background: #e2e8f0; border-radius: 8px 8px 0 0; padding: 8px 16px; color: {GRAY_TEXT};}}
[aria-selected="true"][data-baseweb="tab"] {{background: {SKY}; color: #ffffff;}}
.hero {{background: linear-gradient(120deg, #0369a1, {SKY} 60%, {SKY_LIGHT});
       padding: 22px 26px; border-radius: 14px; color: #fff; margin-bottom: 14px;
       box-shadow: 0 4px 14px rgba(14,165,233,.25);}}
.hero h1 {{margin: 0; font-size: 1.7rem; color: #fff;}}
.hero p {{margin: 4px 0 0 0; color: {SKY_PALE}; font-size: .95rem;}}
.status {{border-radius: 14px; padding: 18px 20px; color: #fff; margin-bottom: 10px;}}
.status .s-title {{font-size: 1.6rem; font-weight: 800; letter-spacing: .5px;}}
.status .s-sub {{font-size: 1rem; margin-top: 4px; opacity: .95;}}
.pulse {{animation: pulse 1.6s infinite;}}
@keyframes pulse {{0% {{box-shadow: 0 0 0 0 rgba(239,68,68,.6);}}
                  70% {{box-shadow: 0 0 0 16px rgba(239,68,68,0);}}
                  100% {{box-shadow: 0 0 0 0 rgba(239,68,68,0);}}}}
@media (prefers-reduced-motion: reduce) {{.pulse {{animation: none;}}}}
.sos a {{display: inline-block; background: #dc2626; color: #fff !important; padding: 8px 14px;
        border-radius: 8px; margin: 4px 6px 4px 0; text-decoration: none; font-weight: 600;}}
.legend span {{display: inline-block; margin-right: 14px; font-size: .85rem; color: {GRAY_TEXT};}}
.legend i {{display: inline-block; width: 12px; height: 12px; border-radius: 3px;
           margin-right: 5px; vertical-align: -1px;}}
</style>
""", unsafe_allow_html=True)


# ---------------- Data ----------------
@st.cache_data
def load_data():
    hot = pd.read_csv("processed/hotspots_all.csv")
    crimes = pd.read_csv("processed/crime_all.csv")
    with open("processed/model_report.json") as f:
        report = json.load(f)
    return hot, crimes, report


try:
    hot, crimes, report = load_data()
except FileNotFoundError:
    st.error("Processed files not found. Run  python build.py  first, then restart the app.")
    st.stop()

# Chicago police district stations (approximate coordinates; verify with the official list)
STATIONS = pd.DataFrame([
    ("Central (1st)", 41.8587, -87.6274), ("Wentworth (2nd)", 41.8014, -87.6309),
    ("Deering (9th)", 41.8363, -87.6463), ("Ogden (10th)", 41.8582, -87.7060),
    ("Harrison (11th)", 41.8736, -87.7035), ("Near North (18th)", 41.9028, -87.6437),
    ("Town Hall (19th)", 41.9468, -87.6538), ("Englewood (7th)", 41.7522, -87.6437),
    ("Chicago Lawn (8th)", 41.7788, -87.7117), ("Austin (15th)", 41.8928, -87.7628),
], columns=["name", "lat", "lon"])

HELPLINES_IN = {"All emergencies": "112", "Police": "100 / 112", "Women safety": "1091 / 181",
                "Child safety": "1098", "Ambulance": "108 / 102", "Fire brigade": "101",
                "Cyber crime": "1930"}
HELPLINES_US = {"Emergency (police, fire, ambulance)": "911"}


# ---------------- Helpers ----------------
def local_now(region):
    """Current time in the region's own time zone (IST for India, Chicago time otherwise)."""
    return datetime.now(IST if region == "India" else CHICAGO_TZ)


def fmt_ist(dt):
    return dt.strftime("%d %b %Y, %I:%M:%S %p IST")


def fmt_hour(hour):
    return datetime(2000, 1, 1, int(hour) % 24).strftime("%I:%M %p")


def show_banner(status, title, subtitle):
    colors = {"RED": ("#ef4444", "#991b1b"), "AMBER": ("#f59e0b", "#92400e"),
              "GREEN": ("#22c55e", "#166534")}
    c1, c2 = colors[status]
    pulse = "pulse" if status == "RED" else ""
    st.markdown(
        f'<div class="status {pulse}" style="background:linear-gradient(135deg,{c1},{c2})">'
        f'<div class="s-title">{title}</div><div class="s-sub">{subtitle}</div></div>',
        unsafe_allow_html=True)


def build_presets(spots, amber_street, amber_city):
    """Demo locations for the dropdown."""
    presets = {"Custom coordinates": None}
    chi = spots[spots["region"] == "Chicago, USA"].sort_values("crimes", ascending=False)
    ind = spots[spots["region"] == "India"].sort_values("crimes", ascending=False)
    for _, r in chi.head(3).iterrows():
        presets[f"Chicago: inside {r['name']} ({r['top_crime']}) - RED"] = (r["lat"], r["lon"])
    if len(chi):
        point = find_amber_point(chi, amber_street, amber_city)
        if point:
            presets["Chicago: just outside a hotspot - AMBER"] = point
    for _, r in ind.iterrows():
        presets[f"India: centre of {r['name']} - RED"] = (r["lat"], r["lon"])
    if len(ind):
        point = find_amber_point(ind, amber_street, amber_city)
        if point:
            presets["India: just outside a city zone - AMBER"] = point
    presets["Remote area (GREEN test)"] = (31.7, 78.0)
    return presets


def build_safety_grid(lat, lon, spots, amber_street, amber_city, level, n=13):
    """Run the same RED/AMBER/GREEN rule on a grid of points around the user.

    Returns a list of (cell_lat, cell_lon, status) plus the cell size in degrees.
    Grid size follows the zone level: ~street scale for Chicago, ~city scale for India.
    """
    half_km = amber_city * 2.5 if level == LEVEL_CITY else max(1.5, amber_street * 3)
    step_km = 2 * half_km / n
    dlat = step_km / 111.0
    dlon = step_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    cells = []
    for i in range(n):
        for j in range(n):
            clat = lat + (i - (n - 1) / 2) * dlat
            clon = lon + (j - (n - 1) / 2) * dlon
            cell_status, _, _ = get_safety(clat, clon, spots, amber_street, amber_city, 0.0)
            cells.append((clat, clon, cell_status))
    return cells, dlat, dlon


def build_crime_rings(lat, lon, spots, amber_street, amber_city, level):
    """Concentric rings around the user. Each ring is coloured by the crime hotspots it touches.

    A hotspot touches a ring when (distance to its centre - its radius) <= ring radius.
    RED   = user is inside a hotspot, or the ring touches a high-volume hotspot (top 25% of its level)
    AMBER = ring touches only lower-volume hotspots
    GREEN = ring touches no hotspot
    """
    factors = (0.5, 1, 2, 3) if level == LEVEL_CITY else (0.5, 1, 2, 4)
    base = amber_city if level == LEVEL_CITY else amber_street
    dist = haversine_km(lat, lon, spots["lat"].values, spots["lon"].values)
    gaps = dist - spots["radius_km"].values
    rings = []
    for f in factors:
        r = base * f
        mask = gaps <= r
        near = spots[mask]
        if near.empty:
            rings.append({"radius_km": r, "status": "GREEN", "hotspots": 0, "crimes": 0, "top_crime": "-"})
            continue
        heavy = bool((near["rank_pct"] >= 0.75).any()) or float(gaps[mask].min()) <= 0
        rings.append({
            "radius_km": r,
            "status": "RED" if heavy else "AMBER",
            "hotspots": len(near),
            "crimes": int(near["crimes"].sum()),
            "top_crime": near.groupby("top_crime")["crimes"].sum().idxmax(),
        })
    return rings


# ---------------- Sidebar ----------------
st.sidebar.title("🛡️ SafeCity Controls")
region_view = st.sidebar.radio("Region", REGIONS)
mode = st.sidebar.radio("Location source", ["Demo location", "Live GPS"])
conservative = st.sidebar.checkbox("Conservative mode (add GPS error to risk)", value=True)
show_grid = st.sidebar.checkbox("Show safety zones around my location", value=True)
show_rings = st.sidebar.checkbox("Show crime rings around my location", value=True)

st.sidebar.subheader("Filters")
types = ["ALL"] + sorted(hot["top_crime"].unique())
selected_type = st.sidebar.selectbox("Hotspot crime type", types)
amber_street = st.sidebar.slider("AMBER distance, Chicago street hotspots (km)", 0.1, 1.5, 0.5, 0.1)
amber_city = st.sidebar.slider("AMBER distance, India city zones (km)", 1, 30, 10)
map_style = st.sidebar.selectbox("Map style", ["CartoDB positron", "CartoDB dark_matter", "OpenStreetMap"])

st.sidebar.subheader("Time")
custom_hour = st.sidebar.checkbox("Use a custom local hour (demo)")
custom_hour_value = st.sidebar.slider("Local hour of day", 0, 23, 18) if custom_hour else None

st.sidebar.subheader("Personal emergency contact")
contact_name = st.sidebar.text_input("Contact name", placeholder="e.g. Mom")
contact_number = st.sidebar.text_input("Contact number", placeholder="e.g. +91XXXXXXXXXX")

spots = hot.copy()
if region_view != "Both":
    spots = spots[spots["region"] == region_view]
if selected_type != "ALL":
    spots = spots[spots["top_crime"] == selected_type]
spots = spots.reset_index(drop=True)
spots["rank_pct"] = spots.groupby("level")["crimes"].rank(pct=True)

crime_view = crimes if region_view == "Both" else crimes[crimes["region"] == region_view]

# ---------------- Header ----------------
st.markdown(
    '<div class="hero"><h1>🛡️ Crime Hotspot & Live Safety Dashboard</h1>'
    '<p>Chicago street-level hotspots (DBSCAN, haversine) merged with India city-level zones, '
    'plus a live GREEN / AMBER / RED location indicator.</p></div>', unsafe_allow_html=True)

now_in, now_us = local_now("India"), local_now("Chicago, USA")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Hotspot zones", len(spots))
m2.metric("Crimes analysed", f"{len(crime_view):,}")
m3.metric("India time (IST)", now_in.strftime("%I:%M %p"), now_in.strftime("%d %b %Y"))
m4.metric("Chicago time", now_us.strftime("%I:%M %p"), now_us.strftime("%d %b %Y"))

tab_live, tab_patterns, tab_hot = st.tabs(
    ["📍 Live Safety", "📊 Crime Patterns", "🗺️ Hotspots"])

# ---------------- Live Safety ----------------
with tab_live:
    lat = lon = None
    acc_m = None
    stamp = now_in

    if mode == "Live GPS":
        st.markdown("Tap the location button, then choose *Allow* in your browser. "
                    "Tap again any time to refresh your position.")
        loc = streamlit_geolocation()
        if loc and loc.get("latitude"):
            lat, lon = loc["latitude"], loc["longitude"]
            acc_m = loc.get("accuracy")
            if loc.get("timestamp"):
                stamp = datetime.fromtimestamp(loc["timestamp"] / 1000, IST)
        else:
            st.info("Waiting for GPS location. Make sure location access is allowed.")
    else:
        presets = build_presets(spots, amber_street, amber_city) if len(spots) else {"Custom coordinates": None}
        names = list(presets.keys())
        choice = st.selectbox("Demo location", names, index=min(1, len(names) - 1))
        if presets[choice]:
            lat, lon = presets[choice]
        else:
            default = (28.6139, 77.2090) if region_view == "India" else (41.8829, -87.6273)
            c_a, c_b = st.columns(2)
            lat = c_a.number_input("Latitude", value=default[0], format="%.5f")
            lon = c_b.number_input("Longitude", value=default[1], format="%.5f")

    status = spot = gap = None
    if lat is not None and len(spots) > 0:
        acc_km = (acc_m / 1000.0) if (acc_m and conservative) else 0.0
        status, spot, gap = get_safety(lat, lon, spots, amber_street, amber_city, acc_km)

    rings = []
    if status and show_rings:
        rings = build_crime_rings(lat, lon, spots, amber_street, amber_city, spot["level"])

    map_col, info_col = st.columns([3, 2])

    with info_col:
        if status:
            is_city = spot["level"] == LEVEL_CITY
            local = local_now(spot["region"])
            hour_now = custom_hour_value if custom_hour else local.hour
            peak = f"{fmt_hour(spot['peak_hour'])} ({spot['peak_day']})"

            if status == "RED":
                show_banner("RED", "🔴 DANGER ZONE",
                            f"{spot['name']}: {spot['top_crime']}, peak {peak}")
            elif status == "AMBER":
                show_banner("AMBER", "🟠 CAUTION",
                            f"{spot['name']} is only {gap:.2f} km away")
            else:
                show_banner("GREEN", "🟢 SAFE ZONE",
                            f"Nearest zone: {spot['name']}, {gap:.1f} km away")

            if status != "GREEN":
                diff = abs(hour_now - int(spot["peak_hour"]))
                diff = min(diff, 24 - diff)
                if diff <= 2:
                    st.error(f"High-risk time: local time is {fmt_hour(hour_now)} and crimes here "
                             f"peak around {fmt_hour(spot['peak_hour'])}.")
                else:
                    st.info(f"Lower-risk time: crimes here peak around {fmt_hour(spot['peak_hour'])}.")
                if is_city:
                    st.caption("City-level zone: RED means you are inside a city with recorded "
                               "crime, not on a specific dangerous street.")

            st.caption(f"🕒 Checked at: {fmt_ist(stamp)}")
            st.caption(f"🌐 Local time at {spot['region']}: {local.strftime('%I:%M %p, %d %b')}")
            if acc_m:
                st.caption(f"📡 GPS accuracy: about {acc_m:.0f} m "
                           f"({'added to risk' if conservative else 'not added to risk'})")
                if acc_m > 100:
                    st.warning("GPS accuracy is low. Move to an open area and tap again.")

            # Police station (data available for Chicago only)
            sd = haversine_km(lat, lon, STATIONS["lat"].values, STATIONS["lon"].values)
            if sd.min() < 30:
                station = STATIONS.iloc[int(sd.argmin())]
                st.markdown(f"**Nearest police station:** {station['name']}, {sd.min():.1f} km away")
                maps_url = (f"https://www.google.com/maps/dir/?api=1&destination="
                            f"{station['lat']},{station['lon']}")
                st.markdown(f"[Get directions in Google Maps]({maps_url})")
            else:
                st.caption("Police-station data is available only for the Chicago demo area. "
                           "In India, call 112.")

            if status == "RED":
                numbers = ["112", "100"] if spot["region"] == "India" else ["911"]
                sos = '<div class="sos">' + "".join(f'<a href="tel:{n}">Call {n}</a>' for n in numbers)
                if contact_number.strip():
                    label = contact_name.strip() or "emergency contact"
                    sos += f'<a href="tel:{contact_number.strip()}">Call {label}</a>'
                st.markdown(sos + "</div>", unsafe_allow_html=True)

            with st.expander("☎️ All helplines", expanded=(status == "RED")):
                st.write("**India**")
                for k, v in HELPLINES_IN.items():
                    st.write(f"{k}: {v}")
                st.write("**Chicago / USA**")
                for k, v in HELPLINES_US.items():
                    st.write(f"{k}: {v}")
        else:
            st.info("Set a location (and make sure at least one hotspot matches the filters).")

    with map_col:
        if status:
            # zoom so the whole safety grid is visible
            center, zoom = [lat, lon], (8 if spot["level"] == LEVEL_CITY else 13)
        elif region_view == "Chicago, USA":
            center, zoom = [41.85, -87.68], 11
        elif region_view == "India":
            center, zoom = [22.5, 79.0], 5
        else:
            center, zoom = [25.0, 0.0], 2
        fmap = folium.Map(location=center, zoom_start=zoom, tiles=map_style)

        # Safety grid: same RED / AMBER / GREEN rule evaluated on points around the user
        if status and show_grid:
            cells, dlat, dlon = build_safety_grid(
                lat, lon, spots, amber_street, amber_city, spot["level"])
            for clat, clon, cstatus in cells:
                folium.Rectangle(
                    bounds=[[clat - dlat / 2, clon - dlon / 2], [clat + dlat / 2, clon + dlon / 2]],
                    color=STATUS_COLOR[cstatus], weight=0, fill=True,
                    fill_color=STATUS_COLOR[cstatus], fill_opacity=0.28,
                    tooltip=cstatus).add_to(fmap)

        # Crime rings around the user (largest first so smaller rings stay on top)
        for ring in reversed(rings):
            rc = STATUS_COLOR[ring["status"]]
            folium.Circle(
                [lat, lon], radius=ring["radius_km"] * 1000, color=rc, weight=2,
                dash_array="6 6", fill=True, fill_color=rc, fill_opacity=0.10,
                tooltip=(f"Within {ring['radius_km']:g} km: {ring['status']}, "
                         f"{ring['hotspots']} hotspot(s), {ring['crimes']:,} crimes, "
                         f"top: {ring['top_crime']}")).add_to(fmap)

        for _, r in spots.iterrows():
            color = "#b91c1c" if r["rank_pct"] >= 0.75 else "#f97316" if r["rank_pct"] >= 0.5 else "#facc15"
            folium.Circle(
                [r["lat"], r["lon"]], radius=r["radius_km"] * 1000, color=color, fill=True,
                fill_color=color, fill_opacity=0.35, weight=2,
                popup=f"{r['name']}: {int(r['crimes'])} crimes, {r['top_crime']}, "
                      f"peak {fmt_hour(r['peak_hour'])} ({r['peak_day']})").add_to(fmap)

        if region_view != "India":
            for _, r in STATIONS.iterrows():
                folium.Marker([r["lat"], r["lon"]], tooltip=r["name"],
                              icon=folium.Icon(color="blue", icon="info-sign")).add_to(fmap)

        if lat is not None:
            ring = STATUS_COLOR.get(status, SKY)
            if acc_m:
                folium.Circle([lat, lon], radius=acc_m, color=ring, weight=2,
                              fill=True, fill_color=ring, fill_opacity=0.2).add_to(fmap)
            folium.Marker([lat, lon], tooltip=f"You are here ({status or 'no data'})",
                          icon=folium.Icon(color=STATUS_ICON.get(status, "blue"),
                                           icon="user", prefix="fa")).add_to(fmap)

        st_folium(fmap, height=520, use_container_width=True, returned_objects=[])
        st.markdown(
            '<div class="legend">'
            f'<span><i style="background:{STATUS_COLOR["RED"]}"></i>RED: inside hotspot</span>'
            f'<span><i style="background:{STATUS_COLOR["AMBER"]}"></i>AMBER: near hotspot</span>'
            f'<span><i style="background:{STATUS_COLOR["GREEN"]}"></i>GREEN: safe</span>'
            '</div>', unsafe_allow_html=True)
        st.caption("The coloured squares around your pin apply the same safety rule to every nearby "
                   "point. Hotspot circles: darker = more crime within its level. "
                   "Blue pins are Chicago police stations.")

        if rings:
            st.markdown("**Crime rings around you**")
            icons = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢"}
            ring_table = pd.DataFrame([{
                "Within (km)": f"{g['radius_km']:g}",
                "Risk": f"{icons[g['status']]} {g['status']}",
                "Hotspots": g["hotspots"],
                "Crimes": g["crimes"],
                "Top crime": g["top_crime"],
            } for g in rings])
            st.dataframe(ring_table, hide_index=True, width="stretch")
            st.caption("Each dashed circle on the map is one ring. Its colour comes from the hotspots "
                       "it touches: RED if you are inside a hotspot or it touches a high-crime one, "
                       "AMBER if it touches only lower-crime hotspots, GREEN if none.")

# ---------------- Crime Patterns ----------------
with tab_patterns:
    scope = st.radio("Region", REGIONS, horizontal=True, key="pattern_scope")
    d = crimes if scope == "Both" else crimes[crimes["region"] == scope]
    if scope == "India":
        city = st.selectbox("City", ["All cities"] + sorted(d["area"].unique()))
        if city != "All cities":
            d = d[d["area"] == city]
    c1, c2 = st.columns(2)
    c1.subheader("Top 10 crime types")
    c1.bar_chart(d["crime_type"].value_counts().head(10), color=SKY)
    c2.subheader("Crimes by hour of day")
    c2.bar_chart(d["hour"].dropna().astype(int).value_counts().sort_index(), color=SKY)
    c3, c4 = st.columns(2)
    c3.subheader("Crimes by month")
    c3.bar_chart(d["month"].dropna().astype(int).value_counts().sort_index(), color=SKY_LIGHT)
    c4.subheader("Crimes by weekday")
    c4.bar_chart(d["weekday"].value_counts().reindex(DAY_ORDER), color=SKY_LIGHT)

    st.subheader("When do crimes happen? (weekday vs hour)")
    heat = (d.dropna(subset=["hour", "weekday"]).groupby(["weekday", "hour"])
            .size().reset_index(name="crimes"))
    heat["hour"] = heat["hour"].astype(int)
    chart = alt.Chart(heat).mark_rect().encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("weekday:N", sort=DAY_ORDER, title=None),
        color=alt.Color("crimes:Q", scale=alt.Scale(scheme="blues"), title="Crimes"),
        tooltip=["weekday", "hour", "crimes"]).properties(height=280)
    st.altair_chart(chart, width="stretch")
    st.caption("Hours are local to each dataset (Chicago local time, India IST). "
               f"Chicago data: year {report['year']}, {report['rows_used']:,}-row sample.")

# ---------------- Hotspots ----------------
with tab_hot:
    st.subheader("Hotspot profiles")
    table = spots[["hotspot_id", "region", "level", "name", "crimes", "radius_km",
                   "top_crime", "peak_hour", "peak_day", "violent_pct"]]
    st.dataframe(table, width="stretch")
    st.download_button("⬇️ Download hotspots (CSV)", table.to_csv(index=False),
                       file_name="hotspots_filtered.csv", mime="text/csv")