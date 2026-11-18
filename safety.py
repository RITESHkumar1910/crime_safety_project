import numpy as np

EARTH_RADIUS_KM = 6371.0088
LEVEL_CITY = "City-level"


def haversine_km(lat, lon, lats, lons):
    """Great-circle distance in km from one point to many points."""
    p1, p2 = np.radians(lat), np.radians(lats)
    dlon = np.radians(lons - lon)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def get_safety(lat, lon, spots, amber_street_km, amber_city_km, accuracy_km=0.0):
    """Return (status, hotspot row, gap in km). `spots` is a DataFrame of hotspots."""
    dist = haversine_km(lat, lon, spots["lat"].values, spots["lon"].values)
    gap = dist - spots["radius_km"].values - accuracy_km
    amber = np.where(spots["level"].values == LEVEL_CITY, amber_city_km, amber_street_km)

    inside = np.where(gap <= 0)[0]
    if len(inside):
        i = inside[np.argmin(gap[inside])]
        return "RED", spots.iloc[i], float(gap[i])
    close = np.where(gap <= amber)[0]
    if len(close):
        i = close[np.argmin(gap[close])]
        return "AMBER", spots.iloc[i], float(gap[i])
    i = int(np.argmin(gap))
    return "GREEN", spots.iloc[i], float(gap[i])


def find_amber_point(spots, amber_street_km, amber_city_km):
    """Find a coordinate that gives AMBER (used for the demo presets). None if not found."""
    for _, r in spots.sort_values("crimes", ascending=False).iterrows():
        amber = amber_city_km if r["level"] == LEVEL_CITY else amber_street_km
        d = r["radius_km"] + amber / 2
        for dlat, dlon in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            la = r["lat"] + dlat * d / 111.0
            lo = r["lon"] + dlon * d / (111.0 * np.cos(np.radians(r["lat"])))
            if get_safety(la, lo, spots, amber_street_km, amber_city_km)[0] == "AMBER":
                return float(la), float(lo)
    return None