import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import haversine_distances

EARTH_RADIUS_KM = 6371.0088
RANDOM_STATE = 42
SAMPLE_SIZE = 50000
LEVEL_STREET = "Street-level (DBSCAN)"
LEVEL_CITY = "City-level"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

os.makedirs("processed", exist_ok=True)

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639), "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319), "Surat": (21.1702, 72.8311),
    "Nagpur": (21.1458, 79.0882), "Agra": (27.1767, 78.0081),
    "Ludhiana": (30.9010, 75.8573), "Patna": (25.5941, 85.1376),
    "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185), "Thane": (19.2183, 72.9781),
    "Ghaziabad": (28.6692, 77.4538), "Varanasi": (25.3176, 82.9739),
    "Meerut": (28.9845, 77.7064), "Rajkot": (22.3039, 70.8022),
    "Vadodara": (22.3072, 73.1812), "Nashik": (19.9975, 73.7898),
    "Faridabad": (28.4089, 77.3178), "Srinagar": (34.0837, 74.7973),
    "Kalyan": (19.2403, 73.1305), "Vasai": (19.3919, 72.8397),
}


def find_file(name):
    """Look for a raw file in data/ first, then in the current folder."""
    for folder in ("data", "."):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found. Put it in the data/ folder.")


def save_eda(df, hour_col, title, path):
    """Save the 4 standard EDA charts (crime type, hour, weekday, month)."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    df["crime_type"].value_counts().head(10).plot.barh(ax=ax[0, 0], title="Top 10 crime types")
    df[hour_col].dropna().astype(int).value_counts().sort_index().plot.bar(
        ax=ax[0, 1], title="Crimes by hour")
    df["weekday"].value_counts().reindex(DAY_ORDER).plot.bar(ax=ax[1, 0], title="Crimes by weekday")
    df["month"].dropna().astype(int).value_counts().sort_index().plot.bar(
        ax=ax[1, 1], title="Crimes by month")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def process_chicago():
    print("\n=== CHICAGO ===")
    path = find_file("Crimes_-_2001_to_Present.csv")
    raw = pd.read_csv(path, usecols=["Date", "Primary Type", "Year", "Latitude", "Longitude"])
    print("1) rows in file:", len(raw))

    raw = raw.dropna(subset=["Latitude", "Longitude"])
    raw = raw[raw["Latitude"].between(41.6, 42.1) & raw["Longitude"].between(-88.0, -87.5)]
    print("2) rows with valid coordinates inside Chicago:", len(raw))


    counts = raw["Year"].value_counts().sort_index()
    recent = counts[counts.index >= 2015]
    year = int(recent.idxmax() if len(recent) else counts.idxmax())
    df = raw[raw["Year"] == year].drop_duplicates().copy()
    print(f"3) chosen year: {year}, rows after removing duplicates: {len(df)}")

    
    s = df["Date"].astype(str).str.strip()
    parsed = pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        parsed[bad] = pd.to_datetime(s[bad], format="mixed", dayfirst=True, errors="coerce")
    print("   dates not parsed:", int(parsed.isna().sum()), "/", len(parsed))
    df["hour"] = parsed.dt.hour
    df["weekday"] = parsed.dt.day_name()
    df["month"] = parsed.dt.month
    df["crime_type"] = df["Primary Type"].str.title()

    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print("4) sample used for modelling:", len(df))
    assert len(df) >= 1000, "Too little Chicago data. Check the raw file."
    save_eda(df, "hour", f"Chicago {year}: exploratory analysis", "processed/eda_chicago.png")

    
    coords = np.radians(df[["Latitude", "Longitude"]].values)
    chosen, labels = None, None
    for eps_km in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        for min_pts in [30, 40, 60, 80]:
            lab = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_pts,
                         metric="haversine", algorithm="ball_tree").fit(coords).labels_
            k = int(lab.max()) + 1
            if k == 0:
                continue
            biggest = np.bincount(lab[lab >= 0]).max() / len(lab)
            print(f"   eps={eps_km} km, min_samples={min_pts} -> hotspots={k}, largest={biggest:.1%}")
            
            if 5 <= k <= 60 and biggest < 0.15:
                chosen, labels = (eps_km, min_pts), lab
                break
        if chosen:
            break
    if chosen is None:
        chosen = (0.10, 60)
        labels = DBSCAN(eps=chosen[0] / EARTH_RADIUS_KM, min_samples=chosen[1],
                        metric="haversine", algorithm="ball_tree").fit(coords).labels_
    k = int(labels.max()) + 1
    assert k > 0, "DBSCAN found no hotspots."
    df["cluster"] = labels
    mask = labels >= 0
    print(f"5) DBSCAN chosen: eps={chosen[0]} km, min_samples={chosen[1]} -> {k} hotspots, "
          f"noise {100 * (~mask).mean():.1f}%")

    
    km = KMeans(n_clusters=max(k, 2), n_init=10, random_state=RANDOM_STATE).fit(coords)
    ss = min(10000, int(mask.sum()))
    report = {
        "year": year, "rows_used": int(len(df)),
        "dbscan_eps_km": chosen[0], "dbscan_min_samples": chosen[1], "dbscan_clusters": k,
        "dbscan_noise_pct": round(100 * float((~mask).mean()), 1),
        "dbscan_silhouette": round(float(silhouette_score(
            coords[mask], labels[mask], metric="haversine", sample_size=ss,
            random_state=RANDOM_STATE)), 3),
        "dbscan_davies_bouldin": round(float(davies_bouldin_score(coords[mask], labels[mask])), 3),
        "kmeans_clusters": max(k, 2),
        "kmeans_silhouette": round(float(silhouette_score(
            coords, km.labels_, metric="haversine", sample_size=min(10000, len(df)),
            random_state=RANDOM_STATE)), 3),
        "kmeans_davies_bouldin": round(float(davies_bouldin_score(coords, km.labels_)), 3),
    }
    print("6) model comparison:", report)

    
    rows = []
    for c in range(k):
        g = df[df["cluster"] == c]
        lat, lon = g["Latitude"].mean(), g["Longitude"].mean()
        d = haversine_distances(np.radians(g[["Latitude", "Longitude"]].values),
                                np.radians([[lat, lon]])).ravel() * EARTH_RADIUS_KM
        hours, days = g["hour"].dropna(), g["weekday"].dropna()
        rows.append(dict(
            hotspot_id=f"CHI-{c + 1}", region="Chicago, USA", level=LEVEL_STREET,
            name=f"Chicago hotspot {c + 1}", lat=lat, lon=lon,
            radius_km=float(np.clip(np.percentile(d, 90), 0.15, 0.8)), crimes=len(g),
            top_crime=g["crime_type"].mode()[0],
            peak_hour=int(hours.mode()[0]) if len(hours) else -1,
            peak_day=days.mode()[0] if len(days) else "NA", violent_pct=np.nan))
    hot = pd.DataFrame(rows).sort_values("crimes", ascending=False).reset_index(drop=True)

    crimes = pd.DataFrame({"region": "Chicago, USA", "area": "Chicago",
                           "crime_type": df["crime_type"], "hour": df["hour"],
                           "weekday": df["weekday"], "month": df["month"]})
    return hot, crimes, report


def process_india():
    print("\n=== INDIA ===")
    ind = pd.read_csv(find_file("crime_dataset_india.csv"))
    print("rows in file:", len(ind))

    t = pd.to_datetime(ind["Time of Occurrence"], dayfirst=True, errors="coerce")
    ind["hour"] = t.dt.hour
    ind["weekday"] = t.dt.day_name()
    ind["month"] = t.dt.month
    ind["crime_type"] = ind["Crime Description"].str.title()
    print("times not parsed:", int(t.isna().sum()), "/", len(t))

    missing = sorted(set(ind["City"]) - set(CITY_COORDS))
    print("cities without coordinates (skipped):", missing)
    ind = ind[ind["City"].isin(CITY_COORDS)].copy()
    save_eda(ind, "hour", "India: exploratory analysis", "processed/eda_india.png")

    rows = []
    max_crimes = ind["City"].value_counts().max()
    for city, g in ind.groupby("City"):
        lat, lon = CITY_COORDS[city]
        rows.append(dict(
            hotspot_id=f"IN-{city}", region="India", level=LEVEL_CITY, name=city,
            lat=lat, lon=lon,
            radius_km=round(8 + 12 * len(g) / max_crimes, 1),  
            crimes=len(g), top_crime=g["crime_type"].mode()[0],
            peak_hour=int(g["hour"].dropna().mode()[0]), peak_day=g["weekday"].dropna().mode()[0],
            violent_pct=round(float((g["Crime Domain"] == "Violent Crime").mean() * 100), 1)))
    hot = pd.DataFrame(rows).sort_values("crimes", ascending=False).reset_index(drop=True)

    crimes = pd.DataFrame({"region": "India", "area": ind["City"], "crime_type": ind["crime_type"],
                           "hour": ind["hour"], "weekday": ind["weekday"], "month": ind["month"]})
    return hot, crimes


if __name__ == "__main__":
    chi_hot, chi_crimes, report = process_chicago()
    in_hot, in_crimes = process_india()

    hotspots_all = pd.concat([chi_hot, in_hot], ignore_index=True)
    crime_all = pd.concat([chi_crimes, in_crimes], ignore_index=True)
    hotspots_all.to_csv("processed/hotspots_all.csv", index=False)
    crime_all.to_csv("processed/crime_all.csv", index=False)
    with open("processed/model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== DONE ===")
    print("hotspots_all.csv:", hotspots_all.shape, "| crime_all.csv:", crime_all.shape)
    print(hotspots_all.groupby("region")["hotspot_id"].count())
    print("Files are in processed/. Now run:  python -m streamlit run app.py")