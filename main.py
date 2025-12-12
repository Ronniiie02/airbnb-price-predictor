from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder

import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import uvicorn
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from pathlib import Path

# ===========================================================
# FastAPI APP
# ===========================================================

app = FastAPI(title="NYC Airbnb Pricing API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
df: Optional[pd.DataFrame] = None
models_dict: Dict[str, Dict[str, Any]] = {}
FEATURE_COLS: List[str] = []
TARGET_COL = "price"

X_test_global: Optional[pd.DataFrame] = None
y_test_global: Optional[pd.Series] = None
preds_test_by_model: Dict[str, List[float]] = {}

# ===========================================================
# Data loading & model training
# ===========================================================

def load_and_clean_data(csv_path: str = "AB_NYC_2019.csv"):
    try:
        d = pd.read_csv(csv_path)

        # Remove non-positive prices
        d = d[d["price"] > 0]

        # Outlier handling: keep prices between 1%–99% quantiles
        q1 = d["price"].quantile(0.01)
        q99 = d["price"].quantile(0.99)
        d = d[(d["price"] >= q1) & (d["price"] <= q99)]

        # Cap extremely long minimum_nights at 95% quantile
        mn_q95 = d["minimum_nights"].quantile(0.95)
        d.loc[d["minimum_nights"] > mn_q95, "minimum_nights"] = mn_q95

        # Fill missing values
        if "reviews_per_month" in d.columns:
            d["reviews_per_month"] = d["reviews_per_month"].fillna(0)

        numeric_fill_zero = [
            "number_of_reviews",
            "calculated_host_listings_count",
            "availability_365",
        ]
        for c in numeric_fill_zero:
            if c in d.columns:
                d[c] = d[c].fillna(0)

        # Drop rows missing essential fields
        d = d.dropna(
            subset=["neighbourhood_group", "neighbourhood", "room_type", "latitude", "longitude"]
        )

        return d, float(q1), float(q99)
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def train_models(data: pd.DataFrame):
    global FEATURE_COLS

    FEATURE_COLS = [
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    X = data[FEATURE_COLS].copy()
    y = data[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = ["neighbourhood_group", "neighbourhood", "room_type"]
    numeric_cols = [c for c in FEATURE_COLS if c not in categorical_cols]

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models: Dict[str, Dict[str, Any]] = {}

    # ---------------- Random Forest ----------------
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_pipe = Pipeline([("preprocess", preprocess), ("model", rf_reg)])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    models["Random Forest"] = {
        "pipeline": rf_pipe,
        "mae": float(mean_absolute_error(y_test, rf_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
    }

    # ---------------- XGBoost ----------------
    xgb_reg = XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    xgb_pipe = Pipeline([("preprocess", preprocess), ("model", xgb_reg)])
    xgb_pipe.fit(X_train, y_train)
    xgb_pred = xgb_pipe.predict(X_test)
    models["XGBoost"] = {
        "pipeline": xgb_pipe,
        "mae": float(mean_absolute_error(y_test, xgb_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, xgb_pred))),
    }

    # ---------------- CatBoost ----------------
    cat_reg = CatBoostRegressor(
        iterations=600,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_state=42,
        verbose=False,
    )
    cat_pipe = Pipeline([("preprocess", preprocess), ("model", cat_reg)])
    cat_pipe.fit(X_train, y_train)
    cat_pred = cat_pipe.predict(X_test)
    models["CatBoost"] = {
        "pipeline": cat_pipe,
        "mae": float(mean_absolute_error(y_test, cat_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, cat_pred))),
    }

    return models, X_train, X_test, y_train, y_test


# ===========================================================
# Request models
# ===========================================================

class FilterRequest(BaseModel):
    neighbourhood_groups: Optional[List[str]] = None
    room_types: Optional[List[str]] = None
    price_range: Optional[List[Optional[float]]] = None
    minimum_nights_range: Optional[List[Optional[int]]] = None


class ModelPerfRequest(BaseModel):
    model: str
    filters: Optional[FilterRequest] = None
    max_points: int = 600


# ===========================================================
# Helpers
# ===========================================================

SAFE_RETURN_COLS = [
    "id", "name",
    "neighbourhood_group", "neighbourhood", "room_type",
    "latitude", "longitude",
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "price"
]

def _safe_records(df_in: pd.DataFrame, limit: int = 1000):
    df_out = df_in.copy()
    cols = [c for c in SAFE_RETURN_COLS if c in df_out.columns]
    df_out = df_out[cols]

    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    df_out = df_out.where(pd.notnull(df_out), None)

    records = df_out.to_dict("records")[:limit]
    return jsonable_encoder(records)

def apply_filters_to_df(base_df: pd.DataFrame, filters: Optional[FilterRequest]) -> pd.DataFrame:
    if filters is None:
        return base_df

    dff = base_df.copy()

    if filters.neighbourhood_groups:
        dff = dff[dff["neighbourhood_group"].isin(filters.neighbourhood_groups)]

    if filters.room_types:
        dff = dff[dff["room_type"].isin(filters.room_types)]

    if filters.price_range and len(filters.price_range) == 2:
        price_min, price_max = filters.price_range
        price_min = float(price_min) if price_min is not None else float(base_df["price"].min())
        price_max = float(price_max) if price_max is not None else float(base_df["price"].max())
        dff = dff[dff["price"].between(price_min, price_max)]

    if filters.minimum_nights_range and len(filters.minimum_nights_range) == 2:
        night_min, night_max = filters.minimum_nights_range
        night_min = int(night_min) if night_min is not None else int(base_df["minimum_nights"].min())
        night_max = int(night_max) if night_max is not None else int(base_df["minimum_nights"].max())
        dff = dff[dff["minimum_nights"].between(night_min, night_max)]

    return dff


# ===========================================================
# Startup
# ===========================================================

@app.on_event("startup")
async def startup_event():
    global df, models_dict, X_test_global, y_test_global, preds_test_by_model
    try:
        df, _, _ = load_and_clean_data()
        models_dict, _, X_test, _, y_test = train_models(df)

        X_test_global = X_test
        y_test_global = y_test

        preds_test_by_model = {}
        for name, info in models_dict.items():
            pipe = info["pipeline"]
            preds_test_by_model[name] = pipe.predict(X_test).tolist()

        print("Data loaded and models trained successfully!")
    except Exception as e:
        print(f"Error during startup: {e}")


# ===========================================================
# Routes
# ===========================================================

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/api/data/summary")
async def get_data_summary():
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    if not models_dict:
        raise HTTPException(status_code=503, detail="Models not loaded")

    p1 = float(df["price"].quantile(0.01))
    p99 = float(df["price"].quantile(0.99))

    maes = [m["mae"] for m in models_dict.values()]
    rmses = [m["rmse"] for m in models_dict.values()]
    avg_mae = float(np.mean(maes))

    best_model_name = min(models_dict.keys(), key=lambda name: models_dict[name]["mae"])
    best_model_mae = float(models_dict[best_model_name]["mae"])
    best_model_rmse = float(models_dict[best_model_name]["rmse"])

    return JSONResponse(
        {
            "total_listings": int(len(df)),
            "avg_price": float(df["price"].mean()),
            "median_price": float(df["price"].median()),
            "price_range": {"min": p1, "max": p99},
            "neighbourhood_groups": df["neighbourhood_group"].unique().tolist(),
            "room_types": df["room_type"].unique().tolist(),
            "ml_models_trained": len(models_dict),
            "avg_mae": avg_mae,
            "avg_rmse": float(np.mean(rmses)),
            "best_model": {"name": best_model_name, "mae": best_model_mae, "rmse": best_model_rmse},
        }
    )


@app.get("/api/filters")
async def get_filter_options():
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    return JSONResponse(
        {
            "neighbourhood_groups": sorted(df["neighbourhood_group"].unique().tolist()),
            "room_types": sorted(df["room_type"].unique().tolist()),
            "price_range": {"min": float(df["price"].min()), "max": float(df["price"].max())},
            "minimum_nights_range": {
                "min": int(df["minimum_nights"].min()),
                "max": int(df["minimum_nights"].quantile(0.95)),
            },
        }
    )


@app.post("/api/data/filter")
async def filter_data(filters: FilterRequest):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    filtered_df = apply_filters_to_df(df, filters)

    return JSONResponse({
        "filtered_count": int(len(filtered_df)),
        "total_count": int(len(df)),
        "data": _safe_records(filtered_df, limit=1000)
    })


# ===========================================================
# Charts (POST + filters)
# ===========================================================

@app.post("/api/charts/price-distribution")
async def chart_price_distribution(filters: Optional[FilterRequest] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    dff = apply_filters_to_df(df, filters)
    fig = px.histogram(dff, x="price", nbins=50, title="Price Distribution")
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis_title="Count",
        template="plotly_white",
        colorway=["#FF5A5F"],
    )
    return JSONResponse(fig.to_json())


@app.post("/api/charts/price-by-room-type")
async def chart_price_by_room_type(filters: Optional[FilterRequest] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    dff = apply_filters_to_df(df, filters)
    fig = px.box(
        dff,
        x="room_type",
        y="price",
        points="suspectedoutliers",
        title="Price by Room Type",
    )
    fig.update_layout(
        xaxis_title="Room Type",
        yaxis_title="Price ($)",
        template="plotly_white",
        colorway=["#00A699"],
    )
    return JSONResponse(fig.to_json())


@app.post("/api/charts/price-by-neighbourhood")
async def chart_price_by_neighbourhood(filters: Optional[FilterRequest] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    dff = apply_filters_to_df(df, filters)
    fig = px.violin(
        dff,
        x="neighbourhood_group",
        y="price",
        box=True,
        points="suspectedoutliers",
        title="Price by Neighbourhood Group",
    )
    fig.update_layout(
        xaxis_title="Neighbourhood Group",
        yaxis_title="Price ($)",
        template="plotly_white",
        colorway=["#FC642D"],
    )
    return JSONResponse(fig.to_json())


@app.post("/api/charts/listings-map")
async def chart_listings_map(filters: Optional[FilterRequest] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    dff = apply_filters_to_df(df, filters)

    # 防止点太多卡死浏览器
    if len(dff) > 5000:
        dff = dff.sample(n=5000, random_state=42)

    fig = px.scatter_mapbox(
        dff,
        lat="latitude",
        lon="longitude",
        color="neighbourhood_group",
        size="price",
        hover_name="neighbourhood",
        hover_data={"price": True, "room_type": True},
        zoom=9.5,
        height=450,
        title="Listings Map (size ~ price)",
    )
    fig.update_layout(mapbox_style="carto-positron")
    return JSONResponse(fig.to_json())


@app.post("/api/charts/reviews-vs-price")
async def chart_reviews_vs_price(filters: Optional[FilterRequest] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    dff = apply_filters_to_df(df, filters)

    fig = px.scatter(
        dff,
        x="number_of_reviews",
        y="price",
        opacity=0.5,
        title="Number of Reviews vs Price",
        trendline="ols",
    )
    fig.update_layout(
        xaxis_title="Number of Reviews",
        yaxis_title="Price ($)",
        template="plotly_white",
        colorway=["#FFB400"],
    )
    return JSONResponse(fig.to_json())


@app.post("/api/charts/price-heatmap")
async def chart_price_heatmap(filters: Optional[FilterRequest] = None):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    dff = apply_filters_to_df(df, filters)

    fig = px.density_heatmap(
        dff,
        x="room_type",
        y="neighbourhood_group",
        z="price",
        histfunc="avg",
        color_continuous_scale="Viridis",
        title="Average Price by Group & Room Type",
    )
    fig.update_layout(
        xaxis_title="Room Type",
        yaxis_title="Neighbourhood Group",
        template="plotly_white",
    )
    return JSONResponse(fig.to_json())


# ===========================================================
# Models & prediction
# ===========================================================

@app.get("/api/models")
async def get_models():
    if not models_dict:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model_info = {}
    for name, info in models_dict.items():
        model_info[name] = {"mae": info["mae"], "rmse": info["rmse"]}
    return JSONResponse(model_info)


@app.post("/api/predict")
async def predict_price(prediction_data: Dict[str, Any]):
    if df is None or not models_dict:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model_name = prediction_data.get("model", "Random Forest")
    features = prediction_data.get("features", {})

    if model_name not in models_dict:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

    try:
        X_pred = pd.DataFrame([features])
        model = models_dict[model_name]["pipeline"]
        pred_price = float(model.predict(X_pred)[0])
        percentile = (df["price"] < pred_price).mean() * 100

        return JSONResponse(
            {
                "predicted_price": pred_price,
                "price_range": {"min": pred_price * 0.85, "max": pred_price * 1.15},
                "percentile": percentile,
                "model": model_name,
                "mae": models_dict[model_name]["mae"],
                "rmse": models_dict[model_name]["rmse"],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/model-performance")
async def model_performance(req: ModelPerfRequest):
    if df is None or X_test_global is None or y_test_global is None:
        raise HTTPException(status_code=503, detail="Data/model not ready")

    model_name = req.model
    if model_name not in models_dict:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

    # 用 holdout test set（标准做法：不建议每次 filter 都 retrain）
    y_true = y_test_global.reset_index(drop=True)
    y_pred = np.array(preds_test_by_model[model_name])

    n = len(y_true)
    k = min(max(50, int(req.max_points)), n)  # 至少 50 点
    if k < n:
        idx = np.random.RandomState(42).choice(n, size=k, replace=False)
        y_true_plot = y_true.iloc[idx].to_numpy()
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true.to_numpy()
        y_pred_plot = y_pred

    fig = px.scatter(
        x=y_true_plot,
        y=y_pred_plot,
        labels={"x": "Actual Price ($)", "y": "Predicted Price ($)"},
        title=f"Actual vs Predicted Prices ({model_name})",
    )

    minv = float(min(np.min(y_true_plot), np.min(y_pred_plot)))
    maxv = float(max(np.max(y_true_plot), np.max(y_pred_plot)))

    fig.add_shape(
        type="line",
        x0=minv, y0=minv,
        x1=maxv, y1=maxv,
        line=dict(dash="dash", width=2),
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return JSONResponse(fig.to_json())


@app.get("/api/neighbourhoods/{group}")
async def get_neighbourhoods(group: str):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    neighbourhoods = (
        df[df["neighbourhood_group"] == group]["neighbourhood"]
        .unique()
        .tolist()
    )
    return JSONResponse(sorted(neighbourhoods))


# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    Path("static").mkdir(exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
