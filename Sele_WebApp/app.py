import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Weather Monitoring - Temp Forecast", layout="wide")
st.title("🌦️ Weather Monitoring Application: Temperature Forecast (Hourly)")

st.markdown("""
**Workflow:** Upload CSV → Data Preparation → Resample 1H → Lag Features → Load model → Predict → Forecast Future → Show results  
ไฟล์ต้องมีคอลัมน์เวลา **convert time** และคอลัมน์อุณหภูมิ **temp**
""")

# ----------------------------
# Model loading
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ ไม่พบไฟล์โมเดลที่: {MODEL_PATH}\n\nตรวจสอบว่าได้วาง model.joblib ไว้ถูกโฟลเดอร์แล้ว")
    st.stop()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


model = load_model(MODEL_PATH)
st.success("✅ โหลดโมเดลสำเร็จ (model.joblib)")


# ----------------------------
# Helpers
# ----------------------------
def prepare_hourly(df_raw: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    df = df_raw.copy()

    df["convert time"] = pd.to_datetime(df["convert time"], errors="coerce")
    df = df.dropna(subset=["convert time"])
    df = df.sort_values("convert time")

    df = df[["convert time", "temp"]].copy()
    df = df.set_index("convert time")

    hourly = df.resample("1h").mean()

    is_nan = hourly["temp"].isna().astype(int)
    groups = (is_nan.diff(1) != 0).cumsum()
    nan_block_sizes = is_nan.groupby(groups).sum()
    block_size_map = nan_block_sizes.reindex(groups).values

    hourly["temp_interp"] = hourly["temp"].interpolate(method="time")

    long_gap_mask = (hourly["temp"].isna()) & (block_size_map > max_gap_hours)
    hourly.loc[long_gap_mask, "temp_interp"] = np.nan

    hourly["temp_final"] = hourly["temp"].copy()
    hourly.loc[hourly["temp_final"].isna(), "temp_final"] = hourly["temp_interp"]

    hourly = hourly.drop(columns=["temp_interp"])
    return hourly


def make_lags(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["temp_lag1"] = df["temp_final"].shift(1)
    df["temp_lag2"] = df["temp_final"].shift(2)
    df["temp_lag3"] = df["temp_final"].shift(3)
    df = df.dropna(subset=["temp_final", "temp_lag1", "temp_lag2", "temp_lag3"])
    return df


def recursive_forecast(model, hourly_df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Forecast future temperatures recursively using last 3 known temp_final values.
    """
    base = hourly_df.copy().sort_index()

    last_valid = base["temp_final"].dropna()

    if len(last_valid) < 3:
        raise ValueError("ข้อมูลไม่พอสำหรับ forecast ล่วงหน้า ต้องมี temp_final อย่างน้อย 3 จุด")

    lag1 = float(last_valid.iloc[-1])
    lag2 = float(last_valid.iloc[-2])
    lag3 = float(last_valid.iloc[-3])

    last_time = last_valid.index[-1]

    future_rows = []

    for step in range(1, horizon_hours + 1):
        future_time = last_time + pd.Timedelta(hours=step)

        X_future = pd.DataFrame(
            [[lag1, lag2, lag3]],
            columns=["temp_lag1", "temp_lag2", "temp_lag3"]
        )

        pred = float(model.predict(X_future)[0])

        future_rows.append({
            "convert time": future_time,
            "forecast_temp": pred,
            "temp_lag1": lag1,
            "temp_lag2": lag2,
            "temp_lag3": lag3,
            "step_ahead": step
        })

        # shift lag
        lag3 = lag2
        lag2 = lag1
        lag1 = pred

    future_df = pd.DataFrame(future_rows).set_index("convert time")
    return future_df


def plot_actual_vs_pred(result_df: pd.DataFrame, daily_mode=False, selected_date=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        result_df.index,
        result_df["actual_temp"],
        label="Actual",
        linewidth=2,
        alpha=0.9
    )

    ax.plot(
        result_df.index,
        result_df["predicted_temp"],
        label="Predicted",
        linewidth=2,
        alpha=0.8
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")

    if daily_mode and selected_date is not None:
        ax.set_title(f"Actual vs Predicted Temperature — {selected_date}")
    else:
        ax.set_title("Actual vs Predicted Temperature (Hourly)")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    if daily_mode:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def plot_history_and_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))

    if not history_df.empty:
        ax.plot(
            history_df.index,
            history_df["actual_temp"],
            label="Historical Actual",
            linewidth=2,
            alpha=0.9
        )

    if not forecast_df.empty:
        ax.plot(
            forecast_df.index,
            forecast_df["forecast_temp"],
            label="Future Forecast",
            linewidth=2,
            linestyle="--",
            alpha=0.9
        )

        ax.axvline(forecast_df.index.min(), linestyle=":", linewidth=1.5, alpha=0.8)

    ax.set_title("Historical Temperature + Future Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    total_span_start = history_df.index.min() if not history_df.empty else forecast_df.index.min()
    total_span_end = forecast_df.index.max() if not forecast_df.empty else history_df.index.max()
    time_span = total_span_end - total_span_start

    if time_span <= pd.Timedelta(days=2):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    elif time_span <= pd.Timedelta(days=7):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


# ----------------------------
# UI
# ----------------------------
uploaded_csv = st.file_uploader("📤 อัปโหลดไฟล์ CSV", type=["csv"])

MAX_GAP_HOURS = 6
SHOW_ROWS = 10

if uploaded_csv:
    progress = st.progress(0)
    status_box = st.empty()

    with st.spinner("กำลังประมวลผล..."):
        try:
            status_box.info("Step 1/6: โหลดไฟล์ CSV ...")
            progress.progress(10)
            df_raw = pd.read_csv(uploaded_csv)

            if "convert time" not in df_raw.columns or "temp" not in df_raw.columns:
                st.error("ไฟล์ CSV ต้องมีคอลัมน์ 'convert time' และ 'temp'")
                st.stop()

            status_box.info("Step 2/6: Data Preparation ...")
            progress.progress(25)
            hourly = prepare_hourly(df_raw, max_gap_hours=MAX_GAP_HOURS)

            status_box.info("Step 3/6: Feature Engineering ...")
            progress.progress(45)
            feat = make_lags(hourly)

            if feat.empty:
                st.error("หลังสร้าง lag แล้วไม่เหลือข้อมูลใช้งาน กรุณาตรวจสอบข้อมูล input")
                st.stop()

            status_box.info("Step 4/6: Predict historical ...")
            progress.progress(65)
            X = feat[["temp_lag1", "temp_lag2", "temp_lag3"]]
            y_pred = model.predict(X)

            result = feat.copy()
            result["predicted_temp"] = y_pred
            result = result.rename(columns={"temp_final": "actual_temp"})

            status_box.info("Step 5/6: Forecast future ...")
            progress.progress(85)

            status_box.info("Step 6/6: เตรียมแสดงผล ...")
            progress.progress(100)
            status_box.success("เสร็จสิ้น! แสดงผลด้านล่าง ✅")

        except Exception as e:
            status_box.error("เกิดข้อผิดพลาดระหว่างประมวลผล")
            st.exception(e)
            st.stop()

    # ----------------------------
    # Filter / View
    # ----------------------------
    st.subheader("📅 Filter / View Options")

    available_dates = sorted(pd.unique(result.index.date))
    view_mode = st.radio("เลือกโหมดการแสดงผล", ["ทั้งหมด", "รายวัน"], horizontal=True)

    if view_mode == "รายวัน":
        selected_date = st.selectbox("เลือกวันที่ที่ต้องการดู", available_dates)
        filtered_result = result[result.index.date == selected_date].copy()
        filtered_hourly = hourly[hourly.index.date == selected_date].copy()
        daily_mode = True
    else:
        selected_date = None
        filtered_result = result.copy()
        filtered_hourly = hourly.copy()
        daily_mode = False

    if filtered_result.empty:
        st.warning("ไม่มีข้อมูลสำหรับวันที่เลือก")
        st.stop()

    # ----------------------------
    # Historical Metrics
    # ----------------------------
    y_true = filtered_result["actual_temp"].values
    y_hat = filtered_result["predicted_temp"].values
    mae = float(mean_absolute_error(y_true, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))

    st.subheader("📌 Model Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (°C)", f"{mae:.3f}")
    m2.metric("RMSE (°C)", f"{rmse:.3f}")
    m3.metric("Predicted points", f"{len(filtered_result)}")

    st.divider()

    # ----------------------------
    # Summary tables
    # ----------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("✅ Data Preparation Summary")
        st.write("ช่วงเวลา:", filtered_hourly.index.min(), "→", filtered_hourly.index.max())
        st.write("จำนวนชั่วโมงทั้งหมด:", len(filtered_hourly))
        st.write(
            "จำนวนชั่วโมงที่ยังว่าง (NaN) หลัง interpolate แบบไม่แต่ง long gap:",
            int(filtered_hourly["temp_final"].isna().sum()),
        )
        st.write("ตัวอย่างข้อมูลรายชั่วโมง:")
        st.dataframe(filtered_hourly.head(SHOW_ROWS), use_container_width=True)

    with colB:
        st.subheader("🔮 Prediction Output")
        st.write("ตัวอย่างผลลัพธ์:")
        st.dataframe(
            filtered_result[["actual_temp", "temp_lag1", "temp_lag2", "temp_lag3", "predicted_temp"]].head(SHOW_ROWS),
            use_container_width=True
        )

    st.divider()

    # ----------------------------
    # Historical Plot
    # ----------------------------
    st.subheader("📈 Actual vs Predicted")
    fig = plot_actual_vs_pred(
        filtered_result,
        daily_mode=daily_mode,
        selected_date=selected_date
    )
    st.pyplot(fig)

    st.divider()

    # ----------------------------
    # Future Forecast Controls
    # ----------------------------
    st.subheader("🔭 Future Forecast")
    forecast_hours = st.slider("เลือกจำนวนชั่วโมงที่ต้องการพยากรณ์ล่วงหน้า", 1, 72, 24)

    try:
        forecast_df = recursive_forecast(model, hourly, horizon_hours=forecast_hours)

        f1, f2, f3 = st.columns(3)
        f1.metric("Forecast horizon", f"{forecast_hours} ชั่วโมง")
        f2.metric("Last known temp", f"{hourly['temp_final'].dropna().iloc[-1]:.2f} °C")
        f3.metric("Forecast points", f"{len(forecast_df)}")

        st.write("ตัวอย่างข้อมูล forecast:")
        st.dataframe(forecast_df.head(SHOW_ROWS), use_container_width=True)

        # รวม history ช่วงท้ายกับ forecast เพื่อดูต่อเนื่อง
        recent_history = result[["actual_temp"]].tail(48).copy()

        st.subheader("📉 Historical + Future Forecast")
        fig2 = plot_history_and_forecast(recent_history, forecast_df)
        st.pyplot(fig2)

        # download
        st.subheader("⬇️ Download Forecast")
        forecast_out = forecast_df.reset_index()
        forecast_bytes = forecast_out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "ดาวน์โหลด forecast_future.csv",
            data=forecast_bytes,
            file_name="forecast_future.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.warning(f"ไม่สามารถ forecast ล่วงหน้าได้: {e}")

    st.divider()

    # ----------------------------
    # Download Historical Prediction
    # ----------------------------
    st.subheader("⬇️ Download Predictions")
    out_csv = filtered_result.reset_index()
    csv_bytes = out_csv.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "ดาวน์โหลด predictions_filtered.csv",
        data=csv_bytes,
        file_name="predictions_filtered.csv",
        mime="text/csv",
    )

else:
    st.info("อัปโหลดไฟล์ CSV เพื่อเริ่มทำนาย")
