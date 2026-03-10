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
**Workflow:** Upload CSV → Data Preparation → Resample 1H → Lag Features → Load model → Predict → Show results  
ไฟล์ต้องมีคอลัมน์เวลา **convert time** และคอลัมน์อุณหภูมิ **temp**
""")

# ----------------------------
# Model loading (from file)
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
    """
    Data Preparation:
    - parse datetime
    - sort
    - resample hourly mean
    - interpolate ONLY short gaps (<= max_gap_hours)
    """
    df = df_raw.copy()

    # Parse time
    df["convert time"] = pd.to_datetime(df["convert time"], errors="coerce")
    df = df.dropna(subset=["convert time"])
    df = df.sort_values("convert time")

    # Keep needed cols
    df = df[["convert time", "temp"]].copy()
    df = df.set_index("convert time")

    # Resample hourly mean
    hourly = df.resample("1h").mean()

    # Interpolate only short gaps
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
    """Feature Engineering: create temp lag1/2/3"""
    df = hourly_df.copy()
    df["temp_lag1"] = df["temp_final"].shift(1)
    df["temp_lag2"] = df["temp_final"].shift(2)
    df["temp_lag3"] = df["temp_final"].shift(3)
    df = df.dropna(subset=["temp_final", "temp_lag1", "temp_lag2", "temp_lag3"])
    return df


def plot_actual_vs_pred(result_df: pd.DataFrame, daily_mode=False, selected_date=None):

    fig, ax = plt.subplots(figsize=(14,5))

    # เส้น Actual
    ax.plot(
        result_df.index,
        result_df["actual_temp"],
        label="Actual",
        linewidth=2,
        alpha=0.9
    )

    # เส้น Predicted
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

    # -------------------------
    # X axis
    # -------------------------
    if daily_mode:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.xticks(rotation=30)

    fig.tight_layout()

    return fig


# ----------------------------
# UI: Upload
# ----------------------------
uploaded_csv = st.file_uploader("📤 อัปโหลดไฟล์ CSV", type=["csv"])

MAX_GAP_HOURS = 6
SHOW_ROWS = 10

if uploaded_csv:
    progress = st.progress(0)
    status_box = st.empty()

    with st.spinner("กำลังประมวลผล..."):
        try:
            status_box.info("Step 1/5: โหลดไฟล์ CSV ...")
            progress.progress(10)
            df_raw = pd.read_csv(uploaded_csv)

            if "convert time" not in df_raw.columns or "temp" not in df_raw.columns:
                st.error("ไฟล์ CSV ต้องมีคอลัมน์ 'convert time' และ 'temp'")
                st.stop()

            status_box.info("Step 2/5: Data Preparation ...")
            progress.progress(35)
            hourly = prepare_hourly(df_raw, max_gap_hours=MAX_GAP_HOURS)

            status_box.info("Step 3/5: Feature Engineering ...")
            progress.progress(55)
            feat = make_lags(hourly)

            status_box.info("Step 4/5: Predict ...")
            progress.progress(75)
            X = feat[["temp_lag1", "temp_lag2", "temp_lag3"]]
            y_pred = model.predict(X)

            status_box.info("Step 5/5: สร้างผลลัพธ์ ...")
            progress.progress(90)

            result = feat.copy()
            result["predicted_temp"] = y_pred
            result = result.rename(columns={"temp_final": "actual_temp"})

            progress.progress(100)
            status_box.success("เสร็จสิ้น! แสดงผลด้านล่าง ✅")

        except Exception as e:
            status_box.error("เกิดข้อผิดพลาดระหว่างประมวลผล")
            st.exception(e)
            st.stop()

    # ----------------------------
    # Filter section
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

    # Metrics ตามช่วงที่เลือก
    y_true = filtered_result["actual_temp"].values
    y_hat = filtered_result["predicted_temp"].values
    mae = float(mean_absolute_error(y_true, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))

    # ----------------------------
    # Metrics section
    # ----------------------------
    st.subheader("📌 Model Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (°C)", f"{mae:.3f}")
    m2.metric("RMSE (°C)", f"{rmse:.3f}")
    m3.metric("Predicted points", f"{len(filtered_result)}")

    st.divider()

    # ----------------------------
    # Show summaries
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
        st.dataframe(filtered_hourly.head(SHOW_ROWS))

    with colB:
        st.subheader("🔮 Prediction Output")
        st.write("ตัวอย่างผลลัพธ์:")
        st.dataframe(
            filtered_result[["actual_temp", "temp_lag1", "temp_lag2", "temp_lag3", "predicted_temp"]].head(SHOW_ROWS)
        )

    st.divider()

    # ----------------------------
    # Plot
    # ----------------------------
    st.subheader("📈 Actual vs Predicted")
    fig = plot_actual_vs_pred(
        filtered_result,
        daily_mode=daily_mode,
        selected_date=selected_date
    )
    st.pyplot(fig)

    # ----------------------------
    # Download
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


