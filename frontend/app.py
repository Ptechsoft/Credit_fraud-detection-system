import streamlit as st
import pandas as pd
import requests

## CONFIG NETWORK PORT

API_URL_SINGLE = "http://127.0.0.1:8007/predict"
API_URL_BATCH = "http://127.0.0.1:8007/predict_batch"

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide"
)

## HEADER

st.title("💳 Credit Card Fraud Detection Dashboard")
st.caption("Analyze transactions and detect fraud using Machine Learning")

## SIDEBAR

st.sidebar.header("⚙️ Controls")

mode = st.sidebar.radio(
    "Choose Input Method",
    ["Manual Entry", "CSV Upload"]
)

## FUNCTIONS

def show_kpis(total, fraud, legit, avg_risk):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📊 Total Transactions", total)
    col2.metric("🚨 Fraud Cases", fraud)
    col3.metric("✅ Legitimate", legit)
    col4.metric("📈 Avg Risk %", f"{avg_risk:.2f}%")


def show_risk(score):
    percent = int(score * 100)

    st.subheader("🔍 Risk Score")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.progress(percent)

    with col2:
        st.metric("Risk %", f"{percent}%")

    if percent < 30:
        st.success("Low Risk")
    elif percent < 70:
        st.warning("Medium Risk")
    else:
        st.error("High Risk 🚨")


## MANUAL ENTRY MODE

if mode == "Manual Entry":

    st.header("🧾 Manual Transaction Input")

    input_data = {}

    ## Create columns layout
    cols = st.columns(4)

    ## TIME INPUT    
    with cols[0]:
        input_data["Time"] = st.number_input(
            "Time",
            value=0.0,
            format="%.6f"
        )

    ## V1 TO V28 INPUTS    
    for i in range(28):
        with cols[(i + 1) % 4]:
            input_data[f"V{i+1}"] = st.number_input(
                f"V{i+1}",
                value=0.0,
                format="%.6f"
            )

    ## AMOUNT INPUT
    input_data["Amount"] = st.number_input(
        "Amount",
        value=0.0,
        format="%.2f"
    )

    ## PREDICT BUTTON
    if st.button("🔍 Predict Transaction"):

        try:

            with st.spinner("Analyzing transaction..."):

                response = requests.post(
                    API_URL_SINGLE,
                    json=input_data
                )

            ## SUCCESS RESPONSE
            if response.status_code == 200:

                result = response.json()

                prediction = result.get("prediction", 0)
                probability = result.get("probability", 0)

                st.subheader("🧠 Prediction Result")

                if prediction == 1:
                    st.error("🚨 Fraudulent Transaction Detected")
                else:
                    st.success("✅ Legitimate Transaction")

                show_risk(probability)

                st.subheader("📋 API Response")
                st.json(result)

            ## API ERROR
            else:
                st.error(f"API Error: {response.text}")

        ## CONNECTION ERROR
        except Exception as e:
            st.error(f"Connection Error: {e}")


## CSV UPLOAD MODE
elif mode == "CSV Upload":

    st.header("📂 Upload Transactions CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    ## FILE UPLOADED
    if uploaded_file is not None:

        st.success(f"Uploaded File: {uploaded_file.name}")

        try:

            ## Read CSV
            df = pd.read_csv(uploaded_file)

            ## PREVIEW
            st.subheader("👀 CSV Preview")
            st.dataframe(df.head())

            ## REQUIRED COLUMNS
            expected_cols = (
                ["Time"] +
                [f"V{i}" for i in range(1, 29)] +
                ["Amount"]
            )

            ## VALIDATION
            missing_cols = [
                col for col in expected_cols
                if col not in df.columns
            ]

            if missing_cols:

                st.error(
                    f"❌ Missing Required Columns: {missing_cols}"
                )

                st.info(
                    "CSV must contain: Time, V1-V28, Amount"
                )

            else:

                st.success("✅ CSV Structure Validated")

                st.write(f"Rows Detected: {len(df)}")

                ## RUN DETECTION
                if st.button("🚀 Run Fraud Detection"):

                    try:

                        with st.spinner(
                            "Analyzing transactions..."
                        ):

                            response = requests.post(
                                API_URL_BATCH,
                                json=df.to_dict(
                                    orient="records"
                                )
                            )

                        ## SUCCESS RESPONSE
                        if response.status_code == 200:

                            results = response.json()

                            ## ADD RESULTS TO DATAFRAME
                            df["Prediction"] = [
                                item["prediction"]
                                for item in results
                            ]

                            df["Risk Score"] = [
                                round(
                                    item["probability"] * 100,
                                    2
                                )
                                for item in results
                            ]

                            ## KPI SECTION
                            total = len(df)
                            fraud = int(df["Prediction"].sum())
                            legit = total - fraud
                            avg_risk = df["Risk Score"].mean()

                            st.subheader("📊 Key Metrics")

                            show_kpis(
                                total,
                                fraud,
                                legit,
                                avg_risk
                            )

                            ## RESULTS TABLE
                            st.subheader("📋 Detailed Results")

                            st.dataframe(
                                df,
                                use_container_width=True
                            )

                            ## DOWNLOAD BUTTON
                            csv_data = df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="⬇️ Download Results CSV",
                                data=csv_data,
                                file_name="fraud_results.csv",
                                mime="text/csv"
                            )

                            ## FRAUD DISTRIBUTION
                            st.subheader(
                                "📊 Fraud Distribution"
                            )

                            chart_data = (
                                df["Prediction"]
                                .value_counts()
                            )

                            st.bar_chart(chart_data)

                            ## RISK TREND
                            st.subheader(
                                "📈 Risk Score Trend"
                            )

                            st.line_chart(
                                df["Risk Score"]
                            )

                        ## API ERROR
                        else:
                            st.error(
                                f"API Error: {response.text}"
                            )

                    ## CONNECTION ERROR
                    except Exception as e:
                        st.error(
                            f"Connection Error: {e}"
                        )

        ## FILE ERROR
        except Exception as e:
            st.error(f"File Error: {e}")