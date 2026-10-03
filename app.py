"""DeliverWise Solo Streamlit demonstration."""

from pathlib import Path

import streamlit as st

from prediction import load_pipeline, predict_late


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "deliverwise_logistic_pipeline.joblib"
STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
    "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
    "RS", "SC", "SE", "SP", "TO",
]
WEEKDAYS = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


st.set_page_config(
    page_title="DeliverWise Solo",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_pipeline(MODEL_PATH)


st.title("DeliverWise Solo")
st.subheader("Late delivery prediction with Logistic Regression")
st.write(
    "Enter information available around order approval. The demonstration uses "
    "the same eleven features as the verified Kaggle notebook."
)

with st.expander("How to interpret the result"):
    st.write(
        "The model returns the estimated probability that an eventually delivered "
        "order will arrive after its promised date. A probability of 50% or more "
        "is labelled as a late-delivery warning. This is an academic prototype "
        "trained on historical Brazilian orders from 2016 to 2018."
    )

try:
    model = get_model()
except Exception as exc:
    st.error(f"The saved model could not be loaded. {exc}")
    st.stop()

with st.form("prediction_form"):
    st.markdown("### Order information")
    col1, col2, col3 = st.columns(3)
    with col1:
        item_count = st.number_input("Number of items", min_value=1, max_value=100, value=1)
        seller_count = st.number_input("Number of sellers", min_value=1, max_value=20, value=1)
        total_price = st.number_input("Total item price (BRL)", min_value=0.0, value=120.0, step=10.0)
        total_freight = st.number_input("Total freight (BRL)", min_value=0.0, value=20.0, step=2.0)
    with col2:
        approval_delay_hours = st.number_input(
            "Purchase-to-approval delay (hours)", min_value=0.0, value=1.0, step=0.5
        )
        promised_days = st.number_input(
            "Days from approval to promised delivery", min_value=0, max_value=120, value=20
        )
        approval_hour = st.slider("Approval hour", min_value=0, max_value=23, value=12)
        weekday_name = st.selectbox("Approval weekday", list(WEEKDAYS))
    with col3:
        approval_month = st.selectbox("Approval month", list(range(1, 13)), index=5)
        customer_state = st.selectbox("Customer state", STATES, index=24)
        seller_state = st.selectbox("Seller state", STATES, index=24)
        threshold = st.slider(
            "Warning threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="The project evaluation uses 0.50. Changing this value is for demonstration only.",
        )
    submitted = st.form_submit_button("Predict delivery risk", type="primary")

if submitted:
    values = {
        "item_count": item_count,
        "seller_count": seller_count,
        "total_price": total_price,
        "total_freight": total_freight,
        "approval_delay_hours": approval_delay_hours,
        "promised_days": promised_days,
        "approval_hour": approval_hour,
        "approval_weekday": WEEKDAYS[weekday_name],
        "approval_month": approval_month,
        "customer_state": customer_state,
        "seller_state": seller_state,
    }
    try:
        result = predict_late(model, values, threshold)
    except Exception as exc:
        st.error(f"Prediction failed. {exc}")
    else:
        probability = result["probability"]
        st.markdown("### Prediction")
        left, right = st.columns([1, 2])
        with left:
            st.metric("Estimated late-delivery probability", f"{probability:.1%}")
        with right:
            if result["predicted_late"]:
                st.warning(
                    f"Late-delivery warning: probability is at or above the "
                    f"{threshold:.0%} threshold."
                )
            else:
                st.success(
                    f"No late-delivery warning: probability is below the "
                    f"{threshold:.0%} threshold."
                )
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "Use this result for demonstration only. The model has low precision at "
            "the fixed evaluation threshold and should not make operational decisions."
        )
        with st.expander("Model input row"):
            st.dataframe(result["input"], use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "DeliverWise Solo | Olist historical dataset | scikit-learn Logistic Regression"
)

