import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Startup Assistant", layout="centered")

page = st.sidebar.selectbox("Choose Feature", ["💬 Chatbot", "📊 Valuation Predictor"])

st.title("💡🧠 AI Startup Assistant App")

if page == "💬 Chatbot":
    st.subheader("💬 Ask Your Question")

    user_query = st.text_input("Enter your question:")
    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Getting response..."):
                response = requests.post(f"{API_URL}/chatbot-query", json={"query": user_query})
                if response.status_code == 200:
                    st.success(response.json()["response"])
                else:
                    st.error("Error communicating with chatbot API.")

elif page == "📊 Valuation Predictor":
    st.subheader("📊 Predict Startup Valuation")

    amount = st.number_input("Funding Amount", min_value=0.0, format="%.2f")
    unit = st.selectbox("Unit", ["millions", "billions"])
    growth_rate = st.number_input("Growth Rate (%)", min_value=0.0, format="%.2f")
    industry = st.text_input("Industry (Select-Fintech, AI, SaaS, Sustainability & energy, Health & biotech, Consumer & retail tech)")
    country = st.text_input("Country (e.g., USA, India)")

    if st.button("Predict Valuation"):
        if amount == 0 or growth_rate == 0 or not industry or not country:
            st.warning("Please fill all inputs.")
        else:
            with st.spinner("Predicting valuation..."):
                payload = {
                    "amount": amount,
                    "growth_rate": growth_rate,
                    "industry": industry,
                    "country": country,
                    "unit": unit
                }
                response = requests.post(f"{API_URL}/predict-valuation", json=payload)
                if response.status_code == 200 and "valuation" in response.json():
                    result = response.json()["valuation"]
                    st.success(f"💰 Estimated Valuation: {result:.2f} {unit}")
                else:
                    st.error("Error predicting valuation.")

    st.markdown(
        """
        <div style="margin-top:2rem; font-size:0.8rem; color:gray;">
        ⚙️ <strong>Data Sources & Assumptions</strong><br>
        – Funding & valuation benchmarks aggregated from TechCrunch, Crunchbase, PitchBook, etc.<br>
        – Industry growth rates set by category (e.g. deep tech/AI = 20%, aerospace = 7.8%).<br>
        </div>
        """,
        unsafe_allow_html=True
    )
