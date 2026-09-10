import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(
    page_title="FinSight MY",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# File paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# Model results
MODEL_ACCURACY = {
    "Logistic Regression": 56.67,
    "Random Forest": 55.33,
    "XGBoost": 52.00
}


MODEL_METRICS = {
    "Logistic Regression": {
        "Accuracy": 56.67,
        "Precision": 56.17,
        "Recall": 56.67,
        "F1 Score": 55.03
    },
    "Random Forest": {
        "Accuracy": 55.33,
        "Precision": 54.50,
        "Recall": 55.33,
        "F1 Score": 54.58
    },
    "XGBoost": {
        "Accuracy": 52.00,
        "Precision": 51.42,
        "Recall": 52.00,
        "F1 Score": 51.69
    }
}


RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}


# Load the saved models
@st.cache_resource
def load_models():

    return {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_model.pkl"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest_model.pkl"
        ),
        "XGBoost": joblib.load(
            MODEL_DIR / "xgboost_model.pkl"
        )
    }


models = load_models()


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# Sidebar
with st.sidebar:

    st.title("FinSight MY")

    st.caption(
        "Malaysian Household Financial Risk Prediction"
    )

    st.divider()

    st.subheader("Navigation")

    selected_page = st.radio(
        "Go to",
        [
            "Risk Prediction",
            "Model Performance",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Prediction Model")

    selected_model = st.selectbox(
        "Choose model",
        [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ]
    )

    accuracy = MODEL_ACCURACY[selected_model]

    st.metric(
        "Test Accuracy",
        f"{accuracy:.2f}%"
    )

    st.progress(accuracy / 100)

    st.divider()

    st.caption("FinSight MY")
    st.caption("Machine Learning Project")


# =========================================================
# Risk Prediction
# =========================================================

if selected_page == "Risk Prediction":

    st.title("FinSight MY")

    st.caption(
        "Malaysian Financial Risk Prediction"
    )

    st.divider()

    st.subheader("Household Financial Profile")

    st.write(
        "Enter the socioeconomic indicators to estimate "
        "the household financial risk level."
    )


    # Preset values
    scenarios = {

        "Custom": {
            "poverty_relative": 7.0,
            "poverty_absolute": 5.0,
            "income_median": 5000.0,
            "income_mean": 6000.0,
            "gini": 0.35
        },

        "Lower Income Household": {
            "poverty_relative": 15.0,
            "poverty_absolute": 10.0,
            "income_median": 2500.0,
            "income_mean": 3000.0,
            "gini": 0.45
        },

        "Middle Income Household": {
            "poverty_relative": 7.0,
            "poverty_absolute": 4.0,
            "income_median": 5000.0,
            "income_mean": 6000.0,
            "gini": 0.38
        },

        "Higher Income Household": {
            "poverty_relative": 2.0,
            "poverty_absolute": 1.0,
            "income_median": 9000.0,
            "income_mean": 11000.0,
            "gini": 0.30
        }
    }


    scenario = st.selectbox(
        "Scenario Preset",
        list(scenarios.keys())
    )


    values = scenarios[scenario]


    # Input fields
    col1, col2 = st.columns(2)


    with col1:

        poverty_relative = st.number_input(
            "Relative Poverty Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(values["poverty_relative"]),
            step=0.1
        )

        poverty_absolute = st.number_input(
            "Absolute Poverty Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(values["poverty_absolute"]),
            step=0.1
        )

        income_median = st.number_input(
            "Median Monthly Household Income (RM)",
            min_value=0.0,
            max_value=100000.0,
            value=float(values["income_median"]),
            step=100.0
        )


    with col2:

        income_mean = st.number_input(
            "Mean Monthly Household Income (RM)",
            min_value=0.0,
            max_value=100000.0,
            value=float(values["income_mean"]),
            step=100.0
        )

        gini = st.number_input(
            "Gini Coefficient",
            min_value=0.0,
            max_value=1.0,
            value=float(values["gini"]),
            step=0.01,
            format="%.2f",
            help="Higher values indicate greater income inequality."
        )

        st.info(
            f"Selected Model: {selected_model}\n\n"
            f"Test Accuracy: {accuracy:.2f}%"
        )


    # Prediction buttons
    col1, col2 = st.columns(2)


    with col1:

        predict = st.button(
            "🔮 Predict Financial Risk",
            use_container_width=True
        )


    with col2:

        reset = st.button(
            "↺ Reset Profile",
            use_container_width=True
        )


    if reset:

        st.session_state.prediction_result = None

        st.rerun()


    # Make prediction
    if predict:

        model = models[selected_model]


        input_data = pd.DataFrame(
            [[
                poverty_relative,
                poverty_absolute,
                income_median,
                income_mean,
                gini
            ]],
            columns=[
                "poverty_relative",
                "poverty_absolute",
                "income_median",
                "income_mean",
                "gini"
            ]
        )


        # Match the order used when training the model
        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

            missing_features = [
                feature
                for feature in expected_features
                if feature not in input_data.columns
            ]

            if missing_features:

                st.error(
                    f"Model expects features that are missing: "
                    f"{missing_features}"
                )

                st.stop()

            input_data = input_data[
                expected_features
            ]


        prediction = int(
            model.predict(input_data)[0]
        )


        # Get prediction probabilities
        if hasattr(model, "predict_proba"):

            raw_probabilities = model.predict_proba(
                input_data
            )[0]

            probabilities = [
                0.0,
                0.0,
                0.0
            ]

            for class_value, probability in zip(
                model.classes_,
                raw_probabilities
            ):

                class_value = int(class_value)

                if class_value in [0, 1, 2]:

                    probabilities[class_value] = float(
                        probability
                    )

        else:

            probabilities = [
                0.0,
                0.0,
                0.0
            ]


        risk_label = RISK_LABELS.get(
            prediction,
            "Unknown Risk"
        )


        confidence = max(probabilities) * 100


        st.session_state.prediction_result = {
            "prediction": prediction,
            "risk_label": risk_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "model": selected_model
        }


    # Show prediction
    result = st.session_state.prediction_result


    if result is not None:

        prediction = result["prediction"]
        risk_label = result["risk_label"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]


        st.divider()

        st.subheader("Prediction Result")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Financial Risk",
                risk_label
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        with col3:

            st.metric(
                "Model",
                selected_model
            )


        # Risk message
        if prediction == 0:

            st.success(
                "The model estimates a Low Risk level based "
                "on the socioeconomic indicators provided."
            )

        elif prediction == 1:

            st.warning(
                "The model estimates a Medium Risk level. "
                "The household profile shows some signs of "
                "financial vulnerability."
            )

        else:

            st.error(
                "The model estimates a High Risk level based "
                "on the socioeconomic indicators provided."
            )


        # Probability
        st.subheader("Risk Probability")


        probability_cols = st.columns(3)


        for i, column in enumerate(probability_cols):

            with column:

                st.metric(
                    RISK_LABELS[i],
                    f"{probabilities[i] * 100:.2f}%"
                )


        probability_df = pd.DataFrame({

            "Risk Level": [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],

            "Probability": [
                probabilities[0] * 100,
                probabilities[1] * 100,
                probabilities[2] * 100
            ]
        })


        fig = px.bar(
            probability_df,
            x="Risk Level",
            y="Probability",
            text="Probability",
            title="Risk Probability Distribution"
        )


        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )


        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="",
            yaxis_range=[
                0,
                max(
                    100,
                    max(probability_df["Probability"]) + 15
                )
            ],
            height=400
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # Household values
        st.subheader("Indicator Values")


        value_col1, value_col2, value_col3 = st.columns(3)


        with value_col1:

            st.metric(
                "Relative Poverty",
                f"{poverty_relative:.1f}%"
            )

            st.metric(
                "Absolute Poverty",
                f"{poverty_absolute:.1f}%"
            )


        with value_col2:

            st.metric(
                "Median Income",
                f"RM {income_median:,.0f}"
            )

            st.metric(
                "Mean Income",
                f"RM {income_mean:,.0f}"
            )


        with value_col3:

            st.metric(
                "Gini Coefficient",
                f"{gini:.2f}"
            )

            st.write(
                "Higher Gini values indicate greater "
                "income inequality."
            )


        # Financial profile chart
        st.subheader("Household Financial Profile")


        # Values are scaled to 0-100 for the radar chart
        radar_values = [

            min(
                poverty_relative / 20,
                1
            ) * 100,

            min(
                poverty_absolute / 20,
                1
            ) * 100,

            min(
                income_median / 10000,
                1
            ) * 100,

            min(
                income_mean / 12000,
                1
            ) * 100,

            gini * 100
        ]


        radar_categories = [

            "Relative Poverty",

            "Absolute Poverty",

            "Median Income",

            "Mean Income",

            "Income Inequality"
        ]


        actual_values = [

            f"{poverty_relative:.1f}%",

            f"{poverty_absolute:.1f}%",

            f"RM {income_median:,.0f}",

            f"RM {income_mean:,.0f}",

            f"{gini:.2f}"
        ]


        # Close the radar shape
        radar_values.append(
            radar_values[0]
        )

        radar_categories.append(
            radar_categories[0]
        )

        actual_values.append(
            actual_values[0]
        )


        fig_radar = go.Figure()


        fig_radar.add_trace(
            go.Scatterpolar(

                r=radar_values,

                theta=radar_categories,

                mode="lines+markers+text",

                text=actual_values,

                textposition="top center",

                fill="toself",

                marker=dict(
                    size=9
                ),

                line=dict(
                    width=2
                ),

                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    "Value: %{text}"
                    "<extra></extra>"
                )
            )
        )


        fig_radar.update_layout(

            title={
                "text": "Financial Indicator Overview",
                "x": 0.5
            },

            template="plotly_dark",

            polar=dict(

                bgcolor="rgba(0,0,0,0)",

                radialaxis=dict(

                    visible=True,

                    range=[0, 100],

                    tickvals=[
                        0,
                        20,
                        40,
                        60,
                        80,
                        100
                    ],

                    ticktext=[
                        "0",
                        "20",
                        "40",
                        "60",
                        "80",
                        "100"
                    ],

                    gridcolor="gray",

                    linecolor="gray",

                    tickfont=dict(
                        size=11
                    )
                ),

                angularaxis=dict(

                    gridcolor="gray",

                    linecolor="gray",

                    tickfont=dict(
                        size=13
                    )
                )
            ),

            showlegend=False,

            height=500,

            margin=dict(
                l=80,
                r=80,
                t=80,
                b=80
            )
        )


        st.plotly_chart(
            fig_radar,
            use_container_width=True
        )


        st.caption(
            "The radar chart uses a 0–100 scale to visually "
            "compare the five indicators. The values shown "
            "above are the actual household values."
        )


        # Interpretation
        st.subheader("Interpretation")


        if prediction == 0:

            st.write(
                """
                The model classifies this household profile
                as Low Risk. The combination of poverty rates,
                household income and income inequality indicators
                is associated with a lower predicted financial
                risk level.
                """
            )

        elif prediction == 1:

            st.write(
                """
                The model classifies this household profile
                as Medium Risk. The household profile contains
                some socioeconomic indicators associated with
                moderate financial vulnerability.
                """
            )

        else:

            st.write(
                """
                The model classifies this household profile
                as High Risk. The socioeconomic indicators
                provided are associated with a higher predicted
                level of financial vulnerability.
                """
            )


        # Download prediction
        result_df = pd.DataFrame({

            "Model": [
                selected_model
            ],

            "Relative Poverty (%)": [
                poverty_relative
            ],

            "Absolute Poverty (%)": [
                poverty_absolute
            ],

            "Median Income (RM)": [
                income_median
            ],

            "Mean Income (RM)": [
                income_mean
            ],

            "Gini": [
                gini
            ],

            "Predicted Risk": [
                risk_label
            ],

            "Confidence (%)": [
                confidence
            ]
        })


        csv = result_df.to_csv(
            index=False
        )


        st.download_button(
            "⬇ Download Prediction Results",
            data=csv,
            file_name="finsight_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# Model Performance
# =========================================================

elif selected_page == "Model Performance":

    st.title("Model Performance")

    st.caption(
        "Comparison of the three machine learning models."
    )

    st.divider()


    st.subheader("Best Performing Model")


    st.success(
        "Logistic Regression achieved the highest test "
        "accuracy among the three models."
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            "56.67%"
        )


    with col2:

        st.metric(
            "Precision",
            "56.17%"
        )


    with col3:

        st.metric(
            "Recall",
            "56.67%"
        )


    with col4:

        st.metric(
            "F1 Score",
            "55.03%"
        )


    st.subheader("Model Comparison")


    performance_df = pd.DataFrame(
        MODEL_METRICS
    ).T


    performance_df.index.name = "Model"


    st.dataframe(
        performance_df.style.format(
            "{:.2f}%"
        ),
        use_container_width=True
    )


    st.subheader("Performance Comparison")


    chart_df = performance_df.reset_index()


    chart_long = chart_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )


    fig = px.bar(
        chart_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        title="Machine Learning Model Performance"
    )


    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig.update_layout(
        yaxis_title="Score (%)",
        xaxis_title="",
        yaxis_range=[0, 70],
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader("Model Selection")


    st.write(
        """
        Logistic Regression was selected as the main model
        because it achieved the highest overall test accuracy.

        Random Forest and XGBoost were used as comparison
        models to evaluate different machine learning approaches.
        """
    )


    st.info(
        """
        The model performance is moderate, showing that
        financial risk prediction is a challenging problem.
        The system should be used as an analytical tool rather
        than as a definitive financial assessment.
        """
    )


# =========================================================
# About
# =========================================================

elif selected_page == "About":

    st.title("About FinSight MY")

    st.caption(
        "Malaysian Household Financial Risk Prediction"
    )

    st.divider()


    st.subheader("Project Overview")


    st.write(
        """
        FinSight MY is a machine learning project that uses
        Malaysian household socioeconomic indicators to
        estimate financial risk levels.

        The system classifies a household profile into
        Low Risk, Medium Risk or High Risk.
        """
    )


    st.subheader("Project Objective")


    st.write(
        """
        The main objective is to develop a simple machine
        learning system that can analyse socioeconomic
        indicators and classify household financial risk.
        """
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Risk Level 1",
            "Low Risk"
        )


    with col2:

        st.metric(
            "Risk Level 2",
            "Medium Risk"
        )


    with col3:

        st.metric(
            "Risk Level 3",
            "High Risk"
        )


    st.subheader("Input Indicators")


    indicator_df = pd.DataFrame({

        "Indicator": [

            "Relative Poverty",

            "Absolute Poverty",

            "Median Income",

            "Mean Income",

            "Gini Coefficient"
        ],

        "Description": [

            "Relative poverty rate",

            "Absolute poverty rate",

            "Median monthly household income",

            "Mean monthly household income",

            "Income inequality measure"
        ]
    })


    st.dataframe(
        indicator_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Machine Learning Models")


    model_df = pd.DataFrame({

        "Model": [

            "Logistic Regression",

            "Random Forest",

            "XGBoost"
        ],

        "Test Accuracy": [

            "56.67%",

            "55.33%",

            "52.00%"
        ],

        "Role": [

            "Main model",

            "Comparison model",

            "Comparison model"
        ]
    })


    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Technologies Used")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Language",
            "Python"
        )


    with col2:

        st.metric(
            "Machine Learning",
            "Scikit-learn"
        )


    with col3:

        st.metric(
            "Dashboard",
            "Streamlit"
        )


    with col4:

        st.metric(
            "Visualisation",
            "Plotly"
        )


    st.subheader("Project Limitations")


    st.warning(
        """
        FinSight MY is an academic machine learning project.
        The predictions are based on the socioeconomic indicators
        included in the dataset and should not be treated as
        professional financial advice.

        Model performance is also affected by the available
        dataset, selected features and sample size.
        """
    )


    st.divider()


    st.caption(
        "FinSight MY — Malaysian Household Financial Risk Prediction"
    )

   