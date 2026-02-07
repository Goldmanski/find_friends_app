import json
import streamlit as st
import pandas as pd # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import time

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Dopasowanie użytkowników", layout="wide")
st.title("Znajdź swoją grupę")
st.caption("Prosta segmentacja użytkowników na podstawie ankiety")

# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# ============================================================
# CACHE FUNCTIONS
# ============================================================

@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

# ============================================================
# LOAD RESOURCES
# ============================================================

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# ============================================================
# SIDEBAR INPUT
# ============================================================

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    st.divider()
    age_options = ['<18','18-24','25-34','35-44','45-54','55-64','>=65','unknown']
    age = st.selectbox("Wiek", age_options)
    #age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    animal_options = ["Brak ulubionych", "Psy", "Koty", "Koty i Psy", "Inne"]
    fav_animals = st.selectbox("Ulubione zwierzęta", animal_options)
    #fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

    if st.button("Zapisz moje odpowiedzi do datasetu"):
        person_df.to_csv(DATA, sep=";", mode="a", header=False, index=False)
        get_all_participants.clear()
        st.session_state.saved = True
        st.rerun()

    if st.session_state.get("saved", False):
        st.success("Twoje odpowiedzi zostały zapisane!")
        st.session_state.saved = False

    st.caption(f"Liczba uczestników datasetu: {len(all_df)}")

# ============================================================
# PREDICTION
# ============================================================

with st.spinner("Analizujemy odpowiedzi..."):
    time.sleep(2)  # sztuczne opóźnienie
    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# ============================================================
# DATA PREPARATION
# ============================================================    

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# ============================================================
# UI
# ============================================================

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Wynik dopasowania")   
    st.success(f"Twoja grupa: {predicted_cluster_data['name']}")
    #st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
    st.caption(f"ID Klastra: {predicted_cluster_id}")
    st.markdown(predicted_cluster_data['description'])

    st.divider()

    st.metric(
        "Osoby podobne do Ciebie",
        len(same_cluster_df),
        f"{len(same_cluster_df)/len(all_df):.0%} uczestników"
    )

with col2:
    st.subheader("Statystyki grupy")

    age_order = ['<18','18-24','25-34','35-44','45-54','55-64','>=65','unknown']
    fig = px.histogram(same_cluster_df, x="age", category_orders={"age": age_order})
    #fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
    fig.update_layout(
        title="Rozkład wieku w grupie",
        xaxis_title="Wiek",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="edu_level")
    fig.update_layout(
        title="Rozkład wykształcenia w grupie",
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="fav_animals")
    fig.update_layout(
        title="Rozkład ulubionych zwierząt w grupie",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="fav_place")
    fig.update_layout(
        title="Rozkład ulubionych miejsc w grupie",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="gender")
    fig.update_layout(
        title="Rozkład płci w grupie",
        xaxis_title="Płeć",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig, use_container_width=True)

