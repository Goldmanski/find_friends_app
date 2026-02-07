# User Clustering App (Streamlit)

## Overview

This project is an interactive **Streamlit application** that assigns a user to a cluster based on survey responses.
The clustering model was trained using **PyCaret**, and the app visualizes statistics for users with similar profiles.

The goal of the project is to demonstrate how a simple **machine learning model can be integrated into a web application** with user interaction and data visualization.

---

## Features

* Survey form in the sidebar
* User cluster prediction using a trained model
* Cluster name and description
* Number of similar users in the dataset
* Group statistics visualization:

  * age distribution
  * education level
  * favorite animals
  * favorite place
  * gender distribution
* Option to append user responses to the dataset
* Cached model and dataset loading
* Dashboard-style layout

---

## Tech Stack

* Python
* Streamlit
* Pandas
* PyCaret
* Plotly

---

## Project Structure

```
.
├── app.py
├── welcome_survey_simple_v2.csv
├── welcome_survey_clustering_pipeline_v2.pkl
├── welcome_survey_cluster_names_and_descriptions_v2.json
└── requirements.txt
```

---

## Run Locally

### Install dependencies

```
pip install -r requirements.txt
```

### Start the app

```
streamlit run app.py
```

---

## How It Works

1. The user fills out a survey in the sidebar.
2. Responses are converted into a DataFrame.
3. The clustering model assigns the user to a cluster.
4. The app displays:

   * cluster name
   * cluster description
   * number of similar users
   * group statistics
5. Optionally, responses can be saved to the dataset.

---

## Possible Improvements

* Similarity search between users
* Recommendation system
* Database integration
* Model retraining pipeline
* Dataset dashboard
* Production deployment

---

## Project Goal

This project demonstrates:

* clustering workflow
* data preprocessing
* integration of ML models with Streamlit
* interactive dashboards
* user data handling
* visualization of categorical data

---

## Author

Created as a machine learning + Streamlit practice project.
