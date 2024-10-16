import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Customize Matplotlib style
plt.style.use('dark_background')
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Function to shorten categories
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# Load and process data for exploration
@st.cache
def load_data():
    df = pd.read_csv("jobs_in_data.csv")

    columns_to_drop = ['salary', 'salary_currency', 'job_category', 'work_year', 'company_size', 'employment_type', 'employee_residence']
    existing_columns = df.columns.tolist()
    columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df.rename(columns={'salary_in_usd': 'salary'}, inplace=True)
    df['work_setting'] = df['work_setting'].replace('In-person', 'Office')

    country_map_job_title = shorten_categories(df['job_title'].value_counts(), 150)
    df['job_title'] = df['job_title'].map(country_map_job_title)

    country_map_company_location = shorten_categories(df['company_location'].value_counts(), 20)
    df['company_location'] = df['company_location'].map(country_map_company_location)

    return df

# Load the CSV file for reference
df = load_data()

def load_model():
    with open('./saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

# Extract the regressor and label encoders
regressor_loaded = data["model"]
le_job_title = data["le_job_title"]
le_experience_level = data["le_experience_level"]
le_work_setting = data["le_work_setting"]
le_company_location = data["le_company_location"]

# Function to create the prediction page
def show_predict_page():
    st.title("📊Salary Prediction")

    st.info('This app predicts the salary for various data job roles based on input features.')

    st.write("### We need some information to predict the salary:")
    
    job_titles = ('Data Analyst', 'Data Scientist', 'Data Engineer', 'Machine Learning Engineer', 'Data Architect', 'Analytics Engineer', 'Applied Scientist', 'Research Scientist')
    experience_levels = ('Entry-level', 'Mid-level', 'Senior', 'Executive')
    work_settings = ('Office', 'Hybrid', 'Remote')
    company_locations = ('United States', 'United Kingdom', 'Canada', 'Spain', 'Germany', 'France', 'Netherlands', 'Portugal', 'Australia', 'Other')

    job_title = st.selectbox("Job Title", job_titles)
    experience_level = st.selectbox("Experience Level", experience_levels)
    work_setting = st.selectbox("Work Type", work_settings)
    company_location = st.selectbox("Company Location", company_locations)

    if st.button("Calculate Salary"):
        x = np.array([[job_title, experience_level, work_setting, company_location]])
        x[:, 0] = le_job_title.transform(x[:, 0])
        x[:, 1] = le_experience_level.transform(x[:, 1])
        x[:, 2] = le_work_setting.transform(x[:, 2])
        x[:, 3] = le_company_location.transform(x[:, 3])
        x = x.astype(float)

        salary = regressor_loaded.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:,.2f}")

# Function to explore data
def show_explore_page(job_title=None):
    st.title("Explore Data Field Jobs Salaries")

    if job_title:
        st.write(f"### {job_title}")
        job_df = df[df["job_title"] == job_title]

        fig, ax = plt.subplots(facecolor=(0, 0, 0, 0))
        ax.hist(job_df["salary"], bins=20, edgecolor='white', alpha=0.7)
        ax.set_xlabel("Salary")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Salaries")
        st.pyplot(fig)

        fig, ax = plt.subplots(facecolor=(0, 0, 0, 0))
        ax.pie(job_df['work_setting'].value_counts(), labels=job_df['work_setting'].value_counts().index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title("Distribution of Work Settings")
        st.pyplot(fig)

        fig, ax = plt.subplots(facecolor=(0, 0, 0, 0))
        sns.countplot(y="company_location", data=job_df[job_df['company_location'] != 'Other'],
                      order=job_df[job_df['company_location'] != 'Other']['company_location'].value_counts().index,
                      ax=ax, alpha=0.7)
        ax.set_xlabel("Count")
        ax.set_ylabel("Company Location")
        ax.set_title("Count of Company Locations")
        st.pyplot(fig)
    else:
        job_titles_counts = df[df['job_title'] != 'Other']['job_title'].value_counts()
        sorted_job_titles = job_titles_counts.index.tolist()

        selected_job_title = st.selectbox("Select a job title", sorted_job_titles)
        if selected_job_title:
            show_explore_page(selected_job_title)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Salary Prediction", "Data Exploration"])

if page == "Salary Prediction":
    show_predict_page()
else:
    show_explore_page()
