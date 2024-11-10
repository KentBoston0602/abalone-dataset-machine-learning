#######################
# Library Imports

# Streamlit
import streamlit as st

# Data Analysis
import pandas as pd
import altair as alt

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

# Images
from PIL import Image

#######################
# Page configuration
st.set_page_config(
    page_title="Abalone Dataset", # Replace this with your Project's Title
    page_icon="assets/icon/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Abalone Dataset')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training three supervised machine learning models using the Abalone dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset/data)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/11jZ-5Eq3cjT9kt7EX_36JPg9EIEuK3QG?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/KentBoston0602/abalone-dataset-machine-learning)")

    # Project Members
    st.subheader("Members")
    st.markdown("1. Kent Patrick **BOSTON**\n2. Luis Frederick **CONDA**\n3. Chaze Kyle **FIDELINO**\n4. Joseph Isaac **ZAMORA**")

#######################
# Data

# Load data
abalone_df = pd.read_csv("data/abalone.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to predict the age of abalone from physical measurements using **Linear Regression**, **Random Forest Regressor** and **Support Vector Regression**.

    #### Pages
    1. `Dataset` - The Abalone Dataset contains information about abalone, including its sex, length, diameter, height, whole weight, shucked weight, viscera weight, shell weight, and rings.
    2. `EDA` - Exploratory Data Analysis of the Abalone dataset. The analysis involved examining data both individually and in pairs. Histograms and pie charts were used to understand the distribution of numerical and categorical features, respectively. Pair plots and heatmaps were employed to visualize relationships and correlations between numerical features. Box plots were used to identify and potentially address outliers in the dataset.
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the 'Sex' column and splitting the dataset into training and testing sets.
    4. `Machine Learning` - Training three supervised ML models: Linear Regression, Random Forest Regressor and Support Vector Regression. Includes model evaluation, model prediction visualizations and feature importance.
    5. `Prediction` - Prediction page where users can input values to predict the age of abalone using the trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.


    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""

    The Abalone dataset was uploaded to UCI Machine Learning Repository. It is a dataset used widely in machine learning.

    From the original data, examples with missing values, primarily those with missing predicted values, were removed. The ranges of continuous values were scaled by dividing by 200 for use with an ANN.  

    For each sample, eight features are measured: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, and Shell weight. This dataset is commonly used to test regression techniques to predict the Age of the abalone. The same dataset that is used for this data science activity was uploaded to Kaggle by the user named Rodolfo Mendes.

    **Content**  
    The dataset has 4177 rows containing 8 numeric attributes that are related to abalone, the columns are as follows: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight and Age.

    `Link:` https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset            
                
    """)

    col_abalone = st.columns((3, 3, 3), gap='medium')

    # Define the new dimensions (width, height)
    resize_dimensions = (500, 300)  # Example dimensions, adjust as needed

    with col_abalone[0]:
        abalone1_image = Image.open('assets/abalone_pictures/abalone1.jpg')
        abalone1_image = abalone1_image.resize(resize_dimensions)
        st.image(abalone1_image)

    with col_abalone[1]:
        abalone2_image = Image.open('assets/abalone_pictures/abalone2.jpg')
        abalone2_image = abalone2_image.resize(resize_dimensions)
        st.image(abalone2_image)

    with col_abalone[2]:

        abalone3_image = Image.open('assets/abalone_pictures/abalone3.jpg')
        abalone3_image = abalone3_image.resize(resize_dimensions)
        st.image(abalone3_image)

    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(abalone_df, use_container_width=True, hide_index=True)

    st.write("""

    The Abalone dataset contains 4,177 entries with nine columns, where each column represents a feature of an abalone.

    1. **Sex:** The sex of the abalone ('M', 'F', or 'I' for immature).
    2. **Length:** The longest shell measurement (in millimeters).
    3. **Diameter:** The shell measurement perpendicular to the length (in millimeters).
    4. **Height:** The height of the abalone with the meat included (in millimeters).
    5. **Whole weight:** The whole abalone’s weight (in grams).
    6. **Shucked weight:** The weight of the abalone's meat (in grams) after it has been removed from the shell.
    7. **Viscera weight:** The weight of the gut of the abalone after bleeding (in grams).
    8. **Shell weight:** The weight of the abalone’s dried shell (in grams).
    9. **Age:** The age of an abalone represented by its number of rings plus 1.5, indicating the number of years it has lived.
    """)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(abalone_df.describe(), use_container_width=True)

    st.markdown("""
    * No missing values in the dataset.
    * All features are numerical except for Sex.
    * Although the features are not normally distributed, they are close to normality.
    * None of the features have a minimum of 0 except for Height (requires re-check).
    * Each feature has a different scale range.
    """)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here