#######################
# Library Imports

# Streamlit
import streamlit as st
import io

# Data Analysis
import pandas as pd
import numpy as np
from scipy.stats import skew
from collections import Counter

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Importing Models
import joblib

# Images
from PIL import Image

# Importing models

lr = joblib.load('assets/models/abalone_linear_regression.joblib')
rf = joblib.load('assets/models/abalone_random_forest_regressor.joblib')
svr = joblib.load('assets/models/abalone_support_vector_regression.joblib')

#######################
# Page configuration
st.set_page_config(
    page_title="Abalone Dataset", # Replace this with your Project's Title
    page_icon="assets/icon/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.image("assets/abalone_pictures/sidebar_abalone.png")

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

    # Univariate Analysis
    st.subheader("Univariate Analysis")
    st.write("Univariate analysis aims to summarize and understand the distribution and characteristics of individual features in the Abalone dataset.")
    st.write("Based on the problem statement and feature description, let's first compute the target variable of the problem, 'Age,' and assign it to the dataset. **Age = 1.5 + Rings**.")
    
    # Calculate 'Age' and drop 'Rings' column
    abalone_df['Age'] = abalone_df['Rings'] + 1.5
    abalone_df.drop('Rings', axis=1, inplace=True)

    # Display the modified DataFrame
    st.dataframe(abalone_df.head(), use_container_width=True, hide_index=True)

    # Histogram plot
    fig, ax = plt.subplots(figsize=(20, 10))
    abalone_df.hist(ax=ax, grid=False, layout=(2, 4), bins=30)
    st.pyplot(fig)

    # Categorical and Numerical Columns
    cat_col = [col for col in abalone_df.columns if abalone_df[col].dtype == 'object']
    num_col = [col for col in abalone_df.columns if abalone_df[col].dtype != 'object']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Categorical Columns:**")
        st.write(cat_col)

    with col2:
        st.markdown("**Numerical Columns:**")
        st.write(num_col)

    with col3:
        # Skewness of Numerical Columns
        skew_values = skew(abalone_df[num_col], nan_policy='omit')
        dummy = pd.concat([pd.DataFrame(list(num_col), columns=['Features']),
                            pd.DataFrame(list(skew_values), columns=['Skewness Degree'])], axis=1)
        dummy = dummy.sort_values(by='Skewness Degree', ascending=False)

        # Display the DataFrame
        st.dataframe(dummy, hide_index=True)

    st.markdown("""
    * The features **Height**, **Age**, **Shucked Weight**, **Shell Weight**, **Viscera Weight**, and **Whole Weight** all exhibit positive skewness, indicating a prevalence of lower values with a few higher values influencing the distribution.
    * In contrast, **Diameter** and **Length** show negative skewness, suggesting that larger values are more common compared to smaller ones.
    """)
    
    # Pie chart
    col1, col2, col3 = st.columns(3)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))  # Adjust the figure size as needed
        plt.pie(abalone_df['Sex'].value_counts(), labels=['M', 'F', 'I'], autopct='%.00f%%')
        plt.title("Sex Distribution")
        st.pyplot(fig)

    abalone_sex_stat = abalone_df.groupby('Sex')[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Age']].mean().sort_values('Age')
    st.dataframe(abalone_sex_stat, use_container_width=True)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    st.write("Bivariate analysis aims to examine the relationship between two variables in the Abalone dataset to identify potential correlations or associations.")

    # Pairplot
    fig = sns.pairplot(abalone_df)
    st.pyplot(fig)

    st.markdown("""
    *   **Length and Diameter**: A strong positive correlation indicates that larger abalone tend to have both greater length and diameter.
    *   **Length**, **Diameter**, and **Whole Weight**: These features exhibit a moderate to strong positive correlation, suggesting that larger abalone generally weigh more.
    *   **Height**: The overall relationship between height and other features seems to be weak. This suggests that height is not a strong predictor of other characteristics of the abalone.
    *   **Age**: The distribution of age appears to be skewed to the right, with a majority of abalone being younger.
    *   **Outliers**: The presence of outliers in some features can impact the correlation analysis and overall interpretation of the data.
    """)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(abalone_df[num_col].corr(), annot=True, ax=ax)
    plt.title("Correlation Matrix", fontsize=16)
    st.pyplot(fig)

    st.markdown("""
    *   **Whole Weight** does show strong positive correlations with **Length**, **Diameter**, **Shucked Weight**, **Viscera Weight**, and **Shell Weight**.
    *   **Height** has the weakest correlations with other features compared to **Length**, **Diameter**, and **Weight**-related features.
    *   Features like **Shell Weight**, **Diameter**, and **Length** show slightly stronger correlations with **Age** compared to other features.
    """)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Calculate 'Age' and drop 'Rings' column
    abalone_df['Age'] = abalone_df['Rings'] + 1.5
    abalone_df.drop('Rings', axis=1, inplace=True)

    # Categorical and Numerical Columns
    cat_col = [col for col in abalone_df.columns if abalone_df[col].dtype == 'object']
    num_col = [col for col in abalone_df.columns if abalone_df[col].dtype != 'object']

    st.dataframe(abalone_df.head(), use_container_width=True, hide_index=True)

    st.divider()

    # Capture the DataFrame information
    buffer = io.StringIO()
    abalone_df.info(buf=buffer)
    s = buffer.getvalue()

    # Calculate the sum of missing values for each column
    missing_values = abalone_df.isna().sum()

    # Create a row with two columns using st.columns
    col1, col2 = st.columns(2)

    # Display information in left column
    with col1:
        st.subheader("Abalone Dataset Information")
        st.text(s)

    # Display missing values in right column
    with col2:
        st.subheader("Missing Value Summary")
        st.table(missing_values)

    st.write("**The Abalone dataset does not contain any null values.**")

    st.divider()

    # Outliers
    st.subheader("Outliers")

    # Boxplot
    fig, ax = plt.subplots(figsize=(15, 5))
    abalone_df.boxplot(ax=ax)
    st.pyplot(fig)

    st.write("**df.boxplot()** method in pandas primarily operates on numerical features (**Sex**) and does not include categorical features directly in the boxplot visualization.")

    st.write("The **Abalone** dataset contains skewed features (like weights and shell dimensions) and the **IQR (Interquartile Range)** method is likely the most suitable for detecting outliers. It can handle skewed distributions better and doesn't assume a normal distribution.")

    # Columns to center the image
    col_dt_fig = st.columns((2, 4, 2), gap='medium')

    with col_dt_fig[0]:
        st.write(' ')

    with col_dt_fig[1]:
        IQR_image = Image.open('assets/reference_images/IQR.jpg')
        st.image(IQR_image, caption='Box Plot: Visualizing Data Distribution and Outliers')

    with col_dt_fig[2]:
        st.write(' ')

    st.markdown("`Reference:` https://www.simplypsychology.org/boxplots.html")

    st.write("We are dealing with numerical data, so we should drop the gender section.")

    st.code("""
    data=abalone_df.drop(columns=['Sex'],axis=1)
    data
    """)

    # DataFrame excluding 'Sex' 
    data = abalone_df.drop(columns=['Sex'], axis=1)
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    st.markdown("""
        First, we are going to find the first quartile (**Q1**), which represents a quarter of the way through the list of all data. Then, we’ll find the third quartile (**Q3**), which represents three-quarters of the way through the data list.,<br><br>
    Next, we need to find the **median** of the dataset, which represents the midpoint of the entire data list.<br><br>
    Finally, we can determine the **upper and lower ranges** of our data using these formulas:
    *   *IQR = Q3 - Q1*
    *   *lower range = Q1 - (1.5 x IQR)*
    *   *upper range = Q3 + (1.5 x IQR)*

    Once we understand the **upper and lower ranges**, we can detect **outliers**. By cleaning the data of outliers, we can perform a more accurate analysis. Now, we’ll apply the **interquartile range method** to the **Abalone** dataset.
    """)

    def detect_outliers(data, features):
        outlier_indices = []

        for c in features:
            # Calculate Q1, Q3 and IQR
            Q1 = np.percentile(data[c], 25)
            Q3 = np.percentile(data[c], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            lower_range = Q1 - outlier_step
            upper_range = Q3 + outlier_step

            # Determine outliers for this feature
            outliers = data[(data[c] < lower_range) | (data[c] > upper_range)].index
            outlier_indices.extend(outliers)

        # Find samples with multiple outliers
        outlier_counts = Counter(outlier_indices)
        multiple_outliers = [idx for idx, count in outlier_counts.items() if count > 2]

        return multiple_outliers

    st.code("""
    def detect_outliers(data, features):
        outlier_indices = []

        for c in features:
            # Calculate Q1, Q3 and IQR
            Q1 = np.percentile(data[c], 25)
            Q3 = np.percentile(data[c], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            lower_range = Q1 - outlier_step
            upper_range = Q3 + outlier_step

            # Determine outliers for this feature
            outliers = data[(data[c] < lower_range) | (data[c] > upper_range)].index
            outlier_indices.extend(outliers)

        # Find samples with multiple outliers
        outlier_counts = Counter(outlier_indices)
        multiple_outliers = [idx for idx, count in outlier_counts.items() if count > 2]

        return multiple_outliers
    """)

    st.write("Change the column names, as some blanks may cause issues.")

    st.code("""
    data.columns=['length', 'diameter', 'height', 'whole weight', 'shucked weight',
       'viscera weight', 'shell weight', 'age']
    data.info()
    """)

    data.columns=['length', 'diameter', 'height', 'whole weight', 'shucked weight',
       'viscera weight', 'shell weight', 'age']
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    st.code("""
    outliers = detect_outliers(abalone_df, num_col)
    data.loc[outliers]
    """)

    # Detect outliers
    outliers = detect_outliers(abalone_df, num_col)
    st.dataframe(data.loc[outliers].sort_index(), use_container_width=True)

    st.write("Drop the outliers and reset the indices.")

    st.code("""
    data=data.drop(outliers,axis=0).reset_index(drop = True)
    data
    """)

    # Drop outliers
    data = data.drop(outliers, axis=0).reset_index(drop=True)

    # Display the cleaned data
    st.dataframe(data.head(), use_container_width=True, hide_index=True)
    st.write(f"We removed the outliers, resulting in a dataset with **{len(data)}** rows")

    st.session_state['cleaned_data'] = data

    st.divider()

    st.code("""
    encoder = LabelEncoder()
    abalone_df['Sex'] = encoder.fit_transform(abalone_df['Sex'])
    abalone_df.head()
    """)

    # Create a LabelEncoder object
    encoder = LabelEncoder()

    # Encode the 'Sex' column
    abalone_df['Sex'] = encoder.fit_transform(abalone_df['Sex'])

    # Correlation Matrix
    st.dataframe(abalone_df.head(), use_container_width=True, hide_index=True)

    def plot_correlation_matrix(df):
        fig, ax = plt.subplots(figsize=(20, 7))
        sns.heatmap(abalone_df.corr(), annot=True, ax=ax)
        plt.title("Correlation Matrix", fontsize=16)
        st.pyplot(fig)

    plot_correlation_matrix(abalone_df)

    st.markdown("""
    **Strong Positive Correlations:**
    *   **Length**, **Diameter**, **Height**, **Whole weight**, **Shucked weight**, **Viscera weight**, and **Shell weight** are highly correlated with each other. This suggests that these variables are closely related to the overall size and weight of the abalone.
    *   **Age** has a moderate positive correlation with **Shell weight**. This indicates that older abalone tend to have heavier shells.

    **Weak Correlations:**
    *   **Sex** has very weak correlations with all other features, suggesting that sex has little impact on the physical measurements of abalone.
    """)

    st.code("""
    # Select the relevant features including the target (age)
    # Excluding sex from the model can help improve performance
    # Create a dataframe with the selected features
    selected_data = data[['length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'age']]
    selected_data   
    """)

    # Select relevant features, excluding 'sex'
    selected_data = data[['length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'age']]
    st.dataframe(selected_data.head(), use_container_width=True, hide_index=True)

    st.divider()

    # Train-Test Split
    st.subheader("Train-Test Split")

    st.code("""
    # Separate features (X) and target variable (y) for model training

    # X: Features
    X = selected_data.drop('age', axis=1)

    # y: Target variable
    y = data['age']
    """)

    # Separate features (X) and target variable (y) for model training

    # X: Features
    X = selected_data.drop('age', axis=1)

    # y: Target variable
    y = data['age']

    st.write("To train our model, we'll use specific characteristics of the abalone as input features. These features include the abalone's length, diameter, height, whole weight, shucked weight, viscera weight, and shell weight. The target variable we aim to predict is the age of the abalone.")

    st.code("""
    # Split dataset into training and testing sets
    # Training - 80%
    # Test Set - 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """)

    # Split dataset into training and testing sets
    # Training - 80%
    # Test Set - 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)

    st.markdown("After splitting our dataset into `training` and `test` set. We can now proceed with **training our supervised models**.")

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    st.subheader("Cleaned Data")

    ####################################################################################################

    # Redefining data preparation steps for machine learning page
    # Re-initializing due to separate page requirements
    
    # Calculate 'Age' and drop 'Rings' column
    abalone_df['Age'] = abalone_df['Rings'] + 1.5
    abalone_df.drop('Rings', axis=1, inplace=True)

    # Categorical and Numerical Columns
    cat_col = [col for col in abalone_df.columns if abalone_df[col].dtype == 'object']
    num_col = [col for col in abalone_df.columns if abalone_df[col].dtype != 'object']

    # DataFrame excluding 'Sex' 
    data = abalone_df.drop(columns=['Sex'], axis=1)

    def detect_outliers(data, features):
        outlier_indices = []

        for c in features:
            # Calculate Q1, Q3 and IQR
            Q1 = np.percentile(data[c], 25)
            Q3 = np.percentile(data[c], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            lower_range = Q1 - outlier_step
            upper_range = Q3 + outlier_step

            # Determine outliers for this feature
            outliers = data[(data[c] < lower_range) | (data[c] > upper_range)].index
            outlier_indices.extend(outliers)

        # Find samples with multiple outliers
        outlier_counts = Counter(outlier_indices)
        multiple_outliers = [idx for idx, count in outlier_counts.items() if count > 2]

        return multiple_outliers

    data.columns=['length', 'diameter', 'height', 'whole weight', 'shucked weight',
       'viscera weight', 'shell weight', 'age']

    # Detect outliers
    outliers = detect_outliers(abalone_df, num_col)

    # Drop outliers
    data = data.drop(outliers, axis=0).reset_index(drop=True)

    # Display the cleaned data
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    # Select relevant features, excluding 'sex'
    selected_data = data[['length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'age']]

    # Separate features (X) and target variable (y) for model training

    # X: Features
    X = selected_data.drop('age', axis=1)

    # y: Target variable
    y = data['age']

    # Split dataset into training and testing sets
    # Training - 80%
    # Test Set - 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ####################################################################################################

    st.divider()

    # LINEAR REGRESSION
    st.subheader("Supervised Machine Learning - Linear Regression")

    # Training the Model
    st.markdown("### **`Training the Model`**")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    st.code("""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    """)

    # Applying the Model to Make a Prediction
    st.markdown("### **`Applying the Model to Make a Prediction`**")
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)

    st.code("""
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    """)

    # Evaluate Model Performance
    st.markdown("### **`Evaluate Model Performance`**")
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    st.code("""
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)
    """)

    lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    st.dataframe(lr_results)

    train_accuracy = lr.score(X_train, y_train) #train data
    test_accuracy = lr.score(X_test, y_test) #test data

    st.code("""
    train_accuracy = lr.score(X_train, y_train) #train data
    test_accuracy = lr.score(X_test, y_test) #test data

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    """)

    st.write(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    st.write(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    st.markdown("""
    1.   **Model Fit**: The R-squared values for both the training and testing sets are around 0.5, indicating that the model explains about 50% of the variance in the age.
    2.   **Generalization**: The R-squared values for the training and testing sets are relatively close, suggesting that the model generalizes reasonably well to unseen data.
    3.   **Error**: The MSE values are relatively high, indicating that the model's predictions can be quite far off from the actual values.
    """)

    st.image("assets/reference_images/LR_Model_Visualization.png", use_column_width=True)

    st.markdown("""
    *   The scatter plots show that the data points are spread around the perfect prediction line (red dashed line), indicating that the model's predictions are not always perfect but follow a linear trend.
    *   There appears to be a wider spread of data points in the test set (green) compared to the training set (blue). This suggests slightly higher variability in the model's predictions for unseen data.
    """)

    st.divider()

    # RANDOM FOREST (REGRESSOR)
    st.subheader("Supervised Machine Learning - Random Forest (Regressor)")

    # Training the Model
    st.markdown("### **`Training the Model`**")
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    st.code("""
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    """)

    # Applying the Model to Make a Prediction
    st.markdown("### **`Applying the Model to Make a Prediction`**")
    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)

    st.code("""
    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)
    """)

    # Evaluate Model Performance
    st.markdown("### **`Evaluate Model Performance`**")
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)

    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    st.code("""
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)

    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)
    """)

    rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
    rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    st.dataframe(rf_results)

    train_accuracy = rf.score(X_train, y_train) #train data
    test_accuracy = rf.score(X_test, y_test) #test data

    st.code("""
    train_accuracy = rf.score(X_train, y_train) #train data
    test_accuracy = rf.score(X_test, y_test) #test data

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    """)

    st.write(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    st.write(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    st.markdown("""
    1.   **Model Fit**: R-squared close to 1 in the training set indicates that the model captures almost all variance in the training data. This might suggest overfitting.
    2.   **Overfit**: The test R-squared is significantly lower than the training R-squared, suggesting that the model might be overfitting to the training data and not generalizing well to unseen data.
    3.   **Error**: Random Forest's training MSE is lower, but its test MSE is comparable to Linear Regression.
    """)

    st.image("assets/reference_images/RF_Model_FI.png", use_column_width=True)

    st.markdown("""
    **Shell weight** is the most important feature by a significant margin, indicating that it has the strongest predictive power in estimating the target variable (age). This suggests that shell weight plays a crucial role in determining the age of abalones.
    """)

    st.divider()

    # SUPPORT VECTOR REGRESSION
    st.subheader("Supervised Machine Learning - Support Vector Regression")

    # Training the Model
    st.markdown("### **`Training the Model`**")
    svr = SVR()
    svr.fit(X_train, y_train)

    st.code("""
    svr = SVR()
    svr.fit(X_train, y_train)
    """)

    # Applying the Model to Make a Prediction
    st.markdown("### **`Applying the Model to Make a Prediction`**")
    y_svr_train_pred = svr.predict(X_train)
    y_svr_test_pred = svr.predict(X_test)

    st.code("""
    y_svr_train_pred = svr.predict(X_train)
    y_svr_test_pred = svr.predict(X_test)
    """)

    # Evaluate Model Performance
    st.markdown("### **`Evaluate Model Performance`**")
    svr_train_mse = mean_squared_error(y_train, y_svr_train_pred)
    svr_train_r2 = r2_score(y_train, y_svr_train_pred)

    svr_test_mse = mean_squared_error(y_test, y_svr_test_pred)
    svr_test_r2 = r2_score(y_test, y_svr_test_pred)

    st.code("""
    svr_train_mse = mean_squared_error(y_train, y_svr_train_pred)
    svr_train_r2 = r2_score(y_train, y_svr_train_pred)

    svr_test_mse = mean_squared_error(y_test, y_svr_test_pred)
    svr_test_r2 = r2_score(y_test, y_svr_test_pred)
    """)

    svr_results = pd.DataFrame(['Support Vector Regression', svr_train_mse, svr_train_r2, svr_test_mse, svr_test_r2]).transpose()
    svr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    st.dataframe(svr_results)

    train_accuracy = svr.score(X_train, y_train) #train data
    test_accuracy = svr.score(X_test, y_test) #test data

    st.code("""
    train_accuracy = svr.score(X_train, y_train) #train data
    test_accuracy = svr.score(X_test, y_test) #test data

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    """)

    st.write(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    st.write(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    st.markdown("""
    1.   **Model Fit**: The R-squared values for both training and testing sets are around 0.5. This means the SVR model explains about 50% of the variance in the target variable (age), which suggests a moderate fit.
    2.   **Generalization**: The training and testing R-squared values are relatively close. It indicates that the model generalizes reasonably well.
    3.   **Error**: The MSE values for the training (4.70) and testing (5.78) sets are relatively high, similar to Linear Regression and Random Forest models.
    """)

    st.image("assets/reference_images/SVR_Model_Visualization.png", use_column_width=True)

    st.markdown("""
    **Scatter Plot:**
    *   The data points generally follow the red dashed line, indicating a positive correlation between actual and predicted ages. This suggests that the **SVR** model is capturing the underlying relationship between the features and the target variable.
    *   However, there's significant scatter around the line, implying that the model's predictions can be quite variable, especially for higher actual ages.

    **Residual Plot:**
    *   **Positive Residuals**: When a data point lies above the horizontal line at 0, it means the actual age is higher than the predicted age.
    *   **Negative Residuals**: When a data point lies below the horizontal line, it means the actual age is lower than the predicted age.

    In the given residual plot, we can observe that as the predicted age increases, there's a higher concentration of positive residuals. This suggests that the model systematically underestimates the age for older abalones.
    """)

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Initialize cleaned_data in session_state if not present
    if 'cleaned_data' not in st.session_state:
        st.session_state['cleaned_data'] = None
    
    # Show cleaned data
    if st.session_state['cleaned_data'] is not None:
        st.write("Here’s the cleaned data used for prediction:")
        st.dataframe(st.session_state['cleaned_data'], use_container_width=True, hide_index=True)
    else:
        st.write("Cleaned data not found. Please run the Pre-processing page first.")

    # Prompt user to choose a model
    model_choice = st.selectbox("Choose Model", ("Select Model", "Linear Regression", "Random Forest Regressor", "Support Vector Regression"))

    # Input fields for features
    length = st.number_input("Length (mm)", min_value=0.0, step=0.001, format="%.4f")
    diameter = st.number_input("Diameter (mm)", min_value=0.0, step=0.001, format="%.4f")
    height = st.number_input("Height (mm)", min_value=0.0, step=0.001, format="%.4f")
    whole_weight = st.number_input("Whole Weight (grams)", min_value=0.0, step=0.001, format="%.4f")
    shucked_weight = st.number_input("Shucked Weight (grams)", min_value=0.0, step=0.001, format="%.4f")
    viscera_weight = st.number_input("Viscera Weight (grams)", min_value=0.0, step=0.001, format="%.4f")
    shell_weight = st.number_input("Shell Weight (grams)", min_value=0.0, step=0.001, format="%.4f")

    # Collect inputs into an array and reshape for model input
    features = np.array([length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]).reshape(1, -1)

    if st.button("Predict Age"):
        # Check if a model has been selected
        if model_choice == "Select Model":
            st.warning("Please select a model to proceed with the prediction.")
        elif any(feature == 0 for feature in [length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]):
            st.warning("Please provide all feature values before predicting.")
        else:
            # Perform prediction based on the selected model
            if model_choice == "Linear Regression":
                age_prediction = lr.predict(features)[0]
            elif model_choice == "Random Forest Regressor":
                age_prediction = rf.predict(features)[0]
            else:
                age_prediction = svr.predict(features)[0]

            # Display the predicted age
            st.write(f"Predicted Age of Abalone: {age_prediction:.1f} years")


# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here