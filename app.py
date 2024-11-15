import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load dataset for EDA
data = sns.load_dataset('iris')

st.title('Iris Model Deployment')

# Load the pre-trained model
model = joblib.load('iris_model.pkl')

# Sidebar navigation
page = st.sidebar.selectbox('Select Option', ['Home', 'EDA', 'Model'])

if page == 'Home':
    st.title("Iris Dataset Analysis 🌸")
    st.write("Welcome to the Iris Dataset Analysis App! 🍀🍂")
    st.markdown("""
        This app allows you to explore and analyze the famous Iris dataset, which contains data on 
        **150 iris flowers**, each described by **four features**:
        
        - **Sepal Length**
        - **Sepal Width**
        - **Petal Length**
        - **Petal Width**

        The dataset is divided into **three species of iris flowers**:
        - *Setosa*
        - *Versicolor*
        - *Virginica*

        ## Why the Iris Dataset?
        The Iris dataset is one of the most well-known datasets used in data science and machine learning. It's a great starting point for exploring:
        
        - Data visualization
        - Feature relationships
        - Classification tasks
    
        ## Features of this App
        - **Data Overview**: A summary of the dataset, including basic statistics and feature information.
        - **Data Visualization**: Graphs and charts to help visualize patterns between features.
        - **Machine Learning Models**: Interactive options to apply and evaluate classification algorithms.
        
        Dive in and start exploring the unique insights this dataset has to offer! 🌼
    """)

elif page == 'EDA':
    st.title('Exploratory Data Analysis')
    st.write("Here's a quick preview of the Iris dataset:")
    st.write(data.head())
    st.write(f'Shape of the data: {data.shape}')
    
    # Plotting distribution of each feature
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Distribution of Iris Features')
    
    sns.histplot(data['sepal_length'], bins=10, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Sepal Length')
    
    sns.histplot(data['sepal_width'], bins=10, ax=axes[0, 1], kde=True)
    axes[0, 1].set_title('Sepal Width')
    
    sns.histplot(data['petal_length'], bins=10, ax=axes[1, 0], kde=True)
    axes[1, 0].set_title('Petal Length')
    
    sns.histplot(data['petal_width'], bins=10, ax=axes[1, 1], kde=True)
    axes[1, 1].set_title('Petal Width')
    
    st.pyplot(fig)

elif page == 'Model':
    st.header('Prediction Page')
    st.write("Enter the values below to predict the Iris species:")

    # User input for flower features
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.35)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3)

    # After the prediction
    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        predicted_species = prediction[0]  # Get the predicted species

        # Map the numeric prediction to flower species
        species_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        predicted_species_name = species_names[predicted_species]

        # Display the predicted species
        st.write(f"Predicted Species: **{predicted_species_name}**")

        # Display an image based on the predicted species
        image_paths = {
            0: "iris-setosa.jpg",
            1: "iris-versicolor.jpg",
            2: "iris-virginica.jpg "
}
        

        try:
            img = Image.open(image_paths[predicted_species])
            st.image(img, caption=f"Predicted Species: {predicted_species_name}", use_container_width=True)
        except FileNotFoundError:
            st.error(f"Image file for {predicted_species_name} not found. Please check the file path.")
