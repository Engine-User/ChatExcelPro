import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import chardet
import csv
import requests
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Get the Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Function to analyze the CSV file
def analyze_csv(df):
    total_rows, total_columns = df.shape
    column_names = ", ".join(df.columns)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_columns].agg(['mean', 'min', 'max']).to_dict()
    
    stats_str = "\n".join([f"{col}:\n  Mean: {val['mean']:.2f}\n  Min: {val['min']:.2f}\n  Max: {val['max']:.2f}" 
                           for col, val in stats.items()])
    
    return f"Total rows: {total_rows}\nTotal columns: {total_columns}\nColumn names: {column_names}\n\nBasic stats:\n{stats_str}"
    

# Function to query Groq API for chat
def chat_with_csv(df, prompt):
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    # Limit the data to the first 100 rows and all columns
    limited_df = df.head(200)
    df_str = limited_df.to_string()
    
    # Prepare a summary of the data
    summary = f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
    summary += f"Columns: {', '.join(df.columns)}. "
    summary += "Here's a sample of the data (first 200 rows):\n\n"
    # Prepare the payload for the API request
    data = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant analyzing CSV data. Use the provided data sample and summary to answer questions as accurately as possible. If the answer cannot be definitively found in the sample, say so and provide the best possible answer based on the available information."},
            {"role": "user", "content": f"{summary}{df_str}\n\nQuestion: {prompt}"}
        ],
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        return result
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 413:
            return "Error: The dataset is too large to process. Please try with a smaller dataset or more specific question."
        elif response.status_code == 401:
            return "Unauthorized access. Please check your API key and ensure it's valid."
        else:
            return f"HTTP error occurred: {str(http_err)}"
    except requests.exceptions.RequestException as req_err:
        return f"An error occurred: {str(req_err)}"

# Function to load data

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            file.seek(0)
            return pd.read_csv(io.StringIO(raw_data.decode(encoding)))
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        return None

    

# Streamlit app layout and functionality
st.markdown("<h1 style='color: green;'>ChatExcel Pro: LLM powered Excel Analytics</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color: green;'>Chat with your Excel sheets and visualize data!</h3>", unsafe_allow_html=True)

st.markdown("<h6 style='color: red;'>From Sad_Engineer</h6>", unsafe_allow_html=True)
# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    if data is not None:
        st.success("File uploaded successfully!")
        st.subheader("Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        # Chat functionality
        st.markdown("<h3 style='color: green;'>Chat With Your Excel</h3>", unsafe_allow_html=True)
        input_text = st.text_area("Ask you query")
        if input_text and st.button("CHAT"):
            with st.spinner("Analyzing data and generating response..."):
                result = chat_with_csv(data, input_text)
            st.success(result)

        # Data Analysis
        st.markdown("<h3 style='color: green;'>Data Analysis</h3>", unsafe_allow_html=True)
        if st.checkbox("Show Summary for Data Analytics"):
            st.text(analyze_csv(data))

        # Data Visualization
        st.subheader("Data Visualization")
        viz_type = st.selectbox("Select the mode of Visualization", 
                                ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap"])
        
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if viz_type != "Heatmap":
            x_column = st.selectbox("Select X-axis", columns)
            y_column = st.selectbox("Select Y-axis", columns)
        
        if viz_type != "Heatmap" and st.button("Generate Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                if viz_type == "Line Chart":
                    sns.lineplot(data=data, x=x_column, y=y_column, ax=ax)
                elif viz_type == "Bar Chart":
                    sns.barplot(data=data, x=x_column, y=y_column, ax=ax)
                elif viz_type == "Scatter Plot":
                    sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
                elif viz_type == "Histogram":
                    sns.histplot(data=data, x=x_column, ax=ax)
                elif viz_type == "Box Plot":
                    sns.boxplot(data=data, x=x_column, y=y_column, ax=ax)
                
                plt.title(f"{viz_type} of {y_column if viz_type != 'Histogram' else x_column}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")

        # Updated Heatmap logic
        if viz_type == "Heatmap" and st.button("Generate Heatmap"):
            try:
                # Calculate the correlation matrix for numeric columns
                corr = data[columns].corr()
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                plt.title("Heatmap of Correlation Matrix")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")

        # Data Filtering and Export
        st.subheader("Data Filtering")
        filter_column = st.selectbox("Select column to filter", data.columns)
        if data[filter_column].dtype == 'object':
            filter_values = st.multiselect("Select values", data[filter_column].unique())
            filtered_data = data[data[filter_column].isin(filter_values)] if filter_values else data
        else:
            min_val, max_val = st.slider("Select range", 
                                         float(data[filter_column].min()), 
                                         float(data[filter_column].max()),
                                         (float(data[filter_column].min()), float(data[filter_column].max())))
            filtered_data = data[(data[filter_column] >= min_val) & (data[filter_column] <= max_val)]

        st.dataframe(filtered_data, use_container_width=True)

        if st.button("Download Filtered Data"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
else:
    st.info("Please upload a CSV or Excel file to get started.")
