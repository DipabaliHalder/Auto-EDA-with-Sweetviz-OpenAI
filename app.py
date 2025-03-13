import streamlit as st
import pandas as pd
import openai
import os
import sweetviz as sv
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
import json
from io import StringIO

# Helper function to convert numpy datatypes to Python standard types
def convert_dtypes(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Wrapper function for displaying Plotly charts safely
def safe_plotly_chart(fig, use_container_width=True):
    """Safely display a Plotly figure by ensuring all data is JSON serializable"""
    # Convert numpy data types to native Python types for JSON serialization
    for data in fig.data:
        for key in data:
            data[key] = convert_dtypes(data[key])
    
    # Handle layout objects
    for key in fig.layout:
        fig.layout[key] = convert_dtypes(fig.layout[key])
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=use_container_width)

# Set page configuration with improved styling
st.set_page_config(
    page_title="Auto-EDA with Sweetviz & OpenAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        font-size: 1rem;
    }
    h1, h2, h3 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .plot-container {
        box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Insight box styling */
    .insight-box {
        border-left: 4px solid #4b6cb7;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 6px 24px 0px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px;
    }
    .insight-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f3b8d;
        display: flex;
        align-items: center;
    }
    .insight-content {
        margin-left: 0.5rem;
    }
    .insight-box.correlation {
        border-left-color: #4b6cb7;
    }
    .insight-box.distribution {
        border-left-color: #38a169;
    }
    .insight-box.outlier {
        border-left-color: #e53e3e;
    }
    .insight-box.categorical {
        border-left-color: #805ad5;
    }
    .insight-box.summary {
        border-left-color: #dd6b20;
    }
</style>
""", unsafe_allow_html=True)

# Initialize reset state tracking
if 'reset_phase' not in st.session_state:
    st.session_state.reset_phase = 0  # 0=normal, 1=confirmation, 2=resetting

# Add file tracking variable and dynamic key generator
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# Initialize session state variables
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'openai_key' not in st.session_state:
    st.session_state.openai_key = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'sweetviz_html' not in st.session_state:
    st.session_state.sweetviz_html = None
if 'dataset_summary' not in st.session_state:
    st.session_state.dataset_summary = None
if 'max_rows' not in st.session_state:
    st.session_state.max_rows = 500  # Default maximum rows to analyze
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# Function to configure OpenAI API
def configure_openai():
    with st.sidebar.expander("OpenAI API Configuration", expanded=True):
        api_key = st.text_input("Enter your OpenAI API key:", type="password", key="openai_api_key")
        
        if api_key:
            openai.api_key = api_key
            st.session_state.openai_key = api_key
            st.success("✅ API key configured successfully")
            return True
        return False

# Function to generate insights using OpenAI
def generate_insights(prompt, dataset_summary):
    try:
        full_prompt = f"""
        You are a helpful data analyst. Analyze the following dataset summary and answer the question or provide insights:
        
        Dataset Summary:
        {dataset_summary}
        
        Question or Task:
        {prompt}
        
        Provide a clear and concise answer. If the question cannot be answered with the dataset, say "I don't have enough information to answer this question."
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Replace with "gpt-4o-mini" if available
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Function to generate a dataset summary based on column names
def generate_dataset_summary(df):
    try:
        column_list = df.columns.tolist()
        column_types = df.dtypes.to_dict()
        column_info = [f"{col} ({column_types[col]})" for col in column_list]
        
        prompt = f"""
        Based on these column names and types, provide a brief 2-3 sentence summary of what this dataset might be about:
        {column_info}
        
        Make sure your summary describes the potential subject matter and purpose of this dataset in a clear and concise way.
        """
        
        dataset_summary = f"""
        Dataset Summary:
        - Columns: {column_info}
        - Shape: {df.shape}
        - Missing Values: {df.isnull().sum().sum()}
        """
        
        summary = generate_insights(prompt, dataset_summary)
        return summary
    except Exception as e:
        return f"Error generating dataset summary: {str(e)}"

# Function to generate Sweetviz report (only if not already in session state)
def generate_sweetviz_report(df):
    if st.session_state.sweetviz_html is None:
        report = sv.analyze(df)
        report_html = report.show_html(filepath="sweetviz_report.html", open_browser=False)
        
        with open("sweetviz_report.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            st.session_state.sweetviz_html = html_content
    
    return st.session_state.sweetviz_html

# Function to detect outliers in numerical columns
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Function to get top correlation insights (limited to 2-3 bullet points)
def get_correlation_insight(corr_matrix):
    # Fill diagonal with 0 to exclude self-correlations
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Get absolute correlation values for ranking importance
    abs_corr = np.abs(corr_matrix.values)
    
    # Get indices of top 3 correlations
    flat_indices = np.argsort(abs_corr, axis=None)[-3:]
    
    # Convert flat indices to 2D indices
    top_indices = np.array(np.unravel_index(flat_indices, corr_matrix.shape)).T
    
    insights = []
    
    # Process the top correlations (maximum 3)
    for i, j in top_indices:
        value = corr_matrix.values[i, j]
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        
        if abs(value) > 0.7:
            strength = "Strong"
        elif abs(value) > 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
            
        direction = "positive" if value > 0 else "negative"
        
        insights.append(f"{strength} {direction} correlation (r={value:.2f}) between '{col1}' and '{col2}'")
    
    return insights[:3]  # Limit to max 3 insights

# Function to display dataset tab
def dataset_tab(df):
    st.subheader("What is this dataset about?")
    
    if st.session_state.dataset_summary is None:
        with st.spinner("Generating dataset summary..."):
            st.session_state.dataset_summary = generate_dataset_summary(df)
    
    st.markdown(f"""
    <div class="insight-box summary">
        <div class="insight-header">📝 Dataset Summary</div>
        <div class="insight-content">
            {st.session_state.dataset_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display dataset size with styled metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    # Display first 5 rows with better styling
    st.subheader("Dataset Preview:")
    st.dataframe(df.head(), use_container_width=True)

# Function to display overview tab
def overview_tab(df):
    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Data types with improved visualization
    st.subheader("Data Types")
    
    # Count data types for a pie chart - using string representation
    type_counts = pd.DataFrame({
        'Data Type': [str(dtype) for dtype in df.dtypes.values],
        'Count': [1] * len(df.dtypes)
    })
    type_counts = type_counts.groupby('Data Type').sum().reset_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Convert dtypes to DataFrame for better display
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': [str(dtype) for dtype in df.dtypes.values]
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        # Add a pie chart of data types
        fig = px.pie(
            type_counts, 
            values='Count', 
            names='Data Type',
            title='Distribution of Data Types',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        safe_plotly_chart(fig)
    
    st.subheader("Missing Value Analysis")
    
    # Create a more detailed missing values DataFrame with explicit conversion
    missing_df = pd.DataFrame({
        'Column': df.columns.tolist(),
        'Missing Values': [int(val) for val in df.isnull().sum().values],
        'Missing (%)': [float(val) for val in (df.isnull().sum() / len(df) * 100).round(2).values]
    })
    
    # Add color coding based on percentage of missing values
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing (%)',
        hover_data=['Missing Values'],
        title='Missing Values by Column',
        color='Missing (%)',
        color_continuous_scale='Viridis',
        labels={'Missing (%)': 'Missing Values (%)'}
    )
    
    # Improve figure layout
    fig.update_layout(
        xaxis_title="Column Name",
        yaxis_title="Missing Values (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(missing_df.sort_values('Missing (%)', ascending=False), use_container_width=True)

# Function to display enhanced visualizations 
def visualizations_tab(df):
    st.subheader("Visualizations & Insights")
    
    # Correlation Matrix for numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numerical_cols) > 1:
        st.subheader("Correlation Matrix")
        corr_matrix = df[numerical_cols].corr()
        
        # Create correlation heatmap with explicit conversion to standard Python types
        z_values = corr_matrix.values.tolist()  # Convert numpy array to Python list
        annotation_text = np.around(corr_matrix.values, decimals=2).astype(str).tolist()
        
        fig = ff.create_annotated_heatmap(
            z=z_values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            annotation_text=annotation_text,
            colorscale='Viridis'
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate limited insights from correlation matrix
        correlation_insights = get_correlation_insight(corr_matrix.copy())
        if correlation_insights:
            # Build the HTML for all insights in one string
            insights_html = ""
            for insight in correlation_insights:
                insights_html += f"<li>{insight}</li>"
            
            # Display everything in a single markdown call
            st.markdown(f"""
            <div class="insight-box correlation">
                <div class="insight-header">🔗 Key Correlation Insights</div>
                <div class="insight-content">
                    <ul>
                        {insights_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Scatter Plot Section
    if len(numerical_cols) >= 2:
        st.subheader("Interactive Scatter Plot Analysis")
        
        # Create 3 columns for better layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            x_axis = st.selectbox("Select X-axis:", numerical_cols)
        
        with col2:
            y_axis = st.selectbox("Select Y-axis:", numerical_cols, index=min(1, len(numerical_cols)-1))
        
        # Option for color coding by a categorical variable
        with col3:
            color_by = st.selectbox(
                "Color points by (optional):",
                ["None"] + categorical_cols,
                index=0
            )
        
        if x_axis and y_axis:
            # Enhanced scatter plot with trend line and optional coloring
            if color_by != "None" and color_by in categorical_cols:
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    color=color_by,
                    trendline="ols",
                    opacity=0.7,
                    title=f"Relationship between {x_axis} and {y_axis}",
                    hover_data=[x_axis, y_axis] + ([color_by] if color_by != "None" else [])
                )
            else:
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    trendline="ols",
                    opacity=0.7,
                    title=f"Relationship between {x_axis} and {y_axis}"
                )
            
            # Update layout for better appearance
            fig.update_layout(
                height=500,
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                legend_title=color_by if color_by != "None" else None
            )
            
            safe_plotly_chart(fig)
            
            # Enhanced correlation insight with interpretation
            corr = df[x_axis].corr(df[y_axis])
            
            # Create an info box with user-friendly insights
            insight_box = st.container()
            with insight_box:
                # Determine correlation strength
                if abs(corr) < 0.3:
                    strength = "Weak"
                elif abs(corr) < 0.7:
                    strength = "Moderate"
                else:
                    strength = "Strong"
                
                # Determine direction
                direction = "positive" if corr > 0 else "negative"
                
                # First determine the interpretation text based on correlation
                if corr > 0.7:
                    interpretation = "As values in one variable increase, the values in the other variable strongly tend to increase as well."
                elif corr > 0.3:
                    interpretation = "As values in one variable increase, the values in the other variable moderately tend to increase as well."
                elif corr > -0.3:
                    interpretation = "There is little to no consistent relationship between these variables."
                elif corr > -0.7:
                    interpretation = "As values in one variable increase, the values in the other variable moderately tend to decrease."
                else:
                    interpretation = "As values in one variable increase, the values in the other variable strongly tend to decrease."

                # Create a complete insight box in a single markdown call
                st.markdown(f"""
                <div class="insight-box correlation">
                    <div class="insight-header">📊 Correlation Analysis</div>
                    <div class="insight-content">
                        <ul>
                            <li><strong>Correlation Strength:</strong> {strength} ({corr:.3f})</li>
                            <li><strong>Correlation Direction:</strong> {direction}</li>
                            <li><strong>Interpretation:</strong> {interpretation}</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)    
    # Outlier Analysis
    if numerical_cols:
        st.subheader("Outlier Analysis")
        selected_col = st.selectbox("Select column for outlier analysis:", numerical_cols)
        
        if selected_col:
            outliers, lower, upper = detect_outliers(df, selected_col)
            
            # Plot showing outlier boundaries
            outlier_fig = px.histogram(
                df, 
                x=selected_col,
                title=f"Distribution with Outlier Boundaries for {selected_col}",
                color_discrete_sequence=['#636EFA']
            )
            
            # Add vertical lines for lower and upper bounds
            outlier_fig.add_vline(x=lower, line_dash="dash", line_color="red", annotation_text=f"Lower Bound: {lower:.2f}")
            outlier_fig.add_vline(x=upper, line_dash="dash", line_color="red", annotation_text=f"Upper Bound: {upper:.2f}")
            
            # Enhance outlier visualization with box plot
            box_fig = px.box(
                df, 
                y=selected_col,
                title=f"Box Plot for {selected_col}",
                color_discrete_sequence=['#636EFA']
            )
            
            # Display the two figures side by side
            col1, col2 = st.columns(2)
            with col1:
                safe_plotly_chart(outlier_fig)
            with col2:
                safe_plotly_chart(box_fig)
            
            # Enhanced insight for outliers with better styling
            outlier_percentage = (len(outliers) / len(df)) * 100
            st.markdown(f"""
            <div class="insight-box correlation">
                <div class="insight-header">⚠️ Outlier Analysis Results</div>
                <div class="insight-content">
                    <ul>
                        <li><strong>Number of outliers:</strong> {len(outliers)} ({outlier_percentage:.2f}% of data)</li>
                        <li><strong>Outlier range:</strong> Values < {lower:.2f} or > {upper:.2f}</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if len(outliers) > 0:
                st.markdown("<h4>Sample of Detected Outliers:</h4>", unsafe_allow_html=True)
                st.dataframe(outliers.head(min(5, len(outliers))), use_container_width=True)
    
    # Categorical Analysis
    if categorical_cols:
        st.subheader("Categorical Analysis")
        selected_cat_col = st.selectbox("Select a categorical column:", categorical_cols)
        if selected_cat_col:
            # Create enhanced categorical visualization
            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = [selected_cat_col, 'Count']
            value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
            
            # Sort by count for better visualization
            value_counts = value_counts.sort_values('Count', ascending=False)
            
            # Limit to top 15 categories if there are many
            if len(value_counts) > 15:
                st.info(f"Showing top 15 out of {len(value_counts)} categories.")
                display_counts = value_counts.head(15).copy()
            else:
                display_counts = value_counts.copy()
            
            # Create enhanced bar chart
            fig = px.bar(
                display_counts, 
                x=selected_cat_col, 
                y='Count',
                text='Percentage',
                color='Count',
                labels={selected_cat_col: selected_cat_col, 'Count': 'Frequency'},
                title=f"Distribution of {selected_cat_col}",
                color_continuous_scale='Viridis'
            )
            
            # Format text to show percentages
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
            # Improve layout
            fig.update_layout(
                xaxis_title=selected_cat_col,
                yaxis_title='Frequency',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced insights for categorical variable with better styling
            most_common = df[selected_cat_col].value_counts().idxmax()
            most_common_pct = (df[selected_cat_col].value_counts().max() / len(df)) * 100
            least_common = df[selected_cat_col].value_counts().idxmin()
            least_common_pct = (df[selected_cat_col].value_counts().min() / len(df)) * 100
            unique_count = df[selected_cat_col].nunique()
            
            st.markdown(f"""
            <div class="insight-box categorical">
                <div class="insight-header">📊 Category Distribution Insights</div>
                <div class="insight-content">
                    <ul>
                        <li><strong>Total unique values:</strong> {unique_count}</li>
                        <li><strong>Most common:</strong> '{most_common}' ({most_common_pct:.2f}%)</li>
                        <li><strong>Least common:</strong> '{least_common}' ({least_common_pct:.2f}%)</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            

# Improved function to handle file processing that checks for new files
def process_uploaded_file(uploaded_file, max_rows):
    """Process an uploaded file (CSV or Excel) and return a dataframe"""
    try:
        # Check if this is a new file
        current_filename = uploaded_file.name if uploaded_file else None
        
        # If this is a new file or we don't have a dataframe, process it
        if current_filename != st.session_state.current_file_name or st.session_state.dataframe is None:
            # Update the current file name
            st.session_state.current_file_name = current_filename
            
            # Clear the dataset summary when a new file is uploaded
            st.session_state.dataset_summary = None
            st.session_state.sweetviz_html = None
            
            # Get file extension
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            # Process based on file type
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
                file_type = "CSV"
            elif file_extension in ['xlsx', 'xls']:
                # For Excel files, get all sheet names
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # If there are multiple sheets, let the user choose
                if len(sheet_names) > 1:
                    # Store the Excel file info in session state for later use
                    if 'excel_sheets' not in st.session_state:
                        st.session_state.excel_sheets = sheet_names
                    
                    selected_sheet = st.sidebar.selectbox(
                        "Select Excel sheet:",
                        options=sheet_names,
                        key="excel_sheet_selector"
                    )
                else:
                    selected_sheet = sheet_names[0]
                
                # Load the selected sheet
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                file_type = f"Excel ({selected_sheet})"
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Apply row limit if needed
            original_row_count = len(df)
            if len(df) > max_rows:
                df = df.head(max_rows)
                limit_message = f"⚠️ The {file_type} file contains {original_row_count:,} rows. Using only the first {max_rows:,} rows for analysis."
            else:
                limit_message = None
            
            return df, file_type, original_row_count, limit_message
        else:
            # Return the existing dataframe if no new file is uploaded
            return st.session_state.dataframe, st.session_state.file_type, len(st.session_state.dataframe), None
    
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

# Improved function to completely reset the app state
def clear_all_data():
    """Function to completely reset the app state and remove temporary files"""
    # Keep only specific keys but modify them 
    preserved_keys = ['openai_key', 'openai_api_key', 'reset_phase', 'file_uploader_key']
    
    # Get all keys to delete
    keys_to_delete = [key for key in st.session_state.keys() if key not in preserved_keys]
    
    # Delete those keys
    for key in keys_to_delete:
        del st.session_state[key]
    
    # Increment the file uploader key to create a fresh file uploader
    st.session_state.file_uploader_key += 1
    
    # Reinitialize essential session state variables with defaults
    st.session_state.dataframe = None
    st.session_state.analysis_complete = False
    st.session_state.sweetviz_html = None
    st.session_state.dataset_summary = None
    st.session_state.max_rows = 500
    st.session_state.current_file_name = None
    
    # Remove cached Sweetviz file if it exists
    if os.path.exists("sweetviz_report.html"):
        try:
            os.remove("sweetviz_report.html")
        except Exception:
            pass
            
    # Also try to clear any other temporary files
    for temp_file in ["SWEETVIZ_REPORT.html", "SWEETVIZ_REPORT.html.bak"]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    # Set phase to trigger a rerun
    st.session_state.reset_phase = 2

# Main Streamlit UI with sidebar
def main():
    # Handle reset phases
    if st.session_state.reset_phase == 2:
        # Just finished resetting, now do a clean rerun
        st.session_state.reset_phase = 0
        st.rerun()
    
    # Main title with a colorful header
    st.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 1.5rem; border-radius: 10px; background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);">
        <h1 style="color: white;">📊 Auto-EDA with Sweetviz & OpenAI</h1>
        <p style="color: white;">Upload your dataset and get instant visual insights powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Configuration")
    
    # OpenAI API configuration in sidebar
    api_configured = configure_openai()
    
    # File upload in sidebar - Using dynamic key that changes on reset
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a dataset (CSV or Excel)", 
        type=["csv", "xlsx", "xls"],
        key=f"file_uploader_{st.session_state.file_uploader_key}"  # Dynamic key
    )
    
    # Row limit configuration
    st.sidebar.subheader("Analysis Settings")
    max_rows = st.sidebar.slider("Max rows to analyze", min_value=100, max_value=10000, value=st.session_state.max_rows, step=100)
    st.session_state.max_rows = max_rows
    
    # Add note about row limit
    if max_rows < 10000:
        st.sidebar.info(f"Note: For performance reasons, only the first {max_rows} rows will be used for analysis.")
    
    # Add a separator for the reset button section to make it stand out
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reset Application")
    
    # Reset button handling
    if st.session_state.reset_phase == 0:
        # Normal state - show reset button
        if st.sidebar.button("🗑️ Reset App", type="primary", use_container_width=True):
            st.session_state.reset_phase = 1
            st.rerun()
    
    elif st.session_state.reset_phase == 1:
        # Confirmation state
        st.sidebar.warning("⚠️ Are you sure? All data and analysis will be lost.")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.reset_phase = 0
                st.rerun()
        
        with col2:
            if st.button("Yes, Reset", type="primary", use_container_width=True):
                with st.spinner("Resetting application..."):
                    clear_all_data()
                # The rerun will happen via reset_phase=2
    
    # Show app info in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("About this app"):
        st.markdown("""
        This app provides automated exploratory data analysis of CSV and Excel files with:
        
        - AI-powered dataset summaries
        - Statistical analysis & visualizations
        - Missing value detection
        - Correlation analysis
        - Outlier detection
        - Categorical data insights
        """)
    
    if not api_configured:
        # Display welcome message when no API key is provided
        st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
        
        # Add some helpful getting started content
        st.markdown("""
        ## Welcome to Auto-EDA! 
        
        This tool helps you explore and understand your data quickly with AI-powered insights.
        
        ### Getting Started:
        1. Enter your OpenAI API key in the sidebar
        2. Upload a CSV or Excel file
        3. Explore the automatically generated visualizations and insights
        
        ### Features:
        - **AI-Generated Summaries**: Get an instant understanding of your dataset
        - **Interactive Visualizations**: Explore relationships in your data
        - **Statistical Analysis**: Understand distributions and outliers
        - **Correlation Analysis**: Discover hidden patterns
        """)
        
        return
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                # Use the improved file processing function
                df, file_type, original_row_count, limit_message = process_uploaded_file(uploaded_file, max_rows)
                
                # Display limit message if needed
                if limit_message:
                    st.warning(limit_message)
                
                # Provide feedback about the loaded file
                st.success(f"✅ Successfully loaded {file_type} file with {df.shape[0]:,} rows and {df.shape[1]:,} columns")
                
                st.session_state.dataframe = df
                st.session_state.file_type = file_type
                st.session_state.analysis_complete = True
                
                # Generate Sweetviz report if not already in session state
                if st.session_state.sweetviz_html is None:
                    with st.spinner(f"Generating Sweetviz report for {file_type} data (this may take a moment)..."):
                        st.session_state.sweetviz_html = generate_sweetviz_report(df)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            if "Excel" in str(e):
                st.info("💡 Tip: If you're having issues with Excel files, try saving as CSV format first or choose a different sheet.")
    
    # If no file is uploaded yet, show a prompt
    if not st.session_state.analysis_complete and uploaded_file is None:
        st.info("👈 Please upload a CSV or Excel file in the sidebar to begin analysis.")
    
    # Display analysis if complete
    if st.session_state.analysis_complete:
        df = st.session_state.dataframe
        
        # Show dataset overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        col4.metric("Data Types", f"{df.dtypes.nunique():,}")
        
        # Create tabs with the new structure
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📊 Overview", "🔍 Visualizations", "📈 Sweetviz Report"])
        
        with tab1:
            dataset_tab(df)
        
        with tab2:
            overview_tab(df)
        
        with tab3:
            visualizations_tab(df)
            
        with tab4:
            st.subheader("Sweetviz Automated EDA Report")
            st.markdown("This report provides a comprehensive overview of your dataset with visualizations and statistics.")
            
            # Display the Sweetviz HTML report
            if st.session_state.sweetviz_html:
                st.components.v1.html(st.session_state.sweetviz_html, height=800, scrolling=True)
            else:
                with st.spinner("Generating Sweetviz report..."):
                    html_content = generate_sweetviz_report(df)
                    st.components.v1.html(html_content, height=800, scrolling=True)

# Application entry point
if __name__ == "__main__":
    main()
