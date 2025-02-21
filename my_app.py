
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Titanic Dataset Analyzer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF4B4B;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_and_process_data():
    """
    Load and preprocess the Titanic dataset
    """
    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 35, 50, 65, 100],
                           labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
    
    # Create fare categories
    df['FareCategory'] = pd.qcut(df['Fare'], 
                                q=4, 
                                labels=['Economy', 'Standard', 'Premium', 'Luxury'])
    
    return df

def create_visualization(df, query):
    """
    Create appropriate visualization based on the query
    """
    query = query.lower()

    try:
        if 'age distribution' in query:
            fig = px.histogram(df, x='Age', 
                             title='Age Distribution of Passengers',
                             nbins=30,
                             color_discrete_sequence=['#FF4B4B'],
                             labels={'Age': 'Age (years)', 'count': 'Number of Passengers'})
            fig.update_traces(opacity=0.75)

        elif 'survival rate by class' in query:
            survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
            fig = px.bar(x=['First', 'Second', 'Third'], 
                        y=survival_by_class.values,
                        title='Survival Rate by Passenger Class',
                        labels={'x': 'Passenger Class', 'y': 'Survival Rate (%)'},
                        color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(opacity=0.75)

        elif 'pie chart' in query and 'class' in query:
            class_dist = df['Pclass'].value_counts()
            fig = px.pie(values=class_dist.values, 
                        names=['First', 'Second', 'Third'],
                        title='Distribution of Passenger Classes',
                        color_discrete_sequence=['#FF4B4B', '#FF8B8B', '#FFB4B4'])

        elif 'survival' in query and ('gender' in query or 'male' in query or 'female' in query):
            survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
            fig = px.bar(x=survival_by_sex.index,
                        y=survival_by_sex.values,
                        title='Survival Rate by Gender',
                        labels={'x': 'Gender', 'y': 'Survival Rate (%)'},
                        color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(opacity=0.75)

        else:
            return None

        # Apply common layout updates
        fig.update_layout(
            template='plotly_white',
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            margin=dict(t=50, l=0, r=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

class TitanicAgent:
    def __init__(self, df):
        self.df = df

    def process_query(self, query):
        try:
            query_lower = query.lower()

            if "survived" in query_lower:
                survived_count = self.df['Survived'].sum()
                total_count = len(self.df)
                return f"🚢 {survived_count} passengers survived out of {total_count} ({(survived_count/total_count)*100:.1f}%)"

            elif "average age" in query_lower:
                avg_age = self.df['Age'].mean()
                return f"👥 The average age of passengers was {avg_age:.1f} years"

            elif "male" in query_lower or "female" in query_lower or "gender" in query_lower:
                gender_stats = self.df['Sex'].value_counts()
                return f"👥 Passenger gender distribution:\nMale: {gender_stats.get('male', 0)}\nFemale: {gender_stats.get('female', 0)}"

            elif "class" in query_lower and not any(term in query_lower for term in ['show', 'plot', 'chart', 'graph']):
                class_stats = self.df['Pclass'].value_counts().sort_index()
                return f"🎫 Passenger class distribution:\nFirst Class: {class_stats.get(1, 0)}\nSecond Class: {class_stats.get(2, 0)}\nThird Class: {class_stats.get(3, 0)}"

            elif "embarked" in query_lower or "port" in query_lower:
                embarked_stats = self.df['Embarked'].value_counts()
                port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
                response = "🚢 Embarkation ports:\n"
                for port, count in embarked_stats.items():
                    response += f"{port_names.get(port, port)}: {count} passengers\n"
                return response

            elif "fare" in query_lower and not any(term in query_lower for term in ['show', 'plot', 'chart', 'graph']):
                avg_fare = self.df['Fare'].mean()
                max_fare = self.df['Fare'].max()
                min_fare = self.df['Fare'].min()
                return f"💰 Fare statistics:\nAverage: ${avg_fare:.2f}\nHighest: ${max_fare:.2f}\nLowest: ${min_fare:.2f}"

            else:
                return ("💡 Try asking about:\n"
                       "- Survival statistics\n"
                       "- Passenger demographics (age, gender)\n"
                       "- Ticket class distribution\n"
                       "- Embarkation ports\n"
                       "- Fare information\n"
                       "Or use keywords like 'show', 'plot', or 'chart' for visualizations!")

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return "❌ I apologize, but I couldn't process that query. Please try rephrasing or use one of the example queries."

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    # Header with custom styling
    st.title("🚢 Titanic Dataset Analysis Chatbot")
    st.markdown("---")

    # Load data
    with st.spinner("🔄 Loading dataset..."):
        df = load_and_process_data()
        agent = TitanicAgent(df)

    # Example queries in sidebar
    with st.sidebar:
        st.header("💡 Example Queries")
        example_queries = [
            "How many passengers survived?",
            "Show me the age distribution of passengers",
            "What was the survival rate by class?",
            "Create a pie chart of passenger classes",
            "Compare survival rates between males and females",
            "What was the most common age group?",
            "Show me the fare distribution by class",
            "How many passengers embarked from each port?",
            "What was the average ticket fare?"
        ]

        for query in example_queries:
            if st.button(f"🔍 {query}", key=f"btn_{query}"):
                st.session_state.messages.append({"role": "user", "content": query})

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if any(viz_term in message["content"].lower() 
                  for viz_term in ['show', 'plot', 'chart', 'graph', 'visualize']):
                with st.spinner("📊 Creating visualization..."):
                    fig = create_visualization(df, message["content"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # Query input
    query = st.chat_input("💭 Ask about the Titanic dataset...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                response = agent.process_query(query)
                st.write(response)

                if any(viz_term in query.lower() 
                      for viz_term in ['show', 'plot', 'chart', 'graph', 'visualize']):
                    fig = create_visualization(df, query)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # Dataset overview
    with st.expander("📊 Dataset Overview", expanded=True):
        st.markdown("""
        <style>
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Passengers", len(df))
        with col2:
            st.metric("Survival Rate", f"{(df['Survived'].mean()*100):.1f}%")
        with col3:
            st.metric("Average Age", f"{df['Age'].mean():.1f} years")

if __name__ == "__main__":
    main()
