import streamlit as st
import plotly.express as px
from utils.data_manager import DataManager


def get_df():
    dm = DataManager()
    dm.load_data()
    dm.process_kpi_data()
    dm.generate_kpi()
    return dm.data


def main():
    st.title("Dashboard 0")

    # Read and process the data
    df = get_df()
    X = df['FinalGrade']
    Y = df['ImporvabilityScore']

    # Create Plotly scatter plot
    fig = px.scatter(df,
                     x='FinalGrade',
                     y='ImprovabilityScore',
                     title='Dashboard',
                     hover_data=['StudentID', 'FirstName', 'FamilyName'],
                     color='sex',
                     symbol='address',
                     size=df['age']-df['age'].min())
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis=dict(autorange='reversed'))

    fig2 = px.histogram(df, x='ImporvabilityScore',
                        color='sex', marginal='box', title='Distribution by sex')
    # Display scatter plot in Streamlit
    st.plotly_chart(fig)
    st.plotly_chart(fig2)


if __name__ == "__main__":
    main()
