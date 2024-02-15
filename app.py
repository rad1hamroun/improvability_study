import plotly.express as px
import streamlit as st

from utils.data_manager import DataManager


def main():
    # initialize dashboard and data manager
    st.title("Ekinox Students Improvability Study")
    with st.sidebar:
        data_directory = st.text_input("Data directory:", "")
        dm = DataManager(data_directory=data_directory)
        kpi_columns = st.multiselect("Improvability features:", dm.columns)
        dm.reset_kpi_columns(kpi_columns)
        filter_by = st.selectbox("Filter by:", dm.columns, index=dm.columns.index('sex'))

    # load the data
    dm.load_data()
    dm.process_kpi_data()
    dm.generate_kpi()

    # 1st chart: FinalGrade x ImprovabilityScore
    chart1 = px.scatter(dm.data,
                        x='FinalGrade',
                        y='ImprovabilityScore',
                        title='FinalGrade x ImrovabilityScore',
                        hover_data=['StudentID', 'FirstName', 'FamilyName'],
                        color=filter_by)
    chart1.update_layout(xaxis=dict(autorange='reversed'))

    # 2nd chart: FinalGrade Histogram
    chart2 = px.histogram(dm.data,
                          x='FinalGrade',
                          color=filter_by,
                          marginal='box',
                          title=f'FinalGrade distribution by {filter_by}')

    # 3rd chart: ImprovabilityScore Histogram
    chart3 = px.histogram(dm.data,
                          x='ImprovabilityScore',
                          color=filter_by,
                          marginal='box',
                          title=f'ImprovabilityScore distribution by {filter_by}')

    # Display charts
    st.plotly_chart(chart1)
    st.plotly_chart(chart2)
    st.plotly_chart(chart3)


if __name__ == "__main__":
    main()
