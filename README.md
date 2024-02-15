# Ekinox Students Improvability Study

This project is a Streamlit dashboard designed to study students improvability based on various collected variables.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Charts](#charts)

## Installation

To run this dashboard locally, you'll need Python installed on your machine. Clone this repository, navigate to the project directory, and install the required dependencies using pip:

```bash
git clone https://github.com/rad1hamroun/ekinox_improvability_study.git
cd ekinox_improvability_study
pip install -r requirements.txt
```

## Usage

To start the Streamlit dashboard, run the following command in your terminal:

```bash
streamlit run app.py
```

This will launch the dashboard in your default web browser. You can now interact with the dashboard by following the on-screen instructions.

![Dashboard Preview](Screenshot%20from%202024-02-15%2011-00-08.png)

## Features

- **Data directory:** You can set up a custom data directory. If the given data directory doesn't exist or empty, 
the data will be loaded from the default directory within the repository
- **Improvability features:** You can multi-select the columns on which the improvability score will be calculated 
- **Filter by:** Choose a column with which you want to filter the dashboard

## Charts

- **FinalGrade x ImprovabilityScore:** Scatter plot showing students according to their FinalGrade and ImprovabilityScore 
- **FinalGrade Distribution**: Histogram of the FinalGrade variable
- **ImprovabilityScore Distribution**: Histogram of the ImprovabilityScore KPI