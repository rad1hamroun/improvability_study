import os

import pandas as pd
from data import DATA_DIR
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataManager:
    """
    A class for managing student data, including loading, validating,
    processing Key Performance Indicators (KPIs), and generating KPI scores.

    Attributes:
        data_directory (str): Path to the directory containing CSV files.
        files (list): List of CSV file paths in the data directory.
        columns (list): List of expected column names in the CSV files.
        kpi_columns (list): List of column names considered as Key Performance Indicators.
        data (DataFrame): DataFrame containing the loaded and validated data.
        kpi_data (DataFrame): DataFrame containing processed KPI data.

    Methods:
        __init__: Initializes the DataManager object.
        check_data_directory: Checks if the given data directory exists and contains CSV files.
        validate_data: Validates the structure and content of the data.
        load_data: Loads and validates data from CSV files.
        process_kpi_data: Processes KPI data, encoding categorical variables and scaling numerical ones.
        generate_kpi: Generates KPI scores using Principal Component Analysis (PCA).
        reset_kpi_columns: Resets the KPI columns with new ones.

    Example Usage:
        manager = DataManager(data_directory='path/to/data')
        manager.load_data()
        manager.process_kpi_data()
        manager.generate_kpi()
    """

    def __init__(self, data_directory: str):
        """
        Initializes a DataManager object with optional data directory.
        :param data_directory: (optional) Path to CSV data directory. Defaults to DATA_DIR.
        """
        if self.check_data_directory(data_directory):
            self.data_directory = data_directory
        else:
            self.data_directory = DATA_DIR
        self.files = [os.path.join(self.data_directory, f) for f in os.listdir(self.data_directory) if
                      f.endswith('.csv')]
        self.columns = ['StudentID', 'FirstName', 'FamilyName', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu',
                        'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'FinalGrade']
        self.kpi_columns = ['absences', 'Dalc', 'Walc', 'studytime']
        self.data = None
        self.kpi_data = None

    @staticmethod
    def check_data_directory(data_directory: str) -> bool:
        """
        Check if the given data directory exists and contains CSV files.
        :param data_directory: Path to the directory to check.
        :return bool: True if the directory exists and contains CSV files, False otherwise.
        """
        if os.path.exists(data_directory):
            files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
            if len(files) > 0:
                return True
            else:
                print('The given data directory is empty.')
                return False
        else:
            print("The given data directory doesn't exist.")
            return False

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the structure and content of the provided DataFrame
        :param df: DataFrame to be validated
        :return DataFrame: Validated DataFrame
        """
        assert not df.empty, 'Dataframe is empty'
        columns = df.columns.tolist()
        assert all([c in self.columns for c in columns]), 'Some columns are missing'
        df = df[self.columns].copy()
        assert df['StudentID'].is_unique, 'StudentID must be unique'
        assert df['sex'].isin(['M', 'F']).all()
        assert ((df['age'] >= 15) & (df['age'] <= 22)).all(), 'Age must be between 15 and 22'
        assert df['address'].isin(['U', 'R']).all(), 'Address must be either U or R'
        assert df['famsize'].isin(['LE3', 'GT3']).all(), 'Family size must be either LE3 or GT3'
        assert df['Pstatus'].isin(['T', 'A']).all(), 'Pstatus must be either T or A'
        assert df['Medu'].isin(list(range(5))).all(), 'Medu must be in (0, 1, 2, 3, 4)'
        assert df['Fedu'].isin(list(range(5))).all(), 'Fedu must be in (0, 1, 2, 3, 4)'
        assert df['Mjob'].isin(['teacher', 'health', 'services', 'at_home', 'other']).all(), "Mjob isn't recognized"
        assert df['Fjob'].isin(['teacher', 'health', 'services', 'at_home', 'other']).all(), "Fjob isn't recognized"
        assert df['reason'].isin(['home', 'reputation', 'course', 'other']).all(), "reason isn't recognized"
        assert df['guardian'].isin(['mother', 'father', 'other']).all(), "guardian isn't recognized"
        assert df['traveltime'].isin(list(range(1, 5))).all(), "traveltime must be in (1, 2, 3, 4)"
        assert df['studytime'].isin(list(range(1, 5))).all(), "studytime must be in (1, 2, 3, 4)"
        assert df['failures'].isin(list(range(5))).all(), "failures must be in (1, 2, 3, 4)"
        assert df['schoolsup'].isin(['yes', 'no']).all(), "schoolsup must be yes or no"
        assert df['famsup'].isin(['yes', 'no']).all(), "famsup must be yes or no"
        assert df['paid'].isin(['yes', 'no']).all(), "paid must be yes or no"
        assert df['activities'].isin(['yes', 'no']).all(), "activities must be yes or no"
        assert df['nursery'].isin(['yes', 'no']).all(), "nursery must be yes or no"
        assert df['higher'].isin(['yes', 'no']).all(), "higher must be yes or no"
        assert df['internet'].isin(['yes', 'no']).all(), "internet must be yes or no"
        assert df['romantic'].isin(['yes', 'no']).all(), "romantic must be yes or no"
        assert df['famrel'].isin(list(range(1, 6))).all(), "famrel must be between 1 and 5"
        assert df['freetime'].isin(list(range(1, 6))).all(), "freetime must be between 1 and 5"
        assert df['goout'].isin(list(range(1, 6))).all(), "goout must be between 1 and 5"
        assert df['Dalc'].isin(list(range(1, 6))).all(), "Dalc must be between 1 and 5"
        assert df['Walc'].isin(list(range(1, 6))).all(), "Dalc must be between 1 and 5"
        assert df['health'].isin(list(range(1, 6))).all(), "health must be between 1 and 5"
        assert ((df['absences'] >= 0) & (df['absences'] <= 93)).all(), "absences must be between 0 and 93"
        assert ((df['FinalGrade'] >= 0) & (df['FinalGrade'] <= 20)).all(), "FinalGrade must be between 0 and 20"
        return df

    def load_data(self):
        """
        Load and validate data from CSV files in the data directory.

        Notes:
        - Iterates over CSV files in the data directory, loads each file as a DataFrame, and validates it.
        - Validated DataFrames are concatenated into one DataFrame and assigned to the 'data' attribute.
        - Ensures the uniqueness of StudentID in the final DataFrame.
        """
        dfs = []
        for filename in self.files:
            print('Loading data from', filename)
            df = pd.read_csv(filename).dropna()
            try:
                df = self.validate_data(df)
            except AssertionError as e:
                print("Couldn't validate data:", e)
                print("skip file")
                continue
            print(f'Data checked and loaded')
            dfs.append(df)
        assert len(dfs) > 0, "No valid data found"
        self.data = pd.concat(dfs).drop_duplicates()
        assert self.data['StudentID'].is_unique, 'StudentID must be unique.'
        print('Full Data ready')

    def process_kpi_data(self):
        """
        Process Key Performance Indicator (KPI) data.

        Notes:
        - Initializes an empty DataFrame for KPI data with specified columns.
        - For each KPI column, encodes categorical variables if necessary using LabelEncoder.
        - Computes the correlation between each KPI and FinalGrade.
        - Adjusts KPI values based on correlation: negates if positive, drops if correlation is zero.
        - Updates 'kpi_data' and 'kpi_columns' attributes accordingly.
        """
        self.kpi_data = pd.DataFrame([], columns=self.kpi_columns)
        for c in self.kpi_columns:
            if self.data[c].dtype == 'int':
                self.kpi_data[c] = self.data[c]
            else:
                label_encoder = LabelEncoder()
                self.kpi_data[c] = label_encoder.fit_transform(self.data[c])
            correlation = self.data['FinalGrade'].corr(self.kpi_data[c])
            if correlation > 0:
                self.kpi_data[c] *= -1
            elif correlation == 0:
                self.kpi_data.drop(columns=[c])
                self.kpi_columns.remove(c)

    def generate_kpi(self):
        """
        Generate Key Performance Indicator (KPI) scores.
        Notes:
        - Standardizes the KPI data using StandardScaler.
        - Applies Principal Component Analysis (PCA) to reduce dimensionality to one component.
        - Calculates 'ImprovabilityScore' based on the transformed PCA component.
        - Adjusts 'ImprovabilityScore' to have a minimum value of zero.
        :return:
        """
        scaler = StandardScaler()
        self.kpi_data[self.kpi_columns] = scaler.fit_transform(self.kpi_data)
        pca = PCA(n_components=1)
        pca.fit(self.kpi_data)
        self.data['ImprovabilityScore'] = pca.transform(self.kpi_data)[:, 0]
        self.data['ImprovabilityScore'] = self.data['ImprovabilityScore'] - self.data['ImprovabilityScore'].min()

    def reset_kpi_columns(self, new_kpi_columns: list):
        """
        Reset Key Performance Indicator (KPI) columns.
        :param new_kpi_columns: (list) New list of KPI column names.
        """
        if len(new_kpi_columns) > 0:
            self.kpi_columns = new_kpi_columns
