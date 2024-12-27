import pandas as pd
from sklearn.preprocessing import LabelEncoder


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Read dataset, remove unwanted columns, rename columns and prepare for cleaning
def read_and_prepare_dataset(logger, path):
    try:
        # Load the dataset
        df = pd.read_csv(path)
        df = df.drop(columns=['salary', 'salary_currency'])

        # Rename columns for better readability
        df.rename(columns={
            'work_year': 'Work Year',
            'experience_level': 'Experience Level',
            'employment_type': 'Employment Type',
            'job_title': 'Job Title',
            'salary_in_usd': 'Salary in USD',
            'employee_residence': 'Employee Residence',
            'work_setting': 'Work Setting',
            'company_location': 'Company Location',
            'company_size': 'Company Size',
            'job_category': 'Job Category'
        }, inplace=True)

        # Define the custom ordering for 'Experience Level'
        experience_order = ['Entry-level', 'Mid-level', 'Senior', 'Executive']

        # Convert 'Experience Level' to a categorical data type with a custom order
        df['Experience Level'] = pd.Categorical(df['Experience Level'], categories=experience_order, ordered=True)
        df = df.sort_values(by=['Work Year', 'Job Category', 'Experience Level'])

        return df
    except Exception as e:
        logger.error(f"Error during reading and preparation of dataset: {e}")
        print(f"Error during reading and preparation of dataset: {e}")


# Dynamically remove outliers for specified groups
def remove_outliers(logger, df, salary_column, group_columns=None):
    def outlier_removal(group):
        Q1 = group[salary_column].quantile(0.25)
        Q3 = group[salary_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group[salary_column] >= lower_bound) & (group[salary_column] <= upper_bound)]

    try:
        # Apply the outlier removal function to each group dynamically
        if group_columns:
            df_cleaned = df.groupby(group_columns).apply(outlier_removal).reset_index(drop=True)
        else:
            df_cleaned = outlier_removal(df)

        return df_cleaned
    except Exception as e:
        logger.error(f"Error during outlier removal: {e}")
        print(f"Error during outlier removal: {e}")


# Data cleaning function
def data_cleaning(logger, df):
    try:
        # Convert 'Salary in USD' to numeric, invalid parsing will be set as NaN
        df['Salary in USD'] = pd.to_numeric(df['Salary in USD'], errors='coerce')
        # Fill missing values in 'Salary in USD' with the median salary
        df['Salary in USD'] = df['Salary in USD'].fillna(df['Salary in USD'].median())

        # Calculate median salaries for each experience level
        median_salaries = df.groupby('Experience Level', observed=True)['Salary in USD'].median().reset_index()

        # Function to determine the closest experience level based on salary
        def closest_experience_level(row):
            if pd.isna(row['Experience Level']):
                salary = row['Salary in USD']
                if not median_salaries.empty:
                    differences = abs(median_salaries['Salary in USD'] - salary)
                    closest_index = differences.idxmin()
                    return median_salaries.loc[closest_index, 'Experience Level']
            return row['Experience Level']

        # Apply the function to fill missing values in 'Experience Level'
        df['Experience Level'] = df.apply(closest_experience_level, axis=1)

        # Fill missing values in 'Employment Type' with the most frequent value and ensure it's not numeric
        df['Employment Type'] = df['Employment Type'].apply(lambda x: x if not is_numeric(x) else None)
        df['Employment Type'] = df['Employment Type'].fillna(df['Employment Type'].mode()[0])

        # Fill missing values in other columns with a placeholder and ensure they are not numeric
        text_columns = ['Job Title', 'Employee Residence', 'Work Setting', 'Company Location', 'Company Size',
                        'Job Category']
        for col in text_columns:
            df[col] = df[col].apply(lambda x: x if not is_numeric(x) else None)
            df[col] = df[col].fillna('Unknown')

        return df
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        print(f"Error during data cleaning: {e}")


# Function to get top and bottom salaries for each job category during each year
def get_top_bottom_salaries(logger, df, group_columns, salary_column, count, additional_return_columns=None):
    try:
        if additional_return_columns is None:
            additional_return_columns = []

        result = pd.DataFrame()

        # Group by the specified columns dynamically
        groupby_obj = df.groupby(group_columns)

        # Iterate through each group
        for group_keys, group in groupby_obj:
            # Get top salaries
            top_n = group.nlargest(count, salary_column)
            top_n['Position'] = f'Top {count}'

            # Get bottom salaries
            bottom_n = group.nsmallest(count, salary_column)
            bottom_n['Position'] = f'Bottom {count}'

            # Select the specified columns plus 'Position'
            top_n = top_n[group_columns + [salary_column] + additional_return_columns + ['Position']]
            bottom_n = bottom_n[group_columns + [salary_column] + additional_return_columns + ['Position']]

            # Concatenate the results
            result = pd.concat([result, top_n, bottom_n])

        return result
    except Exception as e:
        logger.error(f"Error while getting {count} top and bottom salaries: {e}")
        print(f"Error while getting {count} top and bottom salaries: {e}")


def get_mean_salary_by_group(logger, df, group_columns, salary_column):
    try:
        mean_salary_df = df.groupby(group_columns)[salary_column].mean().reset_index()
        return mean_salary_df
    except Exception as e:
        logger.error(f"Error while getting mean salary: {e}")
        print(f"Error while getting mean salary: {e}")


def encode_categorical_columns(logger, df, custom_mappings):
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Apply custom mappings
        for column, mapping in custom_mappings.items():
            df_copy[column] = df_copy[column].map(mapping, na_action='ignore')
            df_copy[column] = df_copy[column].fillna(-1)

        # Encode other categorical columns
        label_encoders = {}
        for column in df_copy.columns:
            if df_copy[column].dtype == 'object':
                le = LabelEncoder()
                df_copy[column] = le.fit_transform(df_copy[column])
                label_encoders[column] = {cls: int(label) for cls, label in zip(le.classes_, le.transform(le.classes_))}

        # Add custom mappings to label_encoders
        for column, mapping in custom_mappings.items():
            label_encoders[column] = mapping

        # Display the encoding mappings
        print("These are the encoded values of categorical columns:")
        for column, mapping in label_encoders.items():
            print(f"Encoding for {column}: {mapping}")
        print("-------------------------------------\n")
        return df_copy, label_encoders
    except Exception as e:
        logger.error(f"Error while encoding categorical columns: {e}")
        print(f"Error while encoding categorical columns: {e}")
