import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error, \
    explained_variance_score, accuracy_score

from project.utility import (
    read_and_prepare_dataset,
    data_cleaning,
    remove_outliers,
    get_top_bottom_salaries,
    get_mean_salary_by_group,
    encode_categorical_columns
)


def main_program(path, logger):
    dataframe = read_and_prepare_dataset(logger, path)

    # Display top 5 entries for fun
    print("Top 5 entries:")
    print(
        dataframe.head(),
        end='\n-------------------------------------\n\n'
    )

    # Check for missing values
    print("Display number of missing values")
    print(
        dataframe.isnull().sum(),
        end='\n-------------------------------------\n\n'
    )

    # Clean Data
    dataframe = data_cleaning(logger, dataframe)

    # Check for missing values after cleaning
    print("Display number of missing values after cleaning")
    print(
        dataframe.isnull().sum(),
        end='\n-------------------------------------\n\n'
    )

    # Part one
    def simple_calculations(df):
        # Find top and bottom salaries for each year and job category
        top_bottom_salaries = get_top_bottom_salaries(
            logger,
            df,
            ['Work Year', 'Job Category'],
            'Salary in USD',
            1
        )
        print("Display top and bottom salaries for each year and job category")
        print(
            top_bottom_salaries,
            end='\n-------------------------------------\n\n'
        )

        # Detect and filter out outliers using the IQR method
        df = remove_outliers(logger, df, 'Salary in USD')

        # Find top and bottom salaries for each year and job category, after removing outliers
        top_bottom_salaries_after_outlier_removal = get_top_bottom_salaries(
            logger,
            df,
            ['Work Year', 'Job Category'],
            'Salary in USD',
            1
        )
        print("Display top and bottom salaries for each year and job category, after removing outliers")
        print(
            top_bottom_salaries_after_outlier_removal,
            end='\n-------------------------------------\n\n'
        )

        # Find average salary for each job category
        average_salary_by_job_category = get_mean_salary_by_group(
            logger,
            df,
            'Job Category',
            'Salary in USD'
        )
        print("Display average salary for each job category")
        print(
            average_salary_by_job_category,
            end='\n-------------------------------------\n\n'
        )

        # Find average salary for each job category, every year
        average_salary_by_job_category_every_year = get_mean_salary_by_group(
            logger,
            df,
            ['Work Year', 'Job Category'],
            'Salary in USD'
        )
        print("Display average salary for each job, category every year")
        print(
            average_salary_by_job_category_every_year,
            end='\n-------------------------------------\n\n'
        )

        # Find average salary for each job category and experience level, every year
        average_salary_by_job_category_every_year = get_mean_salary_by_group(
            logger,
            df,
            ['Work Year', 'Job Category', 'Experience Level'],
            'Salary in USD'
        )
        print("Display average salary for each job, category every year")
        print(
            average_salary_by_job_category_every_year,
            end='\n-------------------------------------\n\n'
        )

        # Find average salary for each job category and experience level, every year
        average_salary_by_job_category_every_year = get_mean_salary_by_group(
            logger,
            df,
            ['Work Year', 'Work Setting'],
            'Salary in USD'
        )
        print("Display average salary for each job, category every year")
        print(
            average_salary_by_job_category_every_year,
            end='\n-------------------------------------\n\n'
        )

        # Find average salary for each job category and experience level, every year
        average_salary_by_job_category_every_year = get_mean_salary_by_group(
            logger,
            df,
            ['Work Year', 'Job Category', 'Work Setting'],
            'Salary in USD'
        )
        print("Display average salary for each job, category every year")
        print(
            average_salary_by_job_category_every_year,
            end='\n-------------------------------------\n\n'
        )

    def visualizations_and_analysis(data):
        def create_box_plots(df):
            try:
                # Create subplots
                fig, axes = plt.subplots(3, 2, figsize=(20, 15))
                # Categories for which we want to create box plots
                box_plots = ['Experience Level', 'Work Setting', 'Employment Type', 'Job Category', 'Work Year',
                             'Company Size']

                # Box plots for various metrics
                for ax, metric_column in zip(axes.flatten(), box_plots):
                    sns.boxplot(ax=ax, x=metric_column, y='Salary in USD', data=df)
                    ax.set_title(f'Salary Distribution by {metric_column}')
                    ax.tick_params(axis='x', rotation=45)

                # Adjust layout
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.error(f"Error creating box plots: {e}")
                print(f"Error creating box plots: {e}")

        def create_bar_charts_for_categories(df):
            try:
                # Create subplots
                fig, axes = plt.subplots(3, 2, figsize=(20, 15))
                box_plots = ['Experience Level', 'Work Setting', 'Employment Type', 'Job Category', 'Work Year',
                             'Company Size']

                # Salary bar charts for various categories
                for ax, category in zip(axes.flatten(), box_plots):
                    avg_salary = df.groupby(category)['Salary in USD'].mean()
                    bars = avg_salary.plot(kind='bar', ax=ax)
                    ax.set_title(f'Average Salary by {category}')
                    ax.set_ylabel('Average Salary in USD')
                    ax.set_xlabel(category)
                    ax.tick_params(axis='x', rotation=45)

                    # Annotate bars with the exact numbers
                    for bar in bars.patches:
                        ax.annotate(f'{bar.get_height():.0f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    xytext=(0, 0),
                                    textcoords='offset points',
                                    ha='center', va='bottom')

                # Adjust layout
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.error(f"Error creating bar charts: {e}")
                print(f"Error creating bar charts: {e}")

        def create_bar_charts_and_pie_charts_for_categories(df):
            try:
                fig, axes = plt.subplots(4, 2, figsize=(20, 15))
                categories = ['Experience Level', 'Company Size', 'Work Setting', 'Employment Type']

                # Bar charts and pie charts for number of jobs by various categories
                for i, category in enumerate(categories):
                    count = df[category].value_counts()
                    bars = count.plot(kind='bar', ax=axes[i, 0], color='skyblue')
                    axes[i, 0].set_title(f'Number of Jobs by {category}')
                    axes[i, 0].set_ylabel('Count')
                    axes[i, 0].set_xlabel(category)
                    axes[i, 0].tick_params(axis='x', rotation=45)

                    # Annotate bars with the exact numbers
                    for bar in bars.patches:
                        axes[i, 0].annotate(f'{bar.get_height():.0f}',
                                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                            xytext=(0, 0),
                                            textcoords='offset points',
                                            ha='center', va='bottom')

                    # Pie chart
                    wedges, texts = axes[i, 1].pie(count, startangle=90, colors=sns.color_palette('pastel'))
                    axes[i, 1].set_title(f'Number of Jobs by {category}')
                    axes[i, 1].set_ylabel('')

                    # Add legend
                    axes[i, 1].legend(wedges,
                                      [f'{cat} - {pct:.1f}%' for cat, pct in
                                       zip(count.index, 100 * count / count.sum())],
                                      title=f'{category}', loc='center left', bbox_to_anchor=(1, 0.5))

                # Adjust layout
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.error(f"Error creating bart charts and pie charts: {e}")
                print(f"Error creating bart charts and pie charts: {e}")

        def create_violin_plots(df):
            try:
                # Categories for which we want to create violin plots
                categories = ['Experience Level', 'Work Setting', 'Employment Type', 'Job Category', 'Work Year',
                              'Company Size']

                # Create subplots
                fig, axes = plt.subplots(3, 2, figsize=(18, 15))

                # Violin plots for various categories
                for ax, category in zip(axes.flatten(), categories):
                    sns.violinplot(ax=ax, x=category, y='Salary in USD', data=df, hue=category, palette='Set2',
                                   legend=False)
                    ax.set_title(f'Salary Distribution by {category}')
                    ax.set_xlabel(category)
                    ax.set_ylabel('Salary in USD')
                    ax.tick_params(axis='x', rotation=45)

                # Adjust layout
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.error(f"Error creating violin plots: {e}")
                print(f"Error creating violin plots: {e}")

        def create_correlation_heatmap(df):
            try:
                # Calculate the correlation matrix
                correlation_matrix = df.corr()

                # Visualize the correlation heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Heatmap of All Relevant Columns')
                plt.show()
            except Exception as e:
                logger.error(f"Error creating correlation heatmap: {e}")
                print(f"Error creating correlation heatmap: {e}")

        def create_pairpot(df):
            try:
                # Select relevant columns for the pair plot
                pair_plot_columns = ['Work Year', 'Salary in USD', 'Experience Level', 'Employment Type',
                                     'Company Size',
                                     'Job Category']

                # Create a pair plot
                sns.pairplot(df[pair_plot_columns], hue="Experience Level", palette='Set2', diag_kind='kde')
                plt.suptitle('Pair Plot of Relevant Columns', y=1.02)
                plt.show()
            except Exception as e:
                logger.error(f"Error creating pairplots: {e}")
                print(f"Error creating pairplots: {e}")

        def conduct_anova(df):
            try:
                formula = ('Q("Salary in USD") ~ C(Q("Experience Level")) + C(Q("Employment Type")) + ' +
                           'C(Q("Job Category")) + C(Q("Work Setting")) + C(Q("Company Size"))')

                model = smf.ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)

                print("ANOVA table:")
                print(anova_table)
                print("-------------------------------------\n")
            except Exception as e:
                logger.error(f"Error during conducting ANOVA {e}")
                print(f"Error during conducting ANOVA {e}")

        def analyse_linear_regression(df):
            try:
                # Convert to numpy arrays
                X = np.column_stack(
                    (
                        df['Work Year'],
                        df['Experience Level'],
                        df['Work Setting'],
                        df['Company Size'],
                        df['Employment Type'],
                    )
                )
                y = np.array(df['Salary in USD'])

                # Add a column of ones to X for the intercept term
                X = np.column_stack([np.ones(X.shape[0]), X])

                # Calculate the coefficients using the normal equation
                coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

                # Print the coefficients with corresponding column names
                column_names = ['Intercept', 'Work Year', 'Experience Level', 'Work Setting', 'Company Size',
                                'Employment Type']

                coefficients_dict = {name: coef for name, coef in zip(column_names, coefficients)}

                print("Coefficients:")
                for name, coef in coefficients_dict.items():
                    print(f"{name}: {coef}")
                print("-------------------------------------\n")

                # Formula for determining probable salary: Salary=Intercept+(β1⋅Work Year)+
                # (β2⋅Experience Level)+(β3⋅Work Setting)+(β4⋅Company Size)+(β5⋅Employment Type)
            except Exception as e:
                logger.error(f"Error in linear regression analysis: {e}")
                print(f"Error in linear regression analysis: {e}")

        create_box_plots(data)
        create_bar_charts_for_categories(data)
        create_bar_charts_and_pie_charts_for_categories(data)
        create_violin_plots(data)

        custom_mappings_for_visualization = {
            'Experience Level': {'Entry-level': 0, 'Mid-level': 1, 'Senior': 2, 'Executive': 3},
            'Work Setting': {'Remote': 0, 'Hybrid': 1, 'In-person': 2},
            'Company Size': {'S': 0, 'M': 1, 'L': 2},
            'Employment Type': {'Freelance': 0, 'Part-time': 1, 'Contract': 2, 'Full-time': 3}
        }

        data_encoded, _ = encode_categorical_columns(logger, data, custom_mappings_for_visualization)
        create_correlation_heatmap(data_encoded)
        create_pairpot(data_encoded)
        conduct_anova(data_encoded)

        custom_mappings_for_linear_regression = {
            'Work Year': {2020: 0, 2021: 1, 2022: 2, 2023: 3, 2024: 4},
            'Experience Level': {'Entry-level': 0, 'Mid-level': 1, 'Senior': 2, 'Executive': 3},
            'Work Setting': {'Remote': 0, 'Hybrid': 1, 'In-person': 2},
            'Company Size': {'S': 0, 'M': 1, 'L': 2},
            'Employment Type': {'Freelance': 0, 'Part-time': 1, 'Contract': 2, 'Full-time': 3}
        }

        data_encoded, _ = encode_categorical_columns(logger, data, custom_mappings_for_linear_regression)
        analyse_linear_regression(data_encoded)

    def machine_learning_models(data):
        def random_forests(df):
            try:
                # Define the features and the target
                X = df.drop('Salary in USD', axis=1)
                y = df['Salary in USD']

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

                # Create and train the Random Forest Regressor model
                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=30,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    random_state=35
                )

                rf_model.fit(X_train, y_train)

                # Predict and evaluate the model
                y_pred = rf_model.predict(X_test)
                print(f"Random Forest Regression R^2: {r2_score(y_test, y_pred)}")
                print(f"Random Forest Regression MSE: {mean_squared_error(y_test, y_pred)}")
                print(f"Random Forest Regression MAE: {mean_absolute_error(y_test, y_pred)}")
                print(f"Random Forest Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
                print(f"Random Forest Regression MAPE: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100}")
                print(f"Random Forest Regression Explained Variance: {explained_variance_score(y_test, y_pred)}")
                print(f"Random Forest Regression MSLE: {mean_squared_log_error(y_test, y_pred)}")
                print("-------------------------------------\n")

                # Feature importance
                feature_importances = rf_model.feature_importances_
                feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                print("Feature Importances:\n", feature_importance_df)
                print("-------------------------------------\n")

                # Visualize feature importance
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = feature_importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False, ax=ax)
                ax.set_title('Feature Importance for Decision Trees')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

                return rf_model
            except Exception as e:
                logger.error(f"Error in random_forests function: {e}")
                print(f"Error in random_forests function: {e}")

        def decision_trees(df):
            try:
                # Define the features and the target
                X = df.drop('Salary in USD', axis=1)
                y = df['Salary in USD']

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

                # Create and train the Decision Tree Regressor model
                dt_model = DecisionTreeRegressor(
                    max_depth=30,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    random_state=35
                )

                dt_model.fit(X_train, y_train)

                # Predict and evaluate the model
                y_pred = dt_model.predict(X_test)
                print(f"Decision Tree Regression R^2: {r2_score(y_test, y_pred)}")
                print(f"Decision Tree Regression MSE: {mean_squared_error(y_test, y_pred)}")
                print(f"Decision Tree Regression MAE: {mean_absolute_error(y_test, y_pred)}")
                print(f"Decision Tree Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
                print(f"Decision Tree Regression MAPE: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100}")
                print(f"Decision Tree Regression Explained Variance: {explained_variance_score(y_test, y_pred)}")
                print(f"Decision Tree Regression MSLE: {mean_squared_log_error(y_test, y_pred)}")
                print("-------------------------------------\n")

                # Feature importance
                feature_importances = dt_model.feature_importances_
                feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                print("Feature Importances:\n", feature_importance_df)
                print("-------------------------------------\n")

                # Visualize feature importance
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = feature_importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False, ax=ax)
                ax.set_title('Feature Importance for Decision Trees')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

                return dt_model
            except Exception as e:
                logger.error(f"Error in decision_trees function: {e}")
                print(f"Error in decision_trees function: {e}")

        custom_mappings = {
            'Experience Level': {'Entry-level': 0, 'Mid-level': 1, 'Senior': 2, 'Executive': 3},
            'Work Setting': {'Remote': 0, 'Hybrid': 1, 'In-person': 2},
            'Company Size': {'S': 0, 'M': 1, 'L': 2},
            'Employment Type': {'Freelance': 0, 'Part-time': 1, 'Contract': 2, 'Full-time': 3}
        }

        data_encoded, all_mappings = encode_categorical_columns(logger, data, custom_mappings)
        # Interaction Feature
        data_encoded['Experience Employment Type Interaction'] = (
                data_encoded['Experience Level'] * data_encoded['Employment Type']
        )

        data_encoded['Setting Employment Type Interaction'] = (
                data_encoded['Work Setting'] * data_encoded['Employment Type']
        )

        data_encoded['Company Size Employment Type Interaction'] = (
                data_encoded['Company Size'] * data_encoded['Employment Type']
        )

        # Binning Salary in USD
        bins = [0, 40000, 80000, 150000, np.inf]
        labels = ['Low', 'Medium', 'High', 'Very High']
        bins_label_mapping = {label: idx for idx, label in enumerate(labels)}
        data_encoded['Salary Bin'] = pd.cut(data_encoded['Salary in USD'], bins=bins, labels=labels)

        # Directly map Salary Bin to numerical values and handle missing values
        data_encoded['Salary Bin'] = (data_encoded['Salary Bin'].astype(str).map(
            bins_label_mapping
        ).fillna(-1).astype(int))

        random_forests(data_encoded)
        decision_trees(data_encoded)

    simple_calculations(dataframe)
    visualizations_and_analysis(dataframe)
    machine_learning_models(dataframe)
