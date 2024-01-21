from machine_learning.model_implementation.utils.model_implementation_utils import min_max_scale


def preprocessing_pipeline(X_train, X_test, y_train, y_test):
    """Applies each pre-processing function in the training and test datas."""
    boolean_features = ['PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION',
                       'CARDIOVASCULAR', 'RENAL_CHRONIC', 'OTHER_DISEASE', 'OBESITY', 'TOBACCO',
                       'INTUBED', 'ICU']
        
    X_train = create_boolean_columns(X_train, boolean_features)
    X_test = create_boolean_columns(X_test, boolean_features)
    
    X_train = correct_pregnant_for_men(X_train)
    X_test = correct_pregnant_for_men(X_test)
    
    pre_imputation_X_train = X_train.copy()
    X_train = mode_imputation(X_train, pre_imputation_X_train, boolean_features)
    X_test = mode_imputation(X_test, pre_imputation_X_train, boolean_features)
    
    X_train = intubed_and_icu_imputation(X_train)
    X_test = intubed_and_icu_imputation(X_test)
    
    X_train = covid_degree(X_train)
    X_test = covid_degree(X_test)

    features = X_train.columns
    
    unscaled_X_train = X_train.copy()
    for feature in features:
        X_train[feature] = min_max_scale(X_train[feature], unscaled_X_train[feature])
        X_test[feature] = min_max_scale(X_test[feature], unscaled_X_train[feature])
    
    y_train = binary_change(y_train)
    y_test = binary_change(y_test)
    
    return X_train, X_test, y_train, y_test

def create_boolean_columns(data, boolean_features):
    """Given a Pandas DataFrame and a list of string feature names, this function creates
    new boolean columns for each feature in the list. The new columns are 1 if the value
    of the feature is less than 3, and 0 otherwise."""
    new_data = data.copy()
    for feature in boolean_features:
        new_data.loc[new_data[feature] < 3, f'is_{feature}_defined'] = 1
        new_data.loc[new_data[feature] >= 3, f'is_{feature}_defined'] = 2
    return new_data

def correct_pregnant_for_men(data):
    """Given a Pandas DataFrame, this function sets the value of the 'PREGNANT' feature to 0
    for all rows where the value of the 'SEX' feature is 2 (corresponding to men)."""
    new_data = data.copy()
    new_data.loc[new_data['SEX'] == 2, 'PREGNANT'] = 0
    return new_data

def mode_imputation(data, pre_imputation_train_data, boolean_features):
    """Given a Pandas DataFrame, a Pandas DataFrame with the original training data used to
    create the model, and a list of string feature names, this function imputes the mode
    value for each feature in the list for all rows where the value of the feature is 3
    or above (corresponding to missing values)."""
    new_data = data.copy()
    for feature in boolean_features:
        most_common = pre_imputation_train_data[feature].mode()[0]
        new_data.loc[new_data[feature] >= 3, feature] = most_common
    return new_data

def intubed_and_icu_imputation(data):
    """Given a Pandas DataFrame, this function sets the value of the 'INTUBED' and 'ICU'
    features to 3 (corresponding to missing values) for all rows where the value is 3 or above."""
    new_data = data.copy()
    more_nan_features = ['INTUBED', 'ICU']
    for feature in more_nan_features:
        new_data.loc[new_data[feature] >= 3, feature] = 3
    return new_data

def covid_degree(data):
    """takes in a pandas DataFrame and returns a copy of the DataFrame with an added column called 'covid_degree'.
    The 'covid_degree' column is based on the 'CLASSIFICATION_FINAL' column in the input DataFrame.
    If the value in the 'CLASSIFICATION_FINAL' column is greater than or equal to 4, the corresponding value in the 'covid_degree' column is 0.
    If the value in the 'CLASSIFICATION_FINAL' column is less than 4, the corresponding value in the 'covid_degree' column
    is the same as the value in the 'CLASSIFICATION_FINAL' column.
    The 'CLASSIFICATION_FINAL' column is then dropped from the DataFrame."""
    new_data = data.copy()
    new_data.loc[new_data['CLASSIFICATION_FINAL'] >= 4, 'covid_degree'] = 0
    new_data.loc[new_data['CLASSIFICATION_FINAL'] < 4, 'covid_degree'] = new_data['CLASSIFICATION_FINAL']
    new_data.drop('CLASSIFICATION_FINAL', axis = 1, inplace = True)
    return new_data

def binary_change(data):
    """This function takes in a pandas DataFrame or a list or numpy array
    and returns a copy of the DataFrame or array with all 2's replaced with 0's."""
    new_data = data.copy()
    new_data.loc[new_data == 2] = 0
    return new_data