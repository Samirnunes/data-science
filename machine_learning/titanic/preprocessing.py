import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder


def get_num_cols(X):
    return X.select_dtypes(['int32', 'float32', 'int64', 'float64']).columns


def get_cat_cols(X):
    return X.select_dtypes(['object']).columns


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.type == 'num':
            return X[get_num_cols(X)]
        elif self.type == 'cat':
            return X[get_cat_cols(X)]
        elif self.type == 'all':
            return X
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names):
        self.columns_names = columns_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(self.columns_names) > 0:
            return X.drop(self.columns_names, axis=1)
        return X


class ImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        imputer = SimpleImputer(strategy=self.strategy)
        return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='min_max'):
        self.scaler = None
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.strategy == 'min_max':
            self.scaler = MinMaxScaler()
        elif self.strategy == 'standard':
            self.scaler = StandardScaler()
        elif self.strategy == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        return pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)


class AgeClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, classify_ages=True):
        self.classify_ages = classify_ages

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.classify_ages:
            def classify_age(age):
                if age < 6:
                    return 0
                if 6 <= age < 12:
                    return 1
                if 12 < age <= 18:
                    return 2
                if 18 < age <= 60:
                    return 3
                else:
                    return 4

            X['AgeClass'] = X['Age'].apply(classify_age)
            return X.drop(['Age'], axis=1)
        return X


class FareClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, classify_fares=True):
        self.classify_fares = classify_fares

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.classify_fares:
            def classify_fare(fare):
                if fare <= 20:
                    return 0
                if 20 < fare < 50:
                    return 1
                else:
                    return 2

            X['FareClass'] = X['Fare'].apply(classify_fare)
            return X.drop(['Fare'], axis=1)
        return X


class TitleExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def extract_title(name):
            if type(name) == float:
                return 'Undefined'
            else:
                return name.split(',')[1].split('.')[0].replace(" ", "")

        X['Title'] = X['Name'].apply(extract_title)
        title_in_order = ['Sir', 'theCountess', 'Don', 'Capt', 'Rev', 'Col', 'Lady', 'Major', 'Dr', 'Mlle', 'Ms', 'Mme',
                          'Mrs', 'Miss', 'Master', 'Mr', 'Jonkheer', 'Undefined'][::-1]
        ordinal_encoder = OrdinalEncoder(categories=[title_in_order], unknown_value=len(title_in_order),
                                         handle_unknown='use_encoded_value')
        X['TitleImportance'] = ordinal_encoder.fit_transform(X[['Title']])
        return X.drop(columns=['Title', 'Name'])


class CabinClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def extract_cabin_letter(cabin):
            return str(cabin)[0]

        X['CabinLetter'] = X['Cabin'].apply(extract_cabin_letter)
        letters_in_order = ['n', 'T', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
        ordinal_encoder = OrdinalEncoder(categories=[letters_in_order])
        X['CabinClass'] = ordinal_encoder.fit_transform(X[['CabinLetter']])
        return X.drop(['Cabin', 'CabinLetter'], axis=1)


class EmbarkClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
        ordinal_encoder = OrdinalEncoder(categories=[['S', 'Q', 'C']])  # PoorerPlace, MediumPlace, RicherPlace
        X['EmbarkClass'] = ordinal_encoder.fit_transform(X[['Embarked']])
        return X.drop(['Embarked'], axis=1)


class SexExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        one_hot_encoder = OneHotEncoder()
        one_hot_df = pd.DataFrame(one_hot_encoder.fit_transform(X[['Sex']]).toarray(),
                                  columns=one_hot_encoder.categories_[0])
        X = pd.concat([X.drop(['Sex'], axis=1), one_hot_df.drop(['female'], axis=1)], axis=1)
        return X.rename(columns={'male': 'Male'})


class AloneExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['TotalFamily'] = X['SibSp'] + X['Parch']
        X['Alone'] = X['TotalFamily'].apply(lambda x: 1 if x == 0 else 0)
        return X.drop(['TotalFamily', 'SibSp', 'Parch'], axis=1)


class DataFrameConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=['Pclass', 'AgeClass', 'FareClass', 'Alone', 'CabinClass', 'EmbarkClass', 'Male',
                                        'TitleImportance'])


class PclassFareCabinExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, activate=True):
        self.activate = activate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.activate:
            total = len(X['Pclass'].unique()) + len(X['FareClass'].unique()) + len(
                X['CabinClass'].unique())
            pclass_correction = -total / len(
                X['Pclass'].unique())  # Minus to correct the relative order (-pclass -> better)
            fare_correction = total / len(X['FareClass'].unique())
            cabin_correction = total / len(X['CabinClass'].unique())

            X['PclassFareCabin'] = pclass_correction * X['Pclass'] + fare_correction * X[
                'FareClass'] + cabin_correction * X['CabinClass']
            return X.drop(['Pclass', 'FareClass', 'CabinClass'], axis=1)
        return X
