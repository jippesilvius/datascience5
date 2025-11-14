#!/usr/bin/env python3


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from LogisticRegression import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

class MethodRecommender:
    def __init__(self, df=None, target=None,):
        """
        df: pandas DataFrame
        target: name of the target column
        """
        self.df = df
        self.target = target

        self.target_type = None
        self.num_classes = None
        self.data_size = None
        self.linear = None
        self.interpretability = None


    # -------------------------------------------------------
    # 1. AUTO ANALYSIS FROM DATAFRAME
    # -------------------------------------------------------
    def analyze_dataframe(self):
        if self.df is None or self.target is None:
            return  # nothing to analyze

        y = self.df[self.target]

        # Target type
        if pd.api.types.is_numeric_dtype(y):
            self.target_type = "num"
        else:
            self.target_type = "cat"

        # Number of classes for classification
        if self.target_type == "cat":
            self.num_classes = y.nunique()

        # Dataset size
        n_rows = len(self.df)
        if n_rows < 500:
            self.data_size = "small"
        elif n_rows < 5000:
            self.data_size = "medium"
        else:
            self.data_size = "large"

        # Linearity heuristic (for regression only)
        if self.target_type == "num":
            numeric_features = self.df.drop(columns=[self.target]).select_dtypes(include=np.number)
            if numeric_features.empty:
                self.linear = "no"  # no numeric features to check
            else:
                corr = abs(numeric_features.corrwith(y)).mean()
                self.linear = "yes" if corr > 0.4 else "no"

        # Warn about missing values
        missing_cols = self.df.columns[self.df.isna().any()].tolist()
        if missing_cols:
            print(f"⚠️ Warning: The following columns have NA values: {missing_cols}")

    # -------------------------------------------------------
    # 2. QUESTIONNAIRE FOR MISSING INFORMATION
    # -------------------------------------------------------
    def ask_options(self, question, options):
        print("\n" + question)
        for i, opt in enumerate(options, 1):
            print(f"{i}. {opt}")

        choice = None
        while choice not in range(1, len(options) + 1):
            try:
                choice = int(input("Choose a number: "))
            except ValueError:
                continue

        return options[choice - 1]

    def ask_questions(self):
        print("\n--- Answer the following questions ---")

        if self.target_type is None:
            self.target_type = self.ask_options(
                "What is the target type?",
                ["num (regression)", "cat (classification)"]
            ).split()[0]

        if self.target_type == "cat" and self.num_classes is None:
            self.num_classes = int(self.ask_options(
                "How many classes does the target have?",
                ["2", "3", "4", "5", "more than 5"]
            ).split()[0])

        if self.data_size is None:
            self.data_size = self.ask_options(
                "How large is your dataset?",
                ["small", "medium", "large"]
            )

        if self.linear is None:
            self.linear = self.ask_options(
                "Do you believe the relationship is linear?",
                ["yes", "no"]
            )

        if self.interpretability is None:
            self.interpretability = self.ask_options(
                "Do you need high interpretability?",
                ["yes", "no"]
            )

    # -------------------------------------------------------
    # 3. RECOMMENDATION LOGIC
    # -------------------------------------------------------
    def recommend(self):
        # Regression
        if self.target_type == "num":
            if self.linear == "yes":
                return "Linear Regression"
            else:
                return "SVM (regression) or Ensemble Methods"

        # Classification
        if self.target_type == "cat":
            if self.num_classes > 2:
                if self.linear == "yes":
                    return "Multiclass Logistic Regression or LDA"
                else:
                    return "Multiclass SVM or Ensemble Learning Methods"

            # binary
            if self.linear == "yes":
                if self.interpretability == "yes":
                    return "Logistic Regression"
                else:
                    return "LDA or SVM"
            else:
                if self.data_size == "small":
                    return "Naive Bayes or ID3"
                elif self.data_size == "large":
                    return "SVM or Ensemble Learning"
                else:
                    return "KNN or QDA"

        return "No recommendation available."

    def convert_columns(self, df):
        new_df = df.copy()

        for col in new_df.columns:
            # Try converting to numeric (coerce errors to NaN)
            numeric_col = pd.to_numeric(new_df[col], errors='coerce')

            # If after conversion, all non-NaN values are numeric, keep as float
            if numeric_col.notna().all():
                new_df[col] = numeric_col.astype(float)
            else:
                # Otherwise, convert everything to string
                new_df[col] = new_df[col].astype(str)

        return new_df

    def encode_column(self, df, columns_to_encode=None):
        """
        Encode specified categorical column(s) using LabelEncoder.
        :param df: pandas DataFrame
        :param columns_to_encode: list or str of columns to encode; if None, encode all object columns except target
        :return: df_encoded, encoders dictionary
        """
        df_encoded = df.copy()
        encoders = {}

        # Determine columns to encode
        if columns_to_encode is None:
            # Encode all object columns except the target
            columns_to_encode = df_encoded.select_dtypes(include=['object']).columns.tolist()
            if self.target:
                columns_to_encode = [col for col in columns_to_encode if col != self.target]
        elif isinstance(columns_to_encode, str):
            columns_to_encode = [columns_to_encode]

        # Encode each column
        for col in columns_to_encode:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder

        return df_encoded, encoders

    def look_at_encode(self, encoders, column_to_see):
        """
        View the mapping of values for a specific encoded column.
        :param encoders: dictionary of LabelEncoders
        :param column_to_see: column name
        :return: dictionary of value -> encoded integer
        """
        if column_to_see not in encoders:
            raise ValueError(f"No encoder found for column: {column_to_see}")

        encoder_look = encoders[column_to_see]
        dict_enc = dict(zip(encoder_look.classes_, range(len(encoder_look.classes_))))
        print(f"Encoding for '{column_to_see}': {dict_enc}")
        return dict_enc

    def return_na_columns(self, data):
        df = data.isna().sum()[data.isna().sum() > 0]

        return df

    # -------------------------------------------------------
    # 4. MAIN WORKFLOW
    # -------------------------------------------------------
    def run(self):
        self.df = self.convert_columns(self.df)
        self.analyze_dataframe()  # auto-analysis if df provided
        self.ask_questions()  # ask for missing details
        method = self.recommend()  # final recommendation

        enocded_df, encoders = encode_column(self.df, columns_to_encode=None)
        look_at_encode(self, encoders, column_to_see)

        print("\n------------------------------------")
        print(f" Recommended Method: {method}")
        print("------------------------------------")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class Imputate:
    def __init__(self, data: pd.DataFrame, target: str, model):
        """
        Parameters:
            data : pd.DataFrame
                The full dataframe.
            target : str
                Column name to impute.
            model : object
                A supervised model instance with fit(X, y) and predict(X)
        """
        self.data = data.copy()
        self.target = target
        self.model = model

        # Placeholders for intermediate steps
        self.df_train = None
        self.df_missing = None
        self.X_train_numeric = None
        self.X_missing_numeric = None
        self.X_train_values = None
        self.X_missing_values = None
        self.y_train = None
        self.y_missing_pred = None

    # -------------------------------
    # Step 1: Split data
    # -------------------------------
    def split_data(self):
        self.df_train = self.data[self.data[self.target].notna()]
        self.df_missing = self.data[self.data[self.target].isna()]

        if self.df_missing.empty:
            print(f"No missing values in '{self.target}'.")
        return self.df_train, self.df_missing

    # -------------------------------
    # Step 2: Process target
    # -------------------------------
    def process_target(self):
        y_train = self.df_train[self.target]

        # Automatically convert categorical/string target to numeric
        if pd.api.types.is_categorical_dtype(y_train):
            y_train = y_train.cat.codes
        elif y_train.dtype == object:
            y_train = pd.factorize(y_train)[0]

        self.y_train = y_train.values.reshape(-1,1).astype(float)
        return self.y_train

    # -------------------------------
    # Step 3: Process predictors
    # -------------------------------
    def process_predictors(self):
        X_train = self.df_train.drop(columns=[self.target])
        X_missing = self.df_missing.drop(columns=[self.target])

        # One-hot encode categorical predictors
        X_train_numeric = pd.get_dummies(X_train, drop_first=True)
        X_missing_numeric = pd.get_dummies(X_missing, drop_first=True)

        # Align missing rows columns with training
        X_missing_numeric = X_missing_numeric.reindex(columns=X_train_numeric.columns, fill_value=0)

        # Convert to numpy arrays
        self.X_train_numeric = X_train_numeric
        self.X_missing_numeric = X_missing_numeric
        self.X_train_values = X_train_numeric.values.astype(float)
        self.X_missing_values = X_missing_numeric.values.astype(float)

        return self.X_train_values, self.X_missing_values

    # -------------------------------
    # Step 4: Evaluate model (optional)
    # -------------------------------
    def evaluate(self, test_size=0.2, random_state=42):
        if self.df_train is None or self.X_train_values is None or self.y_train is None:
            raise ValueError("Run split_data, process_target, and process_predictors first.")

        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X_train_values, self.y_train, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_tr, y_tr)
        y_pred_test = self.model.predict(X_te)

        # Choose metric based on target type
        if len(np.unique(self.y_train)) <= 2:
            acc = accuracy_score(y_te, y_pred_test)
            print(f"Validation Accuracy for '{self.target}': {acc:.3f}")
            return acc
        else:
            rmse = np.sqrt(mean_squared_error(y_te, y_pred_test))
            print(f"Validation RMSE for '{self.target}': {rmse:.2f}")
            return rmse

    # -------------------------------
    # Step 5: Impute missing values
    # -------------------------------
    def impute(self):
        if self.X_train_values is None or self.y_train is None:
            raise ValueError("Run split_data, process_target, and process_predictors first.")

        self.model.fit(self.X_train_values, self.y_train)
        self.y_missing_pred = self.model.predict(self.X_missing_values)

        # Fill missing values
        self.data.loc[self.data[self.target].isna(), self.target] = self.y_missing_pred
        print(f"Imputed {len(self.df_missing)} missing values in '{self.target}'.")
        return self.data


    def run_all(self):
        self.split_data()
        self.process_target()
        self.process_predictors()
        self.evaluate()  # optional, prints accuracy
        data = self.impute()  # fills missing values
        return data