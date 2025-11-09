
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import openml

class DataLoader:
    def __init__(self, dataset_id=45069, test_size=0.2, random_state=42):
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.selected_features = None
    
    def load_data(self):
        print(f"Loading dataset {self.dataset_id} from OpenML...")
        dataset = openml.datasets.get_dataset(self.dataset_id)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )
        
        if y is not None:
            self.df = pd.concat([X, y], axis=1)
        else:
            self.df = X
        
        print(f"Loaded {self.df.shape[0]} records with {self.df.shape[1]} features")
        return self

    def drop_unnecessary_columns(self):
        """Drop optional high-missing/low-information columns if present; ignore if absent."""
        columns_to_drop = ['examide', 'citoglipton', 'weight']
        existing = [c for c in columns_to_drop if c in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)
            print(f"Dropped columns: {existing}")
        else:
            print("No optional columns to drop.")
        return self
    
    def convert_ids_to_categorical(self):
        columns_to_convert = [
            'admission_type_id',
            'discharge_disposition_id',
            'admission_source_id',
            'class',
            'diabetesMed',
        ]
        for col in columns_to_convert:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        return self
    
    def handle_missing_values(self):
        categorical_cols_with_missing = ['payer_code', 'medical_specialty']
        
        for col in categorical_cols_with_missing:
            if col in self.df.columns:
                if pd.api.types.is_categorical_dtype(self.df[col].dtype):
                    if 'Missing' not in self.df[col].cat.categories:
                        self.df[col] = self.df[col].cat.add_categories('Missing')
                self.df[col].fillna('Missing', inplace=True)
        return self
    
    def create_diagnosis_columns(self):
        all_diag_codes = pd.concat([
            self.df['diag_1'],
            self.df['diag_2'],
            self.df['diag_3']
        ]).unique()
        all_diag_codes = [code for code in all_diag_codes if pd.notna(code)]
        
        print(f"Creating {len(all_diag_codes)} diagnosis binary columns...")
        
        diag_cols_dict = {
            f'diag|{code}': (
                (self.df['diag_1'] == code) |
                (self.df['diag_2'] == code) |
                (self.df['diag_3'] == code)
            ).astype(int)
            for code in all_diag_codes
        }
        
        diag_cols_df = pd.DataFrame(diag_cols_dict, index=self.df.index)
        self.df = self.df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
        self.df = pd.concat([self.df, diag_cols_df], axis=1)
        return self
    
    def cap_outliers_iqr(self):
        columns_to_cap = [
            'number_diagnoses',
            'num_lab_procedures',
            'num_medications',
            'num_procedures',
            'time_in_hospital'
        ]
        
        for col in columns_to_cap:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df[col] = self.df[col].apply(
                lambda x: lower_bound if x < lower_bound else (
                    upper_bound if x > upper_bound else x
                )
            )
        
        print(f"Capped outliers for {len(columns_to_cap)} columns")
        return self
    
    def one_hot_encode(self):
        categorical_cols = self.df.select_dtypes(include=['category']).columns.tolist()
        
        cols_to_drop_first = [
            col for col in categorical_cols
            if self.df[col].nunique() == 2
        ]
        
        cols_to_keep_all = [
            col for col in categorical_cols
            if self.df[col].nunique() > 2
        ]
        
        self.df = pd.get_dummies(
            self.df,
            columns=cols_to_drop_first,
            drop_first=True,
            prefix_sep='|'
        )
        self.df = pd.get_dummies(
            self.df,
            columns=cols_to_keep_all,
            drop_first=False,
            prefix_sep='|'
        )
        
        encoded_cols = [
            col for col in self.df.columns
            if self.df[col].dtype != 'float'
        ]
        self.df[encoded_cols] = self.df[encoded_cols].astype(int)
        
        print(f"One-hot encoded: {self.df.shape[1]} columns")
        return self
    
    def select_features(self, correlation_threshold=0.05, importance_threshold=0.005):
        print("Performing feature selection")
        
        # Define target and remove ALL diabetesMed-derived columns from features to prevent leakage
        y = self.df['diabetesMed|Yes']
        diabetes_cols = [c for c in self.df.columns if c.startswith('diabetesMed|')]
        X = self.df.drop(columns=diabetes_cols)
        
        correlation_matrix = self.df.corrwith(self.df['diabetesMed|Yes'])
        correlated_features = correlation_matrix[
            abs(correlation_matrix) > correlation_threshold
        ].index.tolist()
        
        X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train_fs, y_train_fs)
        
        importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        important_features = importances[
            importances > importance_threshold
        ].index.tolist()
        
        self.selected_features = list(set(correlated_features) | set(important_features))
        # Remove any diabetesMed derived one-hot columns to avoid target leakage
        self.selected_features = [f for f in self.selected_features if not f.startswith('diabetesMed|')]
            
        print(f"Selected {len(self.selected_features)} features")
        print(f"  - Correlated features: {len(correlated_features)}")
        print(f"  - Important features: {len(important_features)}")
        
        return self
    
    def prepare_data(self):
        X = self.df[self.selected_features]
        y = self.df['diabetesMed|Yes']
        
        print(f"Shape before dropping duplicates: {X.shape}")
        X = X.drop_duplicates()
        y = y.loc[X.index]
        print(f"Shape after dropping duplicates: {X.shape}")
        
        return X, y
    
    def create_train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)
        
        print(f"\nFinal tensor shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_and_preprocess(self):
        print("="*80)
        print("DIABETES DATA PROCESSING PIPELINE")
        print("="*80)

        # 1) Raw load + core preprocessing
        (self.load_data()
             .drop_unnecessary_columns()
             .convert_ids_to_categorical()
             .handle_missing_values()
             .create_diagnosis_columns()
             .cap_outliers_iqr()
             .one_hot_encode())

        # 2) Define target and remove leaky columns BEFORE split
        if 'diabetesMed|Yes' not in self.df.columns:
            raise ValueError("Target column 'diabetesMed|Yes' not found after encoding.")

        y_full = self.df['diabetesMed|Yes']

        # drop any one-hot columns derived from specific diabetes medications or the target/change
        leaky_prefixes = [
            'diabetesMed', 'change', 'insulin', 'metformin', 'glyburide', 'glipizide',
            'glyburide.metformin', 'nateglinide', 'pioglitazone', 'rosiglitazone',
            'glimepiride', 'repaglinide', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'tolbutamide', 'chlorpropamide', 'sitagliptin'
        ]
        leaky_cols = [c for c in self.df.columns if any(c.startswith(prefix + '|') for prefix in leaky_prefixes)]
        if leaky_cols:
            print(f"Dropping leaky medication-related columns: {len(leaky_cols)}")
        X_full = self.df.drop(columns=leaky_cols + ['diabetesMed|Yes'])

        # 3) Train/Test split BEFORE feature selection
        X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
            X_full, y_full, test_size=self.test_size, random_state=self.random_state, stratify=y_full
        )

        # 4) Feature selection using ONLY training data
        selected = self._select_features_train(X_train_df, y_train_s)
        self.selected_features = selected

        # 5) Apply selected features to both splits
        X_train_df = X_train_df[selected]
        X_test_df = X_test_df[selected]

        # 6) Optionally drop duplicates on train only
        print(f"Shape before dropping duplicates (train): {X_train_df.shape}")
        X_train_df = X_train_df.drop_duplicates()
        y_train_s = y_train_s.loc[X_train_df.index]
        print(f"Shape after  dropping duplicates (train): {X_train_df.shape}")

        # 7) Convert to tensors
        X_train = torch.tensor(X_train_df.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train_s.to_numpy(), dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test_df.to_numpy(), dtype=torch.float32)
        y_test = torch.tensor(y_test_s.to_numpy(), dtype=torch.float32).unsqueeze(1)

        n_features = X_train.shape[1]

        print("="*80)
        print(f"PREPROCESSING COMPLETE - {n_features} features selected (train-only selection, no leakage)")
        print("="*80)

        return X_train, X_test, y_train, y_test, n_features

    def _select_features_train(self, X_train_df: pd.DataFrame, y_train_s: pd.Series,
                                correlation_threshold: float = 0.05,
                                importance_threshold: float = 0.005) -> list:
        """Select features based on correlation and RandomForest importance using ONLY training data."""
        # Drop zero-variance (constant) columns to avoid NaN/inf in correlation
        std = X_train_df.std(numeric_only=True)
        non_constant_cols = std[std > 0].index
        dropped_zero_var = X_train_df.shape[1] - len(non_constant_cols)
        if dropped_zero_var > 0:
            print(f"Dropped {dropped_zero_var} zero-variance columns from train before correlation")
        X_train_nc = X_train_df[non_constant_cols]

        # Correlation with target (absolute), suppress runtime warnings from divide-by-zero internally
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = X_train_nc.corrwith(y_train_s).abs().fillna(0.0).sort_values(ascending=False)
        corr_features = corr[corr > correlation_threshold].index.tolist()

        # RandomForest feature importance
        rf_model = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rf_model.fit(X_train_nc, y_train_s)
        importances = pd.Series(rf_model.feature_importances_, index=X_train_nc.columns)
        imp_features = importances[importances > importance_threshold].index.tolist()

        selected = list(set(corr_features) | set(imp_features))
        print(f"Selected {len(selected)} features (train-only)")
        print(f"  - Correlated: {len(corr_features)}  Important: {len(imp_features)}")
        return selected
