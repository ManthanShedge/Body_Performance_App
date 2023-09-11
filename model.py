import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
# Load the data
def load_data():
    df = pd.read_csv("bodyPerformance.csv")
    return df

def main():
    df = load_data()

    # Data Preprocessing
    X_ = df.loc[:, df.columns != "class"]
    Y = df["class"]

    enc = OneHotEncoder(handle_unknown="ignore")
    val = pd.get_dummies(X_['gender'])
    X_ = pd.concat([X_, val], axis=1)
    X = X_.drop('gender', axis=1)
    Y = LabelEncoder().fit_transform(Y)

    # Train_Test_Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Feature Selection
    X_new = SelectKBest(f_classif, k=10).fit(X_train, Y_train)
    cols_idxs = X_new.get_support(indices=True)
    features_df_new = X.iloc[:, cols_idxs]

    X_train_10f = X_train.iloc[:, cols_idxs]
    X_test_10f = X_test.iloc[:, cols_idxs]

    # LightGBM Model
    model = lgb.LGBMClassifier(learning_rate=0.09, random_state=42,num_leaves= 1200,
    min_data_in_leaf= 90,
    max_depth=10,
    feature_fraction= 0.8,
    bagging_frequency= 5,
    bagging_fraction=0.5)
    model.fit(X_train_10f, Y_train)
    joblib.dump(model, "lightgbm_model.pkl")

if __name__ == "__main__":
    main()

    # Pickle the model

