import pandas as pd

# Load the training data
train_path = os.path.join(extract_path, 'train.csv')
train_df = pd.read_csv(train_path)

# Show basic info and first few rows
train_info = train_df.info()
train_head = train_df.head()

train_df.shape, train_df.columns[:10], train_head

from sklearn.preprocessing import LabelEncoder

# Step 1: Drop Irrelevant Columns
df = train_df.drop(columns=['Id'])

# Step 2: Encode Categorical Variables
# Identify all categorical columns
cat_cols = df.select_dtypes(include='object').columns

# Label Encoding for all categorical variables (as a basic, quick encoding method)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # convert all to string to handle NaNs as 'nan'
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Handle Missing Values
# Drop columns with more than 80% missing data
threshold = 0.8 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# For remaining missing values, fill numerics with median, categoricals (now encoded) with mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Check shape and missing values summary after preprocessing
df.shape, df.isnull().sum().sum()
