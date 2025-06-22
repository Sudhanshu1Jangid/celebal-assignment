df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['Age'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# Map quality ratings to numerical scale (ordinal encoding)
quality_map = {
    'Ex': 5,  # Excellent
    'Gd': 4,  # Good
    'TA': 3,  # Typical/Average
    'Fa': 2,  # Fair
    'Po': 1,  # Poor
    'nan': 0  # Missing/None
}

# Identify and reverse-map encoded quality columns
quality_columns = ['ExterQual', 'KitchenQual', 'HeatingQC', 'BsmtQual', 'BsmtCond', 'FireplaceQu',
                   'GarageQual', 'GarageCond', 'PoolQC']

# Apply quality map by inverse transforming original string values back from labels
for col in quality_columns:
    if col in df.columns:
        # Decode labels back to original text using stored label encoders
        le = label_encoders[col]
        df[col] = df[col].apply(lambda x: le.inverse_transform([x])[0])  # decode label
        df[col] = df[col].map(quality_map).fillna(0).astype(int)

# Confirm addition of new features
df[['TotalSF', 'Age', 'RemodAge'] + [col for col in quality_columns if col in df.columns]].head()
