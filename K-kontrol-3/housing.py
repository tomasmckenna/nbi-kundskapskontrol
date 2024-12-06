import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# CSV 
file_path = '/home/tomas/GitHub/HandelsAI/KK-kontrol/NBI/K-kontrol-3/housing.csv'  # Replace with your actual file path
output_folder = os.path.dirname(file_path)  # Get the directory of the file
df = pd.read_csv(file_path)

# print("Initial Data Inspection:")
# print(df.head())
# print(df.info())

# nan filter
df.dropna(inplace=True) 

# formatting
numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# unique values for ocean_proximity
unique_ocean_proximity = df['ocean_proximity'].unique()
print("\nUnique values for ocean_proximity:")
print(unique_ocean_proximity)

# Boxplot: median_house_value  vs.  ocean_proximity
plt.figure(figsize=(10, 6))
df.boxplot(column='median_house_value', by='ocean_proximity', grid=False)
plt.title("House Prices by Ocean Proximity")
plt.suptitle("") 
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'house_prices_by_ocean_proximity.png'))
plt.close()

# Correlation
numeric_df = df.select_dtypes(include=['float64', 'int64']) #  only numeric columns
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Correlation Heatmap   
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
plt.close()

# Scatter plots 
scatter_features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
for feature in scatter_features:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[feature], df['median_house_value'], alpha=0.5)
    plt.title(f"House Prices vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Median House Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'house_prices_vs_{feature}.png'))
    plt.close()

# Age Analysis
plt.figure(figsize=(8, 5))
df['housing_median_age'].plot(kind='hist', bins=20, title='Age Distribution of Houses')
plt.xlabel('Housing Median Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'housing_age_distribution.png'))
plt.close()

# income analysis
plt.figure(figsize=(8, 5))
df['median_income'].plot(kind='hist', bins=20, title='Income Distribution')
plt.xlabel('Median Income')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'income_distribution.png'))
plt.close()

# regression analysis
X = df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train LinearRegression  
model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate 
r2 = r2_score(y_test, y_pred)
print(f"\nLinear Regression R-squared score: {r2:.2f}")

# featureimportance
coefficients = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(coefficients)

# Bubbleplot  featureimportance 
coefficients['Bubble_Size'] = abs(coefficients['Importance']) * 10
# Scatterplot
plt.figure(figsize=(20, 12))
plt.scatter(
    x=coefficients['Feature'],
    y=coefficients['Importance'],
    s=coefficients['Bubble_Size'], 
    alpha=0.6,
    edgecolors='w'
)
plt.title("Feature Importance Visualization")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# label bubbles with importance values
for i, row in coefficients.iterrows():
    plt.text(row['Feature'], row['Importance'], f"{row['Importance']:.2f}", ha='center', va='center')

# save  plot
bubble_plot_path = os.path.join(output_folder, 'feature_importance_bubble_plot.png')
plt.savefig(bubble_plot_path)
plt.show()

# featureimportance
coefficients = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_}).sort_values(by='Importance', ascending=False)

# Imost significant factors
top_features = coefficients.sort_values(by='Importance', key=abs, ascending=False).head(3)
print("\nTop Factors Influencing House Prices:")
for idx, row in top_features.iterrows():
    direction = "positive" if row['Importance'] > 0 else "negative"
    print(
        f"- {row['Feature']}: This feature has a {direction} relationship with house prices "
        f"and an importance coefficient of {row['Importance']:.2f}. "
        f"A higher value for '{row['Feature']}' is likely to {'increase' if direction == 'positive' else 'decrease'} house prices."
    )

print(f"\nModel R-squared score: {r2:.2f}")
print("This indicates the proportion of variance in house prices explained by the selected features.")

# Encode 'ocean_proximity' as dummy variables
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Define features (including encoded ocean proximity variables) and target
X = df_encoded.drop(columns=['median_house_value'])
y = df_encoded['median_house_value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
print(f"\nLinear Regression R-squared score: {r2:.2f}")

# Feature Importance
coefficients = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_}).sort_values(by='Importance', ascending=False)

# Separate and rank the influence of 'ocean_proximity' variables
ocean_proximity_features = coefficients[coefficients['Feature'].str.contains('ocean_proximity')]

print("\nInfluence of Ocean Proximity Categories on House Prices:")
for idx, row in ocean_proximity_features.iterrows():
    direction = "positive" if row['Importance'] > 0 else "negative"
    print(
        f"- {row['Feature']}: This category has a {direction} relationship with house prices "
        f"and an importance coefficient of {row['Importance']:.2f}. "
        f"A property in this category is likely to {'increase' if direction == 'positive' else 'decrease'} house prices."
    )

# Overall most important factors
top_features = coefficients.sort_values(by='Importance', key=abs, ascending=False).head(3)
print("\nTop Factors Influencing House Prices:")
for idx, row in top_features.iterrows():
    direction = "positive" if row['Importance'] > 0 else "negative"
    print(
        f"- {row['Feature']}: This feature has a {direction} relationship with house prices "
        f"and an importance coefficient of {row['Importance']:.2f}. "
        f"A higher value for '{row['Feature']}' is likely to {'increase' if direction == 'positive' else 'decrease'} house prices."
    )

print(f"\nModel R-squared score: {r2:.2f}")
print("This indicates the proportion of variance in house prices explained by the selected features.")
