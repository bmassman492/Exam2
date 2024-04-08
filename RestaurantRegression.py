import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Excel file into a DataFrame
file_path = r'C:\Users\MassmanBS23\Downloads\Restaurant Revenue.xlsx'
df = pd.read_excel(file_path)

# Extract independent and dependent variables
X = df[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
        'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = df['Monthly_Revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Create a multiple regression formula
formula = "Monthly_Revenue = "
for i, col in enumerate(X.columns):
    if i != 0:
        formula += f" + {model.coef_[i]:.2f} * {col}"
    else:
        formula += f"{model.coef_[i]:.2f} * {col}"
formula += f" + {model.intercept_:.2f}"

print("Multiple Regression Formula:", formula)

print ("Go Brewers!")
