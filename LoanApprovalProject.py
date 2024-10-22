import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Load dataset
data = pd.read_csv("LoanApprovalPrediction.csv")

# Save Loan_ID for later use and drop it for model training
loan_ids = data['Loan_ID']
data.drop(['Loan_ID'], axis=1, inplace=True)

# Handle missing values: fill categorical with mode, numerical with mean
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
object_cols = data.select_dtypes(include='object').columns
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Split data into features (X) and target (Y)
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test, loan_ids_train, loan_ids_test = train_test_split(
    X, Y, loan_ids, test_size=0.4, random_state=1
)

# Reset the index of loan_ids_test to align with test predictions
loan_ids_test = loan_ids_test.reset_index(drop=True)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
rfc.fit(X_train, Y_train)

# Train SVM Classifier
svc = SVC()
svc.fit(X_train, Y_train)

# Predict loan eligibility on the test set for both classifiers
Y_pred_rfc = rfc.predict(X_test)
Y_pred_svc = svc.predict(X_test)

# Create a DataFrame to compare actual vs predicted eligibility
results = pd.DataFrame({
    'Loan_ID': loan_ids_test,
    'Actual Loan Status': Y_test.map({0: 'N', 1: 'Y'}),
    'Predicted Loan Status (Random Forest)': pd.Series(Y_pred_rfc).map({0: 'N', 1: 'Y'}),
    'Predicted Loan Status (SVM)': pd.Series(Y_pred_svc).map({0: 'N', 1: 'Y'})
})

# Show eligible applicants predicted by Random Forest
eligible_applicants_rf = results[results['Predicted Loan Status (Random Forest)'] == 'Y']
print("Eligible Applicants for Loan (Loan_IDs) using Random Forest:")
print(eligible_applicants_rf[['Loan_ID', 'Predicted Loan Status (Random Forest)']])

# Show eligible applicants predicted by SVM
eligible_applicants_svc = results[results['Predicted Loan Status (SVM)'] == 'Y']
print("\nEligible Applicants for Loan (Loan_IDs) using SVM:")
print(eligible_applicants_svc[['Loan_ID', 'Predicted Loan Status (SVM)']])

# Calculate and print accuracy for both classifiers
accuracy_rfc = metrics.accuracy_score(Y_test, Y_pred_rfc)
accuracy_svc = metrics.accuracy_score(Y_test, Y_pred_svc)

print("\nAccuracy of Random Forest Classifier:", accuracy_rfc * 100)
print("Accuracy of SVM Classifier:", accuracy_svc * 100)

# Visualizing categorical variables
plt.figure(figsize=(18, 36))
n_cols = 3
n_rows = (len(object_cols) + n_cols - 1) // n_cols  # Calculate number of rows needed
for index, col in enumerate(object_cols, 1):
    plt.subplot(n_rows, n_cols, index)
    y = data[col].value_counts()
    sns.barplot(x=y.index, y=y.values)
    plt.xticks(rotation=45)
    plt.title(f'Count plot of {col}')
plt.tight_layout()
plt.show()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='BrBG', annot=True, fmt='.2f', linewidths=2)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics

# Load dataset
data = pd.read_csv("LoanApprovalPrediction.csv")

# Backup original data for displaying after prediction
original_data = data.copy()

# Drop Loan_ID column from training data
loan_ids = data['Loan_ID']  # Save Loan_ID for later use
data.drop(['Loan_ID'], axis=1, inplace=True)

# Handle missing values: fill categorical columns with mode, numerical columns with mean
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_encoder.fit_transform(data[col])

# Split the dataset into features (X) and target (Y)
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test, loan_ids_train, loan_ids_test = train_test_split(
    X, Y, loan_ids, test_size=0.4, random_state=1)

# Reset index for loan_ids_test and original_data for proper alignment
loan_ids_test = loan_ids_test.reset_index(drop=True)
original_test_data = original_data.loc[loan_ids_test.index].reset_index(drop=True)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
rfc.fit(X_train, Y_train)

# Predict loan eligibility on the test set
Y_pred_rfc = rfc.predict(X_test)

# Convert Y_test to a Pandas Series to use map()
Y_test_series = pd.Series(Y_test).reset_index(drop=True)

# Add predictions and actual loan status to the original test data for comparison
original_test_data['Predicted Loan Status'] = pd.Series(Y_pred_rfc).map({0: 'N', 1: 'Y'})
original_test_data['Actual Loan Status'] = Y_test_series.map({0: 'N', 1: 'Y'})

# Filter eligible applicants predicted by the model
eligible_applicants = original_test_data[original_test_data['Predicted Loan Status'] == 'Y']

# Show eligible applicants with all their attributes
print("Eligible Applicants for Loan (with full attributes):")
print(eligible_applicants[['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                           'Credit_History', 'Property_Area', 'Predicted Loan Status']])

# Calculate and print accuracy of the Random Forest model
accuracy_rfc = metrics.accuracy_score(Y_test, Y_pred_rfc)
print("\nAccuracy of Random Forest Classifier:", accuracy_rfc * 100)
