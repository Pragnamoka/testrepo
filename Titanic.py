import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = sns.load_dataset("titanic")
print(df.head())
print(df.info)
print(df.describe(include="object").T)
print(df['survived'].value_counts(normalize=True))
sns.countplot(x='survived', data=df)
plt.title("survival Distribution")
plt.show()
sns.countplot(data=df, x='pclass', hue='survived')
plt.title("Survived by passenger class")
plt.show()

sns.countplot(data=df, x='sex', hue='survived')
plt.title("Survival by gender")
plt.show()

sns.histplot(data=df, x='age', hue='survived', kde=True, bins=30)
plt.title("Age distribution by Survival")
plt.show()

sns.boxplot(data=df, x='pclass', y='fare', hue='survived')
plt.title("Fare by passenger Class and Survival")
plt.show()

df['age'] = df['age'].fillna(df["age"].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)
X = df[['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male', 'embarked_Q', 'embarked_S']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Data Preprocessing complete. Ready for model training!")

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

df["FamilySize"] = df['sibsp'] + df['parch']

X = df[['age', 'FamilySize', 'pclass', 'sibsp', 'parch','fare', 'sex_male', 'embarked_Q','embarked_S']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Updated Model Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print("\n updated classification report")
print(classification_report(y_test, y_pred))
print("\n updated confusion matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, 'titanic_survival_model.pkl')
print("Model saved as 'titanic_survival_model.pkl'")

loaded_model = joblib.load('titanic_survival_model.pkl')
loaded_model.predict(X_test)



