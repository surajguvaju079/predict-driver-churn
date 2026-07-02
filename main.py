import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)

df = pd.read_csv("churn.csv");

X = df[[
    "loads_completed",
    "last_login_days",
    "rating"
]]

y = df["churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)


predictions = model.predict(X_test)
print(predictions)

print(accuracy_score(y_test,predictions));
print(confusion_matrix(y_test,predictions));
print(classification_report(y_test,predictions))
