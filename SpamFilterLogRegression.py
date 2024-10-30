import os
import emailReadUtility
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

os.getcwd()
DATA_DIR = 'trec07p/data/'
LABELS_FILE = 'trec07p/full/index'
TESTING_SET_RATIO = 0.2

labels = {}
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

def read_email_files():
    X = []
    y = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = emailReadUtility.extract_email_text(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y

X, y = read_email_files()

pd.DataFrame(X).head()
pd.DataFrame(y).head()

X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, range(len(y)),
    train_size=TESTING_SET_RATIO, random_state=2)

vectorizer = TfidfVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

cl_mlp = MLPClassifier(random_state=101, max_iter=300)
cl_mlp.fit(X_train_vector, y_train)
y_pred = cl_mlp.predict(X_test_vector)

cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
