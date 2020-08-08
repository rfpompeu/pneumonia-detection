from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

def acurracy(x_train, x_test, y_train, y_test):
    print("Treinando!")
    model = SVC()
    x_train = scaler.fit_transform(x_train)
    model.fit(x_train, y_train)
    x_test = scaler.fit_transform(x_test)
    prevision = model.predict(x_test)
    return accuracy_score(y_test, prevision)
