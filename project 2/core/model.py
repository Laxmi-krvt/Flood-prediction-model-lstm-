from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(data):
    X = data[['NORMAL(mm)', 'ACTUAL(mm)']]
    y = data['Flood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, conf_matrix

def plot_data(data, district_name, flood_percentage):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Dates'], data['ACTUAL(mm)'], label='Actual Rainfall (mm)')
    plt.plot(data['Dates'], data['NORMAL(mm)'], label='Normal Rainfall (mm)')
    plt.title(f'Rainfall Data for {district_name} & Flood % = {flood_percentage:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rainfall_plot.png')
    plt.close()
    return 'rainfall_plot.png'

def plot_confusion_matrix(conf_matrix, district_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'])
    plt.title(f'Confusion Matrix for {district_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('conf_matrix_plot.png')
    plt.close()
    return 'conf_matrix_plot.png'


