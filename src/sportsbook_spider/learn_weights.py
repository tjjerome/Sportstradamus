from sportsbook_spider.stats import statsNBA, statsMLB, statsNHL
import pickle
import importlib.resources as pkg_resources
from sportsbook_spider import data
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, accuracy_score, fbeta_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

mlb = statsMLB()
mlb.load()

market = 'pitcher strikeouts'

X, y = mlb.get_training_matrix(market)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

search = False
model = MLPClassifier(hidden_layer_sizes=(256, 128),
                      batch_size=16, tol=1e-6, max_iter=200)

if search:

    params = {
        'hidden_layer_sizes': [(256, 128), (256, 256), (256, 128, 32)],
        'tol': [1e-6, 1e-8],
        'max_iter': [200, 300, 400, 500],
        # 'batch_size': [16, 32, 64],
        # 'alpha': [0.0001, 0.001, 0.01],
        # 'learning_rate_init': [0.0001, 0.001, 0.01]
    }

    f_5score = make_scorer(fbeta_score, beta=0.5, average='weighted')

    grid_search = GridSearchCV(
        model, params, scoring=f_5score, n_jobs=-1, cv=5)

    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and the best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
else:

    filepath = (pkg_resources.files(data) / "brains.skl")
    with open(filepath, 'rb') as infile:
        models = pickle.load(infile)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print(classification_report(y_test,
          y_pred, target_names=['Over', 'Under']))

    print("Accuracy      " + str(accuracy_score(y_test, y_pred)))

    y_proba = model.predict_proba(X_test_scaled)

    score = 0
    for threshold in np.arange(.6, .95, .05):
        acc = accuracy_score(y_test, (y_proba > threshold).astype(int))
        prec = precision_score(
            y_test, (y_proba > threshold).astype(int), average='weighted')
        newscore = (2*prec*acc)/(prec+acc)
        if newscore > score:
            t = threshold
            score = newscore

    print(f"Threshold: {t}")
    print(classification_report(y_test,
                                (y_proba > t).astype(int), target_names=['Over', 'Under']))
    print("Accuracy      " + str(accuracy_score(y_test,
          (y_proba > t).astype(int))))

    models['MLB'][market] = model

    filepath = (pkg_resources.files(data) / "brains.skl")
    with open(filepath, 'wb') as outfile:
        pickle.dump(models, outfile)
