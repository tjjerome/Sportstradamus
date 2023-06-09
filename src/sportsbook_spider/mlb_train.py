from sportsbook_spider.stats import statsMLB
import pickle
import importlib.resources as pkg_resources
from sportsbook_spider import data
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, accuracy_score, roc_auc_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from skopt import BayesSearchCV
from matplotlib import pyplot as plt

mlb = statsMLB()
mlb.load()
mlb.update()

league = 'MLB'
markets = ['total bases', 'pitcher strikeouts', 'runs allowed',
           'hits', 'runs', 'rbi', 'singles', 'hits+runs+rbi']
for market in markets:

    X, y = mlb.get_training_matrix(market)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if len(X_train) < 100:
        continue

    scaler = MaxAbsScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = np.ravel(y_train.to_numpy())
    y_test = np.ravel(y_test.to_numpy())

    # model = MLPClassifier(hidden_layer_sizes=(472, 112, 64),  # (472,112,64), (448,96,64), (416, 352, 56)
    #                       batch_size=128, tol=3.86e-5, max_iter=1000, alpha=0.001375,
    #                       beta_1=.96, beta_2=.958, learning_rate_init=.005685,
    #                       solver='adam', early_stopping=True, n_iter_no_change=300)  # .553r, .40a, .54u, .64o

    model = GradientBoostingClassifier(
        loss='log_loss', learning_rate=0.1, n_estimators=500, max_depth=3, max_features='log2')  # .547r, .39a, .61u, .55o

    # model = RandomForestClassifier(
    #     criterion='entropy', n_estimators=500, max_features=0.25, max_depth=26)  # .564r, .40a, .58u, .60o

    model = CalibratedClassifierCV(
        model, cv=50, n_jobs=-1, method='isotonic')
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)

    thresholds = np.arange(0.54, 0.6, 0.001)
    acc = np.zeros_like(thresholds)
    preco = np.zeros_like(thresholds)
    precu = np.zeros_like(thresholds)
    prec = np.zeros_like(thresholds)
    roc = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        y_pred = (y_proba > threshold).astype(int)
        preco[i] = precision_score((y_test == 1).astype(int), y_pred[:, 1])
        precu[i] = precision_score(
            (y_test == -1).astype(int), y_pred[:, 0])
        y_pred = y_pred[:, 1]-y_pred[:, 0]
        acc[i] = accuracy_score(y_test, y_pred)
        prec[i] = precision_score(y_test, y_pred, average='weighted')
        roc[i] = roc_auc_score(
            y_test, y_pred, average='weighted')

    i = np.argmax(prec)
    t = thresholds[i]

    # y_pred = (y_proba > t).astype(int)
    # y_pred = y_pred[:, 1]-y_pred[:, 0]
    # print(f"Threshold: {t}")
    # print(classification_report(y_test, y_pred))

    # fop, mpv = calibration_curve(y_test, y_proba[:, 1], n_bins=10)
    # bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)

    # f, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(thresholds, acc, color='r', label='Accuracy')
    # ax1.plot(thresholds, preco, color='b', label='Precision - Over')
    # ax1.plot(thresholds, precu, color='c', label='Precision - Under')
    # ax1.plot(thresholds, roc, color='g', label='ROC AUC')
    # ax1.axvline(t, linestyle='--', label="T=%.3f" % t)
    # ax1.set_title("Validation Scores")
    # ax1.legend()

    # # plot perfectly calibrated
    # ax2.plot([0, 1], [0, 1], linestyle='--')
    # # plot model reliability
    # ax2.plot(mpv, fop, marker='.')
    # ax2.set_title("Probability Calibration, Breier=%.3f" % bs)
    # plt.show()

    filedict = {'model': model,
                'scaler': scaler,
                'threshold': t,
                'edges': [np.floor(i*2)/2 for i in mlb.edges][:-1],
                'stats': {
                    'Accuracy': acc[i],
                    'Precision_Over': preco[i],
                    'Precision_Under': precu[i],
                    'ROC_AUC': roc[i]
                }}

filename = "_".join([league, market]).replace(" ", "-")+'.skl'

filepath = (pkg_resources.files(data) / filename)
with open(filepath, 'wb') as outfile:
    pickle.dump(filedict, outfile, -1)
    del filedict
    del model
    del scaler
