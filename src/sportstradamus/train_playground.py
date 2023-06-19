from sportstradamus.stats import StatsNBA, StatsMLB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from matplotlib import pyplot as plt
import pandas as pd

# nba = StatsNBA()
# nba.load()
# nba.update()

league = 'NBA'
markets = ['PRA']
for market in markets:

    # X, y = nba.get_training_matrix(market)
    X = pd.read_csv('src/sportstradamus/data/X.csv', index_col=0)
    y = pd.read_csv('src/sportstradamus/data/y.csv', index_col=0)

    X = X.drop(columns=['Game 6', 'Game 7', 'Game 8', 'Game 9'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25)

    if len(X_train) < 1000:
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
        loss='log_loss', learning_rate=0.1, n_estimators=500, max_depth=10, max_features='sqrt')  # .547r, .39a, .61u, .55o

    # model = RandomForestClassifier(
    #     criterion='entropy', n_estimators=500, max_features=0.25, max_depth=26)  # .564r, .40a, .58u, .60o

    # model.fit(X_train, y_train)

    model = CalibratedClassifierCV(
        model, cv=10, n_jobs=-1, method='sigmoid')
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)

    thresholds = np.arange(0.54, 0.58, 0.001)
    acc = np.zeros_like(thresholds)
    null = np.zeros_like(thresholds)
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
        acc[i] = accuracy_score(
            y_test[np.abs(y_pred) > 0], y_pred[np.abs(y_pred) > 0])
        prec[i] = precision_score(y_test, y_pred, average='weighted')
        roc[i] = roc_auc_score(
            y_test, y_pred, average='weighted')

    i = np.argmax(prec)
    j = np.argmax(acc)
    t1 = thresholds[i]
    t2 = thresholds[j]

    # y_pred = (y_proba > t).astype(int)
    # y_pred = y_pred[:, 1]-y_pred[:, 0]
    # print(f"Threshold: {t}")
    # print(classification_report(y_test, y_pred))

    fop, mpv = calibration_curve(y_test, y_proba[:, 1], n_bins=50)
    bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(thresholds, acc, color='r', label='Accuracy')
    ax1.plot(thresholds, preco, color='b', label='Precision - Over')
    ax1.plot(thresholds, precu, color='c', label='Precision - Under')
    ax1.plot(thresholds, roc, color='g', label='ROC AUC')
    ax1.axvline(t1, linestyle='--', color='b', label="T1=%.3f" % t1)
    ax1.axvline(t2, linestyle='--', color='c', label="T2=%.3f" % t2)
    ax1.set_title("Validation Scores")
    ax1.legend()

    # plot perfectly calibrated
    ax2.plot([0.4, 0.6], [0.4, 0.6], linestyle='--')
    # plot model reliability
    ax2.plot(mpv, fop, marker='.')
    ax2.set_title("Probability Calibration, Breier=%.3f" % bs)
    plt.show()

    model
