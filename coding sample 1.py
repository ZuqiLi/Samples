import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, average_precision_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel

from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler


def plot_roc_curves(fpr, tpr, roc_auc, n_classes):
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


lb = LabelBinarizer()
scaler = StandardScaler()

pair = 'mRNA-PROMPT'
params = np.loadtxt("F:\\courses\\Project\\data\\ASAP\\final\\param_matrix_"+pair+"_new.txt",
                    dtype='float', delimiter="\t")

f_f = open("F:\\courses\\Project\\data\\bigWigAverageOverBed\\sum of signal\\sum_CAGER_"+pair+"_forward.out", 'r')
f_r = open("F:\\courses\\Project\\data\\bigWigAverageOverBed\\sum of signal\\sum_CAGER_"+pair+"_reverse.out", 'r')

# generate labels

labels = []
for forward in f_f:
    reverse = f_r.readline()
    v1 = float(forward.strip().split()[1])
    v2 = float(reverse.strip().split()[1])
    if v1 == v2 == 0:
        stat = 0
    else:
        stat = (v1 - v2) / (v1 + v2)
    if stat <= -0.4:
        labels.append(0)
    elif stat >= 0.4:
        labels.append(1)
    else:
        labels.append(2)
'''
plus = []
minus = []
i = 0
for forward in f_f:
    reverse = f_r.readline()
    v1 = float(forward.strip().split()[1])
    v2 = float(reverse.strip().split()[1])
    if i < 572:
        plus.append(v1)
        minus.append(v2)
    else:
        minus.append(v1)
        plus.append(v2)
    i += 1
labels = []
for i in range(1085):
    if plus[i] == minus[i] == 0:
        stat = 0
    else:
        stat = (plus[i] - minus[i]) / (plus[i] + minus[i])
    if stat <= -0.2:
        labels.append(0)
    elif stat >= 0.2:
        labels.append(1)
    else:
        labels.append(2)
'''
f_f.close()
f_r.close()

# reindex labels for mRNA-PROMPT pairs
if pair == 'mRNA-PROMPT':
    ind_p = np.loadtxt("F:\\courses\\Project\\data\\index_mRNA-PROMPT_plus.txt", dtype='int')
    ind_m = np.loadtxt("F:\\courses\\Project\\data\\index_mRNA-PROMPT_minus.txt", dtype='int')

    ind_p = ind_p - 1
    ind_m = ind_m - 1

    tmp = np.zeros((1, len(labels)))
    tmp[0, ind_p] = labels[0:len(ind_p)]
    tmp[0, ind_m] = labels[len(ind_p):]
    labels = tmp
else:
    labels = np.array(labels, ndmin=2)

dataset = np.concatenate((params, labels.T), axis=1)
labels_flat = labels.flatten()
'''
# remove the 3rd label
ind = []
for i in range(params.shape[0]):
    if labels_flat[i] != 2:
        ind.append(i)
params = params[ind, :]
labels_flat = labels_flat[ind]
'''
# summary of classes
class_1 = labels_flat[labels_flat == 0]
class_2 = labels_flat[labels_flat == 1]
class_3 = labels_flat[labels_flat == 2]
print(class_1.shape[0])
print(class_2.shape[0])
print(class_3.shape[0])

# under-sample dataset
params, labels_flat = RandomUnderSampler(random_state=0).fit_resample(params, labels_flat)

# label binarization
lb.fit(labels_flat)

# feature selection
# params = SelectKBest(chi2, k=10).fit_transform(params, labels_flat)
print(params.shape)
# create classifier
clf = RandomForestClassifier(n_estimators=500, random_state=666, min_samples_split=2,
                             min_samples_leaf=1, n_jobs=-1, criterion='entropy',
                             max_depth=None, max_features='sqrt',
                             class_weight=[{0:1,1:1},{0:1,1:1},{0:1,1:0}])
# storing variables
n = 3
if n == 3:
    y_tests = np.array([], dtype=np.int64).reshape(0, n)
    y_preds = np.array([], dtype=np.int64).reshape(0, n)
else:
    y_tests = []
    y_preds = []
y_scores = np.array([], dtype=np.int64).reshape(0, n)
conf_mat = np.zeros([n, n])
selected = []
importance = []

# Cross-Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
for train, test in kf.split(params, labels_flat):
    X_train, X_test, y_train, y_test = params[train, :], params[test, :], labels_flat[train], labels_flat[test]

    # scale features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    # feature selection
    sel = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight={0:1, 1:1, 2:0})
    sel = sel.fit(X_train, y_train)
    model = SelectFromModel(sel, prefit=True, threshold="mean")
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    selected.append(model.get_support(indices=True))
    '''
    # binarize labels
    if n == 3:
        y_train = lb.transform(y_train)
        y_test = lb.transform(y_test)

    # fit model
    clf.fit(X_train, y_train)
    # predict by model
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    if n == 3:
        tmp = np.asarray(y_score)
        y_score = np.column_stack((tmp[0, :, 1], tmp[1, :, 1], tmp[2, :, 1]))
        y_tests = np.concatenate((y_tests, y_test), axis=0)
        y_preds = np.concatenate((y_preds, y_pred), axis=0)
    else:
        y_tests.extend(y_test)
        y_preds.extend(y_pred)
    y_scores = np.concatenate((y_scores, y_score), axis=0)
    importance.append(clf.feature_importances_)

    y_test = lb.inverse_transform(y_test)
    y_pred = lb.inverse_transform(y_pred)
    conf_mat += confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)

# remove the 3rd label
ind = []
for i in range(y_tests.shape[0]):
    if y_tests[i, 2] != 1 and y_preds[i, 2] != 1:
        ind.append(i)
y_tests = y_tests[ind, 0:-1]
y_preds = y_preds[ind, 0:-1]
y_scores = y_scores[ind, 0:-1]

if n == 3:
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_scores.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_tests[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_tests.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plot ROC curves
    plot_roc_curves(fpr, tpr, roc_auc, y_scores.shape[1])
else:
    y_tests = np.array(y_tests, dtype=np.int8)
    fpr, tpr, _ = roc_curve(y_tests, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

ap = average_precision_score(y_tests, y_preds, average='micro')

if n == 3:
    y_tests = lb.inverse_transform(y_tests)
    y_preds = lb.inverse_transform(y_preds)
accuracy = accuracy_score(y_tests, y_preds)

print("Accuracy score: " + str(accuracy))
print("AP score: " + str(ap))
'''
# print(np.mean(importance, axis=0))
selected = np.array(selected, dtype=object)
importance = np.array(importance, dtype=object)
np.savetxt('selected_features.txt', selected, fmt='%s')
np.savetxt('feature_importance.txt', importance, fmt='%s')
'''
