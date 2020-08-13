import numpy as np
from model import build_tuned_model, build_tuned_model_skopt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(features, target, name, model, param_grid, scorer, n_iter=10, cv_folds=5, pipeline=None):
  tuned_model = build_tuned_model(name, model, features, target, param_grid, scorer, n_iter=n_iter, cv_folds=cv_folds, pipeline=pipeline)
  results = tuned_model.results
  best_result = results.query('rank_test_score == 1')
  test_mean = best_result['mean_test_score'].values[0]
  test_std = best_result['std_test_score'].values[0]
  return (tuned_model, name, test_mean, test_std)



def evaluate_model_skopt(features, target, name, model, param_grid, scorer, n_iter=10, cv_folds=5, pipeline=None):
  tuned_model = build_tuned_model_skopt(name, model, features, target, param_grid, scorer, n_iter=n_iter, cv_folds=cv_folds, pipeline=pipeline)
  results = tuned_model.results
  best_result = results.query('rank_test_score == 1')
  test_mean = best_result['mean_test_score'].values[0]
  test_std = best_result['std_test_score'].values[0]
  return (tuned_model, name, test_mean, test_std)



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
  """
  Generate a simple plot of the test and training learning curve.

  Parameters
  ----------
  estimator : object type that implements the "fit" and "predict" methods
      An object of that type which is cloned for each validation.

  title : string
      Title for the chart.

  X : array-like, shape (n_samples, n_features)
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.

  y : array-like, shape (n_samples) or (n_samples, n_features), optional
      Target relative to X for classification or regression;
      None for unsupervised learning.

  ylim : tuple, shape (ymin, ymax), optional
      Defines minimum and maximum yvalues plotted.

  cv : int, cross-validation generator or an iterable, optional
      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

      For integer/None inputs, if ``y`` is binary or multiclass,
      :class:`StratifiedKFold` used. If the estimator is not a classifier
      or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

      Refer :ref:`User Guide <cross_validation>` for the various
      cross-validators that can be used here.

  n_jobs : integer, optional
      Number of jobs to run in parallel (default 1).
  """
  import matplotlib.pyplot as plt
  from sklearn.model_selection import learning_curve
  import numpy as np
  
  plt.figure()
  plt.title(title)
  if ylim is not None:
    plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)

  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  plt.grid()

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

  plt.legend(loc="best");
  return 




def plot_confusion_matrix(y, y_pred):
    print(classification_report(y, y_pred))
    #plt.rcParams['figure.figsize'] = (6, 4)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, ax=ax, fmt='d', cmap='Reds')
    ax.set_title("Confusion Matrix", fontsize=18)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    return



from sklearn.inspection import permutation_importance
from sklearn.base import clone
def plot_feature_permutation(pipeline, model, X_train, y_train, X_test, y_test):
    
    full_pipeline = clone(pipeline)
    full_pipeline.steps.append(['classifier',model])
    full_pipeline.fit(X_train, y_train)

    result = permutation_importance(full_pipeline, X_test, y_test, n_repeats=10, random_state=20, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()







from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix

def plot_calibration_curve(model, X_train, X_test, y_train, y_test, plot_=True):
    """Plot calibration curve for est w/o and with calibration. """
    
    
    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)
    lr_pipe = clone(model.pipeline)
    lr_pipe.steps.append(['lr', lr])
    
    
    # Calibrated with isotonic calibration
    iso_model = CalibratedClassifierCV(base_estimator=model.model, cv=5, method='isotonic')
    iso_pipe = clone(model.pipeline)
    iso_pipe.steps.append([model.name+'_iso', iso_model])
    
    # Calibrated with sigmoid calibration
    sig_model = CalibratedClassifierCV(base_estimator=model.model, cv=5, method='isotonic')
    sig_pipe = clone(model.pipeline)
    sig_pipe.steps.append([model.name+'_sig', sig_model])


    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [#(lr_pipe, 'Logistic'),
                      (model.get_model_pipeline(), model.name)
                      #,(sig_pipe, model.name + ' + Sigmoid')
                      ,(iso_pipe, model.name + ' + Isotonic')]:


        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]   

        
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_train.max())
        clf_score2 = roc_auc_score(y_test, prob_pos)
        clf_score3 = precision_score(y_test, y_pred, average='binary')
        if (plot_!=False or name == model.name + ' + Isotonic' or name == model.name):
          print()
          print("======== %s (test):" % name)    
          print(">Brier Score: %1.3f" % (clf_score))
          print(">ROC AUC: %1.3f" % (clf_score2))
          print(">Precision class 1: %1.3f" % (clf_score3))
          print(classification_report(y_test, y_pred))
          print("Confusion matrix:")
          print(confusion_matrix(y_test, y_pred, labels=[0,1]))
          print()
    
        #print("%s:" % name)
        #print("\tBrier: %1.3f" % (clf_score))
        #print("\tPrecision: %1.3f" % precision_score(y_test, y_pred, average='macro'))
        #print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        #print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.show()
    return (iso_model, name, clf_score2, clf_score3)


def get_cv_scores(model):
  test_score_cols = ['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']
  test_scores = model.results.sort_values('mean_test_score', ascending=False).head(1)[test_score_cols].values[0]
  return test_scores




