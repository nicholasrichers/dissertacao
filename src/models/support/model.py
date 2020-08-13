import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

def build_tuned_model(name, base_model, X_train, y_train, hparams, scorer, n_iter, cv_folds, pipeline):
  from time import time
  start = time()
  print('==> Starting {}-fold cross validation for {} model, {} examples'.format(str(cv_folds), name, len(X_train)))
  model = TunedModel(hparams, name=name, model=base_model, n_iter=n_iter, cv_folds=cv_folds, pipeline=pipeline)
  model.train(X_train, y_train, scorer, n_iter, cv_folds, pipeline)
  elapsed = time() - start
  print("==> Elapsed seconds: {:.3f}".format(elapsed))
  
  print('Best {} model: {}'.format(model.name, model.model))
  print('Best {} score (val): {:.3f}'.format(
    model.name,
    model.results.sort_values('mean_test_score', ascending=False).head(1).mean_test_score.values[0]))

  return model



def build_tuned_model_skopt(name, base_model, X_train, y_train, hparams, scorer, n_iter, cv_folds, pipeline):
  from time import time
  start = time()
  print('==> Starting {}-fold cross validation for {} model, {} examples'.format(str(cv_folds), name, len(X_train)))
  model = TunedModel_Skopt(hparams, name=name, model=base_model, n_iter=n_iter, cv_folds=cv_folds, pipeline=pipeline)
  model.train(X_train, y_train, scorer, n_iter, cv_folds, pipeline)
  elapsed = time() - start
  print("==> Elapsed seconds: {:.3f}".format(elapsed))
  
  print('Best {} model: {}'.format(model.name, model.model))
  print('Best {} score (val): {:.3f}'.format(
    model.name,
    model.results.sort_values('mean_test_score', ascending=False).head(1).mean_test_score.values[0]))

  return model


# ============================================================================================================
# Model
# ============================================================================================================
class Model(object):
  def __init__(self, name, model, n_iter, cv_folds, pipeline):
    self.name = name
    self.model = model
    self.pipeline = pipeline
    self.n_iter = n_iter
    self.cv_folds = cv_folds

  def train(self, X, y):
    """ Fits the model and builds the full pipeline """
    if self.pipeline is None:
      X_transformed = X
      self.model_pipeline = make_pipeline(self.model)
    else:
      X_transformed = self.pipeline.fit_transform(X)
      self.model_pipeline = make_pipeline(self.pipeline, self.model)
      
    self.model.fit(X_transformed, y)
    
    return self


  def predict_calibrated(self, X_test, X, y):
    """ calibrated predictions """

    cal_model = CalibratedClassifierCV(base_estimator=self.model, cv=5, method='isotonic')
    cal_pipe = clone(self.pipeline)
    cal_pipe.steps.append([self.name+'_iso', cal_model])

    cal_pipe.fit(X, y)
    preds = cal_pipe.predict(X_test)
    probs = cal_pipe.predict_proba(X_test)  
    
    return preds, probs



  def predict(self, X):
    """ Fits the model and builds the full pipeline 
    TODO: Make sure the model was fitted
    """
    # if self.pipeline is None:
    #   X_transformed = X
    # else:
    #   X_transformed = self.pipeline.fit_transform(X)
      
    preds = self.model_pipeline.predict(X)
    
    return preds

  def get_model_pipeline(self):
    """ Useful for cross validation to refit the pipeline on every round """
    full_pipeline = clone(self.pipeline)
    full_pipeline.steps.append((self.name, self.model))
    return full_pipeline

  def score(self, X, y, scorer):
    """ Scores the model using the scorer
  
    Postcondititions: 
      - score should not be 0 
      - model.predictions should have elements
    """
    score = 0
    
    if self.pipeline is None:
      model.predictions = self.model.predict(X)
      score = scorer(self.model, X, y)
    else:
      model_predictions = self.model_pipeline.predict(X)
      score = scorer(self.model_pipeline, X, y)

    return score

  def save(self, file_path):
    from joblib import dump
    dump(self, file_path)
  
  @staticmethod
  def load(file_path):
    from joblib import load
    model = load(file_path)
    return model
    
      
  def score_cv(self, X, y, scorer):
    """ Scores the model using the scorer
  
    Postcondititions: 
      - score should not be 0 
      - model.predictions should have elements
    """
    from sklearn.model_selection import cross_val_score
    
    score = 0
    
    if self.pipeline is None:
      score = cross_val_score(self.model, X, y, scoring=scorer, cv=self.cv_folds, n_jobs=-1)
    else:
      score = scorer(self.model_pipeline, X, y)
      
    return (score.mean(), score.std())

# ============================================================================================================
# TunedModel
# ============================================================================================================
class TunedModel(Model):
  """ A class used to optimize the hyperparameters for a machine learning algorithm

  Parameters
  ----------
  name : string
      The name of a model
      
  param_distributions : dict
      A dict of (parameter, values) pairs to optimize
      
  pipeline : object
      A pipeline to apply to the data before fitting the model
  """

  def __init__(self, param_distributions, **kwargs):
      Model.__init__(self, **kwargs)
      self.param_distributions = param_distributions

  def train(self, X, y, scorer, n_iter, cv_folds, pipeline):
      """ Tunes a model using the parameter grid that this class was initialized with.
      
      Parameters
      ----------
      X : array-like, matrix
          Input data
          
      y : array-like
          Targets for input data
          
      cv_folds : int, optional, default: 5
          The number of cross-validation folds to use in the optimization process.
      """
      if not self.pipeline:
        # Setup
        grid_search = RandomizedSearchCV(
            self.model, 
            self.param_distributions, 
            cv=cv_folds,
            n_iter=n_iter,
            scoring=scorer, 
            return_train_score=True, 
            n_jobs=-1, verbose=5)
        
        # Run it
        grid_search.fit(X, y)
        
        # Save the model
        self.model = grid_search.best_estimator_
      else:
        # Setup
        grid_search = RandomizedSearchCV(
            self.get_model_pipeline(), 
            self.param_distributions, 
            cv=cv_folds,
            n_iter=n_iter,
            scoring=scorer, 
            return_train_score=True, 
            n_jobs=-1, verbose=5)
        
        # Run it
        grid_search.fit(X, y)
        
        # Save the model and pipeline
        self.model = grid_search.best_estimator_.steps[-1][1]
        self.pipeline = Pipeline(grid_search.best_estimator_.steps[:-1])

      self.results = pd.DataFrame(grid_search.cv_results_)


# ============================================================================================================
# TunedModel
# ============================================================================================================
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
class TunedModel_Skopt(Model):
  """ A class used to optimize the hyperparameters for a machine learning algorithm

  Parameters
  ----------
  name : string
      The name of a model
      
  param_distributions : dict
      A dict of (parameter, values) pairs to optimize
      
  pipeline : object
      A pipeline to apply to the data before fitting the model
  """

  def __init__(self, param_distributions, **kwargs):
      Model.__init__(self, **kwargs)
      self.param_distributions = param_distributions

  def train(self, X, y, scorer, n_iter, cv_folds, pipeline):
      """ Tunes a model using the parameter grid that this class was initialized with.
      
      Parameters
      ----------
      X : array-like, matrix
          Input data
          
      y : array-like
          Targets for input data
          
      cv_folds : int, optional, default: 5
          The number of cross-validation folds to use in the optimization process.
      """
      if not self.pipeline:
        # Setup
        grid_search = BayesSearchCV(
            self.model, 
            self.param_distributions, 
            cv=cv_folds,
            n_iter=n_iter,
            scoring=scorer, 
            return_train_score=True, 
            n_jobs=-1)
        
        # Run it
        grid_search.fit(X, y)
        
        # Save the model
        self.model = grid_search.best_estimator_
      else:
        # Setup
        grid_search = BayesSearchCV(
            self.get_model_pipeline(), 
            self.param_distributions, 
            cv=cv_folds,
            n_iter=n_iter,
            scoring=scorer, 
            return_train_score=True,
            n_jobs=-1)
        
        # Run it
        grid_search.fit(X, y)
        
        # Save the model and pipeline
        self.model = grid_search.best_estimator_.steps[-1][1]
        self.pipeline = Pipeline(grid_search.best_estimator_.steps[:-1])

      self.results = pd.DataFrame(grid_search.cv_results_)


