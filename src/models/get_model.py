
from joblib import load
def get_trained_model(path, pipe=True):


  model = load(path)
  if pipe==False:
    params_dict = model.get_params()

  else:
    params_dict = model.model.get_params()


  return model, params_dict