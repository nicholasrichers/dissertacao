from sklearn.metrics import roc_auc_score, make_scorer

def get_scorer():
  scorer = make_scorer(roc_auc_score)
  return scorer


  from sklearn.metrics import brier_score_loss

# calculate brier skill score (BSS)
def brier_skill_score(y_true, y_prob):
    # calculate reference brier score
    pos_prob = 0.17036 ###y.sum/len(X)
    ref_probs = [pos_prob for _ in range(len(y_true))] 
    bs_ref = brier_score_loss(y_true, ref_probs)
    # calculate model brier score
    bs_model = brier_score_loss(y_true, y_prob)
    # calculate skill score
    return 1.0 - (bs_model / bs_ref)

scorer = make_scorer(brier_skill_score, needs_proba=True)
scorer

def get_scorer_brier():
  scorer = make_scorer(brier_skill_score, needs_proba=True)
  return scorer