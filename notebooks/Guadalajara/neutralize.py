

########################################
######################################## Neutralize (tips and tricks)


from sklearn.preprocessing import MinMaxScaler
def _neutralize(df, columns, by, proportion=1.0):
    scores = df[columns]
    exposures = df[by].values
    scores = scores - proportion * exposures.dot(numpy.linalg.pinv(exposures).dot(scores))
    return scores / scores.std()
def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)
def normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]



df2["preds"] = xgb_preds
df2["preds_neutralized"] = df2.groupby("era").apply(
    lambda x: normalize_and_neutralize(x, ["preds"], features, 0.5) # neutralize by 50% within each era
)
scaler = MinMaxScaler()
df2["preds_neutralized"] = scaler.fit_transform(df2[["preds_neutralized"]]) # transform back to 0-1



corr_list2 = []
for feature in features:
    corr_list2.append(numpy.corrcoef(df2[feature], df2["preds_neutralized"])[0,1])
corr_series2 = pandas.Series(corr_list2, index=features)
corr_series2.describe()


unbalanced_scores_per_era = df2.groupby("era").apply(lambda d: np.corrcoef(d["preds"], d[target])[0,1])
balanced_scores_per_era = df2.groupby("era").apply(lambda d: np.corrcoef(d["preds_neutralized"], d[target])[0,1])

print(f"score for high feature exposure: {unbalanced_scores_per_era.mean()}")
print(f"score for balanced feature expo: {balanced_scores_per_era.mean()}")

print(f"std for high feature exposure: {unbalanced_scores_per_era.std()}")
print(f"std for balanced feature expo: {balanced_scores_per_era.std()}")

print(f"sharpe for high feature exposure: {unbalanced_scores_per_era.mean()/unbalanced_scores_per_era.std()}")
print(f"sharpe for balanced feature expo: {balanced_scores_per_era.mean()/balanced_scores_per_era.std()}")



balanced_scores_per_era.describe()
unbalanced_scores_per_era.describe()

