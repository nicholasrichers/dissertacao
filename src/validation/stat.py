from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
from itertools import combinations 

from string import ascii_letters
import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns


def plot_corr_matrix(df):


    sns.set(style="white")
    #pyplot.rcParams['axes.titlesize'] = 18
    #pyplot.rcParams['axes.labelsize'] = 12

    # Compute the correlation matrix
    corr = df.corr(method="spearman")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = pyplot.subplots(figsize=(8, 8) )


    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax.set_title('Spearman Corr Matrix', fontsize=18)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.99, center=0.75, 
                square=True, linewidths=.75, cbar_kws={"shrink": .8},  annot=True)


def wilconxon(df, s1, s2): #use era_scores

    stat, p = wilcoxon(df[s1], df[s2])

    # interpret
    alpha = 0.05
    if p < alpha:

        print('\nTesting: {} & {}'.format(s1, s2))
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        print('Different distribution (reject H0)')

    #else: print('Same distribution (fail to reject H0)')
    
    


def friedman(df): #use era_scores
    print("Friedman Test")
    scores = [i for i in df.T.values]
    stat, p = friedmanchisquare(*scores)

    # test results
    print('Testing: {}'.format(list(df.columns)))
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    #interpret
    alpha = 0.05
    if p > alpha: print('Same distributions (fail to reject H0)\n')
    else: print('Different distributions (reject H0)\n')



def diversity_test(df_scores, df_preds, plot=False):
    
    if plot==True: plot_corr_matrix(df_preds)
    
    friedman(df_scores)

    colums = combinations(df_scores.columns, 2)

    for cols in colums:
        wilconxon(df_scores, cols[0], cols[1])
    print('Todas as Combinações de colunas testadas para Wilcoxon signed-rank test')
    
    

    
#diversity_test(df_era_scores_nr, df_preds_nr)