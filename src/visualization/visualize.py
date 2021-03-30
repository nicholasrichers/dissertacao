
##########
# File: plot_libraries.py
# Description:
#    Coleção de funções para criar gráficos!
##########


from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats




#https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

#print(color.UNDERLINE + 'Hello World !' + color.END)


#configuracao basica dos graficos
def setup_graphics():
  # Plotting options


  sns.set(style='whitegrid')
  pyplot.rcParams['savefig.dpi'] = 75
  pyplot.rcParams['figure.autolayout'] = False
  pyplot.rcParams['figure.figsize'] = 10, 6
  pyplot.rcParams['axes.labelsize'] = 18
  pyplot.rcParams['axes.titlesize'] = 20
  pyplot.rcParams['font.size'] = 16
  pyplot.rcParams['lines.linewidth'] = 2.0
  pyplot.rcParams['lines.markersize'] = 8
  pyplot.rcParams['legend.fontsize'] = 14


#verificar residuo gaussiano
def plot_residuos(X, Xt):

  fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)
  #fig.suptitle('Sharing x per column, y per row')

  X.hist(ax=ax1) 
  Xt.hist(ax=ax2, color='coral') 

  X.plot(kind='kde', ax=ax3)
  Xt.plot(kind='kde', ax=ax4, color='coral')
  
  ax1.set_title("Antes")
  ax2.set_title("Depois")

  #for ax in fig.get_axes(): ax.label_outer()
  pyplot.tight_layout()
  pyplot.show()

  #some statistics
  stat_X = X.groupby(X.index.year).agg(['mean', 'std']).T
  stat_Xt = Xt.groupby(Xt.index.year).agg(['mean', 'std']).T
  return stat_Xt #, stat_Xt



  #verificar residuo gaussiano
def plot_error(X, Xt):

  fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)
  #fig.suptitle('Sharing x per column, y per row')

  X.hist(ax=ax1) 
  Xt.hist(ax=ax2, color='coral') 

  X.plot(kind='kde', ax=ax3)
  Xt.plot(kind='kde', ax=ax4, color='coral')
  
  ax1.set_title("svr")
  ax2.set_title("lgbm")

  pyplot.tight_layout()
  pyplot.show()


#plot graficos EDA
def plot_var(df, y_axis, stack, x_axis1, x_axis2, agg='Mean'):

    #-----------------------------
    sns.set(style='whitegrid')
    pyplot.rcParams['savefig.dpi'] = 75
    pyplot.rcParams['figure.autolayout'] = False
    pyplot.rcParams['figure.figsize'] = 10, 6
    pyplot.rcParams['axes.labelsize'] = 8
    pyplot.rcParams['axes.titlesize'] = 12
    pyplot.rcParams['font.size'] = 8
    pyplot.rcParams['lines.linewidth'] = 2.5
    pyplot.rcParams['lines.markersize'] = 8
    pyplot.rcParams['legend.fontsize'] = 8
    pyplot.rcParams['xtick.labelsize'] = 8
    pyplot.rcParams['ytick.labelsize'] = 8
    #-----------------------------


    #SETUP GRAPHIC SPACE (1X2)
    f, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    cmap = pyplot.cm.inferno


    #DATA
    data = df.groupby([x_axis1, stack])

    #Aggregate by mean
    if agg == 'Mean':
       data = data[y_axis].mean().unstack()

    #Aggregate by sum
    else: data = data[y_axis].sum().unstack()
    

    #PLOT GRAPHIC 1
    data.plot(kind='area', stacked=True,
                        colormap=cmap, grid=False, 
                        legend=False, ax=ax1,
                        figsize=(12,3))
    

    #SETUP GRAPHIC 1
    ax1.set_title(agg +' of ' + y_axis + ' by ' + stack)
    ax1.set_xlabel(x_axis1)
    ax1.set_ylabel(y_axis)

    ax1.legend(bbox_to_anchor=(0.2, 0.9, 0.6, 0.1), 
               loc=10, prop={'size':7},
               ncol=len(list(data.columns)),
               mode="expand", borderaxespad=0.0)
  

#-----------------------------

    #PLOT GRAPHIC 2
    sns.boxplot(y=y_axis, x=x_axis2, data=df, ax=ax2)


    #SETUP GRAPHIC 2
    ax2.set_ylabel('')
    ax2.set_title(y_axis + ' by ' + x_axis2)
    ax2.set_xlabel(x_axis2)

    pyplot.tight_layout()
    return data
    #return df.groupby(x_axis2)[y_axis].describe()
    
    
    
def dist_plot(df, attr, log=False):
#setup graphics
    #pyplot.rcParams['axes.titlesize'] = 14
    #pyplot.rcParams['axes.labelsize'] = 12
    fig, (ax1) = pyplot.subplots(1, 1, figsize=(12,2))
    
    if(log==True):
        df[attr] = np.log1p(df[attr])
        pyplot.title('Log({}) Distribution by Outcome'.format(attr))
    else:
        pyplot.title('{} Distribution by Outcome'.format(attr))

        
    sns.distplot(df.loc[df.y == 1, attr],
                 hist=False, color='steelblue', label='yes');
    
    sns.distplot(df.loc[df.y == 0, attr],
                 hist=False, color='red', label='no');
    
    pyplot.legend()
    pyplot.tight_layout





def plot_corr_matrix(df):

    from string import ascii_letters
    import numpy as np
    import pandas as pd

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




import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px

def plot_feat_cors(df, model_names):
    data=[]
    cores = px.colors.cyclical.HSV
    for cor, model in enumerate(model_names):

        # Gerando gráficos para cada status de pgto.
        trace = go.Box(y = df[model],
                       name = model,
                       #boxpoints="all",
                       marker = {'color': cores[cor]})

        data.append(trace)


    #Criando do layout da imagem
    layout = go.Layout(title = 'Correlação das features com o Target',
                       titlefont = {'family': 'Arial',
                                    'size': 22,
                                    'color': '#7f7f7f'},
                       xaxis = {'title': 'Models'},
                       yaxis = {'title': 'Feat_corr w/ target'},
                       paper_bgcolor = 'rgb(243, 243, 243)', #também podemos usar RGB na imagem
                       plot_bgcolor = 'rgb(243, 243, 243)')

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)




def plot_era_scores(df, model_names, xaxis='era_'):

  cores = px.colors.cyclical.HSV
  data=[]


  #lista com as eras
  eras =  [xaxis+era for era in df.index.astype('str')]

  #note que usamos a lista de status do gráfico de boxplot
  for cor, model in enumerate(model_names):


    # Gerando gráficos para cada modelo
    trace =  go.Bar(y = df[model],
                    x = eras,
                    name = model,
                    marker = {'color': cores[cor]})
    
    data.append(trace)


  layout = go.Layout(title = 'Era Scores por Modelo',
                     xaxis = {'title': 'Eras'},
                     yaxis = {'title': 'Spearman Score'},
                     barmode = 'relative') #Atenção a opção "stack"
  fig = go.Figure(data=data, layout=layout)
  py.iplot(fig)





import datetime
def live_eras_perfomance(models_dict, round_range):

   # import warnings
    import numerapi
    #warnings.filterwarnings('ignore')
    api = numerapi.NumerAPI()
    #last_closed_round = api.get_current_round()-4
    rounds = np.arange(round_range[0], round_range[-1]+1)
    df_models_rounds = pd.DataFrame()

    for model_name, model_alias in models_dict.items():

        df = pd.DataFrame(api.daily_submissions_performances(model_name))
        df_model = pd.DataFrame()

        for round_ in rounds:
            temp_df = pd.DataFrame(api.round_details(int(round_)))
            temp_df['round_number'] = round_
            temp_df['percentile_rank'] = temp_df.groupby(['round_number','date']).rank(pct=True)

            complete_round = df[df['roundNumber']==round_].head(20)
            complete_round['percentile_rank'] = temp_df[temp_df['username'] == model_name].percentile_rank.values


            last_day_round = complete_round.tail(1)
            last_day_round['std_correlation'] = np.std(complete_round['correlation'])
            last_day_round['std_mmc'] = np.std(complete_round['mmc'])
            last_day_round['std_fnc'] = complete_round[np.logical_not(np.isnan(complete_round['fnc']))].fnc.values.std()
            last_day_round['std_correlationWithMetamodel'] = np.std(complete_round['correlationWithMetamodel'])
            last_day_round['std_percentile_rank'] = np.std(complete_round['percentile_rank'])


            df_model = pd.concat([df_model,last_day_round], axis=0)

        
        df_model['model_name'] = model_alias
        df_models_rounds = pd.concat([df_models_rounds,df_model], axis=0)
    

    return df_models_rounds



def plot_live_scores_line(df, column, base="acum", return_data = False):

    data=[]
    cores = px.colors.cyclical.HSV
    cores = px.colors.qualitative.Plotly
    #df = live_eras_perfomance(models_dict, round_range)


    #note que usamos a lista de status do gráfico de boxplot
    for cor, model in enumerate(df['model_name'].unique()):
        
        returns = df[df['model_name']==model][column]*1
        acumulated = [(sum(returns[:i+1])/1)+1 for i in range(len(returns))]

        if base=="acum": 
          line_values = [1]+acumulated
          dates = df['date'].unique()
          dates = np.insert(dates, 0, dates[0] - pd.Timedelta("1 day"))

        else: 
          line_values = returns[:]
          dates = df['date'].unique()[:]


      # Gerando gráficos para cada modelo
        trace =  go.Scatter(y = line_values,
                      x = dates,
                      name = model,
                      line_shape='linear',
                      marker = {'color': cores[cor]}) 
      
        data.append(trace)


    layout = go.Layout(title = 'Live Scores by Model',
                       xaxis = {'title': 'Round Closing Date'},
                       yaxis = {'title': column+' (%)'}
                       #barmode = 'stack') #Atenção a opção "stack"
    )
    fig = go.Figure(data=data, layout=layout)

    if return_data==True: return fig, data
    py.iplot(fig)



def plot_live_scores_bar(df, column, return_data=False):

  cores = px.colors.cyclical.HSV
  cores = px.colors.qualitative.Plotly
  data=[]


  for cor, model in enumerate(df['model_name'].unique()):
        
        returns = df[df['model_name']==model][column]*1
        #acumulated = [(sum(returns[:i+1])/1)+1 for i in range(len(returns))]

        line_values = returns[:]
        dates = df['date'].unique()[:]


      # Gerando gráficos para cada modelo
        trace =  go.Bar(y = line_values,
                        x = dates,
                        name = model,
                        #line_shape='linear',
                        marker = {'color': cores[cor]}) 
      
        data.append(trace)


  layout = go.Layout(title = 'Live Scores by Model',
                       xaxis = {'title': 'Closing Round Date'},
                       yaxis = {'title': column+' (%)'}
                       #barmode = 'stack') #Atenção a opção "stack"
  )
  fig = go.Figure(data=data, layout=layout)

  if return_data==True: return fig, data
  py.iplot(fig)


from plotly.subplots import make_subplots
def plot_live_scores_stacked(df, base, column1, column2):
    fig1, data1 = plot_live_scores_line(df, column1, base=base, return_data=True)
    fig2, data2 = plot_live_scores_bar(df, column2, return_data=True)

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        x_title='Round Closing Date'
                        )

    for dt1 in data1: fig.add_trace(dt1, row=1, col=1)
    for dt2 in data2: fig.add_trace(dt2, row=2, col=1)

    fig['layout']['yaxis']['title']= column1+' (%)'
    fig['layout']['yaxis2']['title']=column2+' (%)'

    fig.update_layout(height=600, width=900, title_text=" Live Eras " + column1 +" & "+ column2)
    fig.show()







def highlight_max_(s):
    min_cols = ["Feat_exp_max", 'Validation_SD', 'corr_with_example_preds', 'corr_with_ex_FN100', 
                'std_percentile_rank', 'std_correlation', 'std_mmc', 'std_fnc'] 


    if s.name in min_cols:
        #print(s.name)
        # Get the smallest values of the column
        is_small = s.nsmallest(1).values
        # Apply style is the current value is among the biggest values
        return ['background-color: yellow' if v in is_small else '' for v in s]
    
    else:
        # Get the largest values of the column
        is_large = s.nlargest(1).values
        # Apply style is the current value is among the smallest values
        return ['background-color: yellow' if v in is_large else '' for v in s]



#compare metrics
def highlight_top3(s):
    # Get 3 largest values of the column
    is_large = s.nlargest(3).values
    # Apply style is the current value is among the 5 biggest values
    return ['background-color: lightgreen' if v in is_large else '' for v in s]

def leaderboard_test_val(metrics_test, metrics_val, model_names, cols):
    
    leaderboard_val = metrics_val.loc[:,model_names].loc[cols, :].T
    leaderboard_test = metrics_test.loc[:,model_names].loc[cols, :].T
    
    test_cols = [str(c)+'_on_Test' for c in cols]
    leaderboard_test.set_axis(test_cols, axis=1, inplace=True)
    leaderboard = pd.concat([leaderboard_val, leaderboard_test], axis=1)
    
    
    l=leaderboard.sort_values(cols[0], ascending=False).astype(float).style.apply(highlight_top3, axis = 0)
    
    return l


def highlight_max(s):

    if s.name=='Feat_exp_max_': return ['background-color: lightgrey' if v in s.nsmallest(1).values else '' for v in s]
    min_cols = ["Feat_exp_max", 'Validation_SD', 'Live_SD','corr_with_example_preds', 'AR(1)', 'AR(1)_sign', 'Preds_Dependence']  

    if s.name in min_cols:
        #print(s.abs())
        # Get the smallest values of the column
        is_small = s.abs().nsmallest(1).values
        # Apply style is the current value is among the biggest values
        return ['background-color: yellow' if v in is_small else '' for v in s]
    
    else:
        # Get the largest values of the column
        is_large = s.nlargest(1).values
        # Apply style is the current value is among the smallest values
        return ['background-color: yellow' if v in is_large else '' for v in s]



def highlight_top10(s):
    
    if s.dtype == 'object':
        return ['background-color: lightblue' if v in [''] else '' for v in s]       
        
    else:
        # Get the largest values of the column
        is_large = s.nlargest(10).values
        # Apply style is the current value is among the smallest values
        return ['background-color: lightblue' if v in is_large else '' for v in s]





VALIDATION_METRIC_INTERVALS = {
    "Validation_Mean": (0.013, 0.028),
    "Validation_Sharpe": (0.53, 1.24),
    "Validation_SD": (0.0303, 0.0168),
    "Feat_exp_max": (0.4, 0.0661),
    "val_mmc_mean": (-0.008, 0.008),
    "corr_plus_mmc_sharpe": (0.41, 1.34),
    "Max_Drawdown": (-0.115, -0.025),
    "Feat_neutral_mean": (0.006, 0.022),
    "corr_with_example_preds": (1, 0.4),

    "Live_Mean": (0.013, 0.028),
    "Live_Sharpe": (0.53, 1.24),
    "Live_SD": (0.0303, 0.0168),
    "Live_mmc_mean": (-0.008, 0.008),
    "Feat_exp_max_": (0.1, -0.1),



}


def color_metric(metric):
    
    #get interval and percentiles
    low, high = VALIDATION_METRIC_INTERVALS[metric.name]
    pct = stats.percentileofscore(np.linspace(low, high, 100), metric.values)
    
    #to min is best
    if high <= low: pct = 100 - pct
        
    if pct > 95: return "lime"
    elif pct > 75: return "darkgreen"
    elif pct > 35: return "black"
    else: return "red"

    


def diagnostic_colors(col):
    
    #get colors list
    colors = col.to_frame().apply(color_metric, axis=1)
    return ['color: '+str(c) for c in colors]
  





