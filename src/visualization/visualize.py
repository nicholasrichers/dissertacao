
##########
# File: plot_libraries.py
# Description:
#    Coleção de funções para criar gráficos!
##########


from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import numpy as np




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
    f, ax = pyplot.subplots(figsize=(11, 9) )


    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax.set_title('Spearman Corr Matrix', fontsize=18)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.7, center=0.5, 
                square=True, linewidths=.75, cbar_kws={"shrink": .5},  annot=True)




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
                       xaxis = {'title': 'Status'},
                       yaxis = {'title': 'Taxa de Juros'},
                       paper_bgcolor = 'rgb(243, 243, 243)', #também podemos usar RGB na imagem
                       plot_bgcolor = 'rgb(243, 243, 243)')

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)




def plot_era_scores(df, model_names):

  cores = px.colors.cyclical.HSV
  data=[]


  #lista com as eras
  eras = eras = ["era_"+era for era in df.index.astype('str')]

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
                     barmode = 'stack') #Atenção a opção "stack"
  fig = go.Figure(data=data, layout=layout)
  py.iplot(fig)


def highlight_max(s):
    min_cols = ["Feat_exp_max", 'Std_Dev', 'corr_with_example_preds']    
    if s.name in min_cols:
        # Get the smallest values of the column
        is_large = s.nsmallest(1).values
        # Apply style is the current value is among the biggest values
        return ['background-color: yellow' if v in is_large else '' for v in s]
    
    else:
        # Get the largest values of the column
        is_large = s.nlargest(1).values
        # Apply style is the current value is among the smallest values
        return ['background-color: yellow' if v in is_large else '' for v in s]







