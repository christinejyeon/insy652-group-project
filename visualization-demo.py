import pandas
import numpy

pokemon = pandas.read_csv("/Users/Christine/Documents/INSY 652/Pokemon.csv")

import plotly
from plotly.offline import init_notebook_mode
from plotly import graph_objs
plotly.offline.init_notebook_mode(connected=True)

x_data = ['Generation 1','Generation 2','Generation 3','Generation 4','Generation 5','Generation 6']
# map(str,list(pokemon.Generation.unique()))

gen1 = pokemon.loc[pokemon['Generation']==1, 'Total']
gen2 = pokemon.loc[pokemon['Generation']==2, 'Total']
gen3 = pokemon.loc[pokemon['Generation']==3, 'Total']
gen4 = pokemon.loc[pokemon['Generation']==4, 'Total']
gen5 = pokemon.loc[pokemon['Generation']==5, 'Total']
gen6 = pokemon.loc[pokemon['Generation']==6, 'Total']

y_data = [gen1,gen2,gen3,gen4,gen5,gen6]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

traces = []

for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(graph_objs.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,),
            line=dict(width=1),
        ))

layout = graph_objs.Layout(
    title='Pokemon Total Stats ',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = graph_objs.Figure(data=traces, layout=layout)
plotly.offline.plot(fig) # plot will be drawn offline

