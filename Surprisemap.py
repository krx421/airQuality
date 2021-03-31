from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/krx421/airQuality/main/UHF42.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/krx421/airQuality/main/Data.csv",
                   dtype={"id": str})

possibleInMeter = 1*10**18/ 8.18

pollution = df.data_value.to_list()

pollutionProbability = [i/possibleInMeter for i in pollution]

print(pollutionProbability)

import plotly.express as px

fig = px.choropleth(df, geojson=counties, locations='geo_entity_id', color='data_value',
                           color_continuous_scale="blues",
                           range_color=(0, 11),
                           scope="usa",
                           hover_data=['geo_entity_name'],
                           labels={'data_value':'Particulate Matter Average per cubic meter', 'geo_entity_name':'Neigborhood ', 'geo_entity_id ':'ID'}
                          )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()




import numpy as np
from scipy.stats import entropy

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


p = [0.1, 0.9]
q = [0.1, 0.8]

entropy(p, q) == kl(p, q)

print(entropy(p,q))


import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/krx421/airQuality/main/Data.csv",
                   dtype={"id": str})

df2 = pd.read_csv("https://raw.githubusercontent.com/krx421/airQuality/main/Asthma%20Hospitalizations%20(Children%200%20to%204%20Yrs%20Old)%20(1).csv")

    
possibleInMeter = 1*10**18/ 8.18

pollution = df.data_value.to_list()
asthma = df2.EstimatedAnnualRate.to_list()

pollutionProb = [i/possibleInMeter for i in pollution]
asthmaProb = [i/10000 for i in asthma]

print(pollutionProb)
print(asthmaProb)

surprise = []
for i in range(len(asthmaProb)):
    surprise.append(entropy([.1,pollutionProb[i]],[.1,asthmaProb[i]]))

    
print(min(surprise))
print(max(surprise))
print(max(surprise) - min(surprise))
