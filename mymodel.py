import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

class FirstModel():
  def __init__(self, cluster):
      cluster['o_tract10'] = cluster['TRACT'].astype(float)
      self.cluster = cluster
      self.pls = PLSRegression(n_components=10)
      self.encoder = OneHotEncoder()
      self.linear = LinearRegression()
      self.knn = KNeighborsRegressor()

  def train(self, x, y, w):
    x['o_tract10'] = x[['o_tract10']].merge(self.cluster[['o_tract10', 'Cluster']], on='o_tract10', how="left")['Cluster'].fillna(-1)
    x,y,w = x.iloc[:,1:], y.iloc[:,1:].values, w.iloc[:,1:].values

    x = self.encoder.fit_transform(x.values).toarray()
    x, _ = self.pls.fit_transform(x, y)
    self.linear.fit(x,y,sample_weight = w.ravel())

  def predict(self, features):
    features[-1] = self.cluster[self.cluster['o_tract10']==features[-1]]['Cluster'].fillna(-1).values[0]
    x = self.pls.transform(self.encoder.transform([features]).toarray())
    return self.linear.predict(x).astype(int)[0]

    

class SecondModel():
  def __init__(self, toy, coord, coord_bg):
    self.toy = toy
    self.coord = coord
    self.coord_bg = coord_bg

  def predict(self, origin):
    destination  = self.toy[self.toy['o_tract10']==float(np.floor(origin/10))]['d_tract10'].values[0]
    o_coord = self.coord_bg[self.coord_bg['TRACT']==origin][["centroid_y", "centroid_x"]].values
    d_coord = self.coord[self.coord['TRACT']==destination][["centroid_y", "centroid_x"]].values
    return o_coord, d_coord, destination

    

class ThirdModel():
    def __init__(self, cluster, inter, outer):
        self.cluster = cluster
        self.inter = inter
        self.outer = outer
        self.inter_model = PLSRegression(n_components=10)
        self.outer_model = PLSRegression(n_components=10)
        self.encoder_x = OneHotEncoder()
        self.encoder_y = OneHotEncoder()

    def train(self, inter_x, inter_y, outer_x, outer_y):
      self.encoder_x.fit_transform(pd.concat([inter_x.iloc[:,:10], outer_x.iloc[:,:10]], axis=0).values)
      inter_x_soc = pd.DataFrame(self.encoder_x.transform(inter_x.iloc[:,:10].values).toarray())
      inter_x = pd.concat([inter_x_soc, inter_x.iloc[:,10:]], axis=1)
      outer_x_soc = pd.DataFrame(self.encoder_x.transform(outer_x.iloc[:,:10].values).toarray()).reset_index().drop(columns="index")
      outer_x = outer_x.reset_index().drop(columns="index")
      outer_x = pd.concat([outer_x_soc, outer_x.iloc[:,10:]], axis=1)
      y = pd.concat([inter_y, outer_y], axis=0).reset_index()
      y = y.pivot_table(index=y.index,columns="main_mode",values="weight").fillna(0)
      inter_len = len(inter_y)
      inter_y = y.iloc[:inter_len, :]
      outer_y = y.iloc[inter_len:, :]
      self.inter_model.fit(inter_x.values, inter_y.values)
      self.outer_model.fit(outer_x.values, outer_y.values)

    def predict(self, features):
      val = self.encoder_x.transform([features[:-2]])[0].toarray()[0].tolist()
      if features[-2] == features[-1]:
        att = self.cluster.merge(self.inter, on='Cluster')
        value = att[att['TRACT'] == features[-1]].iloc[:,2:].values.tolist()[0]
        for ele in value:
          val.append(ele)
        result = self.inter_model.predict([val])
      else:
        att = self.cluster.merge(self.outer, on='TRACT', how='left')
        ct = features[-2]
        value_o = att[att['TRACT']==ct][['centroid_x','centroid_y']].values[0]
        ct = features[-1]
        value_d = att[att['TRACT']==ct][['centroid_x','centroid_y']].values[0]
        value = np.sqrt(sum((value_o-value_d)**2))
        val.append(value)
        result = self.outer_model.predict([val])
      return result[0]