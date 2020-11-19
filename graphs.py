#%%

# Network Science Take Home Exam Part2 Problem1
import networkx as nx
import json
import matplotlib.pyplot as plt
import numpy as np
from random import randint, seed
import csv
import time

def plot(data):
    """ Plot Distribution """
    # plt.plot(range(len(data)),data,'bo')
    # plt.yscale('linear')
    # plt.xscale('linear')
    # plt.ylabel('Freq')
    # plt.xlabel('Degree')
    #plt.show()

    """ Plot CDF """
    s = float(data.sum())
    print(s)
    print("This is cumsum: ", data.cumsum(0))
    cdf = data.cumsum(0)/s
    #plt.plot(range(len(cdf)),cdf)
    #plt.xscale('linear')
    #plt.ylim([0,1])
    #plt.ylabel('CDF')
    #plt.xlabel('Degree')
    #plt.show()

    """ Plot CCDF """
    print("[0] ", cdf[0])
    print("[1] ", cdf[1])
    print("[len - 1] ", cdf[len(cdf) - 1])
    ccdf = 1-cdf
    plt.plot(range(len(ccdf)),ccdf)
    plt.xscale('linear')
    plt.yscale('log')
    #plt.ylim([0,1])
    plt.ylabel('CCDF')
    plt.xlabel('Degree')

    plt.show()

def get_degree_array(G):
  degrees = []
  for i in range(len(G.nodes)):
    degrees.append(G.degree[i])
  return degrees

def get_eigenvalue_centrality_array(G):
  eigenvalue_centralities = nx.eigenvector_centrality(G, max_iter=500)
  egv_cent_array = list(eigenvalue_centralities.values())
  return egv_cent_array

def get_pagerank_array(G):
  pageranks = nx.pagerank(G, alpha=.85)
  pageranks_array = list(pageranks.values())
  return pageranks_array

def get_closeness_array(G):
  closeness = nx.closeness_centrality(G)
  closeness_array = list(closeness.values())
  return closeness_array  

def get_betweenness_array(G):
  betweenness = nx.betweenness_centrality(G)
  betweenness_array = list(betweenness.values())
  return betweenness_array  

## assumes we are using a continusous random variable, ignores duplicates
## dont use for degree distribution
def plot_ccdf(np_vector, data, scale):
  sorted_data = np.sort(np_vector)
  print("Sorting")
  print(sorted_data[0])
  print(sorted_data[len(sorted_data) - 2])
  print(sorted_data[len(sorted_data) - 1])
  yvals = np.arange(1, len(sorted_data) + 1)/float(len(sorted_data))
  print("yval")
  print(yvals[0])
  print(yvals[len(yvals) - 1])
  plt.plot(sorted_data, 1 - yvals)
  if scale == 'linear':
    plt.yscale('linear')
    plt.xscale('linear')
    plt.ylabel('CCDF')
    plt.xlabel(data)
  elif scale == 'log':
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Log(CCDF)')
    plt.xlabel('Log(%s)' %data)
  elif scale == 'semi-log':
    plt.yscale('log')
    plt.xscale('linear')
    plt.ylabel('Log(CCDF)')
    plt.xlabel(data)
  else:
    print("Error with scale")
    return
  plt.show()

def distance(vectorx, vectory):
  sum = 0
  for i in range(len(vectorx)):
    sum = sum + abs(vectorx[i] - vectory[i])
  return sum / len(vectorx)   

def  calculate_expected_xt(vector_x0):
  expected_xt = [sum(vector_x0)/len(vector_x0)] * len(vector_x0)
  return expected_xt

def plot_distance_vs_time(G, vector_x0, expected_vector):
  seed(1)
  time_points = []
  distance_points = []
  xt = vector_x0
  i = 0
  #initial_time = time.perf_counter()
  #print("initial vector ", vector_x0)
  while i < 5000000:
    index = randint(0, len(xt) - 1)
    #print("index ", index)
    neighbors = nx.neighbors(G, index)
    sum = 0
    for neighbor in neighbors:
      #print("neighbor ", neighbor)
      sum = sum + xt[neighbor]
    #print("sum ", sum)
    #print("degree of i ", G.degree[index])
    #print(xt[index], " replaced by ", sum / (G.degree[index]))
    xt[index] = sum / (G.degree[index])
    #print("updated xt: ", xt)
    current = time.perf_counter()
    #time_points.append(current - initial_time)
    if i % 100 == 0:
      time_points.append(i)
      distance_points.append(distance(xt, expected_vector))  
    i = i + 1  
  #print("final xt ", xt)
  plt.plot(time_points, distance_points)
  plt.ylabel("distance")
  plt.xlabel("time")
  plt.show() 
  plt.clf() 

def find_num_elelements_greater_than_num(vectorv, num):
  i = -1
  while i < len(vectorv) - 1:
    if vectorv[i+1] > num:
      return (len(vectorv)-i-1)
    i = i + 1
  return 0

def get_xvals_yvals(np_degree_vector):
  vals = []
  max_degree = max(np_degree_vector)
  sorted_degrees = sorted(np_degree_vector)
  xvals = np.linspace(0, max_degree, len(np_degree_vector))
  yvals = []
  for xval in xvals:
    yval = find_num_elelements_greater_than_num(sorted_degrees, xval) / float(len(np_degree_vector))
    yvals.append(yval)
  vals.append(xvals)
  vals.append(yvals)
  return vals

def plot_good_ccdf(xvals, yvals, data, scale):
  plt.plot(xvals, yvals)
  if scale == 'linear':
    plt.yscale('linear')
    plt.xscale('linear')
    plt.ylabel('CCDF')
    plt.xlabel(data)
  elif scale == 'log':
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Log(CCDF)')
    plt.xlabel('Log(%s)' %data)
  elif scale == 'semi-log':
    plt.yscale('log')
    plt.xscale('linear')
    plt.ylabel('Log(CCDF)')
    plt.xlabel(data)
  else:
    print("Error with scale")
    return
  plt.show()  

# -----------------------------------------------------------------------------
# starting main

# building Graph 1
with open('deezer.json') as f:
  data = json.load(f)
G = nx.Graph()
G.add_edges_from(data["0"])

# Building Graph 2
G2 = nx.Graph()
G2_edges = []
with open('chameleon.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for row in csv_reader:
    G2_edges.append([int(row[0]), int(row[1])])
G2.add_edges_from(G2_edges)

# Building Graph 3
G3 = nx.Graph()
G3_edges = []
with open('crocodiles.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for row in csv_reader:
    G3_edges.append([int(row[0]), int(row[1])])
G3.add_edges_from(G3_edges)

print("G edges", nx.number_of_edges(G))
print("G nodes", nx.number_of_nodes(G))
# finding degree distributions and plotting them
gdegrees = get_degree_array(G)
gdegrees_np = np.asarray(gdegrees)

g2degrees = get_degree_array(G2)
g2degrees_np = np.asarray(g2degrees)

g3degrees = get_degree_array(G3)
g3degrees_np = np.asarray(g3degrees)

# print("Plotting good cdf for g1")
# vals = get_xvals_yvals(gdegrees_np)
# xvals = vals[0]
# yvals = vals[1]
# plot_good_ccdf(xvals, yvals, 'degree', 'linear')
# plot_good_ccdf(xvals, yvals, 'degree', 'semi-log')
# plot_good_ccdf(xvals, yvals, 'degree', 'log')

# print("Plotting good cdf for g2")
# vals = get_xvals_yvals(g2degrees_np)
# xvals = vals[0]
# yvals = vals[1]
# plot_good_ccdf(xvals, yvals, 'degree', 'linear')
# plot_good_ccdf(xvals, yvals, 'degree', 'semi-log')
# plot_good_ccdf(xvals, yvals, 'degree', 'log')


# print("Plotting good cdf for g3")
# vals = get_xvals_yvals(g3degrees_np)
# xvals = vals[0]
# yvals = vals[1]
# plot_good_ccdf(xvals, yvals, 'degree', 'linear')
# plot_good_ccdf(xvals, yvals, 'degree', 'semi-log')
# plot_good_ccdf(xvals, yvals, 'degree', 'log')

#-----------------------------------------------------------

# G1 centrality graphs

print("G1 eigenvalue centrality")
geigenvalues = get_eigenvalue_centrality_array(G)
geigenvalues_np = np.asarray(geigenvalues)
vals = get_xvals_yvals(geigenvalues_np)
xvals = vals[0]
yvals = vals[1]
plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'linear')
plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'semi-log')
plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'log')

print("G1 pagerank centrality")
gpageranks = get_pagerank_array(G)
gpageranks_np = np.asarray(gpageranks)
vals = get_xvals_yvals(gpageranks_np)
xvals = vals[0]
yvals = vals[1]
plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'linear')
plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'semi-log')
plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'log')

print("G1 closeness centrality")
gcloseness = get_closeness_array(G)
gcloseness_np = np.asarray(gcloseness)
vals = get_xvals_yvals(gcloseness_np)
xvals = vals[0]
yvals = vals[1]
plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'linear')
plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'semi-log')
plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'log')

print("----------------------------------")

# G2 centrality graphs
print("G2 eigenvalue centrality")
geigenvalues = get_eigenvalue_centrality_array(G2)
geigenvalues_np = np.asarray(geigenvalues)
vals = get_xvals_yvals(geigenvalues_np)
xvals = vals[0]
yvals = vals[1]
plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'linear')
plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'semi-log')
plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'log')

print("G2 pagerank centrality")
gpageranks = get_pagerank_array(G2)
gpageranks_np = np.asarray(gpageranks)
vals = get_xvals_yvals(gpageranks_np)
xvals = vals[0]
yvals = vals[1]
plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'linear')
plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'semi-log')
plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'log')

print("G2 closeness centrality")
gcloseness = get_closeness_array(G2)
gcloseness_np = np.asarray(gcloseness)
vals = get_xvals_yvals(gcloseness_np)
xvals = vals[0]
yvals = vals[1]
plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'linear')
plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'semi-log')
plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'log')


print("----------------------------------")

# # G3 centrality graphs
# print("G3 eigenvalue centrality")
# geigenvalues = get_eigenvalue_centrality_array(G3)
# geigenvalues_np = np.asarray(geigenvalues)
# vals = get_xvals_yvals(geigenvalues_np)
# xvals = vals[0]
# yvals = vals[1]
# plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'linear')
# plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'semi-log')
# plot_good_ccdf(xvals, yvals, 'eigenvalue_centrality', 'log')

# print("G3 pagerank centrality")
# gpageranks = get_pagerank_array(G3)
# gpageranks_np = np.asarray(gpageranks)
# vals = get_xvals_yvals(gpageranks_np)
# xvals = vals[0]
# yvals = vals[1]
# plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'linear')
# plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'semi-log')
# plot_good_ccdf(xvals, yvals, 'pagerank_centrality', 'log')

# print("G3 closeness centrality")
# gcloseness = get_closeness_array(G3)
# gcloseness_np = np.asarray(gcloseness)
# vals = get_xvals_yvals(gcloseness_np)
# xvals = vals[0]
# yvals = vals[1]
# plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'linear')
# plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'semi-log')
# plot_good_ccdf(xvals, yvals, 'closeness_centrality', 'log')

print("-------------------------------------------------")

#calculating algebraic connectivity

# print("")
# print("G1 algebraic connectivity")
# print(nx.algebraic_connectivity(G))
# print("")

# print("")
# print("G2 algebraic connectivity")
# print(nx.algebraic_connectivity(G2))
# print("")

# print("")
# print("G3 algebraic connectivity")
# print(nx.algebraic_connectivity(G3))
# print("")

print("------------------------------------------------------")

# print("G1 distance vs. time")
# expected = [30] * len(gdegrees)
# plot_distance_vs_time(G, gdegrees, expected) 

# print("G2 distance vs. time")
# expected = [45] * len(g2degrees)
# plot_distance_vs_time(G2, g2degrees, expected)

# print("G3 distance vs. time")
# expected = [260] * len(g3degrees)
# plot_distance_vs_time(G3, g3degrees, expected)  

# print("Testing small graph -------------------------------")
# test_edges = [[0, 2], [0, 3], [0, 4], [0, 5], [1, 5], [1, 6], [1, 2], [2, 3], [3, 4]]
# testg = nx.Graph()
# testg.add_edges_from(test_edges)
# print("edges ", testg.number_of_edges())
# print("nodes ", testg.number_of_nodes())
# degree_vector = get_degree_array(testg)
# expected = [2.5] * 7

#plot_distance_vs_time(testg, degree_vector, expected)
#expected_g1 = [30] * len(gdegrees_np)
#plot_distance_vs_time(G, gdegrees_np, expected_g1)

#plot_distance_vs_time(G3, g3degrees)  
#plot_distance_vs_time(G2, g2degrees)

#print("average degree ", np.average(g3degrees_np))
#print(np.sum(g3degrees_np))

#print("average degree g2 ", np.average(g2degrees))
# %%
