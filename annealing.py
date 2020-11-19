#%%
# Carl Klier - Graph Coloring Annealing for Network Science Final Exam
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import json
import sys

# nodes = ['a', 'b', 'c']
# edges = [('a', 'b'), ('a', 'c'), ('b', 'c')]

# nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
# edges = [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('b', 'e'),
#               ('e', 'f'), ('d', 'f'), ('e', 'g'), ('g', 'h'), ('f', 'h'), ('g', 'i'),
#                ('i', 'j'), ('h', 'j'), ('i', 'k'), ('k', 'l'), ('j', 'l')]

# building Graph 1
with open('deezer.json') as f:
  data = json.load(f)
G = nx.Graph()
G.add_edges_from(data["0"])

nodes = list(G.nodes)
edges = list(G.edges)

# print("nodes: ", nodes)
# print("edges: ", edges)
# sys.exit()

# G = nx.Graph()
# G.add_nodes_from(nodes)
# G.add_edges_from(edges)

colors = ['Red', 'Green']
temperature = 5

random.seed(100)

def get_initial_color_mapping(nodes):
  color_mapping = {}
  for node in nodes:
    color_mapping[node] = colors[0]
  return color_mapping

color_mapping = get_initial_color_mapping(nodes)  

def update_color_mapping(node, color):
  color_mapping[node] = color 

def get_energy_value(color, neighbors):
  energy = 0
  for neighbor in neighbors:
    if color == color_mapping[neighbor]:
      energy = energy + 1
  return energy

# def get_next_energy_value(node, current_energy):
#   neighbors = list(G.neighbors(node))
#   current_color = color_mapping[node]
#   lowest_energy = get_energy_value(current_color, neighbors)
#   lowest_energy_color = current_color
#   for color in colors:
#     possible_lowest_energy = get_energy_value(color, neighbors)
#     if possible_lowest_energy < lowest_energy:
#       lowest_energy = possible_lowest_energy
#       lowest_energy_color = color
#   update_color_mapping(node, color)

plt.figure(figsize=(50, 50))
print("initial color mapping: ", color_mapping)
nx.draw_networkx(G, with_labels=True, node_color=list(color_mapping.values()))
plt.show()

for i in range(100):
  node_to_pick = random.randint(0, len(nodes) - 1)
  color_to_pick = random.randint(0, len(colors) - 1)
  node = nodes[node_to_pick]
  color = colors[color_to_pick]

  neighbors = list(G.neighbors(node))
  current_color = color_mapping[node]
  current_energy_value = get_energy_value(current_color, neighbors)
  proposed_energy_value = get_energy_value(color, neighbors)

  exponential = math.exp((-proposed_energy_value + current_energy_value)/temperature)
  accept_prob = min(1, exponential)
  sample = random.random()
  if sample < accept_prob:
    update_color_mapping(node, color)

plt.figure(figsize=(50, 50))
print("final color mapping: ", color_mapping)
nx.draw_networkx(G, with_labels=True, node_color=list(color_mapping.values()))
plt.show()

#%%