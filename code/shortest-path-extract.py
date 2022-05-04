from ast import parse
import numpy as np
from igraph import * 
from tqdm import tqdm
import pandas as pd
from shutil import which
import json
import os
from itertools import chain
from multiprocessing import Pool as pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--langs", nargs="+", default=["english", "hindi", "telugu"])
parser.add_argument('--cores', required=False, type=int, default='4')

args = parser.parse_args()

# if not os.path.exists('wikidata5m'):
#     if which("axel") is not None:
#         !axel --quiet https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m.tar.gz 
#     else:
#         !wget https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m.tar.gz 
#     !tar -xvf wikidata5m.tar.gz
#     !rm wikidata5m.tar.gz
#     !cat wikidata5m/train.txt wikidata5m/valid.txt wikidata5m/test.txt > wikidata5m/all.txt


def mp_get_paths(x):
    start, end = x
    if start == None:
        return []
    
    s_path = wiki_graph.get_shortest_paths(start, to=end, mode="all")[0]
    if len(s_path) <= 3:
        paths = wiki_graph.get_all_shortest_paths(start, to=end, mode="all")
    else:
        # paths = [s_path]
        paths = wiki_graph.get_all_shortest_paths(start, to=end, mode="out")
        paths += wiki_graph.get_all_shortest_paths(end, to=start, mode="out")
        if s_path not in paths:
            paths += [s_path]
    # paths += [x[::-1] for x in out_paths]
    return paths

print("Loading wiki data")
train_df = pd.read_csv("wikidata5m/all.txt", sep="\t", header=None)
tuples = [tuple([x[0], x[2], x[1]]) for x in train_df.values]
print("Creating wiki graph")
wiki_graph = Graph.TupleList(tuples, directed = True, edge_attrs = ['weight'])
wid_vid_map = {x["name"]: id for id, x in enumerate(wiki_graph.vs)}
# wid_eid_map = {x["weight"]: id for id, x in enumerate(wiki_graph.es)}

graph_entities = {x:0 for x in (set(list(set(train_df[0].values)) + list(set(train_df[2].values))))}
p = pool(args.cores)

for lang in args.langs:
    el_data = json.load(open(f"{lang}_indore_el.json"))

    all_es = list(set(list(chain.from_iterable((x["entity1"]["wiki_entities"][0]["id"], x["entity2"]["wiki_entities"][0]["id"]) for x in el_data))))
    present_es = [x for x in all_es if x in graph_entities]

    s_e_pairs = [(wid_vid_map[x["entity1"]["wiki_entities"][0]["id"]], wid_vid_map[x["entity2"]["wiki_entities"][0]["id"]]) if x["entity1"]["wiki_entities"][0]["id"] in wid_vid_map and x["entity2"]["wiki_entities"][0]["id"] in wid_vid_map else (None, None) for x in el_data]
    print("Computing shortest paths for ", lang)
    results = list(tqdm(p.imap(mp_get_paths, s_e_pairs[:]), total=len(el_data[:])))

    edge_pos_map = {}
# x = results[0]
    edges = []
    final_graph = []
    print("Getting edge data of the shortest paths for ", lang)
    for eid, x in tqdm(enumerate(results)):
        final_graph.append({"paths": []})
        for pid, path in enumerate(x):
            final_graph[eid]["paths"].append({"edges": []})
            for e1 in range(len(path)-1):
                final_graph[eid]["paths"][pid]["edges"].append([])
                edges.append((path[e1], path[e1+1]))
                edges.append((path[e1+1], path[e1]))
                if (path[e1], path[e1+1]) not in edge_pos_map:
                    edge_pos_map[(path[e1], path[e1+1])] = []
                if (path[e1+1], path[e1]) not in edge_pos_map:
                    edge_pos_map[(path[e1+1], path[e1])] = []
                edge_pos_map[(path[e1], path[e1+1])].append((eid, pid, e1))
                edge_pos_map[(path[e1+1], path[e1])].append((eid, pid, e1))

    names = [wiki_graph.es[x]["weight"] for x in wiki_graph.get_eids(list(set(edges)), error=False) if x != -1]
    for name, edge in zip(names, list(set(edges))):
        for position in edge_pos_map[edge]:
            eid, pid, id = position
            final_graph[eid]["paths"][pid]["edges"][id] += [[wiki_graph.vs[edge[0]]["name"], name, wiki_graph.vs[edge[1]]["name"]]]

    # store the final_graph into json format

    # print(json.dumps(final_graph, indent=4))
    with open(f"{lang}_indore_el_paths.json", "w") as f:
        json.dump(final_graph, f, indent=4)


    