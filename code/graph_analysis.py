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
parser.add_argument("--langs", nargs="+", default=["telugu", "english", "hindi"])
parser.add_argument('--cores', required=False, type=int, default='4')

args = parser.parse_args()

def mp_get_paths(x):
    start, end = x
    if start == None:
        return []
    
    s_path = wiki_graph.get_shortest_paths(start, to=end, mode="all")[0]
    if len(s_path) <= 3:
        paths = wiki_graph.get_all_shortest_paths(start, to=end, mode="all")
    else:
        paths = []
        paths = wiki_graph.get_all_shortest_paths(start, to=end, mode="out")
        paths += wiki_graph.get_all_shortest_paths(end, to=start, mode="out")
        if s_path not in paths:
            paths += [s_path]
    # paths += [x[::-1] for x in out_paths]
    return paths

print("Loading wiki data")
train_df        = pd.read_csv("/data/multilingual_KGQA/kg_data/wikidata5m/all.txt", sep="\t", header=None)
tuples          = [tuple([x[0], x[2], x[1]]) for x in train_df.values]
print("Creating wiki graph")
wiki_graph      = Graph.TupleList(tuples, directed = True, edge_attrs = ['weight'])
wid_vid_map     = {x["name"]: id for id, x in enumerate(wiki_graph.vs)}
# wid_eid_map = {x["weight"]: id for id, x in enumerate(wiki_graph.es)}

graph_entities = {x:0 for x in (set(list(set(train_df[0].values)) + list(set(train_df[2].values))))}
p = pool(args.cores)

for lang in args.langs:
    el_data = json.load(open(f"/data/multilingual_KGQA/IndoRE/data/{lang}_indore_el.json"))

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

    names = [wiki_graph.es[x]["weight"] if x != -1 else None for x in wiki_graph.get_eids(list(set(edges)), error=False) ]
    for name, edge in zip(names, list(set(edges))):
        if name == None: continue
        for position in edge_pos_map[edge]:
            eid, pid, id = position
            final_graph[eid]["paths"][pid]["edges"][id] += [[wiki_graph.vs[edge[0]]["name"], name, wiki_graph.vs[edge[1]]["name"]]]

    # store the final_graph into json format

    # print(json.dumps(final_graph, indent=4))
    # with open(f"/data/multilingual_KGQA/IndoRE/data/{lang}_indore_el_paths.json", "w") as f:
    #     json.dump(final_graph, f, indent=4)


    data = final_graph
    original_data = json.load(open(f'/data/multilingual_KGQA/IndoRE/data/{lang}_indore_el.json'))
    write_lines = []
    e_c, r_c, p_c = [], [], []
    for id, x in enumerate(data[:]):
        graph = []
        # random sample 25 paths from the list if paths are more than 25
        if len(x["paths"]) > 25:
            x = np.random.choice(x["paths"], 25, replace=False)
        else:
            x = x["paths"]
        p_c.append(len(x))
        for y in x:
            for z in y["edges"]:
                graph += [(e1, r, e2) for (e1, r, e2) in z]
        graph = list(set(graph))
        entities = [y[0] for y in graph] + [y[2] for y in graph]
        entities = list(set(entities))
        relations = [y[1] for y in graph]
        relations = list(set(relations))
        e1, e2 = original_data[id]["entity1"]["wiki_entities"][0]["id"], original_data[id]["entity2"]["wiki_entities"][0]["id"]
        write_line = {"entity1": e1, "entity2": e2, "nodes": entities, "#nodes": len(entities), "#edges": len(relations), "edges": relations, "graph": graph}
        write_lines += [write_line]
        e_c += [len(entities)]
        r_c += [len(relations)]
    # dump write_lines to json lines file
    with open(f'/data/multilingual_KGQA/IndoRE/data/{lang}_indore_el_graph.json', 'w') as f:
        for line in write_lines:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')
            
    
    # # plot histogram of entity counts
    # # plt.figure(figsize=(10, 5))
    # fig = plt.figure(figsize=(10, 5))
    # fig.patch.set_facecolor('white')
    # # plt.text(0.1,0.9, f'mean:{np.mean(e_c):.2f}', ha='center', va='center')
    # plt.text(0.80, 0.70, f'mean:{np.mean(e_c):.2f}\nstd:{np.std(e_c):.2f}\nmedian:{np.median(e_c):.2f}\nmin:{np.min(e_c):.2f}\nmax:{np.max(e_c):.2f}', transform=plt.gca().transAxes)
    # plt.hist(e_c, bins=20, color='#00bcd4', alpha=0.5)
    # plt.title(f'{lang}_indore_el_node_counts')
    # plt.xlabel('Entity Count')
    # plt.ylabel('Frequency')
    # plt.savefig(f'{lang}_indore_el_node_counts.png', dpi=240, transparent=False)
    # # plot histogram of relation counts
    # # plt.figure(figsize=(10, 5))
    # fig = plt.figure(figsize=(10, 5))
    # fig.patch.set_facecolor('white')
    # plt.hist(r_c, bins=20, color='#ff9800', alpha=0.5)
    # plt.title(f'{lang}_indore_el_edge_counts')
    # plt.text(0.80, 0.70, f'mean:{np.mean(r_c):.2f}\nstd:{np.std(r_c):.2f}\nmedian:{np.median(r_c):.2f}\nmin:{np.min(r_c):.2f}\nmax:{np.max(r_c):.2f}', transform=plt.gca().transAxes)
    # plt.xlabel('Relation Count')
    # plt.ylabel('Frequency')
    # plt.savefig(f'{lang}_indore_el_edge_counts.png', dpi=240, transparent=False)
    # fig = plt.figure(figsize=(10, 5))
    # fig.patch.set_facecolor('white')
    # # plt.text(0.1,0.9, f'mean:{np.mean(e_c):.2f}', ha='center', va='center')
    # plt.text(0.80, 0.70, f'mean:{np.mean(p_c):.2f}\nstd:{np.std(p_c):.2f}\nmedian:{np.median(p_c):.2f}\nmin:{np.min(p_c):.2f}\nmax:{np.max(p_c):.2f}', transform=plt.gca().transAxes)
    # plt.hist(p_c, bins=20, color='#00bcd4', alpha=0.5)
    # plt.title(f'{lang}_indore_el_path_counts')
    # plt.xlabel('Path Count')
    # plt.ylabel('Frequency')
    # plt.savefig(f'{lang}_indore_el_path_counts.png', dpi=240, transparent=False)

    





# parser = argparse.ArgumentParser()
# parser.add_argument("--langs", nargs="+", default=["english", "hindi", "telugu"])
# parser.add_argument('--cores', required=False, type=int, default='4')
# args= parser.parse_args()

# # if not os.path.exists('wikidata5m'):
# #     if which("axel") is not None:
# #         !axel --quiet https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m.tar.gz 
# #     else:
# #         !wget https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m.tar.gz 
# #     !tar -xvf wikidata5m.tar.gz
# #     !rm wikidata5m.tar.gz
# #     !cat wikidata5m/train.txt wikidata5m/valid.txt wikidata5m/test.txt > wikidata5m/all.txt


# def mp_get_paths(x):
#     start, end = x
#     if start == None:
#         return []
#     paths = wiki_graph.get_all_shortest_paths(start, to=end, mode="out")
#     paths += wiki_graph.get_all_shortest_paths(end, to=start, mode="out")
#     # paths += [x[::-1] for x in out_paths]
#     return paths


# train_df         = pd.read_csv("/data/multilingual_KGQA/kg_data/wikidata5m/all.txt", sep="\t", header=None)
# tuples           = [tuple([x[0], x[2], x[1]]) for x in train_df.values]
# wiki_graph       = Graph.TupleList(tuples, directed = True, edge_attrs = ['weight'])
# wid_vid_map      = {x["name"]: id for id, x in enumerate(wiki_graph.vs)}
# # wid_eid_map = {x["weight"]: id for id, x in enumerate(wiki_graph.es)}
# graph_entities = {x:0 for x in (set(list(set(train_df[0].values)) + list(set(train_df[2].values))))}
# p = pool(args.cores)


# for lang in args.langs:
#     el_data = json.load(open(f"/data/multilingual_KGQA/IndoRE/data/{lang}_indore_el.json"))
#     all_es = list(set(list(chain.from_iterable((x["entity1"]["wiki_entities"][0]["id"], x["entity2"]["wiki_entities"][0]["id"]) for x in el_data))))
#     present_es = [x for x in all_es if x in graph_entities]
#     s_e_pairs = [(wid_vid_map[x["entity1"]["wiki_entities"][0]["id"]], wid_vid_map[x["entity2"]["wiki_entities"][0]["id"]]) if x["entity1"]["wiki_entities"][0]["id"] in wid_vid_map and x["entity2"]["wiki_entities"][0]["id"] in wid_vid_map else (None, None) for x in el_data]
#     print("Computing shortest paths for ", lang)
#     results = list(tqdm(p.imap(mp_get_paths, s_e_pairs[:]), total=len(el_data[:])))
#     edge_pos_map = {}
#     # x = results[0]
#     edges = []
#     final_graph = []
#     print("Getting edge data of the shortest paths for ", lang)
#     for eid, x in tqdm(enumerate(results)):
#         final_graph.append({"paths": []})
#         for pid, path in enumerate(x):
#             final_graph[eid]["paths"].append({"edges": []})
#             for e1 in range(len(path)-1):
#                 final_graph[eid]["paths"][pid]["edges"].append([])
#                 edges.append((path[e1], path[e1+1]))
#                 if (path[e1], path[e1+1]) not in edge_pos_map:
#                     edge_pos_map[(path[e1], path[e1+1])] = []
#                 edge_pos_map[(path[e1], path[e1+1])].append((eid, pid, e1))
#     names = [wiki_graph.es[x]["weight"] for x in wiki_graph.get_eids(list(set(edges)))]
#     for name, edge in zip(names, list(set(edges))):
#         for position in edge_pos_map[edge]:
#             eid, pid, id = position
#             final_graph[eid]["paths"][pid]["edges"][id] = [wiki_graph.vs[edge[0]]["name"], name, wiki_graph.vs[edge[1]]["name"]]
#     # store the final_graph into json format
#     with open(f"/data/multilingual_KGQA/IndoRE/data/{lang}_indore_el_paths.json", "w") as f:
#         json.dump(final_graph, f, indent=4)
