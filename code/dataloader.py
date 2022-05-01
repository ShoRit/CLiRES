from helper import *
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_geometric.data import Data
import torch

class RelDataset(Dataset):
	def __init__(self, dataset, params, data_idx=0):
		self.dataset		= dataset
		self.p				= params
  
		if self.p.bert_model 		 =='mbert':
			self.tokenizer       	 = 	AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
		elif self.p.bert_model     	 == 'xlmr':
			self.tokenizer         	 = 	AutoTokenizer.from_pretrained('xlm-roberta-base')
			
		self.data_idx 		= data_idx

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		tokens 			= torch.tensor(ele['tokens']	+ [0]*(self.p.max_seq_len-len(ele['tokens'])))
		marked_1		= torch.tensor(ele['arg1_ids']	+ [0]*(self.p.max_seq_len-len(ele['tokens'])))
		marked_2		= torch.tensor(ele['arg2_ids']	+ [0]*(self.p.max_seq_len-len(ele['tokens'])))
		segments		= torch.tensor([0]*len(tokens))
		et_1			= torch.LongTensor(ele['arg1_type'])
		et_2			= torch.LongTensor(ele['arg2_type'])
		desc_emb		= ele['desc_emb']
		label 			= torch.LongTensor(ele['label'])
		dep_data 		= ele['dep_data']
		dep_data 		= Data(x=torch.tensor(dep_data.x), edge_index= torch.tensor(dep_data.edge_index), 
      						edge_type= torch.tensor(dep_data.edge_type), n1_mask=torch.tensor(dep_data.n1_mask), n2_mask=torch.tensor(dep_data.n2_mask))

		ent1_emb		= torch.FloatTensor(ele['arg1_emb'])
		ent2_emb 		= torch.FloatTensor(ele['arg2_emb'])

		return (tokens, segments, marked_1, marked_2, desc_emb, label, et_1, et_2, dep_data, ent1_emb, ent2_emb)

def create_mini_batch(samples):
	import torch
	from torch_geometric.loader import DataLoader 
	tokens_tensors 					= [s[0] for s in samples]
	segments_tensors 				= [s[1] for s in samples]
	marked_e1 						= [s[2] for s in samples]
	marked_e2 						= [s[3] for s in samples]
	relation_emb 					= torch.stack([s[4] for s in samples])
	label_ids 						= torch.stack([s[5] for s in samples])
	et_1 							= torch.stack([s[6] for s in samples])
	et_2 							= torch.stack([s[7] for s in samples])

	tokens_tensors 					= pad_sequence(tokens_tensors, batch_first=True)
	segments_tensors 				= pad_sequence(segments_tensors, batch_first=True)
	marked_e1 						= pad_sequence(marked_e1, batch_first=True)
	marked_e2 						= pad_sequence(marked_e2, batch_first=True)
	masks_tensors 					= torch.zeros(tokens_tensors.shape, dtype=torch.long)
	masks_tensors 					= masks_tensors.masked_fill(tokens_tensors != 0, 1)

	graph_list 						= [s[8] for s in samples]
	graph_loader 					= DataLoader(graph_list, batch_size=len(graph_list))
	graph_tensors 					= [elem for elem in  graph_loader][0]

	ent1_emb 						= torch.stack([s[9] for s in samples])
	ent2_emb 						= torch.stack([s[10] for s in samples])
	
	return tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb


