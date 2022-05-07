from helper import *
import torch, torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter as Param
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import FastRGCNConv, RGCNConv

class GNNDropout(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.drop = nn.Dropout(params.drop)
	
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.drop(x), edge_index, edge_type


class GNNReLu(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.relu = nn.ReLU()
  
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.relu(x), edge_index, edge_type


class GNNRGCNConv(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.gnn 	= RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
  
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.gnn(x, edge_index, edge_type), edge_index, edge_type

    
class GNNRGCNElConv(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.gnn 	= RGCNConv(in_channels= self.params.ent_emb_dim, out_channels=self.params.ent_emb_dim, num_relations=self.params.wiki_rels)
  
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.gnn(x, edge_index, edge_type), edge_index, edge_type

    


class DeepNet(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.rgcn_layers = []
		for i in range(self.params.gnn_depth-1):
			self.rgcn_layers.append(GNNRGCNConv(self.params))
			self.rgcn_layers.append(GNNReLu(self.params))
			self.rgcn_layers.append(GNNDropout(self.params))

		self.rgcn_layers.append(GNNRGCNConv(self.params))
		self.rgcn_layers.append(GNNReLu(self.params))
		self.rgcn_module = nn.Sequential(*self.rgcn_layers)
  
		# self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		# self.drop  = nn.Dropout(self.params.drop)
		# self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		# for id, _ in enumerate(self.rgcn_layers):
		# 	if id%2 == 0:
		# 		x = F.relu(self.rgcn_layers[id](x, edge_index, edge_type))
		# 	else:
		# 		x = self.rgcn_layers[id](x)
		x,edge_index, edge_type = self.rgcn_module((x, edge_index, edge_type))
		# x = F.relu(self.conv1(x, edge_index, edge_type))
		# x = self.drop(x)
		# x = F.relu(self.conv2(x, edge_index, edge_type))
		return x

class DeepElNet(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.rgcn_layers = []
		for i in range(self.params.el_gnn_depth-1):
			self.rgcn_layers.append(GNNRGCNElConv(self.params))
			self.rgcn_layers.append(GNNReLu(self.params))
			self.rgcn_layers.append(GNNDropout(self.params))

		self.rgcn_layers.append(GNNRGCNElConv(self.params))
		self.rgcn_layers.append(GNNReLu(self.params))
		self.rgcn_module = nn.Sequential(*self.rgcn_layers)
  
		# self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		# self.drop  = nn.Dropout(self.params.drop)
		# self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		# for id, _ in enumerate(self.rgcn_layers):
		# 	if id%2 == 0:
		# 		x = F.relu(self.rgcn_layers[id](x, edge_index, edge_type))
		# 	else:
		# 		x = self.rgcn_layers[id](x)
		x,edge_index, edge_type = self.rgcn_module((x, edge_index, edge_type))
		# x = F.relu(self.conv1(x, edge_index, edge_type))
		# x = self.drop(x)
		# x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class Net(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		self.drop  = nn.Dropout(self.params.drop)
		self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class Net5(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		self.drop  = nn.Dropout(self.params.drop)
		self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class Net7(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		self.drop  = nn.Dropout(self.params.drop)
		self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class MLRelClassifier(nn.Module):
	def __init__(self, params):
		super().__init__()
		self.p					= params
		if self.p.bert_model 	=='mbert':
			self.model 			= AutoModel.from_pretrained("bert-base-multilingual-uncased")
		
		if self.p.bert_model 	=='xlmr':
			self.model 			= AutoModel.from_pretrained("xlm-roberta-base")
   
		self.dropout			= nn.Dropout(self.p.drop)
		
		rel_in 	        		= self.model.config.hidden_size *3
		ent_in 					= self.model.config.hidden_size *2
  
		if self.p.dep 			=='1':
			self.rgcn    		= DeepNet(params)
# 			self.rgcn    		= Net(params)
			rel_in 	        	+= self.p.node_emb_dim*2
			ent_in 				+= self.p.node_emb_dim
   
		if self.p.el 			=='1':
			rel_in 				+= self.p.ent_emb_dim*2
			ent_in 				+= self.p.ent_emb_dim
            
		if self.p.el 			=='2':
			self.ent_rgcn    		= DeepElNet(params)
			rel_in 	        	+= self.p.ent_emb_dim*2
			ent_in 				+= self.p.ent_emb_dim
  
		self.rel_classifier		= nn.Linear(rel_in, params.num_rels)
		self.ent_classifier		= nn.Linear(ent_in, params.num_ents)
  
	def extract_entity(self, sequence_output, e_mask):
		extended_e_mask = e_mask.unsqueeze(1)
		extended_e_mask = torch.bmm(
		extended_e_mask.float(), sequence_output).squeeze(1)
		return extended_e_mask.float()
	
  	
	def forward(self, bat):
		tokens_tensors, segments_tensors, e1_mask, e2_mask, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_data, ent1_emb, ent2_emb, ent_data = bat
  
		bert_embs 						= self.model(input_ids=tokens_tensors,attention_mask= masks_tensors, token_type_ids= segments_tensors)
		sequence_output 				= bert_embs[0] # Sequence of hidden-states of the last layer.
		pooled_output   				= bert_embs[1] # Last layer hidden-state of the [CLS] token further processed 
		context 						= self.dropout(pooled_output)
  	
		e1_h 							= self.extract_entity(sequence_output, e1_mask)
		e2_h							= self.extract_entity(sequence_output, e2_mask)

		rel_output 						= [context, e1_h, e2_h]
		ent1_output 					= [context, e1_h]
		ent2_output 					= [context, e2_h]
  
		if self.p.dep == '1':         
			# graph_embs                = self.rgcn(graph_data.x, graph_data.edge_index, graph_data.edge_type)
			n1_mask, n2_mask, batch     = graph_data.n1_mask, graph_data.n2_mask, graph_data.batch
			batch_np 					= batch.cpu().detach().numpy()
			graph_embs 					= []
			for idx in range(0, sequence_output.shape[0]):
				bids					= np.where(batch_np==idx)[0]
				sid, eid 				= bids[0], bids[-1]+1
				graph_embs.append(torch.max(sequence_output[idx]+graph_data.x[sid:eid,:,None], dim=1)[0])



			# import pdb; pdb.set_trace()
			graph_embs 					= torch.vstack(graph_embs)
			graph_embs                  = self.rgcn(graph_embs, graph_data.edge_index, graph_data.edge_type)

			e1_dep, e2_dep              = [],[]
			for idx in range(0,sequence_output.shape[0]):
				mask        			= torch.where(batch==idx, 1,0)
				m1, m2      			= mask*n1_mask, mask*n2_mask
				e1_dep.append(torch.mm(m1.unsqueeze(dim=0).float(),graph_embs))
				e2_dep.append(torch.mm(m2.unsqueeze(dim=0).float(),graph_embs))
	
			# import pdb; pdb.set_trace()
   
			e1_dep          			= torch.cat(e1_dep, dim=0)
			e2_dep          			= torch.cat(e2_dep, dim=0)
	
			rel_output.append(e1_dep)
			rel_output.append(e2_dep)
   
			ent1_output.append(e1_dep)
			ent2_output.append(e2_dep)
   
		if self.p.el 					== '1':
			rel_output.append(ent1_emb)
			rel_output.append(ent2_emb)

			ent1_output.append(ent1_emb)
			ent2_output.append(ent2_emb)

		if self.p.el 					== '2':
			n1_mask, n2_mask, batch     = ent_data.n1_mask, ent_data.n2_mask, ent_data.batch
			batch_np 					= batch.cpu().detach().numpy()
# 			import pdb; pdb.set_trace()
			ent_embs                    = self.ent_rgcn(ent_data.x, ent_data.edge_index, ent_data.edge_type)

			e1_dep, e2_dep              = [],[]
			for idx in range(0,sequence_output.shape[0]):
				mask        			= torch.where(batch==idx, 1,0)
				m1, m2      			= mask*n1_mask, mask*n2_mask
				e1_dep.append(torch.mm(m1.unsqueeze(dim=0).float(),ent_embs))
				e2_dep.append(torch.mm(m2.unsqueeze(dim=0).float(),ent_embs))
	
			# import pdb; pdb.set_trace()
   
			e1_dep          			= torch.cat(e1_dep, dim=0)
			e2_dep          			= torch.cat(e2_dep, dim=0)
	
			rel_output.append(e1_dep)
			rel_output.append(e2_dep)
   
			ent1_output.append(e1_dep)
			ent2_output.append(e2_dep)


		rel_output 						= torch.cat(rel_output,  dim=-1)
		ent1_output 					= torch.cat(ent1_output, dim=-1)
		ent2_output 					= torch.cat(ent2_output, dim=-1)
		
		rel_output 						= torch.tanh(rel_output)
		rel_logits 		  				= self.rel_classifier(rel_output)

		ent1_output 					= torch.tanh(ent1_output)
		ent1_logits 		  			= self.ent_classifier(ent1_output)
  
		ent2_output 					= torch.tanh(ent2_output)
		ent2_logits 		  			= self.ent_classifier(ent2_output)
		
		return rel_logits, ent1_logits, ent2_logits