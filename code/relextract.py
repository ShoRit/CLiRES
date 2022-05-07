from helper import *
from models import *
from dataloader import *
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import AdamW
import torch
import wandb
from igraph import * 


def seed_everything():
	SEED = args.seed
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	return device


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--src_lang",       help="choice of source language",   type=str, default='english')
	parser.add_argument("--tgt_lang",       help="choice of target language",   type=str, default='english')
	# parser.add_argument("--data_dir",       help="choice of target language",   type=str, default='/data/multilingual_KGQA/SMILER/data')

	# parser.add_argument("--data_dir",       help="choice of target language",   type=str, default='/data/multilingual_KGQA/IndoRE/data')
 
	parser.add_argument("--mode",      		help="choice of operation",       	type=str, default='eval')

	parser.add_argument("--max_seq_len",    help="maximum sequence length",  	type=int, default=512)
	parser.add_argument("--bert_model",     help="choice of bert model  ",  	type=str, default='mbert')
	parser.add_argument("--seed", 			help="random seed", 				type=int, default=11737)
	parser.add_argument("--gpu", 			help="choice of device", 			type=str, default='0')
 
	parser.add_argument("--gnn_depth", 		help="layers used in the gnn", 		type=int, default =5)
 
	parser.add_argument("--drop", 			help="dropout_used", 				type=float, default=0.2)
	parser.add_argument("--node_emb_dim", 	help="number of unseen classes", 	type=int, default=768)
	parser.add_argument("--ent_emb_dim", 	help="number of unseen classes", 	type=int, default=128)
	parser.add_argument("--dep", 			help="dependency_parsing", 			type=str, default='1')
	parser.add_argument("--el", 			help="dependency_parsing", 			type=str, default='1')
 
	parser.add_argument("--mtl", 		    help="amount of weightage for MTL", type=float, default=1.0)
	parser.add_argument("--lr", 		    help="learning rate", 				type=float, default=5e-6)
	parser.add_argument('--data', 			help='data', 						type=str, default='IndoRE')
 
	'''	
	In case we need to do ZS BERT for zero-shot relation extraction
	'''
	# parser.add_argument("--n_unseen", 		help="number of unseen classes", 	type=int, default=10)
	# parser.add_argument("--gamma", 			help="margin factor gamma", 		type=float, default=7.5)
	# parser.add_argument("--alpha", 			help="balance coefficient alpha", 	type=float, default=0.5)
	# parser.add_argument("--dist_func", 		help="distance computing function", type=str, default='cosine')
	# parser.add_argument("--num_neighbors", 										type=int, default=2)
	
	parser.add_argument("--batch_size", 										type=int, default=12)
	parser.add_argument("--epochs", 											type=int, default=50)
	parser.add_argument("--patience", 											type=int, default=5)

	args = parser.parse_args()

	return args


def seen_eval(model, loader, device):
	model.eval()
	correct, total = 0, 0
	y_true, y_pred = [],[]
 
	for data in tqdm(loader):
		tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb = [t.to(device) for t in data]
		bat 										= (tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb)
		
		with torch.no_grad():
			rel_logits, ent1_logits, ent2_logits    = model((bat))
		
		_, pred 				= 	torch.max(rel_logits, 1)
		_, labels 				= 	torch.max(label_ids, 1)
  
		y_pred.extend(list(np.array(pred.cpu().detach())))
		y_true.extend(list(np.array(labels.cpu().detach())))

	f1		  					= 	f1_score(y_true,y_pred, average="macro")
	p1, r1 						= 	precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro')
 
	return p1, r1, f1
 


def seen_predict(model, loader, device):
	model.eval()
	correct, total = 0, 0
	y_true, y_pred = [],[]
 
	for data in tqdm(loader):
		tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb = [t.to(device) for t in data]
		bat 										= (tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb)
		
		with torch.no_grad():
			rel_logits, ent1_logits, ent2_logits    = model((bat))
		
		_, pred 				= 	torch.max(rel_logits, 1)
		_, labels 				= 	torch.max(label_ids, 1)
  
		y_pred.extend(list(np.array(pred.cpu().detach())))
		y_true.extend(list(np.array(labels.cpu().detach())))

	return y_pred, y_true


def add_optimizer(model, train_len ):
	warmup_proportion 	= 0.05
	n_train_steps		= int(train_len/args.batch_size) * args.epochs
	num_warmup_steps	= int(float(warmup_proportion) * float(n_train_steps))
	param_optimizer		= list(model.named_parameters())
	param_optimizer		= [n for n in param_optimizer if 'pooler' not in n[0]]
	no_decay			= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr= args.lr)

	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_train_steps)
	scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
	return optimizer, scheduler


def main(args):
	device 												=   seed_everything()
	args.data_dir										=   f'/data/multilingual_KGQA/{args.data}/data'

	src_file 											= 	f'{args.data_dir}/{args.src_lang}_new_rels.dill'
	tgt_file 											= 	f'{args.data_dir}/{args.tgt_lang}_new_rels.dill'

	deprel_dict 										= 	load_deprels(enhanced=False)
	entity_dict             							= 	load_pickle(f'{args.data_dir}/ents.pkl')
	relation_dict          								= 	load_pickle(f'{args.data_dir}/rels.pkl')
 
	args.num_rels 										= 	len(relation_dict)
	args.num_ents 										= 	len(entity_dict)
 
	
	if check_file(src_file):
		src_dataset										=   load_dill(src_file)
	else:
		print('SRC FILE IS NOT CREATED'); exit()
  
	if check_file(tgt_file):
		tgt_dataset										=   load_dill(tgt_file)
	else:
		print('TGT FILE IS NOT CREATED'); exit()

	checkpoint_file 									=  f'../{args.data}_checkpoints/{args.src_lang}_{args.tgt_lang}-model_{args.bert_model}-dep_{args.dep}-gnn-depth_{args.gnn_depth}-el_{args.el}-mtl_{args.mtl}-seed_{args.seed}-lr_{args.lr}'
	
	if args.src_lang == args.tgt_lang:
		train_data, dev_data, test_data     			=   src_dataset['train']['rels'], src_dataset['dev']['rels'], src_dataset['test']['rels']
	else:
		train_data, dev_data, test_data     			=   src_dataset['train']['rels'], tgt_dataset['dev']['rels'], tgt_dataset['test']['rels']


	
 
	print('train size: {}, dev size {}, test size: {}'.format(len(train_data), len(dev_data), len(test_data)))
	print('Data is successfully loaded')
	
	model 												= MLRelClassifier(args)
	model       										= model.to(device)
 
	ce_loss 											= nn.CrossEntropyLoss()
	trainset    										= RelDataset(train_data, args)
	trainloader 										= DataLoader(trainset, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=True)
	model.train()
 
	optimizer 											= torch.optim.AdamW(model.parameters(), lr=args.lr)
	# optimizer, scheduler 								= add_optimizer(model, len(train_data))
 

	best_f1, best_model 								= 0, None
 
	if args.mode 										== 'train':
		wandb.login()
		wandb.init(project="ml-relation", entity="flow-graphs-cmu", name=f'{checkpoint_file.split("/")[-1]}')
	
		wandb.config = {
			"learning_rate" : args.lr,
			"epochs"		: args.epochs,
			"batch_size"	: args.batch_size,
			"node_vec"		: args.node_emb_dim,
			"dep"			: args.dep,
			"tgt_lang"		: args.tgt_lang
		}
  
		devset 											= RelDataset(dev_data, args)
		devloader     							   		= DataLoader(devset, batch_size=args.batch_size, collate_fn=create_mini_batch)
		kill_cnt 										= 0
		for epoch in range(args.epochs):
			print(f'============== TRAIN ON THE {epoch+1}-th EPOCH ==============')
			running_loss, correct, total = 0.0, 0, 0
   
			model.train()
			for data in tqdm(trainloader):
				tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb, _ = [t.to(device) for t in data]
				bat 									= (tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb, _)
				optimizer.zero_grad()
				
				rel_logits, ent1_logits, ent2_logits    = model((bat))
				rel_loss 								= (ce_loss(rel_logits.view(-1, args.num_rels), label_ids.float()))
				et1_loss 								= (ce_loss(ent1_logits.view(-1, args.num_ents), et_1.float()))
				et2_loss 								= (ce_loss(ent2_logits.view(-1, args.num_ents), et_2.float()))

				loss 									= rel_loss*args.mtl + (1-args.mtl)*et1_loss*0.5 + (1-args.mtl)*et2_loss*0.5

				wandb.log({"batch_loss": loss.item()})
	
				loss.backward()
				optimizer.step()
				running_loss += loss.item()


			print('============== EVALUATION ON DEV DATA ==============')

			wandb.log({"loss": running_loss})
			wandb.log({"data": args.data})

			pt, rt, f1t 	 = seen_eval(model, trainloader, device=device)
			print(f'Train data {f1t} \t Prec {pt} \t Rec {rt}')
			wandb.log({"train_f1": f1t})
			pt, rt, f1t 	 = seen_eval(model, devloader, device=device)
			wandb.log({"dev_f1": f1t})
			print(f'Eval data {f1t} \t Prec {pt} \t Rec {rt}')

			if f1t > best_f1:
				best_p, best_r, best_f1 = pt, rt, f1t
				wandb.log({"best_f1": best_f1})
				best_model = model
				kill_cnt = 0
				torch.save(best_model.state_dict(),checkpoint_file)
			else:
				kill_cnt +=1
				if kill_cnt >= args.patience:
					torch.save(best_model.state_dict(),checkpoint_file)
					break
			
			print(f'[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}')
		
		torch.save(best_model.state_dict(),checkpoint_file)
	
		testset 												= RelDataset(test_data, args)
		testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
		best_model.eval()
		pt, rt, test_f1	 										= seen_eval(best_model, testloader, device=device)
		wandb.log({"test_f1": test_f1})


	# Evaluation is done here. 

	if args.mode											== 'eval':
		checkpoint_file 									=  f'../{args.data}_checkpoints/{args.src_lang}_{args.tgt_lang}-model_{args.bert_model}-dep_{args.dep}-gnn-depth_{args.gnn_depth}-el_{args.el}-mtl_{args.mtl}-seed_{args.seed}-lr_{args.lr}'
		if not check_file(checkpoint_file) :
			print('MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS')
			return

		wandb.login()
		wandb.init(project="ml-relation-evaluation", entity="flow-graphs-cmu", name=f'{args.src_lang}_{args.tgt_lang}-dep_{args.dep}-el_{args.el}')
	
		wandb.config = {
			"learning_rate" : args.lr,
			"epochs"		: args.epochs,
			"batch_size"	: args.batch_size,
			"node_vec"		: args.node_emb_dim,
			"tgt_lang"		: args.tgt_lang,
		}		
  
  
		testset 												= RelDataset(test_data, args)
		testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
  
		model.load_state_dict(torch.load(checkpoint_file))
		model.eval()


		pt, rt, f1t 	 = seen_eval(model, testloader, device=device)
		print(f'Test data {f1t} \t Prec {pt} \t Rec {rt}')
	
		wandb.log({"f1": 		f1t})
		wandb.log({"precision": pt})
		wandb.log({"recall": 	rt})
		wandb.log({"seed": 		args.seed})
		wandb.log({"dep": 		args.dep})
		wandb.log({"el": 		args.el})
		wandb.log({"mtl": 		args.mtl})
		wandb.log({"bert": 		args.bert_model})
		wandb.log({"lr": 		args.lr})
		wandb.log({"src_lang": 	args.src_lang})
		wandb.log({"tgt_lang": 	args.tgt_lang})

  
	if args.mode												== 'batch_eval':
	 
		prec_arr, rec_arr, f1_arr								=  [],[],[] 
		wandb.login()
		wandb.init(project="ml-relation-evaluation", 			entity="flow-graphs-cmu", name=f'{args.src_lang}_{args.tgt_lang}-dep_{args.dep}-el_{args.el}-all')
	
		wandb.config = {
			"learning_rate" : args.lr,
			"epochs"		: args.epochs,
			"batch_size"	: args.batch_size,
			"node_vec"		: args.node_emb_dim,
			"tgt_lang"		: args.tgt_lang,
		}
		testset 												= RelDataset(test_data, args)
		testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
	
		res_arr 												= []
		for seed in range(0,5):
			checkpoint_file 								    =  f'../{args.data}_checkpoints/{args.src_lang}_{args.src_lang}-model_{args.bert_model}-dep_{args.dep}-gnn-depth_{args.gnn_depth}-el_{args.el}-mtl_{args.mtl}-seed_{seed}-lr_{args.lr}'
			print(checkpoint_file)
   
			if not check_file(checkpoint_file) :
				print('MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS')
				return
			model.load_state_dict(torch.load(checkpoint_file))
			model.eval()
			pt, rt, f1t 	 = seen_eval(model, testloader, device=device)
			# print(f'Test data {f1t} \t Prec {pt} \t Rec {rt}')
			res_arr.append((f1t,rt,pt))

		res_arr = sorted(res_arr, reverse=True)
		for elem in res_arr:
			f1t, rt, pt = elem
			prec_arr.append(pt); rec_arr.append(rt); f1_arr.append(f1t)
		

		wandb.log({"f1": 		np.mean(f1_arr)})
		wandb.log({"std_f1": 	np.std(f1_arr)})
		wandb.log({"precision": np.mean(prec_arr)})
		wandb.log({"recall": 	np.mean(rec_arr)})
		wandb.log({"seed": 		"all"})
		wandb.log({"dep": 		args.dep})
		wandb.log({"el": 		args.el})
		wandb.log({"mtl": 		args.mtl})
		wandb.log({"bert": 		args.bert_model})
		wandb.log({"lr": 		args.lr})
		wandb.log({"src_lang": 	args.src_lang})
		wandb.log({"tgt_lang": 	args.tgt_lang})
		wandb.log({"gnn_depth": args.gnn_depth})
		wandb.log({"data"	  : args.data})



  

	if args.mode											    == 'predict':	
		testset 												= RelDataset(test_data, args)
		testloader                                              = DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)

		pred_dict 		 	                                    = ddict(dict)
		for seed in range(0,3):
			checkpoint_file 									=  f'../{args.data}_checkpoints/{args.src_lang}_{args.src_lang}-model_{args.bert_model}-dep_{args.dep}-gnn-depth_{args.gnn_depth}-el_{args.el}-mtl_{args.mtl}-seed_{seed}-lr_{args.lr}'
			if not check_file(checkpoint_file) :
				print('MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS')
			model.load_state_dict(torch.load(checkpoint_file))
			model.eval()
			y_pred, y_true 	 = seen_predict(model, testloader, device=device)
			inv_relation_dict = {relation_dict[key]:key for key in relation_dict}

			bad_sents 	     = 0
			for idx in tqdm(range(0, len(test_data))):
				try:                
					dep_data 										  = test_data[idx]['dep_data']
					edges                                             = dep_data.edge_index.cpu().detach().numpy()
					n1_mask, n2_mask                                  = dep_data.n1_mask.cpu().detach().numpy(), dep_data.n2_mask.cpu().detach().numpy() 
					src_nodes                                         = np.where(n1_mask==1)[0]
					tgt_nodes                                         = np.where(n2_mask==1)[0]
					graph                                             = Graph(directed=False)
					graph.add_vertices(list(range(len(n1_mask))))
					graph.add_edges([(n1,n2) for n1,n2 in zip(edges[0],edges[1])])
					try:
						sp = min([min(path_len) for path_len in graph.shortest_paths(src_nodes, tgt_nodes)])
					except Exception as e:
						if len(src_nodes)>0 and len(tgt_nodes) >0 : sp =0
						else: raise AssertionError
					pred_dict[idx + seed*len(test_data)]['sent'] 		= test_data[idx]['sent']
					pred_dict[idx + seed*len(test_data)]['sent_len'] 	= len(test_data[idx]['tokens'])
					arg1_start, arg2_start 		= test_data[idx]['arg1_ids'].index(1), test_data[idx]['arg2_ids'].index(1) 
					arg1_end, arg2_end 			= arg1_start + np.sum(test_data[idx]['arg1_ids'])-1, arg2_start + np.sum(test_data[idx]['arg2_ids'])-1
					pred_dict[idx + seed*len(test_data)]['lex_dist']  = max(arg1_start, arg2_start) - min(arg1_end, arg2_end)
					pred_dict[idx + seed*len(test_data)]['pred_rel']  = inv_relation_dict[y_pred[idx]]
					pred_dict[idx + seed*len(test_data)]['true_rel']  = inv_relation_dict[y_true[idx]]
					pred_dict[idx + seed*len(test_data)]['dep_path']  = sp
					# pred_dict[idx + seed*len(test_data)]['syn_dist']  =                     # import pdb; pdb.set_trace()
					
				except Exception as e:
					print(e)
					bad_sents +=1
					continue

			print(bad_sents)
		pred_file 												= f'../predictions_{args.data}/{args.src_lang}_{args.tgt_lang}-dep_{args.dep}-el_{args.el}-gnn-depth_{args.gnn_depth}.pkl'
		dump_pickle(pred_dict, pred_file)
	



if __name__ =='__main__':	
	args                            =   get_args()
	main(args)