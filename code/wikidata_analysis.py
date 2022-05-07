from helper import *

def create_kg():
    
    if args.kg=='wiki_5M':
        jsonfiles = glob(f'{args.fact_dir}/*')
        
        ent_dict, rel_dict, fact_dict = ddict(set), ddict(set), {}
        bad_facts          = 0
        
        for jsonfile_name in jsonfiles:
            json_lines =  open(jsonfile_name).readlines()
            for obj in tqdm(json_lines):
                
                try:
                    curr_dict = json.loads(obj)
                    e1_id, rel_id, e2_id        = curr_dict['fact'].split(';')
                    e1_name, rel_name, e2_name  = curr_dict['fact_label']['en'].split(';')
                    e1_name, rel_name, e2_name  = e1_name.strip().lower(), rel_name.strip().lower(), e2_name.strip().lower()
                    
                    ent_dict[e1_id].add(e1_name)
                    ent_dict[e2_id].add(e2_name)
                    rel_dict[rel_id].add(rel_name)
                    if (e1_id, rel_id, e2_id) not in fact_dict:
                        fact_dict[(e1_id, rel_id, e2_id)]       = 1
                except Exception as e:
                    bad_facts +=1
                            
            print(f'Done for {jsonfile_name}')
    
    elif args.kg =='wiki_all':
        ent_dict, rel_dict, fact_dict = {}, {}, {}
        ents         = load_pickle(f'{args.wikikg_dir}/item_names.pickle')
        rels         = load_pickle(f'{args.wikikg_dir}/property_names.pickle')
        facts        = load_pickle(f'{args.wikikg_dir}/wikikb_sling.pickle')
        
        for ent in tqdm(ents):
            ent_dict[ent]= set(ents[ent])
        for rel in tqdm(rels):
            rel_dict[rel]= set(rels[rel])
        for ent in tqdm(facts):
            for elems in facts[ent]:
                rid, e2= elems[0], elems[1]
                if (ent,rid,e2) not in fact_dict:
                    fact_dict[(ent,rid,e2)] =1
            
    dump_pickle(ent_dict,   f'{args.wikikg_dir}/ent_dict.pkl')
    dump_pickle(rel_dict,   f'{args.wikikg_dir}/rel_dict.pkl')
    dump_pickle(fact_dict,  f'{args.wikikg_dir}/fact_dict.pkl')
    
    print(f'Entities {len(ent_dict)} \t Relations {len(rel_dict)} \t Facts {len(fact_dict)}')





def el_epitran():
    import epitran
    import epitran.reromanize
    
    wiki_ent_dict           =  load_pickle(f'{args.wikikg_dir}/ent_dict.pkl')
    wiki_rel_dict           =  load_pickle(f'{args.wikikg_dir}/rel_dict.pkl')
    rel_dir                 = f'{args.rel_dir}'
    languages               = ['bengali','hindi','telugu']
    
    lang_ent_dict           = ddict(set)
    for lang in languages:
        lines = open(f'{rel_dir}/{lang}_indore.tsv').readlines()
        for line in lines:
            e1_start,   e2_start    = line.index('<e1>')+4, line.index('<e2>')+4
            e1_end,     e2_end      = line.index('</e1>'), line.index('</e2>')
            e1_span,    e2_span     = line[e1_start:e1_end].strip(), line[e2_start:e2_end].strip()
            
            lang_ent_dict[lang].add(e1_span)
            lang_ent_dict[lang].add(e2_span)
    
    transliterated_ent_dict = ddict(lambda: ddict(set))
    for lang in languages:
        if lang =='bengali': 
            epi                  = epitran.Epitran('ben-Beng')
            rr                   = epitran.reromanize.ReRomanizer('ben-Beng', 'anglocentric')
        elif lang == 'hindi':
            epi                  = epitran.Epitran('hin-Deva')
            rr                   = epitran.reromanize.ReRomanizer('hin-Deva', 'anglocentric')
        elif lang == 'telugu':
            epi                  = epitran.Epitran('tel-Telu')
            rr                   = epitran.reromanize.ReRomanizer('tel-Telu', 'anglocentric')
            
        print(len(lang_ent_dict[lang]), len(set(lang_ent_dict[lang])))        
        ents                                    = list(lang_ent_dict[lang])
        for ent in tqdm(ents):
            phone_ent                           = rr.reromanize(ent)
            # phone_ent =  epi.transliterate(ent)
            transliterated_ent_dict[lang][ent]  = phone_ent
    
    
    def get_eng_phone(pid, ents):
        eng_epi                         = epitran.Epitran('eng-Latn')
        # eng_rr                          = epitran.reromanize.ReRomanizer('eng-Latn', 'anglocentric')
        phone_arr                       = []
        for count, ent in enumerate(ents):
            phone_ent                   = eng_epi.transliterate(ent)
            phone_arr.append((ent, phone_ent))
            if count % 1000 ==0:
                print(f'Done for {count} for {pid}', end='\r')                
        return phone_arr
    
    all_wiki_ents                   = {}
    
    sound_wiki_ents                 = {}
    for eid in tqdm(wiki_ent_dict):
        ents                        = list(wiki_ent_dict[eid])
        for ent in ents:
            all_wiki_ents[ent]      = eid
            # sound_wiki_ents[ent]    = eng_epi.transliterate(ent)
    
    sound_wiki_ents                 = load_pickle(f'{rel_dir}/sound_wiki_ents.pkl')
    '''
    wiki_ents_arr                   = list(all_wiki_ents.keys())
    num_procs                       = args.workers
    chunks                          = partition(wiki_ents_arr, num_procs)
    res_list                        = mergeList(Parallel(n_jobs = num_procs)(delayed(get_eng_phone)(i, chunk) for i, chunk in enumerate(chunks)))
    for elem in tqdm(res_list):
        sound_wiki_ents[elem[1]]   = all_wiki_ents[elem[0]]
    dump_pickle(sound_wiki_ents, f'{rel_dir}/sound_wiki_ents.pkl')
    '''
    all_sound_ents   =  [ent for ent in sound_wiki_ents]
    all_sound_ents   =  set(all_sound_ents)
        
    for lang in transliterated_ent_dict:
        transliterated_match = 0
        for orig_ent in transliterated_ent_dict[lang]:
            trans_ent = transliterated_ent_dict[lang][orig_ent]
            if trans_ent in all_wiki_ents:
                transliterated_match +=1
            # elif trans_ent in all_sound_ents:
            #     import pdb; pdb.set_trace()
            #     transliterated_match +=1
            #     break
                
        print(f'Lang {lang} EM: {transliterated_match} Frac: {transliterated_match/len(transliterated_ent_dict[lang])}')




def el_indic():
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    import phonetics
    from rapidfuzz import process, fuzz
    
    wiki_ent_dict           =  load_pickle(f'{args.wikikg_dir}/ent_dict.pkl')
    wiki_rel_dict           =  load_pickle(f'{args.wikikg_dir}/rel_dict.pkl')
    rel_dir                 =  f'{args.rel_dir}'
    languages               =  ['bengali','hindi','telugu']
    
    lang_ent_dict           = ddict(set)
    for lang in languages:
        lines = open(f'{rel_dir}/{lang}_indore.tsv').readlines()
        for line in lines:
            e1_start,   e2_start    = line.index('<e1>')+4, line.index('<e2>')+4
            e1_end,     e2_end      = line.index('</e1>'), line.index('</e2>')
            e1_span,    e2_span     = line[e1_start:e1_end].strip(), line[e2_start:e2_end].strip()
            
            lang_ent_dict[lang].add(e1_span)
            lang_ent_dict[lang].add(e2_span)
    
    transliterated_ent_dict = ddict(lambda: ddict(set))
    for lang in languages:
        if lang =='bengali': 
            from_script =  sanscript.BENGALI
        elif lang == 'hindi':
            from_script =  sanscript.DEVANAGARI
        elif lang == 'telugu':
            from_script = sanscript.TELUGU
            
        print(len(lang_ent_dict[lang]), len(set(lang_ent_dict[lang])))
        ents    = list(lang_ent_dict[lang])
        for ent in tqdm(ents):
            # trans_ent   =  transliterate(ent, from_script, sanscript.ITRANS)
            # transliterated_ent_dict[lang][ent].add(trans_ent.lower())
            
            # trans_ent =  transliterate(ent, from_script, sanscript.IAST)
            # transliterated_ent_dict[lang][ent].add(trans_ent.lower())
            
            # trans_ent =  transliterate(ent, from_script, sanscript.OPTITRANS)
            # transliterated_ent_dict[lang][ent].add(trans_ent.lower())
            
            trans_ent   =  transliterate(ent, from_script, sanscript.OPTITRANS)
            indian_ent  =  sanscript.SCHEMES[sanscript.OPTITRANS].to_lay_indian(trans_ent)
            transliterated_ent_dict[lang][ent].add(indian_ent.lower())
            
            
    all_wiki_ents     = {}
    sound_wiki_ents   = {}
    
    for eid in tqdm(wiki_ent_dict):
        ents          = list(wiki_ent_dict[eid])
        for ent in ents:
            all_wiki_ents[ent] = eid
            
    print(len(all_wiki_ents))
    
    wiki_ents          =  list(all_wiki_ents.keys())
    # all_sound_ents   =  [ent[0] for ent in sound_wiki_ents]
    # all_sound_ents.extend([ent[1] for ent in sound_wiki_ents])
    # all_sound_ents   = set(all_sound_ents)
    # print(len(all_sound_ents))
    
    sim_matches         = ddict(lambda: ddict(list))        
    
    for lang in transliterated_ent_dict:
        transliterated_match = 0
        for orig_ent in tqdm(transliterated_ent_dict[lang]):
            trans_ents = list(transliterated_ent_dict[lang][orig_ent])
            for trans_ent in trans_ents:
                # import pdb; pdb.set_trace()
                if trans_ent in all_wiki_ents:
                    transliterated_match +=1
                    sim_matches[lang][orig_ent].append((trans_ent,100,0))
                else:
                    sim_trans = process.extract(trans_ent, wiki_ents, scorer=fuzz.ratio, limit=100)
                    sim_matches[lang][orig_ent].extend(sim_trans)
                    
                    if sim_trans[0][1]>= 80:
                        transliterated_match +=1
                        
        print(f'Lang {lang} EM: {transliterated_match} Frac: {transliterated_match/len(transliterated_ent_dict[lang])}')
        
    import pdb; pdb.set_trace()
    dump_dill(sim_matches, f'matching_ents.dill')
    

def verify_quald_ids():
    # log_file                     =   open(f'{args.log_dir}/qald_data.log','w')
    
    ent_ids, rel_ids, other_ents =  ddict(list), ddict(list), ddict(list)
    all_ents, all_rels           = set(), set()
    
    for split in ['train','test']:
        jsonfile                 =   f'{args.qald_dir}/qald_9_plus_{split}_wikidata.json'
        data                     =   json.load(open(jsonfile))['questions']
        for elem in tqdm(data):
            sparql              =   elem['query']['sparql']
            ent_regex           =   r'http://www.wikidata.org/entity/Q[A-z0-9]+'
            rel_regex           =   r'http://www.wikidata.org/prop/[A-z]+/P[A-z0-9]+'
            wd_regex            =   r'wd:Q[A-z0-9]+'
            wdt_regex           =   r'wdt:P[A-z0-9]+'
            
            ents                =   re.findall(ent_regex, sparql)
            ents.extend(re.findall(wd_regex, sparql))
            rels                =   re.findall(rel_regex, sparql)
            rels.extend(re.findall(wdt_regex, sparql))
            ans_ents            =   get_recursively(elem,'value')
            ents.extend(ans_ents)
            
            qid                 =  f"{split}_{elem['id']}"
            for ent in ents:
                if isinstance(ent,dict):
                    other_ents[qid].append(ent)
                else:
                    if ent.startswith('http://www.wikidata.org/entity/'):
                        ent_id      = ent.split('/')[-1]
                    elif ent.startswith('wd'):
                        ent_id      = ent.split(':')[-1]    
                    ent_ids[qid].append(ent_id)
                    
            for rel in rels:
                if rel.startswith('http://www.wikidata.org/prop/'):
                    rel_id      = rel.split('/')[-1]
                elif rel.startswith('wdt'):
                    rel_id      = rel.split(':')[-1]
                rel_ids[qid].append(rel_id)
                
            # log_file.write('\n')
            # log_file.write(qid+'\n')            
            # log_file.write(sparql+'\n')
            # log_file.write(str(ent_ids[qid]))
            # log_file.write(str(rel_ids[qid]))
            # log_file.write('\n')
    
    ent_dict =  load_pickle(f'{args.wikikg_dir}/ent_dict.pkl')
    rel_dict =  load_pickle(f'{args.wikikg_dir}/rel_dict.pkl')
    
    
    for qid in ent_ids:
        for ent in ent_ids[qid]:
            if isinstance(ent,str): all_ents.add(ent)
        
    for qid in rel_ids:
        for rel in rel_ids[qid]:
            if isinstance(rel,str): all_rels.add(rel)
            
    abs_ents = all_ents - set(ent_dict.keys()) 
    abs_rels = all_rels - set(rel_dict.keys()) 
    
    print(f'Number of entities {len(all_ents)}')
    print(f'Number of relations {len(all_rels)}')
    print(f'Number of entities absent {len(abs_ents)}')
    print(f'Number of relations absent {len(abs_rels)}')
    import pdb; pdb.set_trace()
    


def analyse_indore():
    languages                   =  ['bengali','hindi','telugu','english']
    relation_dict               =  {}
    entity_dict                 =  {}
    
    for lang in tqdm(languages):
        lines = open(f'{args.rel_dir}/{lang}_indore.tsv').readlines()
        for line in lines:
            # import pdb; pdb.set_trace()
            # e1_start,   e2_start    = line.index('<e1>')+4, line.index('<e2>')+4
            # e1_end,     e2_end      = line.index('</e1>'), line.index('</e2>')
            # e1_span,    e2_span     = line[e1_start:e1_end].strip(), line[e2_start:e2_end].strip()
            
            rel, sent, ent_1, ent_2   = line.strip().split('\t')
            if rel not in relation_dict: 
                relation_dict[rel] = len(relation_dict)
            if ent_1 not in entity_dict:
                entity_dict[ent_1] = len(entity_dict)
            if ent_2 not in entity_dict:
                entity_dict[ent_2] = len(entity_dict)
    
    dump_pickle(entity_dict,    f'{args.rel_dir}/ents.pkl')
    dump_pickle(relation_dict,  f'{args.rel_dir}/rels.pkl')
    


def analyse_smiler():
    rel_dir                     = '/data/multilingual_KGQA/SMILER/data'
    files 						= os.listdir(f'{rel_dir}')
    languages 					= set([file.split('/')[-1][0:2] for file in files if file.endswith('tsv')])

    lang_dict                   = {}
    langs                       = []

    for lang in languages:
        lang_df 				= pd.read_csv(f'{rel_dir}/{lang}_corpora_train.tsv', sep='\t')
        curr_langs 				= list(lang_df['label'])
        langs.extend(curr_langs)
        
    for lang in langs:
        if lang == 'no_relation': continue
        if lang not in lang_dict: lang_dict[lang] = len(lang_dict)

    print(lang_dict)
    print(Counter(langs))

    test_langs                  = []
    test_lang_dict              = {}
    for lang in languages:
        lang_df 				= pd.read_csv(f'{rel_dir}/{lang}_corpora_test.tsv', sep='\t')
        curr_langs 				= list(lang_df['label'])
        test_langs.extend(curr_langs)
        
    for lang in test_langs:
        if lang == 'no_relation': continue
        if lang not in test_lang_dict: test_lang_dict[lang] = len(test_lang_dict)

    
def create_indore_data():
    from transformers import AutoTokenizer, AutoModel 
    import stanza, torch
    from torch_geometric.data import Data
    deprel_dict 						= load_deprels(enhanced=False)
    entity_dict             			= load_pickle(f'{args.rel_dir}/ents.pkl')
    relation_dict          				= load_pickle(f'{args.rel_dir}/rels.pkl')
    languages               			= ['te','en','hi']#,'english','bengali','hindi']
    lang_code 							= {'bn':'bengali','en':'english','hi':'hindi','te':'telugu'}
 
    ent_vec 							= np.random.randn(128)
 
    lang_sents_path						= f'{args.rel_dir}/lang_sents.dill'
    if os.path.exists(lang_sents_path):
        lines_dict 						= load_dill(lang_sents_path)
    else:
        lines_dict 						= ddict(lambda:ddict(list))
        for lang in tqdm(languages):
            lines 						= open(f'{args.rel_dir}/{lang_code[lang]}_indore.tsv').readlines()
            random.shuffle(lines)
            lines_dict[lang]['train'] 	= lines[0:int(len(lines)*args.train_ratio)]
            lines_dict[lang]['dev'] 	= lines[int(len(lines)*args.train_ratio):int(len(lines)*(args.train_ratio+args.dev_ratio))]
            lines_dict[lang]['test']	= lines[int(len(lines)*(args.train_ratio+args.dev_ratio)):]
        dump_dill(lines_dict, lang_sents_path)
        
    if args.bert_model      			== 'mbert':
        tokenizer         				= 	AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model 							= 	AutoModel.from_pretrained('bert-base-multilingual-uncased')

    if args.bert_model      			== 'xlmr':
        tokenizer         				= 	AutoTokenizer.from_pretrained('xlm-roberta-base')
        model 							= 	AutoModel.from_pretrained('xlm-roberta-base')
    
 
    relation_desc						= {}
    for rel in relation_dict:
        rel_toks 						= tokenizer(rel.replace('_',''), return_tensors='pt')
        relation_desc[rel]				= model(**rel_toks)['pooler_output'].cpu().squeeze(dim=1)

    
    for lang in languages:
        data   			        			= ddict(lambda: ddict(list))
        stanza_nlp 							= stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,depparse')
        for split in ['train','dev','test']:
            for line in tqdm(lines_dict[lang][split]):
                rel, sent, ent_1, ent_2 	= line.strip().split('\t')
                org_sent 					= sent
                e1_start,   e2_start    	= sent.index('<e1>')+4, sent.index('<e2>')+4
                e1_end,     e2_end      	= sent.index('</e1>'), sent.index('</e2>')
                e1_span,    e2_span     	= sent[e1_start:e1_end], sent[e2_start:e2_end]

                if e1_start > e2_end:
                    e2_start				= sent.index('<e2>')
                    sent					= sent.replace('<e2>','')
                    e2_end 					= sent.index('</e2>')
                    sent					= sent.replace('</e2>','')	
                    e1_start				= sent.index('<e1>')
                    sent					= sent.replace('<e1>','')
                    e1_end 					= sent.index('</e1>')
                    sent					= sent.replace('</e1>','')
                else:
                    e1_start				= sent.index('<e1>')
                    sent					= sent.replace('<e1>','')
                    e1_end 					= sent.index('</e1>')
                    sent					= sent.replace('</e1>','')
                    e2_start				= sent.index('<e2>')
                    sent					= sent.replace('<e2>','')
                    e2_end 					= sent.index('</e2>')
                    sent					= sent.replace('</e2>','')	
                
                assert e1_span == sent[e1_start:e1_end] and e2_span == sent[e2_start:e2_end]
    
                e1_map 											= IntervalMapping()				
                e2_map 											= IntervalMapping()
                e1_map[e1_start:e1_end]							= 1
                e2_map[e2_start:e2_end]							= 1
    
                sent_toks 										= tokenizer(sent, return_offsets_mapping=True, max_length=512)
                bert_toks 										= sent_toks['input_ids']
                tok_range 										= sent_toks['offset_mapping']
                
                e1_toks											= [0]+[1 if e1_map.contains(elem[0],elem[1]) else 0 for elem in tok_range[1:-1]]+[0]
                e2_toks 										= [0]+[1 if e2_map.contains(elem[0],elem[1]) else 0 for elem in tok_range[1:-1]]+[0]
                e1_type, e2_type 								= np.zeros(len(entity_dict)), np.zeros(len(entity_dict))
                e1_type[entity_dict[ent_1]] 					= 1 
                e2_type[entity_dict[ent_2]] 					= 1 
                rel_type 										= np.zeros(len(relation_dict))
                rel_type[relation_dict[rel]]					= 1
    
                node_dict 										= {}
                node_idx_dict 									= {}
                node_mask_dict 									= {}
                edge_arr 										= []
                dep_arr 										= []
    
                # Specifically for the root that is attached to the main verb
                # STAR NODE
                node_dict[(-1,-1)]								= 0
                node_idx_dict[(-1,-1)]							= (1,len(bert_toks)-1)
                node_mask_dict[(-1,-1)]							= 0
            
                dep_doc 										= stanza_nlp(sent)	
                num_sents 										= len(dep_doc.sentences)
    
                for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
                    for word in dep_sent.words:
                        if     e1_map.contains(word.start_char, word.end_char) and  e2_map.contains(word.start_char, word.end_char): dep_val =3
                        elif   e2_map.contains(word.start_char, word.end_char): dep_val =2
                        elif   e1_map.contains(word.start_char, word.end_char): dep_val =1 
                        else: dep_val =0
                        # if 		word.start_char	>= e1_start and word.end_char <= e1_end: dep_val = 1
                        # elif	word.start_char >= e2_start and word.end_char <= e2_end: dep_val = 2
                        # else:   dep_val = 0
                        dep_arr.append(((sent_cnt, word.id), word.text, (sent_cnt, word.head), word.deprel, word.start_char, word.end_char, dep_val))
    
                
                last_six, last_eix 							= 1,1
                for elem in dep_arr:
                    start_idx, word, end_idx, deprel, start_char, end_char, mask_val  = elem
                    curr_map 								= IntervalMapping()
                    curr_map[start_char:end_char]			= 1
                    if start_idx not in node_dict		:node_dict[start_idx]= len(node_dict)
                    if end_idx   not in node_dict		:node_dict[end_idx]  = len(node_dict)
    
                    start_flag 								= False
                    start_tok_idx = 1; end_tok_idx =1;
                    for idx in range(start_tok_idx, len(tok_range)-1):
                        curr_start, curr_end 				= tok_range[idx][0], tok_range[idx][1]
                        if curr_map.contains(curr_start,curr_end):
                            if start_flag is False:
                                start_tok_idx 				= idx
                                start_flag 					= True
                            end_tok_idx 					= idx +1
                        
                        # if 		curr_end 	== 0  and idx ==len(tok_range)-1 : end_tok_idx =idx; break
                        # elif 	curr_end 	<= start_char	: start_tok_idx = idx +1; continue
                        # elif 	curr_end	<= end_char		: continue
                        # elif	curr_start 	>= end_char		: end_tok_idx = idx; break
                        
                    # if 	idx == len(tok_range) -2		: end_tok_idx = idx+1
                    # if 	idx == len(tok_range) -1		: end_tok_idx = idx+1
                    if start_tok_idx == 1 and end_tok_idx == 1: 
                        start_tok_idx, end_tok_idx 			= last_six, last_eix
                    node_idx_dict[start_idx]				= (start_tok_idx, end_tok_idx)

                    if ':' in deprel:
                        deprel 								= deprel.split(':')[0] 
                    # edge_index[0].append(start_idx)
                    # edge_index[1].append(end_idx)
                    # edge_type.append(deprel_dict[deprel])
                    edge_arr.append((start_idx, end_idx,deprel_dict[deprel]))
                    node_mask_dict[start_idx]				= mask_val
                    last_six, last_eix 						= start_tok_idx, end_tok_idx
                    # start_tok_idx 		 			= end_tok_idx
    
                ################ CREATED A GLOBAL GRAPH ACROSS ALL SENTENCES #######################

                for sent_num in range(num_sents):
                    tok_idxs 			= [node_idx_dict[elem] for elem in node_idx_dict if elem[0]== sent_num]
                    min_tok_idx 		= min([tok_idx[0] for tok_idx in tok_idxs])
                    max_tok_idx 		= max([tok_idx[1] for tok_idx in tok_idxs])

                    node_idx_dict[(sent_num,0)]					= (min_tok_idx, max_tok_idx)
                    node_mask_dict[(sent_num,0)]				= 0
    
                x, edge_index, edge_type, n1_mask, n2_mask		= [],[[],[]],[],[],[]
                for node in node_dict:
                    six, eix 									= node_idx_dict[node]
                    temp_ones 									= torch.ones((512,))*-torch.inf

                    try:
                        assert six < eix
                    except Exception as e:
                        import pdb; pdb.set_trace()
                    temp_ones[six:eix]							= 0
                    x.append(temp_ones)
    
                    mask = node_mask_dict[node]
                    if mask == 0: n1_mask.append(0); n2_mask.append(0)
                    if mask == 1: n1_mask.append(1); n2_mask.append(0)
                    if mask == 2: n1_mask.append(0); n2_mask.append(1)
                    if mask == 3: n2_mask.append(1); n1_mask.append(1)

                for edge in edge_arr:
                    n1, n2, rel_idx 							= edge
                    edge_index[0].append(node_dict[n1])
                    edge_index[1].append(node_dict[n2])
                    edge_type.append(rel_idx)
    
                for sent_num in range(num_sents):
                    edge_index[0].append(node_dict[(sent_num,0)])
                    edge_index[1].append(node_dict[(-1,-1)])
                    edge_type.append(deprel_dict['STAR'])

                try:
                    assert sum(n1_mask)> 0 and sum(n2_mask) >0
                except Exception as e:
                    import pdb; pdb.set_trace()
     
                try:
                    x, edge_index, edge_type, n1_mask, n2_mask		= torch.stack(x, dim=0), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
                    dep_data 										= Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
                except Exception as e:
                    import pdb; pdb.set_trace()



                data[split]['rels'].append({
                    'tokens'	: bert_toks,
                    'tok_range'	: tok_range,	
                    'arg1_ids'	: e1_toks,
                    'arg2_ids'	: e2_toks,
                    'arg1_type' : e1_type,
                    'arg2_type' : e2_type,
                    'desc_emb'	: relation_desc[rel],
                    'label'		: rel_type,
                    'dep_data'	: dep_data,
                    'sent'		: sent,
                    'orig_sent'	: org_sent,
                    'arg1_emb'	: ent_vec,
                    'arg2_emb'	: ent_vec,
                    'info'		: (e1_span, e2_span),
                    'ent_data'	: []
                })
                # import pdb; pdb.set_trace()
        dump_dill(data, f'{args.rel_dir}/{lang}_rels.dill')



def add_ent_data():
    import torch
    from torch_geometric.data import Data
    rand_vec 							= np.random.randn(128)
    data_dir    = args.rel_dir
    langs       = ['hindi','english','bengali','telugu']
    lang_codes 	= {'hi':'hindi', 'en':'english','te': 'telugu'}

    split_langs 	=    f'{data_dir}/lang_sents.dill'
    uniq_dict       =   ddict(dict)
    rel_dict        =   {}

    for line in open(f'{data_dir}/kge_rels.del').readlines():
        id, rel_name =  line.strip().split('\t')   
        rel_dict[rel_name] = int(id)

    ent_embs        =   load_pickle(f'{data_dir}/entity_embeddings.pkl')

    for lang in tqdm(lang_codes):
        cnt                                     =    0
        lines 		                            =    open(f'{data_dir}/{lang_codes[lang]}_indore.tsv').readlines()
        graph_data                              =    open(f'{data_dir}/{lang_codes[lang]}_indore_el_graph.json').readlines()
        
        for line, elem in tqdm(zip(lines, graph_data)):
            elem                                =    json.loads(elem)
            rel, sent, ent_1, ent_2             =    line.strip().split('\t')
            graph_data                          =    elem['graph']
            graph_ent1, graph_ent2              =    elem['entity1'], elem['entity2']
            uniq_dict[lang][sent]               =    (elem['graph'], elem['entity1'],elem['entity2'], elem['nodes'])
        
    
    for lang in lang_codes:
        rels_data = load_dill(f'{data_dir}/{lang}_rels.dill')
        lang_cnts = 0
        data   			        			    = ddict(lambda: ddict(list))
        for split in ['train','test','dev']:
            for elem in tqdm(rels_data[split]['rels']):
                curr_data                = elem
                sent					 = elem['orig_sent']
                if sent not in uniq_dict[lang]: 
                    lang_cnts+=1
                
                x, edge_index, edge_type, n1_mask, n2_mask          = [], [[],[]], [], [],[]
                graph_data, e1, e2, nodes = uniq_dict[lang][sent]
                
                node_dict     = {}
                for i, node in enumerate(nodes): node_dict[node]    = i
                
                if e1 in ent_embs: 
                    curr_data['arg1_emb'] = torch.FloatTensor(ent_embs[e1])
                else:
                    curr_data['arg1_emb'] = torch.tensor(rand_vec).float()
                                                     
                if e2 in ent_embs: 
                    curr_data['arg2_emb'] = torch.FloatTensor(ent_embs[e2])
                else:
                    curr_data['arg2_emb'] = torch.tensor(rand_vec).float()
   
                x             = [torch.FloatTensor(ent_embs[n]) for n in nodes]
                for item in graph_data:
                    n1, rel, n2 = item
                    edge_index[0].append(node_dict[n1])
                    edge_index[1].append(node_dict[n2])
                    edge_type.append(rel_dict[rel])
                    
                n1_mask       = [1 if n==e1 else 0 for n in nodes ]
                n2_mask       = [1 if n==e2 else 0 for n in nodes ]
                
                if len(x) == 0:
                    x       = [torch.FloatTensor(curr_data['arg1_emb']),torch.FloatTensor(curr_data['arg2_emb'])]
                    n1_mask = [1,0]
                    n2_mask = [0,1]
                
                x, edge_index, edge_type, n1_mask, n2_mask		= torch.stack(x, dim=0), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
                ent_data 										= Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
                curr_data['ent_data']                           = ent_data
                
                data[split]['rels'].append(curr_data)
                
        dump_dill(data, f'{data_dir}/{lang}_ent_rels.dill')






        
# def create_indore_el_data():
#     from transformers import AutoTokenizer, AutoModel 
    
#     rand_vec 							= np.random.randn(128)
 
#     import stanza, torch
#     from torch_geometric.data import Data
#     deprel_dict 						= load_deprels(enhanced=False)
#     entity_dict             			= load_pickle(f'{args.rel_dir}/ents.pkl')
#     relation_dict          				= load_pickle(f'{args.rel_dir}/rels.pkl')
#     ent_embs_dict 						= load_pickle(f'{args.rel_dir}/ent_embs.pkl')
#     # languages 					 	= ['te']
#     languages               			= ['te','en','hi']#,'english','bengali','hindi']
#     lang_code 							= {'bn':'bengali','en':'english','hi':'hindi','te':'telugu'}
#     # lang_code 						= {'bengali':'bn','english':'en','hindi':'hi','telugu':'te'}
#     # languages 						= ['bengali']
#     lang_sents_path						= f'{args.rel_dir}/lang_el_sents.dill'
#     if os.path.exists(lang_sents_path):
#         lines_dict 						= load_dill(lang_sents_path)
#     else:
#         lines_dict 						= ddict(lambda:ddict(list))
#         for lang in tqdm(languages):
#             lines 						= open(f'{args.rel_dir}/{lang_code[lang]}_indore.tsv').readlines()
#             random.shuffle(lines)
#             lines_dict[lang]['train'] 	= lines[0:int(len(lines)*args.train_ratio)]
#             lines_dict[lang]['dev'] 	= lines[int(len(lines)*args.train_ratio):int(len(lines)*(args.train_ratio+args.dev_ratio))]
#             lines_dict[lang]['test']	= lines[int(len(lines)*(args.train_ratio+args.dev_ratio)):]
#         dump_dill(lines_dict, lang_sents_path)
        
#     if args.bert_model      			== 'mbert':
#         tokenizer         				= 	AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
#         model 							= 	AutoModel.from_pretrained('bert-base-multilingual-uncased')

#     if args.bert_model     				== 'xlmr':
#         tokenizer         				= 	AutoTokenizer.from_pretrained('xlm-roberta-base')
#         model 							= 	AutoModel.from_pretrained('xlm-roberta-base')
    
 
#     relation_desc						= {}
#     for rel in relation_dict:
#         rel_toks 						= tokenizer(rel.replace('_',''), return_tensors='pt')
#         relation_desc[rel]				= model(**rel_toks)['pooler_output'].cpu().squeeze(dim=1)

    
#     for lang in languages:
#         data   			        			= ddict(lambda: ddict(list))
#         if lang								!='bengali': 
#             stanza_nlp 						= stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,depparse')
   
#         for split in ['train','dev','test']:
#             for line in tqdm(lines_dict[lang][split]):
#                 rel, sent, ent_1, ent_2 	= line.strip().split('\t')
#                 orig_sent 					= sent

#                 e1_start,   e2_start    	= sent.index('<e1>')+4, sent.index('<e2>')+4
#                 e1_end,     e2_end      	= sent.index('</e1>'), sent.index('</e2>')
#                 e1_span,    e2_span     	= sent[e1_start:e1_end], sent[e2_start:e2_end]

#                 if e1_start > e2_end:
#                     e2_start				= sent.index('<e2>')
#                     sent					= sent.replace('<e2>','')
#                     e2_end 					= sent.index('</e2>')
#                     sent					= sent.replace('</e2>','')	
#                     e1_start				= sent.index('<e1>')
#                     sent					= sent.replace('<e1>','')
#                     e1_end 					= sent.index('</e1>')
#                     sent					= sent.replace('</e1>','')
#                 else:
#                     e1_start				= sent.index('<e1>')
#                     sent					= sent.replace('<e1>','')
#                     e1_end 					= sent.index('</e1>')
#                     sent					= sent.replace('</e1>','')
#                     e2_start				= sent.index('<e2>')
#                     sent					= sent.replace('<e2>','')
#                     e2_end 					= sent.index('</e2>')
#                     sent					= sent.replace('</e2>','')	
                
#                 assert e1_span == sent[e1_start:e1_end] and e2_span == sent[e2_start:e2_end]

#                 # sent_toks 									= tokenizer(sent, return_offsets_mapping=True, max_length=512,add_special_tokens=False)
#                 sent_toks 										= tokenizer(sent, return_offsets_mapping=True, max_length=512)
#                 bert_toks 										= sent_toks['input_ids']
#                 tok_range 										= sent_toks['offset_mapping']
#                 e1_toks											= [0]+[1 if elem[0]>=e1_start and elem[1]<=e1_end else 0 for elem in tok_range[1:-1]]+[0]
#                 e2_toks 										= [0]+[1 if elem[0]>=e2_start and elem[1]<=e2_end else 0 for elem in tok_range[1:-1]]+[0]
#                 e1_type, e2_type 								= np.zeros(len(entity_dict)), np.zeros(len(entity_dict))
#                 e1_type[entity_dict[ent_1]] 					= 1 
#                 e2_type[entity_dict[ent_2]] 					= 1 
#                 rel_type 										= np.zeros(len(relation_dict))
#                 rel_type[relation_dict[rel]]					= 1
    
#                 node_dict 										= {}
#                 node_idx_dict 									= {}
#                 node_mask_dict 									= {}
#                 edge_arr 										= []
#                 dep_arr 										= []
    
#                 # Specifically for the root that is attached to the main verb
#                 # STAR NODE
#                 node_dict[(-1,-1)]								= 0
#                 # node_idx_dict[(-1,-1)]						= (0,len(bert_toks))
#                 node_idx_dict[(-1,-1)]							= (1,len(bert_toks)-1)
#                 node_mask_dict[(-1,-1)]							= 0
    
                
#                 if lang == 'bn': continue;
#                     # x, edge_index, edge_type, n1_mask, n2_mask		= torch.FloatTensor(np.array(x)), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
#                     # dep_data 										= Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
#                 else:				
#                     dep_doc 										= stanza_nlp(sent)	
#                     num_sents 										= len(dep_doc.sentences)
     
#                     for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
#                         for word in dep_sent.words:
#                             if 		word.start_char	>= e1_start and word.end_char <= e1_end: dep_val = 1
#                             elif	word.start_char >= e2_start and word.end_char <= e2_end: dep_val = 2
#                             else:   dep_val = 0
#                             dep_arr.append(((sent_cnt, word.id), word.text, (sent_cnt, word.head), word.deprel, word.start_char, word.end_char, dep_val))
     
#                     start_tok_idx = 1; end_tok_idx =1;
     
#                     for elem in dep_arr:
#                         start_idx, word, end_idx, deprel, start_char, end_char, mask_val  = elem
#                         if start_idx not in node_dict		:node_dict[start_idx]= len(node_dict)
#                         if end_idx   not in node_dict		:node_dict[end_idx]  = len(node_dict)
      
#                         # for idx in range(start_tok_idx, len(tok_range)):
#                         for idx in range(start_tok_idx, len(tok_range)):
#                             curr_start, curr_end 				= tok_range[idx][0], tok_range[idx][1]
#                             if 		curr_end 	== 0  and idx ==len(tok_range)-1 : end_tok_idx =idx; break
#                             elif 	curr_end 	<= start_char	: start_tok_idx = idx +1; continue
#                             elif 	curr_end	<= end_char		: continue
#                             elif	curr_start 	>= end_char		: end_tok_idx = idx; break
                            
#                         # if 	idx == len(tok_range) -2		: end_tok_idx = idx+1
#                         # if 	idx == len(tok_range) -1		: end_tok_idx = idx+1
      
#                         node_idx_dict[start_idx]			= (start_tok_idx, end_tok_idx)
#                         if ':' in deprel:
#                             deprel 							= deprel.split(':')[0] 
#                         # edge_index[0].append(start_idx)
#                         # edge_index[1].append(end_idx)
#                         # edge_type.append(deprel_dict[deprel])
#                         edge_arr.append((start_idx, end_idx,deprel_dict[deprel]))
#                         node_mask_dict[start_idx]		= mask_val
#                         start_tok_idx 		 			= end_tok_idx
      
#                     ################ CREATED A GLOBAL GRAPH ACROSS ALL SENTENCES #######################

#                     for sent_num in range(num_sents):
#                         tok_idxs 			= [node_idx_dict[elem] for elem in node_idx_dict if elem[0]== sent_num]
#                         min_tok_idx 		= min([tok_idx[0] for tok_idx in tok_idxs])
#                         max_tok_idx 		= max([tok_idx[1] for tok_idx in tok_idxs])

#                         node_idx_dict[(sent_num,0)]					= (min_tok_idx, max_tok_idx)
#                         node_mask_dict[(sent_num,0)]				= 0
      
#                     x, edge_index, edge_type, n1_mask, n2_mask	= [],[[],[]],[],[],[]
#                     for node in node_dict:
#                         six, eix 									= node_idx_dict[node]
#                         temp_ones 									= torch.ones((512,))*-torch.inf
      
#                         if six < eix	: temp_ones[six:eix]=0
#                         elif six == eix : temp_ones[six]	=0
#                         else: import pdb; pdb.set_trace()
#                         x.append(temp_ones)
      
#                         mask = node_mask_dict[node]
#                         if mask == 0: n1_mask.append(0); n2_mask.append(0)
#                         if mask == 1: n1_mask.append(1); n2_mask.append(0)
#                         if mask == 2: n1_mask.append(0); n2_mask.append(1)

#                     for edge in edge_arr:
#                         n1, n2, rel_idx 							= edge
#                         edge_index[0].append(node_dict[n1])
#                         edge_index[1].append(node_dict[n2])
#                         edge_type.append(rel_idx)
      
#                     for sent_num in range(num_sents):
#                         edge_index[0].append(node_dict[(sent_num,0)])
#                         edge_index[1].append(node_dict[(-1,-1)])
#                         edge_type.append(deprel_dict['STAR'])
      
#                     try:
#                         x, edge_index, edge_type, n1_mask, n2_mask		= torch.stack(x, dim=0), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
#                         dep_data 										= Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
#                     except Exception as e:
#                         import pdb; pdb.set_trace()

#                 if orig_sent in ent_embs_dict[lang]:
#                     ent1_emb 										= ent_embs_dict[lang][orig_sent][0]
#                     ent2_emb 										= ent_embs_dict[lang][orig_sent][1]
#                     if isinstance(ent1_emb, int)                    : ent1_emb = rand_vec
#                     if isinstance(ent2_emb, int) 					: ent2_emb = rand_vec
#                 else:
#                     ent1_emb = rand_vec; ent2_emb = rand_vec					

#                 data[split]['rels'].append({
#                     'tokens'	: bert_toks,
#                     'tok_range'	: tok_range,	
#                     'arg1_ids'	: e1_toks,
#                     'arg2_ids'	: e2_toks,
#                     'arg1_type' : e1_type,
#                     'arg2_type' : e2_type,
#                     'desc_emb'	: relation_desc[rel],
#                     'label'		: rel_type,
#                     'dep_data'	: dep_data,
#                     'sent'		: sent,
#                     'orig_sent'	: orig_sent, 
#                     'arg1_emb'	: ent1_emb,
#                     'arg2_emb'	: ent2_emb
#                 })
#         dump_dill(data, f'{args.rel_dir}/{lang}_el_rels.dill')






    
def create_smiler_data():
    from transformers import AutoTokenizer, AutoModel 
    import stanza, torch
    from torch_geometric.data import Data
    from sklearn.model_selection import train_test_split
 
    deprel_dict 						= load_deprels(enhanced=False)
    relation_dict          				= load_pickle(f'{args.rel_dir}/rels.pkl')
    entity_dict             			= load_pickle(f'{args.rel_dir}/ents.pkl')
    files 								= os.listdir(f'{args.rel_dir}')
    languages 							= set([file.split('/')[-1][0:2] for file in files if file.endswith('tsv')])
    
    ent_vec 							= np.random.randn(128)
 
    lang_sents_path						= f'{args.rel_dir}/lang_dfs.dill'
    if os.path.exists(lang_sents_path):
        lines_dict 						= load_dill(lang_sents_path)
    else:
        lines_dict 						= ddict(lambda:ddict(list))
        for lang in tqdm(languages):
            print(f'INSIDE THIS FUNCTION FOR LANG {lang}')
            df 							= pd.read_csv(f'{args.rel_dir}/{lang}_corpora_train.tsv', sep='\t')
            if len(df)>10000:
                df 						= df.sample(n=10000)
            else:
                df						= df.sample(frac=1)
            
            train_df					= df[:int(len(df)*(args.train_ratio))]
            dev_df 						= df[int(len(df)*(args.train_ratio)):int(len(df)*(args.train_ratio+args.dev_ratio))]
            test_df 				 	= df[int(len(df)*(args.train_ratio+args.dev_ratio)):]
            # train_df, dev_df 			= train_test_split(train_df, test_size=1-args.train_ratio)
            test_df2 					= pd.read_csv(f'{args.rel_dir}/{lang}_corpora_test.tsv', sep='\t')
            test_df						= pd.concat([test_df,test_df2])
   
            lines_dict[lang]['train'] 	= train_df
            lines_dict[lang]['dev'] 	= dev_df
            lines_dict[lang]['test']	= test_df
   
        dump_dill(lines_dict, lang_sents_path)
    
    if args.bert_model      == 'mbert':
        tokenizer         	= 	AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model 				= 	AutoModel.from_pretrained('bert-base-multilingual-uncased')

    if args.bert_model      == 'xlmr':
        tokenizer         	= 	AutoTokenizer.from_pretrained('xlm-roberta-base')
        model 				= 	AutoModel.from_pretrained('xlm-roberta-base')
    
    relation_desc			= {}
    for rel in relation_dict:
        rel_toks 			= tokenizer(rel.replace('_',''), return_tensors='pt')
        relation_desc[rel]	= model(**rel_toks)['pooler_output'].cpu().squeeze(dim=1)

    done_langs 				= [file.split('/')[-1][0:2] for file in os.listdir(f'{args.rel_dir}') if file.endswith('_new_rels.dill')]
    languages				= args.lang_list.split(',')
    for lang in languages:
        if lang in done_langs: continue
        data   			        		= ddict(lambda: ddict(list))
        stanza_nlp 						= stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,depparse')
   
        for split in ['train','dev','test']:
            for index, row in tqdm(lines_dict[lang][split].iterrows(), total=lines_dict[lang][split].shape[0]):
                # print(f"Done for {index}/{len(lines_dict[lang][split])} ", end='\r')
                rel, sent 					= row['label'], row['text']
                org_sent 					= sent
                try:
                    e1_start,   e2_start    = sent.index('<e1>')+4, sent.index('<e2>')+4
                    e1_end,     e2_end      = sent.index('</e1>'), sent.index('</e2>')
                    e1_span,    e2_span     = sent[e1_start:e1_end], sent[e2_start:e2_end]
                    if e1_start == e1_end  or e2_start ==e2_end: continue;
                except Exception as e:
                    continue
    
                if e1_start > e2_end:
                    e2_start				= sent.index('<e2>')
                    sent					= sent.replace('<e2>','')
                    e2_end 					= sent.index('</e2>')
                    sent					= sent.replace('</e2>','')	
                    e1_start				= sent.index('<e1>')
                    sent					= sent.replace('<e1>','')
                    e1_end 					= sent.index('</e1>')
                    sent					= sent.replace('</e1>','')
                else:
                    e1_start				= sent.index('<e1>')
                    sent					= sent.replace('<e1>','')
                    e1_end 					= sent.index('</e1>')
                    sent					= sent.replace('</e1>','')
                    e2_start				= sent.index('<e2>')
                    sent					= sent.replace('<e2>','')
                    e2_end 					= sent.index('</e2>')
                    sent					= sent.replace('</e2>','')	
                
                assert e1_span == sent[e1_start:e1_end] and e2_span == sent[e2_start:e2_end]

                e1_map 											= IntervalMapping()				
                e2_map 											= IntervalMapping()
                e1_map[e1_start:e1_end]							= 1
                e2_map[e2_start:e2_end]							= 1

                sent_toks 										= tokenizer(sent, return_offsets_mapping=True, max_length=512)
                bert_toks 										= sent_toks['input_ids']
                tok_range 										= sent_toks['offset_mapping']

                e1_toks											= [0]+[1 if e1_map.contains(elem[0],elem[1]) else 0 for elem in tok_range[1:-1]]+[0]
                e2_toks 										= [0]+[1 if e2_map.contains(elem[0],elem[1]) else 0 for elem in tok_range[1:-1]]+[0]
    
                rel_type 										= np.zeros(len(relation_dict))
                rel_type[relation_dict[rel]]					= 1

                node_dict 										= {}
                node_idx_dict 									= {}
                node_mask_dict 									= {}
                edge_arr 										= []
                dep_arr 										= []
    
                # Specifically for the root that is attached to the main verb
                # STAR NODE
                node_dict[(-1,-1)]								= 0
                node_idx_dict[(-1,-1)]							= (1,len(bert_toks)-1)
                node_mask_dict[(-1,-1)]							= 0
            
                dep_doc 										= stanza_nlp(sent)	
                num_sents 										= len(dep_doc.sentences)
    
                for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
                    last_start_char 							= 0
                    for word in dep_sent.words:
                        try:
                            if word.start_char is None or word.end_char is None:
                                start_char , end_char 			= last_start_char, last_start_char + len(word.text)
                            else:
                                start_char,  end_char 			= word.start_char, word.end_char
        
                            # if 		start_char	>= e1_start and end_char <= e1_end: dep_val = 1
                            # elif	start_char  >= e2_start and end_char <= e2_end: dep_val = 2
                            # else:   dep_val = 0
                            if     e1_map.contains(start_char, end_char) and  e2_map.contains(start_char,end_char): dep_val =3
                            elif   e2_map.contains(start_char, end_char): dep_val =2
                            elif   e1_map.contains(start_char, end_char): dep_val =1 
                            else: dep_val =0
                            
                            dep_arr.append(((sent_cnt, word.id), word.text, (sent_cnt, word.head), word.deprel, start_char, end_char, dep_val))
                            last_start_char 					= end_char
                        except Exception as e:
                            import pdb; pdb.set_trace()
                
                last_six, last_eix 							= 1,1
                for elem in dep_arr:
                    start_idx, word, end_idx, deprel, start_char, end_char, mask_val  = elem
                    curr_map 								= IntervalMapping()
                    curr_map[start_char:end_char]			= 1
                    if start_idx not in node_dict		:node_dict[start_idx]= len(node_dict)
                    if end_idx   not in node_dict		:node_dict[end_idx]  = len(node_dict)
    
                    start_flag 								= False
                    start_tok_idx = 1; end_tok_idx =1;
                    for idx in range(start_tok_idx, len(tok_range)-1):
                        curr_start, curr_end 				= tok_range[idx][0], tok_range[idx][1]
                        if curr_map.contains(curr_start,curr_end):
                            if start_flag is False:
                                start_tok_idx 				= idx
                                start_flag 					= True
                            end_tok_idx 					= idx +1
                        
                        # if 		curr_end 	== 0  and idx ==len(tok_range)-1 : end_tok_idx =idx; break
                        # elif 	curr_end 	<= start_char	: start_tok_idx = idx +1; continue
                        # elif 	curr_end	<= end_char		: continue
                        # elif	curr_start 	>= end_char		: end_tok_idx = idx; break
                        
                    # if 	idx == len(tok_range) -2		: end_tok_idx = idx+1
                    # if 	idx == len(tok_range) -1		: end_tok_idx = idx+1
                    if start_tok_idx == 1 and end_tok_idx == 1: 
                        start_tok_idx, end_tok_idx 			= last_six, last_eix
                    node_idx_dict[start_idx]				= (start_tok_idx, end_tok_idx)

                    if ':' in deprel:
                        deprel 								= deprel.split(':')[0] 
                    # edge_index[0].append(start_idx)
                    # edge_index[1].append(end_idx)
                    # edge_type.append(deprel_dict[deprel])
                    edge_arr.append((start_idx, end_idx,deprel_dict[deprel]))
                    node_mask_dict[start_idx]				= mask_val
                    last_six, last_eix 						= start_tok_idx, end_tok_idx
                    # start_tok_idx 		 			= end_tok_idx
    
                ################ CREATED A GLOBAL GRAPH ACROSS ALL SENTENCES #######################

                for sent_num in range(num_sents):
                    tok_idxs 			= [node_idx_dict[elem] for elem in node_idx_dict if elem[0]== sent_num]
                    min_tok_idx 		= min([tok_idx[0] for tok_idx in tok_idxs])
                    max_tok_idx 		= max([tok_idx[1] for tok_idx in tok_idxs])

                    node_idx_dict[(sent_num,0)]					= (min_tok_idx, max_tok_idx)
                    node_mask_dict[(sent_num,0)]				= 0
    
                x, edge_index, edge_type, n1_mask, n2_mask		= [],[[],[]],[],[],[]
                for node in node_dict:
                    six, eix 									= node_idx_dict[node]
                    temp_ones 									= torch.ones((512,))*-torch.inf

                    try:
                        assert six < eix
                    except Exception as e:
                        temp_ones[six]							= 0
                        # import pdb; pdb.set_trace()
                    temp_ones[six:eix]							= 0
                    x.append(temp_ones)
    
                    mask = node_mask_dict[node]                    
                    if mask == 0: n1_mask.append(0); n2_mask.append(0)
                    if mask == 1: n1_mask.append(1); n2_mask.append(0)
                    if mask == 2: n1_mask.append(0); n2_mask.append(1)
                    if mask == 3: n2_mask.append(1); n1_mask.append(1)

                for edge in edge_arr:
                    n1, n2, rel_idx 							= edge
                    edge_index[0].append(node_dict[n1])
                    edge_index[1].append(node_dict[n2])
                    edge_type.append(rel_idx)
    
                for sent_num in range(num_sents):
                    edge_index[0].append(node_dict[(sent_num,0)])
                    edge_index[1].append(node_dict[(-1,-1)])
                    edge_type.append(deprel_dict['STAR'])
    
                try:
                    assert sum(n1_mask)> 0 and sum(n2_mask) >0
                except Exception as e:
                    import pdb; pdb.set_trace()
    
                try:
                    x, edge_index, edge_type, n1_mask, n2_mask		= torch.stack(x, dim=0), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
                    dep_data 										= Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
                except Exception as e:
                    import pdb; pdb.set_trace()

                data[split]['rels'].append({
                    'tokens'	: bert_toks,
                    'tok_range'	: tok_range,	
                    'arg1_ids'	: e1_toks,
                    'arg2_ids'	: e2_toks,
                    'arg1_type' : np.zeros(len(entity_dict)),
                    'arg2_type' : np.zeros(len(entity_dict)),
                    'desc_emb'	: relation_desc[rel],
                    'label'		: rel_type,
                    'dep_data'	: dep_data,
                    'sent'		: sent,
                    'orig_sent'	: org_sent,
                    'arg1_emb'	: ent_vec,
                    'arg2_emb'	: ent_vec,
                    'info'		: (e1_span, e2_span),
                    'ent_data'	: []
                })
        dump_dill(data, f'{args.rel_dir}/{lang}_new_rels.dill')





if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Arguments for analysis')
    parser.add_argument('--step',  		        required = True, help= "Which function to call")
    parser.add_argument('--fact_dir',  		    default='/data/multilingual_KGQA/IndicLink/fact_dir')
    parser.add_argument('--kg',                 default='wiki_all')
    parser.add_argument('--wikikg_dir',         default='/data/multilingual_KGQA/kg_data/wikidata_kgFID')
 
    parser.add_argument('--train_ratio',        default = 0.8, type=int)
    parser.add_argument('--dev_ratio',          default = 0.1, type=int)
 
    # parser.add_argument('--rel_dir',            default='/data/multilingual_KGQA/IndoRE/data')
    parser.add_argument('--rel_dir',            default='/data/multilingual_KGQA/SMILER/data')
    parser.add_argument('--bert-model',         default='mbert')
 
    parser.add_argument('--lang_list',          default='ru', type=str)
    parser.add_argument('--qald_dir',           default='/data/multilingual_KGQA/QALD_9_plus/')
    parser.add_argument('--log_dir',            default='../log/')
    parser.add_argument('--workers',            default = 30, type=int)

    
    
    args = parser.parse_args()

    if args.step == 'create':       	create_kg()
    elif args.step == 'qald':       	verify_quald_ids()
    elif args.step == 'rel_epi':    	el_epitran()
    elif args.step == 'rel':        	el_indic()
    elif args.step == 'ind_dump':   	create_indore_data()
    elif args.step == 'smiler_dump': 	create_smiler_data()
    elif args.step == 'add_ents':       add_ent_data()
    # elif args.step == 'ind_el_dump': 	create_indore_el_data()