import ijson
from collections import defaultdict
import numpy as np
from utils import dblp_config
import dgl
import torch
import os
import dill


def get_word_dict_from_file(word_file,feat_len):
    word_map={}
    print("get word dict")
    with open(word_file, mode='r') as f:
        for line in f:
            infos=line.replace("\n","")
            info_tokens=infos.split(" ")
            emb_list= [float(string) for string in info_tokens[1:]]
            emb=np.asarray(emb_list).reshape(-1)
            assert len(emb)==feat_len
            word_map[info_tokens[0]]=emb
    print('save word dict')
    dict_path=word_file.replace('.txt','_dict.pkl')
    dill.dump(word_map,open(dict_path,'wb'))
    return word_map


# keywords_sentences_file=root_path+"keywords.txt"
# phrases_sentences_file = root_path + "phrases.txt"
# mag_fos_sentences_file = root_path + "mag_fos.txt"
#  venue_names_file = root_path + "venue_names.txt"
def get_name_vec_from_dict(dict_path,name_file_path:str):
    emb_path=name_file_path.replace('.txt','_dict.pkl')
    if os.path.isfile(emb_path)==False:
        word_map=dill.load(open(dict_path,'rb'))
        emb_dict={}
        with open(name_file_path, mode='r') as f:
            for line in f:
                emb_vec=torch.zeros(dblp_config.feat_len)
                keyword=line.replace("\n","")
                emb_dict[keyword]=emb_vec
                tokens = keyword.split(" ")
                for word in tokens:
                    if word in word_map.keys():
                        emb_vec+=1.0/len(tokens)*word_map[word]
                emb_dict[keyword]+=emb_vec
        dill.dump(emb_dict,open(emb_path,'wb'))
    else:
        emb_dict=dill.load(open(emb_path,'rb'))
    return emb_dict



def read_dlbp_json(dblp_file:str):
    root_path = dblp_config.root_path + "data/DBLP/"

    keywords_file=root_path+"keywords.txt"
    phrases_file = root_path + "phrases.txt"
    mag_fos_file = root_path + "mag_fos.txt"

    venue_names_file = root_path + "venue_names.txt"

    # local: phrases_sentences, author_sentences
    paper_map=defaultdict(dict) # _id [ind,year_ind,phrases_ids,n_citation]
    paper_id=0
    author_map=defaultdict(dict) # name [ind,total_citation]
    author_id=0
    venue_map=defaultdict(dict) #name ind
    venue_id=0
    phrase_map=defaultdict(dict) # phrase [ind, word_emb]
    phrase_id=0
    keyword_map=defaultdict(dict)
    keyword_id=0
    mag_fos_map=defaultdict(dict)
    mag_fos_id=0
    write_link={'src':[],'dst':[]}
    coauthor_link={'src':[],'dst':[]} # double direction
    in_link={'src':[],'dst':[]}

    with_link={'src':[],'dst':[]}
    study_link={'src':[],'dst':[]}
    cite_link={'src':[],'dst':[]}
    max_author_cite=0
    missing_num={'year':0,'authors':0,'venue':0,'title':0,
                 'phrases':0,'keywords':0,'mag_fos':0,'references':0}
    print("read file start")
    with open(dblp_file,'rt',encoding='utf8') as f:
        objects=ijson.parse(f,multiple_values=True)
        while True:
            try:
                paper_item = {}
                o = objects.__next__()
                while o!=('', 'end_map', None):
                    o = objects.__next__()
                    if o[0]=='_id':
                        paper_item['_id']=o[-1]
                    elif o==('authors', 'start_array', None):
                        paper_item['authors']=[]
                    elif o[0]=="authors.item":
                        paper_item['authors'].append(o[-1])
                    elif o[0] == "n_citation.$numberInt":
                        paper_item['n_citation']=int(o[-1])
                    elif o[0]=='title':
                        paper_item['title']=o[-1]
                    # elif o[0] =='references.item':
                    #     paper_item['references'].append(o[-1])
                    elif o[0] =='venue':
                        paper_item['venue']=o[-1]
                    elif o[0] == "year.$numberInt":
                        paper_item['year']=int(o[-1])
                    elif o==('keywords', 'start_array', None):
                        paper_item['keywords']=[]
                    elif o[0] =='keywords.item':
                        paper_item['keywords'].append(o[-1])
                    elif o==('mag_fos', 'start_array', None):
                        paper_item['mag_fos']=[]
                    elif o[0] =='mag_fos.item':
                        paper_item['mag_fos'].append(o[-1])
                    elif o==('phrases', 'start_array', None):
                        paper_item['phrases']=[]
                    elif o[0] =='phrases.item':
                        str_phrase=str(o[-1])
                        clean_str=str_phrase.replace("<phrase>","")
                        clean_str = clean_str.replace("</phrase>", "")
                        paper_item['phrases'].append(clean_str)

                cur_paper_ind=paper_id
                paper_map[paper_item['_id']]['ind']=cur_paper_ind
                paper_map[paper_item['_id']]['phrases_ids']=[]
                paper_map[paper_item['_id']]['word_emb'] = torch.zeros(dblp_config.feat_len,dtype=torch.float32)
                paper_id+=1

                if 'venue' in paper_item.keys():
                    with open(venue_names_file, 'a+') as f1:
                        f1.write(paper_item['venue']+"\n")
                    if paper_item['venue'] not in venue_map.keys():
                        venue_map[paper_item['venue']]['ind']=venue_id
                        venue_map[paper_item['venue']]['word_emb'] = torch.zeros(dblp_config.feat_len,dtype=torch.float32)
                        venue_id+=1
                    in_link['src'].append(cur_paper_ind)
                    in_link['dst'].append(venue_map[paper_item['venue']]['ind'])
                else:
                    missing_num['venue'] += 1
                    # print('venue missing for paper ' + str(cur_paper_ind))
                paper_map[paper_item['_id']]['n_citation'] = paper_item['n_citation']
                if 'authors' in paper_item.keys():
                    for a_i in paper_item['authors']:
                        if a_i not in author_map.keys():
                            author_map[a_i]['ind'] = author_id
                            author_id += 1
                            author_map[a_i]['n_citation']=0
                        write_link['src'].append(author_map[a_i]['ind'])
                        write_link['dst'].append(cur_paper_ind)

                    for a_i in paper_item['authors']:
                        for b_i in paper_item['authors']:
                            if b_i != a_i:
                                coauthor_link['src'].append(author_map[a_i]['ind'])
                                coauthor_link['dst'].append(author_map[b_i]['ind'])
                        author_map[a_i]['n_citation']+=paper_item['n_citation']
                        if author_map[a_i]['n_citation']>max_author_cite:
                            max_author_cite=author_map[a_i]['n_citation']

                else:
                    missing_num['authors'] += 1
                    # print('authors missing for paper ' + str(cur_paper_ind))
                if 'phrases' in paper_item.keys():
                    for ph_i in paper_item['phrases']:
                        if ph_i not in phrase_map.keys():
                            phrase_map[ph_i]['ind'] = phrase_id
                            phrase_map[ph_i]['word_emb'] = torch.zeros(dblp_config.feat_len,dtype=torch.float32)
                            phrase_id += 1
                            with open(phrases_file, 'a+') as f1:
                                f1.write(str(ph_i)+"\n")
                        paper_map[paper_item['_id']]['phrases_ids'].append(phrase_map[ph_i]['ind'])

                else:
                    missing_num['phrases'] += 1
                    # print('phrases missing for paper '+str(cur_paper_ind))
                if 'keywords' in paper_item.keys():
                    for kw_i in paper_item['keywords']:
                        if kw_i not in keyword_map.keys():
                            keyword_map[kw_i]['ind'] = keyword_id
                            keyword_map[kw_i]['word_emb']= torch.zeros(dblp_config.feat_len,dtype=torch.float32)
                            keyword_id += 1
                            with open(keywords_file, 'a+') as f1:
                                f1.write(str(kw_i)+"\n")
                        with_link['src'].append(cur_paper_ind)
                        with_link['dst'].append(keyword_map[kw_i]['ind'])
                else:
                    missing_num['keywords'] += 1
                    # print('keywords missing for paper '+str(cur_paper_ind))
                if 'mag_fos' in paper_item.keys():
                    for fos_i in paper_item['mag_fos']:
                        if fos_i not in mag_fos_map.keys():
                            mag_fos_map[fos_i]['ind'] = mag_fos_id
                            mag_fos_map[fos_i]['word_emb']= torch.zeros(dblp_config.feat_len,dtype=torch.float32)
                            mag_fos_id += 1
                            with open(mag_fos_file, 'a+') as f1:
                                f1.write(str(fos_i)+"\n")
                        study_link['src'].append(cur_paper_ind)
                        study_link['dst'].append(mag_fos_map[fos_i]['ind'])
                else:
                    missing_num['mag_fos'] += 1
                    # print('mag_fos missing for paper ' + str(cur_paper_ind))

            except StopIteration as e:
                print("Done reading")
                break

    # second go-through -- add cite links
    with open(dblp_file, 'rt', encoding='utf8') as f:
        objects = ijson.parse(f, multiple_values=True)
        while True:
            try:
                o = objects.__next__()
                paper_item = {}
                while o != ('', 'end_map', None):
                    o = objects.__next__()
                    if o[0] == '_id':
                        paper_item['_id'] = o[-1]
                    elif o == ('references', 'start_array', None):
                        paper_item['references'] = []
                    elif o[0] == 'references.item':
                        paper_item['references'].append(o[-1])

                cur_paper_ind=paper_map[paper_item['_id']]['ind']
                if 'references' in paper_item.keys():
                    for p_i in paper_item['references']:
                        cite_link['src'].append(cur_paper_ind)
                        cite_link['dst'].append(paper_map[p_i]['ind'])
                else:
                    missing_num['references'] += 1
                    # print('references missing for paper ' + str(paper_map[paper_item['_id']]['ind']))

            except StopIteration as e:
                print("Done reading")
                break
    print(missing_num)


    return paper_map,author_map,venue_map,mag_fos_map,keyword_map,phrase_map, \
           in_link,with_link,write_link,cite_link,coauthor_link,study_link


def get_whole_graph(dblp_file,path):
    paper_map, author_map, venue_map, mag_fos_map, keyword_map, phrase_map, \
     in_link, with_link, write_link, cite_link, coauthor_link, \
    study_link=read_dlbp_json(dblp_file)
    num_nodes_dict={
        'paper': len(paper_map.keys()),
        'author':len(author_map.keys()),
        'venue':len(venue_map.keys()),
        'keyword':len(keyword_map.keys()),
        'mag_fos':len(mag_fos_map.keys())
    }

    edges={
           ('author', 'coauthor', 'author'):
               (torch.tensor(coauthor_link['src'], dtype=torch.int32),
                torch.tensor(coauthor_link['dst'], dtype=torch.int32)),
           ('author', 'write', 'paper'):
               (torch.tensor(write_link['src'], dtype=torch.int32),
                torch.tensor(write_link['dst'], dtype=torch.int32)),
           ('paper', 'cite', 'paper'):
    (torch.tensor(cite_link['src'], dtype=torch.int32), torch.tensor(cite_link['dst'], dtype=torch.int32)),
           ('paper', 'in', 'venue'):
    (torch.tensor(in_link['src'], dtype=torch.int32),
     torch.tensor(in_link['dst'], dtype=torch.int32)),
           ('paper', 'with', 'keyword'):
    (torch.tensor(with_link['src'], dtype=torch.int32),
     torch.tensor(with_link['dst'],dtype= torch.int32)),
           ('paper', 'study', 'mag_fos'):
    (torch.tensor(study_link['src'], dtype=torch.int32),
     torch.tensor(study_link['dst'], dtype=torch.int32)),

           ('paper', 'belongto', 'author'):
               (torch.tensor(write_link['dst'], dtype=torch.int32),
                torch.tensor(write_link['src'], dtype=torch.int32)),
           ('paper', 'citedby', 'paper'):
               (torch.tensor(cite_link['dst'], dtype=torch.int32), torch.tensor(cite_link['src'], dtype=torch.int32)),
           ('venue', 'have', 'paper'):
               (torch.tensor(in_link['dst'], dtype=torch.int32),
                torch.tensor(in_link['src'], dtype=torch.int32)),
           ('keyword', 'mentionedby', 'paper'):
               (torch.tensor(with_link['dst'], dtype=torch.int32),
                torch.tensor(with_link['src'],dtype= torch.int32)),
           ('mag_fos', 'include', 'paper'):
               (torch.tensor(study_link['dst'], dtype=torch.int32),
                torch.tensor(study_link['src'], dtype=torch.int32)),
           }
    dblp_graph=dgl.heterograph(data_dict=edges,num_nodes_dict=num_nodes_dict)
    dblp_graph.nodes['author'].data['n_citation']=torch.tensor(
        [author_map[i]['n_citation'] for i in author_map.keys()],dtype=torch.int32)

    dblp_graph.nodes['author'].data['ind'] = torch.tensor(
        [author_map[i]['ind'] for i in author_map.keys()],dtype=torch.int32)
    dblp_graph.nodes['author'].data['feat'] = torch.tensor(
        np.random.rand(len(author_map.keys()),dblp_config.feat_len),dtype=torch.float32)

    dblp_graph.nodes['paper'].data['feat'] = torch.tensor(
        np.random.rand(len(paper_map.keys()),dblp_config.feat_len),dtype=torch.float32)

    dblp_graph.nodes['paper'].data['ind'] = torch.tensor(
        [paper_map[i]['ind'] for i in paper_map.keys()],dtype=torch.int32)

    dgl.save_graphs(path,[dblp_graph])
    return dblp_graph


def construct_graph(dblp_file,path):
    if os.path.isfile(path):
        dblp_graph=dgl.load_graphs(path)[0][0]
        return dblp_graph
    else:
        dblp_graph=get_whole_graph(dblp_file,path)
    root_path = dblp_config.root_path + "data/DBLP/"
    keywords_file=root_path+"keywords.txt"
    mag_fos_file = root_path + "mag_fos.txt"
    venue_names_file = root_path + "venue_names.txt"

    glove_txt_file=root_path+"glove_"+str(dblp_config.feat_len)+".txt"
    glove_dict_file=glove_txt_file.replace(".txt","_dict.pkl")
    print("get glove word dict from file")
    if os.path.isfile(glove_dict_file)==False:
        get_word_dict_from_file(glove_txt_file,dblp_config.feat_len)
    print("get keyword_map vec")
    keyword_feat = get_name_vec_from_dict(glove_dict_file, keywords_file)

    print("get mag_fos_map vec")
    mag_fos_feat = get_name_vec_from_dict(glove_dict_file, mag_fos_file)
    print("get venue_map vec")
    venue_feat = get_name_vec_from_dict(glove_dict_file, venue_names_file)

    venue_tensor=torch.stack([v for k,v in venue_feat.items()])
    dblp_graph.nodes['venue'].data['feat'] = torch.tensor(
        venue_tensor.data,dtype=torch.float32)


    dblp_graph.nodes['keyword'].data['feat'] = torch.tensor(
        torch.stack([v for k,v in keyword_feat.items()]).data,dtype=torch.float32)

    dblp_graph.nodes['mag_fos'].data['feat'] = torch.tensor(
        torch.stack([v for k,v in mag_fos_feat.items()]).data,dtype=torch.float32)

    dgl.save_graphs(path,[dblp_graph])

    return dblp_graph