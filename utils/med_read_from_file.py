import numpy as np
from utils import med_config
import dgl
import torch as th
import os

def read_prc_node(node_file:str):
    node_dict={}
    print("read file "+str(node_file))
    file1 = open(node_file, 'r')
    while True:
        line = file1.readline()
        if not line:
            break
        strings=line.replace('\n','').split(" ")
        if len(strings)<2:
            break
        cur_id=int(strings[0])
        cur_id=str(cur_id)
        feature=strings[1:]
        feature_tensor=th.tensor(list(map(np.float32,feature)),dtype=th.float32)
        node_dict[cur_id]=feature_tensor
    file1.close()
    return node_dict

def read_node(node_file:str):
    node_dict={}
    print("read file "+str(node_file))
    file1 = open(node_file, 'r')
    while True:
        line = file1.readline()
        if not line:
            break
        strings=line.replace('\n','').split(" ")
        if len(strings)<2:
            break
        cur_id=strings[0]
        feature=strings[1:]
        feature_tensor=th.tensor(list(map(np.float32,feature)),dtype=th.float32)
        node_dict[cur_id]=feature_tensor
    file1.close()
    return node_dict

def read_link(link_file:str):
    link_list=[[],[]]
    print("read file "+str(link_file))
    file1 = open(link_file, 'r')
    while True:
        line = file1.readline()
        if not line:
            break
        strings=line.replace('\n','').split(",")
        if len(strings)<2:
            break
        link_list[0].append(strings[0])
        link_list[1].append(strings[1])

    file1.close()
    return link_list


def map_link_to_tensor(link_list:list,src_dict:dict,targ_dict:dict):
    link_tensor_s=[]
    link_tensor_t=[]
    for i,j in zip(link_list[0],link_list[1]):
        link_tensor_s.append(src_dict[i])
        link_tensor_t.append(targ_dict[j])
    link_tensors=(th.tensor(link_tensor_s,dtype=th.int32),th.tensor(link_tensor_t,dtype=th.int32))
    return link_tensors


def read_med():
    root_path = med_config.root_path + "data/MIMICIII/"

    admission_file=root_path+"admission.txt"
    patient_file=root_path+"patient.txt"
    procedure_file = root_path + "procedure.txt"
    medicine_file = root_path + "medicine.txt"

    admission=read_node(admission_file)
    adm_type={
        "ELECTIVE":th.tensor([1 ,0 ,0, 0],dtype=th.float32),
    "EMERGENCY":th.tensor([ 0, 1, 0 ,0],dtype=th.float32),
    "URGENT":th.tensor([ 0, 0 ,0, 1],dtype=th.float32),
    "NEWBORN":th.tensor([ 0, 0 ,1, 0],dtype=th.float32)
    }
    insurance= {"Medicaid":th.tensor([0,1,0,0,0],dtype=th.float32),
                "Self Pay":th.tensor([0 ,0, 0, 0, 1],dtype=th.float32),
                "Medicare": th.tensor([0, 0, 1, 0, 0],dtype=th.float32),
                "Private" :th.tensor([0, 0, 0, 1, 0],dtype=th.float32),
                "Government": th.tensor([1, 0, 0, 0, 0],dtype=th.float32)}
    patient=read_node(patient_file)
    procedure=read_prc_node(procedure_file)
    medicine=read_node(medicine_file)

    icu_adm_file=root_path+"labels_icustay_admission.txt"
    icu_adm=read_link(icu_adm_file)




    adm_adm_file=root_path+"admission_admission.txt"
    adm_admtype_file=root_path+"admission_admissiontype.txt"
    adm_insurance_file=root_path+"admission_insurance.txt"
    adm_medicine_file=root_path+"admission_medicine.txt"
    adm_patient_file=root_path+"admission_patient.txt"
    adm_procedure_file=root_path+"admission_procedure.txt"
    patient_patient_file=root_path+"patient_patient.txt"



    adm_adm=read_link(adm_adm_file)
    adm_admtype=read_link(adm_admtype_file)
    adm_insurance=read_link(adm_insurance_file)
    adm_medicine=read_link(adm_medicine_file)
    adm_patient=read_link(adm_patient_file)
    adm_procedure=read_link(adm_procedure_file)
    patient_patient=read_link(patient_patient_file)



    adm_keys=list(admission.keys())
    admission_id_dict={k:i for k,i in zip(adm_keys,range(len(adm_keys)))}

    adm_type_key=list(adm_type.keys())
    adm_type_id_dict={k:i for k,i in zip(adm_type_key,range(len(adm_type_key)))}


    insurance_keys=list(insurance.keys())
    insurance_id_dict={k:i for k,i in zip(insurance_keys,range(len(insurance_keys)))}

    patient_keys=list(patient.keys())
    patient_id_dict={k:i for k,i in zip(patient_keys,range(len(patient_keys)))}

    procedure_keys=list(procedure.keys())
    procedure_id_dict={k:i for k,i in zip(procedure_keys,range(len(procedure_keys)))}

    medicine_keys=list(medicine.keys())
    medicine_id_dict={k:i for k,i in zip(medicine_keys,range(len(medicine_keys)))}


    admission_icu_dict={i:0 for i in range(len(adm_keys))}
    for i,j in zip(icu_adm[0],icu_adm[1]):
        if j=='':
            j='0'
        admission_icu_dict[admission_id_dict[i]]=float(j)
    # print(admission_icu_dict)



    adm_adm=map_link_to_tensor(adm_adm,admission_id_dict,admission_id_dict)
    adm_admtype=map_link_to_tensor(adm_admtype,admission_id_dict,adm_type_id_dict)
    adm_insurance=map_link_to_tensor(adm_insurance,admission_id_dict,insurance_id_dict)
    adm_medicine=map_link_to_tensor(adm_medicine,admission_id_dict,medicine_id_dict)
    adm_patient=map_link_to_tensor(adm_patient,admission_id_dict,patient_id_dict)
    adm_procedure=map_link_to_tensor(adm_procedure,admission_id_dict,procedure_id_dict)
    patient_patient=map_link_to_tensor(patient_patient,patient_id_dict,patient_id_dict)




    return admission,adm_type,insurance,patient,procedure,medicine, \
           admission_icu_dict,\
           adm_adm,adm_admtype,adm_insurance,adm_medicine,adm_patient,\
           adm_procedure,icu_adm,patient_patient



def get_whole_graph(graph_path):
    admission,adm_type,insurance,patient,procedure,medicine, \
    admission_icu_dict, \
    adm_adm,adm_admtype,adm_insurance,adm_medicine,adm_patient, \
    adm_procedure,icu_adm,patient_patient=read_med()
    num_nodes_dict={
        'admission': len(admission.keys()),
        'adm_type':len(adm_type.keys()),
        'insurance':len(insurance.keys()),
        'patient':len(patient.keys()),
        'procedure':len(procedure.keys()),
        'medicine':len(medicine.keys())
    }

    edges={ ('admission', 'seq', 'admission'):adm_adm,
            ('admission', 'in', 'adm_type'):adm_admtype,
            ('adm_type', 'contain', 'admission'):(adm_admtype[1],adm_admtype[0]),
            ('admission', 'belongto', 'patient'):adm_patient,
            ('patient', 'of', 'admission'): (adm_patient[1],adm_patient[0]),
            ('patient', 'codiagnose', 'patient'): patient_patient,
            ('admission', 'with', 'insurance'):adm_insurance,
            ('insurance', 'choosenby', 'admission'):(adm_insurance[1],adm_insurance[0]),
            ('admission', 'use', 'medicine'):adm_medicine,
            ('medicine', 'usedon', 'admission'):(adm_medicine[1],adm_medicine[0]),
            ('admission', 'get', 'procedure'):adm_procedure,
            ('procedure', 'givento', 'admission'):(adm_procedure[1],adm_procedure[0])
            }
    print("('admission', 'get', 'procedure'):"+str(adm_procedure))
    dblp_graph=dgl.heterograph(data_dict=edges,num_nodes_dict=num_nodes_dict)
    print("whole graph")
    print(dblp_graph)
    dblp_graph.nodes['patient'].data['feat'] = th.stack([i for k,i in patient.items()])
    dblp_graph.nodes['admission'].data['feat'] = th.stack([i for k,i in admission.items()])
    dblp_graph.nodes['adm_type'].data['feat'] = th.stack([i for k,i in adm_type.items()])
    dblp_graph.nodes['insurance'].data['feat'] = th.stack([i for k,i in insurance.items()])
    dblp_graph.nodes['procedure'].data['feat'] = th.stack([i for k,i in procedure.items()])
    dblp_graph.nodes['medicine'].data['feat'] = th.stack([i for k,i in medicine.items()])

    dblp_graph.nodes['patient'].data['ind'] = dblp_graph.nodes('patient')
    dblp_graph.nodes['admission'].data['ind'] = dblp_graph.nodes('admission')
    dblp_graph.nodes['adm_type'].data['ind'] = dblp_graph.nodes('adm_type')
    dblp_graph.nodes['insurance'].data['ind'] = dblp_graph.nodes('insurance')
    dblp_graph.nodes['procedure'].data['ind'] = dblp_graph.nodes('procedure')
    dblp_graph.nodes['medicine'].data['ind'] = dblp_graph.nodes('medicine')



    dblp_graph.nodes['admission'].data['n_icu'] = th.tensor(data=[i for k,i in admission_icu_dict.items()])
    dgl.save_graphs(graph_path,[dblp_graph])
    return dblp_graph


def construct_graph(graph_path):
    if os.path.isfile(graph_path):
        dblp_graph=dgl.load_graphs(graph_path)[0][0]
        return dblp_graph
    else:
        dblp_graph=get_whole_graph(graph_path)
    return dblp_graph