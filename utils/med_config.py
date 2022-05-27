
batch_size = 32  # batch size for the generator node v

latent_dim=128
num_pred=5

num_classes=5
delta=10
feat_len=300
private_types=['admission','patient']
public_types=['adm_type','insurance','medicine','procedure']

pred_in_n_rev_etypes={
    ('patient', 'codiagnose', 'patient'):('patient', 'codiagnose', 'patient'),
    ('admission', 'belongto', 'patient'):('patient', 'of', 'admission'),
    ('admission', 'seq', 'admission'):('admission', 'seq', 'admission')
}



root_path="/path/to/kdd2022_fedhg/"
substring=''
def get_path(dataset,num_owners):
    local_info_prefix=root_path +"local/info"+substring+'/'+ dataset+"_"+str(num_owners)
    local_subgraphs_path=local_info_prefix+"_all_subs.bin"
    return local_subgraphs_path

# arg part
seed=42
weight_decay=0.01

# TGCN param
hidden=128
dropout=0.5
n_layers=2
n_bases=-1
l2norm=0
validation=True
use_self_loop=True
gen_self_loop=False
train_portion=0.5
val_portion=0.2
test_portion=0.3
gen_train_portion=0.5
gen_val_portion=0.2
gen_test_portion=0.3
fanout=4
# cuda = not no_cuda and torch.cuda.is_available()
cuda = True
gpu=0
# cuda=False
# gpu=-1
local_epochs=1

# fed settings
fed_local_gen_epochs=1
fed_global_gen_epochs=20


a=1
b=1
c=1
b_fl=1
