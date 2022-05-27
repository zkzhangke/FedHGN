batch_size = 256  # batch size for the generator node v

latent_dim=128
num_pred=5


num_classes=5
delta=10
feat_len=300
private_types=['author','paper']
public_types=['venue','keyword','mag_fos']

pred_in_n_rev_etypes={
    ('author', 'coauthor', 'author'):('author', 'coauthor', 'author'),
    ('paper', 'belongto', 'author'):('author', 'write', 'paper'),
    ('paper', 'citedby', 'paper'):('paper', 'cite', 'paper'),
    ('paper', 'cite', 'paper'):('paper', 'citedby', 'paper')
}




# path settings
root_path="/path/to/FedHGN/"

substring=''
def get_path(dataset,num_owners):
    local_subgraphs_path=root_path +"local/info"+substring+'/'+ dataset+"_"+str(num_owners)+"_all_subs.bin"
    return local_subgraphs_path

# arg part
seed=42
weight_decay=0.01

# TGCN param
hidden=128
dropout=0.5
n_layers=5
n_bases=-1
l2norm=0
validation=True
use_self_loop=True
gen_self_loop=False
train_portion=0.5
val_portion=0.1
test_portion=0.4
gen_train_portion=0.5
gen_val_portion=0.1
gen_test_portion=0.4
fanout=4
# cuda = not no_cuda and torch.cuda.is_available()
cuda = True
gpu=0
local_epochs=1

# fed settings
fed_local_gen_epochs=1
fed_global_gen_epochs=20


a=1
b=1
c=1
b_fl=1
