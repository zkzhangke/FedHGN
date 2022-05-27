# To run the code
1. Download the code to your local directory;
2. Download data from the links in the paper to ~/KDD_2022_FedHG/data;
3. Run 'pip install -r requirements.txt' to download required packages;
4. For the DBLP dataset:
    1. Modify the configuration file:
        1. Open file ~/KDD_2022_FedHG/utils/dblp_config.py;
        2. Replace the "/path/to/kdd2022_fedhg/" in line 24 of dblp_config.py with your current path;
        3. You can change hyper-parameters in dblp_config.py according to different testing scenarios;
    2. Construct the demo heterograph and sub-heterographs by running 'python ~/KDD_2022_FedHG/dblp_build_HGNN.py';
    3. Run the T-GCN_{sin} pipline with 'python ~/KDD_2022_FedHG/dblp_sin_HGNN.py';
    4. Run the T-GCN_{glb} pipline with 'python ~/KDD_2022_FedHG/dblp_glb_HGNN.py'.Run the T-GCN_{sin} pipline with 'python ~/KDD_2022_FedHG/dblp_sin_HGNN.py';
    5. Run the T-GCN_{sin}+ pipline with 'python ~/KDD_2022_FedHG/dblp_sinp_HGNN.py';
    6. Run the FedHG pipline with 'python ~/KDD_2022_FedHG/dblp_fhg_HGNN.py';
    7. Run the FedHG+ pipline with 'python ~/KDD_2022_FedHG/dblp_fhgp_HGNN.py'.

5. For the MIMIC-III dataset:
    1. Modify the configuration file:
        1. Open file ~/KDD_2022_FedHG/utils/med_config.py;
        2. Replace the "/path/to/kdd2022_fedhg/" in line 21 of med_config.py with your current path;
        3. You can change hyper-parameters in dblp_config.py according to different testing scenarios;
    2. Construct the demo heterograph and sub-heterographs by running 'python ~/KDD_2022_FedHG/med_build_HGNN.py';
    3. Run the T-GCN$_{glb}$ pipline with 'python ~/KDD_2022_FedHG/med_glb_HGNN.py';
    4. Run the T-GCN$_{sin}$ pipline with 'python ~/KDD_2022_FedHG/med_sin_HGNN.py';
    5. Run the T-GCN$_{sin}$+ pipline with 'python ~/KDD_2022_FedHG/med_sinp_HGNN.py';
    6. Run the FedHG pipline with 'python ~/KDD_2022_FedHG/med_fhg_HGNN.py';
    7. Run the FedHG+ pipline with 'python ~/KDD_2022_FedHG/med_fhgp_HGNN.py'.
