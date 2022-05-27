# To run the code
1. Download the code to your local directory;
2. Download data from the links in the paper to ~/FedHGN/data;
3. Run 'pip install -r requirements.txt' to download required packages;
4. For the DBLP dataset:
    1. Modify the configuration file:
        1. Open file ~/FedHGN/utils/dblp_config.py;
        2. Replace the "/path/to/FedHGN/" in line 24 of dblp_config.py with your current path;
        3. You can change hyper-parameters in dblp_config.py according to different testing scenarios;
    2. Construct the demo heterograph and sub-heterographs by running 'python ~/FedHGN/dblp_build_HGNN.py';
    3. Run the T-GCN_{sin} pipline with 'python ~/FedHGN/dblp_sin_HGNN.py';
    4. Run the T-GCN_{glb} pipline with 'python ~/FedHGN/dblp_glb_HGNN.py'.
    5. Run the T-GCN_{sin} pipline with 'python ~/FedHGN/dblp_sin_HGNN.py';
    6. Run the T-GCN_{sin}+ pipline with 'python ~/FedHGN/dblp_sinp_HGNN.py';
    7. Run the FedHG pipline with 'python ~/FedHGN/dblp_fhg_HGNN.py';
    8. Run the FedHG+ pipline with 'python ~/FedHGN/dblp_fhgp_HGNN.py'.

5. For the MIMIC-III dataset:
    1. Modify the configuration file:
        1. Open file ~/FedHGN/utils/med_config.py;
        2. Replace the "/path/to/FedHGN/" in line 21 of med_config.py with your current path;
        3. You can change hyper-parameters in dblp_config.py according to different testing scenarios;
    2. Construct the demo heterograph and sub-heterographs by running 'python ~/FedHGN/med_build_HGNN.py';
    3. Run the T-GCN$_{glb}$ pipline with 'python ~/FedHGN/med_glb_HGNN.py';
    4. Run the T-GCN$_{sin}$ pipline with 'python ~/FedHGN/med_sin_HGNN.py';
    5. Run the T-GCN$_{sin}$+ pipline with 'python ~/FedHGN/med_sinp_HGNN.py';
    6. Run the FedHG pipline with 'python ~/FedHGN/med_fhg_HGNN.py';
    7. Run the FedHG+ pipline with 'python ~/FedHGN/med_fhgp_HGNN.py'.
