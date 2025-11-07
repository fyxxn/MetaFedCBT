# MetaFedCBT
Metadata-Driven Federated Learning of Connectional Brain Templates in Non-IID Multi-Domain Scenarios

![main_figure](overview.png)

## Usage

## Data 
This project uses multi-view brain connectivity data from the public ABIDE-I dataset (Autism Brain Imaging Data Exchange), which includes two groups: Normal Control (NC) and Autism Spectrum Disorder (ASD). The data is divided by left hemisphere (LH) and right hemisphere (RH).

## Configs
Different experiments (e.g., dataset selection, training epoch adjustment) are controlled by the config.py file.

## Train
This project adopts an integrated training pipeline of "metadata prediction-connectivity generation-federated aggregation", which needs to be started through the main script and supports switching between training and validation modes:
1. Start Federated Training
```python
# Run the main training script to automatically execute metadata generation, local training, and global aggregation
python train_MetaFedCBT.py
```
2. Model Validation (CBT Topological Soundness Evaluation)
```python
# Evaluate the topological consistency between CBT and real brain networks based on KL divergence
python eval_metaFedCBT_kl.py
```
3. Model Validation (CBT Discriminative Evaluation)
```python
# Evaluate the discriminative ability of CBT for ASD/NC using an SVM classifier (outputs Acc, Prec, Rec, F1)
python eval_metaFedCBT_classify.py
```

## Results
These figures present key experimental results of the MetaFedCBT method:

1. Quantitative comparison of centeredness

  ![Centeredness](Quantitative%20Comparison%20of%20the%20Centeredness.png)

2. Quantitative comparison of topological soundness
   
   ![Topological Soundness](Quantitative%20Comparison%20of%20the%20Topological%20Soundness.png)

3. Visual comparison of topological distributions

   ![Topological Distributions](Visual%20Comparison%20-%20Topological%20Distributions.png)

4. Visual comparison
   ![Visual Comparison](Visual%20Comparison.png)

## Citation
Please cite our paper if you find the work useful:

@article{chen2025metadata,

author = {Geng Chen and Qingyue Wang and Yuan Feng and Islem Rekik},

journal = {IEEE Transactions on Medical Imaging},

title = {Metadata-Driven Federated Learning of Connectional Brain Templates in Non-IID Multi-Domain Scenarios},

year = {2025}}

## Contacts
If you have any question, please contact:

Qingyue Wang: qywang_@mail.nwpu.edu.cn;
Yuan Feng: yuanfeng@mail.nwpu.edu.cn;
Geng Chen: geng.chen@ieee.org
