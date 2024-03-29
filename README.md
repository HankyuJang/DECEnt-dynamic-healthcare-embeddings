# Dynamic Healthcare Embeddings for Improving Patient Care (DECEnt)

__Paper__: H. Jang, S. Lee, D.M.H. Hasan, P.M. Polgreen, S.V. Pemmaraju, B. Adhikari, "Dynamic Healthcare Embeddings for Improving Patient Care," _2022 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)_, 2022.

## Software requirements
- Python3
- PyTorch # GPU
- Numpy
- Pandas
- sklearn # for the classifications (random forest, logistic regression)
- keras # for the classifications
- tensorflow # for the classifications
- matplotlib
- mysql.connector

## DECEnt and DECEnt+

- DECEnt: implemented in `DECEnt_3M_auto_v1/`
- DECEnt+: implemented in `DECEnt_3M_auto_v3/`

## Training DECEnt and DECEnt+
Follow these steps in order to reproduce results in the paper

Model train in folder `DECEnt_3M_auto_v1/`
```
python learn_embeddings.py --network patient_DECEnt_PF_2010-01-01 --gpu 0 --epochs 1000
```

Model train in folder `DECEnt_3M_auto_v3/`
```
python learn_embeddings.py --network patient_DECEnt_PF_2010-01-01 --gpu 1 --epochs 1000
```

### plot loss
```
python plot_loss.py -folder DECEnt_3M_auto_v1 -network patient_DECEnt_PF_2010-01-01
```

## Application 1-3. Predictive modeling using patient embeddings of DECEnt and DECEnt+

### Prepare labels

Prepare labels for the prediction tasks
```
python process_patient_label.py
```

### Prepare features 

Prepare two datasets per trained dynamic embedding: `df_patient_embedding_per_day` and `df_patient_embedding`
```
python process_dynamic_embedding.py
```

### Train and test the patient embeddings on predictive tasks

Train MLP models. Distribute the jobs on the four GPUs. Each GPU gets a task on one of the four classification tasks (MICU transfer, CDI, severity, mortality) 
```
./evaluate_patient_embeddings_MLP0.sh
./evaluate_patient_embeddings_MLP1.sh
./evaluate_patient_embeddings_MLP2.sh
./evaluate_patient_embeddings_MLP3.sh
```

Train random forest and logistic regression models
```
./evaluate_patient_embeddings_rf_logit.sh
```

## Evaluate other entity embeddings of DECEnt

plots saved in `plots/DECEnt_3M_auto_v1/`
```
python validate_item_embeddings.py -folder DECEnt_3M_auto_v1
```

## Application 1-3. Predictive modeling using patient embeddings of baselines

### Domain specific baselines for other applications

Uses the interaction features that we have. Compute feature importance vector, sort by the features by importance in the decreasing order.
Linear search by train starting w/ all the features, then start removing one most unimportant feature at a time. If performance degrades, stop.
Use that feature set, then train the model w/ repetition.
```
./evaluate_baseline_models_for_application_MLP0.sh
./evaluate_baseline_models_for_application_MLP1.sh
./evaluate_baseline_models_for_application_MLP2.sh
./evaluate_baseline_models_for_application_MLP3.sh
./evaluate_baseline_models_for_application_rf_logit.sh
```

### RNN, LSTM

script: `evaluate_RNN_for_application.py`
```
./evaluate_RNN0.sh
./evaluate_RNN1.sh
./evaluate_RNN2.sh
./evaluate_RNN3.sh
```

### node2vec, deepwalk

Trained node2vec and deepwalk using scripts from snap.
```
python prepare_interactions_for_net_emb.py # this script generates data to train node2vec, deepwalk, and CTDNE
python process_node2vec_embeddings.py
./evaluate_node2vec_embeddings_rf_logit.sh
./evaluate_deepwalk_MLP.sh
./evaluate_node2vec_BFS.sh
./evaluate_node2vec_DFS.sh
```

### CTDNE
CTDNE was trained using external script. Following scripts load the learned embeddings in training Application1-3
```
python process_CTDNE_embeddings.py
./evaluate_CTDNE_embeddings_MLP0.sh
./evaluate_CTDNE_embeddings_MLP1.sh
./evaluate_CTDNE_embeddings_MLP2.sh
./evaluate_CTDNE_embeddings_MLP3.sh
./evaluate_CTDNE_embeddings_rf_logit.sh
```

### JODIE

```
./evaluate_jodie_MLP0.sh
./evaluate_jodie_MLP1.sh
./evaluate_jodie_MLP2.sh
./evaluate_jodie_MLP3.sh
./evaluate_jodie_rf_logit.sh
```

## Other scripts... 

### Plotting results of Application 1-3

ROC curves for MICU transfer

Use the following to plot the ROC curves. plots saved in `plots/roc_curve/`
```
python EXP2_plot_ROC.py
```

### Generate Applicaiton 1-3 result tables in paper 

```
python gen_result_tables.py
```

### Generate data summary table

```
python gen_data_statistics.py
```
