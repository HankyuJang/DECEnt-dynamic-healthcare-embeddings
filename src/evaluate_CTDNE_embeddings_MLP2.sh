network=patient_DECEnt_PF_2010-01-01
label=CDI

for folder in CTDNE
do
    for clf in MLP
    do
        CUDA_VISIBLE_DEVICES=2 python -W ignore evaluate_patient_embeddings.py -folder "$folder" -network "$network" -label $label -clf_name $clf
    done
done
