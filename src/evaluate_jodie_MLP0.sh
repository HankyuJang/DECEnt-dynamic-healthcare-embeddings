label=mortality
folder=jodie

for network in patient_jodie_2010-01-01
do
    for clf in MLP
    do
        CUDA_VISIBLE_DEVICES=0 python -W ignore evaluate_patient_embeddings.py -folder "$folder" -network "$network" -label $label -clf_name $clf
    done
done
