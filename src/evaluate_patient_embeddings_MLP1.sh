network=patient_DECEnt_PF_2010-01-01
label=severity

for folder in DECEnt_3M_auto_v1 DECEnt_3M_auto_v3
do
    for clf in MLP
    do
        CUDA_VISIBLE_DEVICES=1 python -W ignore evaluate_patient_embeddings.py -folder "$folder" -network "$network" -label $label -clf_name $clf
    done
done
