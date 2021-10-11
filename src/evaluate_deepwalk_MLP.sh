network=patient_DECEnt_PF_2010-01-01

for label in mortality severity CDI MICU_transfer
do
    for folder in deepwalk
    do
        for clf in MLP
        do
            CUDA_VISIBLE_DEVICES=0 python -W ignore evaluate_patient_embeddings.py -folder "$folder" -network "$network" -label $label -clf_name $clf
        done
    done
done

