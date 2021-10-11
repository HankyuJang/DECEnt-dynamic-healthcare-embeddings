folder=jodie
network=patient_jodie_2010-01-01

for label in mortality severity CDI MICU_transfer
do
    for clf in rf logit
    do
        python -W ignore evaluate_patient_embeddings.py -folder "$folder" -network "$network" -label $label -clf_name $clf
    done
done

