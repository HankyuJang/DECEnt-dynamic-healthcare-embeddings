for clf in rf logit
do
    for label in mortality severity CDI MICU_transfer
    do
        python -W ignore evaluate_baseline_models_for_application.py -label $label -clf_name $clf
    done
done
