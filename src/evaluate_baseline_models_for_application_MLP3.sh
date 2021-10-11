clf=MLP

for label in MICU_transfer
do
    CUDA_VISIBLE_DEVICES=3 python -W ignore evaluate_baseline_models_for_application.py -label $label -clf_name $clf
done
