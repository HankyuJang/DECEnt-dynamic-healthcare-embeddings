clf=MLP

for label in CDI
do
    CUDA_VISIBLE_DEVICES=2 python -W ignore evaluate_baseline_models_for_application.py -label $label -clf_name $clf
done
