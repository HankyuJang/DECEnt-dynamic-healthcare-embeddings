network=patient_DECEnt_PF_2010-01-01
label=mortality

for method in RNN LSTM
do
    python -W ignore evaluate_RNN_for_application.py -method "$method" -network "$network" -label $label -gpu 0
done
