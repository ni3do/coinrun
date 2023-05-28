$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --run-id $model_name --save-interval 4 -uda 1 -dropout $dp -l2 $l_two -eps $epsilon

for ((i=0; i <=127; i++))
do
echo "Iteration $i"
$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id $model_name -num-eval 50 -rep 5
$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --restore-id $model_name --run-id $model_name --save-interval 4 -uda 1 -dropout $dp -l2 $l_two -eps $epsilon
done

echo "Iteration 128"
$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id $model_name -num-eval 50 -rep 5

echo "Finished training at:     $(date)"
total_steps=128
echo "Making copy of model"
cp -r $HOME/coinrun/coinrun/saved_models/sav_reg_${dp}_${l_two}_${epsilon}_0 $HOME/coinrun/models/sav_${model_name}-${total_steps}e6_0

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0