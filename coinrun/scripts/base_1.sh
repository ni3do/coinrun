echo "hyperparameters: $dp $l_two $epsilon"
bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Starting training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Dropout;$dp;false" --field "l2 reg;$l_two;false" --field "Epsilon;$epsilon;false"

$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --run-id $model_name --save-interval 4 -uda 1 -dropout $dp -l2 $l_two -eps $epsilon

for ((i=0; i <=127; i++))
do
echo "Iteration $i"
$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id $model_name -num-eval 50 -rep 5
$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --restore-id $model_name --run-id $model_name --save-interval 4 -uda 1 -dropout $dp -l2 $l_two -eps $epsilon
if [[ $((i % 32)) == 0 ]]
then
bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Training for $USER at iteration $i" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Dropout;$dp;false" --field "l2 reg;$l_two;false" --field "Epsilon;$epsilon;false"
fi
done

echo "Iteration 128"
$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id $model_name -num-eval 50 -rep 5

echo "Finished training at:     $(date)"
total_steps=128
echo "Making copy of model"
cp -r $HOME/coinrun/coinrun/saved_models/sav_reg_${dp}_${l_two}_${epsilon}_0 $HOME/coinrun/models/sav_${model_name}-${total_steps}e6_0

echo "Finished at:     $(date)"

# discord notification on finish
bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Finished training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Dropout;$dp;false" --field "l2 reg;$l_two;false" --field "Epsilon;$epsilon;false"

# End the script with exit code 0
exit 0