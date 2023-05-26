$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --run-id $model_name --save-interval 10 -uda 1 -dropout $dp -l2 $l_two -eps $epsilon

$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id $model_name -num-eval 100 -rep 10

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0