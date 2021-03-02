screen -dmS cnn_transformer_2
screen -x -S cnn_transformer_2 -p 0 -X stuff 'source activate torch
'
screen -x -S cnn_transformer_2 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 3 1 6
'
screen -x -S cnn_transformer_2 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 3 2 6
'
