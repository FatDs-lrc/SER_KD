screen -dmS cnn_transformer_1
screen -x -S cnn_transformer_1 -p 0 -X stuff 'source activate torch
'
screen -x -S cnn_transformer_1 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 2 1 3
'
screen -x -S cnn_transformer_1 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 2 2 3
'
