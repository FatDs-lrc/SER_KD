screen -dmS cnn_transformer_0
screen -x -S cnn_transformer_0 -p 0 -X stuff 'source activate torch
'
screen -x -S cnn_transformer_0 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 1 1 2
'
screen -x -S cnn_transformer_0 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 1 2 2
'
