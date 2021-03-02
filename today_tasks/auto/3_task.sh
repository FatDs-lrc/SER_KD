screen -dmS cnn_transformer_3
screen -x -S cnn_transformer_3 -p 0 -X stuff 'source activate torch
'
screen -x -S cnn_transformer_3 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 4 1 7
'
screen -x -S cnn_transformer_3 -p 0 -X stuff 'sh today_tasks/cnn_transformer.sh 4 2 7
'
