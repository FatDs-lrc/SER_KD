screen -dmS transformer_test_1
screen -x -S transformer_test_1 -p 0 -X stuff 'source activate torch
'
screen -x -S transformer_test_1 -p 0 -X stuff 'sh today_tasks/transformer.sh 2 meanpool 1 1
'
screen -x -S transformer_test_1 -p 0 -X stuff 'sh today_tasks/transformer.sh 2 meanpool 2 1
'
screen -x -S transformer_test_1 -p 0 -X stuff 'sh today_tasks/transformer.sh 2 last 1 1
'
screen -x -S transformer_test_1 -p 0 -X stuff 'sh today_tasks/transformer.sh 2 last 2 1
'
