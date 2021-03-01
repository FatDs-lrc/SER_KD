screen -dmS transformer_test_0
screen -x -S transformer_test_0 -p 0 -X stuff 'source activate torch
'
screen -x -S transformer_test_0 -p 0 -X stuff 'sh today_tasks/transformer.sh 1 meanpool 1 0
'
screen -x -S transformer_test_0 -p 0 -X stuff 'sh today_tasks/transformer.sh 1 meanpool 2 0
'
screen -x -S transformer_test_0 -p 0 -X stuff 'sh today_tasks/transformer.sh 1 last 1 0
'
screen -x -S transformer_test_0 -p 0 -X stuff 'sh today_tasks/transformer.sh 1 last 2 0
'
