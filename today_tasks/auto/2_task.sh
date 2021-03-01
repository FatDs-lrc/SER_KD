screen -dmS transformer_test_2
screen -x -S transformer_test_2 -p 0 -X stuff 'source activate torch
'
screen -x -S transformer_test_2 -p 0 -X stuff 'sh today_tasks/transformer.sh 3 meanpool 1 2
'
screen -x -S transformer_test_2 -p 0 -X stuff 'sh today_tasks/transformer.sh 3 meanpool 2 2
'
screen -x -S transformer_test_2 -p 0 -X stuff 'sh today_tasks/transformer.sh 3 last 1 2
'
screen -x -S transformer_test_2 -p 0 -X stuff 'sh today_tasks/transformer.sh 3 last 2 2
'
