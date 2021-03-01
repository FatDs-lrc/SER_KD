screen -dmS transformer_test_3
screen -x -S transformer_test_3 -p 0 -X stuff 'source activate torch
'
screen -x -S transformer_test_3 -p 0 -X stuff 'sh today_tasks/transformer.sh 6 meanpool 1 3
'
screen -x -S transformer_test_3 -p 0 -X stuff 'sh today_tasks/transformer.sh 6 meanpool 2 3
'
screen -x -S transformer_test_3 -p 0 -X stuff 'sh today_tasks/transformer.sh 6 last 1 3
'
screen -x -S transformer_test_3 -p 0 -X stuff 'sh today_tasks/transformer.sh 6 last 2 3
'
