screen -dmS bert_explore_2
screen -x -S bert_explore_2 -p 0 -X stuff 'source activate torch
'
screen -x -S bert_explore_2 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh cls 1 2
'
screen -x -S bert_explore_2 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh cls 2 2
'
