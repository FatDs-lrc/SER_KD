screen -dmS bert_explore_1
screen -x -S bert_explore_1 -p 0 -X stuff 'source activate torch
'
screen -x -S bert_explore_1 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh mean 2 1
'
screen -x -S bert_explore_1 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh cls 1 1
'
screen -x -S bert_explore_1 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh cls 2 1
'
