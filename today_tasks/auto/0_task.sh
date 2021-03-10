screen -dmS bert_explore_0
screen -x -S bert_explore_0 -p 0 -X stuff 'source activate torch
'
screen -x -S bert_explore_0 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh max 1 0
'
screen -x -S bert_explore_0 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh max 2 0
'
screen -x -S bert_explore_0 -p 0 -X stuff 'sh today_tasks/bert_externel_text.sh mean 1 0
'
