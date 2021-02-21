screen -dmS emo_ablation_1
screen -x -S emo_ablation_1 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh multi A 1 3
'
screen -x -S emo_ablation_1 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh multi A 2 3
'
screen -x -S emo_ablation_1 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh multi V 1 3
'
screen -x -S emo_ablation_1 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh multi V 2 3
'
screen -x -S emo_ablation_1 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh multi L 1 3
'
screen -x -S emo_ablation_1 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh multi L 2 3
'
screen -x -S emo_ablation_1 -p 0 -X stuff 'exit
'
