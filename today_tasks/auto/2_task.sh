screen -dmS emo_ablation_2
screen -x -S emo_ablation_2 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh single A 1 4
'
screen -x -S emo_ablation_2 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh single A 2 4
'
screen -x -S emo_ablation_2 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh single V 1 4
'
screen -x -S emo_ablation_2 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh single V 2 4
'
screen -x -S emo_ablation_2 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh single L 1 4
'
screen -x -S emo_ablation_2 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh single L 2 4
'
screen -x -S emo_ablation_2 -p 0 -X stuff 'exit
'
