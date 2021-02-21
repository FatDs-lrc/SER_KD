screen -dmS emo_ablation_0
screen -x -S emo_ablation_0 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh ef A 1 2
'
screen -x -S emo_ablation_0 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh ef A 2 2
'
screen -x -S emo_ablation_0 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh ef V 1 2
'
screen -x -S emo_ablation_0 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh ef V 2 2
'
screen -x -S emo_ablation_0 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh ef L 1 2
'
screen -x -S emo_ablation_0 -p 0 -X stuff 'sh today_tasks/emo_ablation.sh ef L 2 2
'
screen -x -S emo_ablation_0 -p 0 -X stuff 'exit
'
