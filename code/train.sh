# !/bin/bash 

# Train for INDORE


# for (( seed = 0 ; seed <= 4; seed++ )) ### Inner for loop ###
# do
#     for lang in en hi te   ### Outer for loop ###    
#     do
#         CUDA_VISIBLE_DEVICES=$seed   python3 relextract.py --src_lang $lang  --tgt_lang  $lang --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 0 --seed $seed
#         CUDA_VISIBLE_DEVICES=$seed   python3 relextract.py --src_lang $lang  --tgt_lang  $lang --mode train --mtl 0.5 --batch_size 6 --dep 1 --el 0 --seed $seed
#         #   echo -n "Do training for $lang on $seed "
#     done &
#   echo "" #### print the new line ###
# done




# Train for SMILER
for (( seed = 0 ; seed <= 4; seed++ )) ### Inner for loop ###
do
    for lang in pt #de fr pl nl  #fa uk ar ru en  ### Outer for loop ###    
    do
        # CUDA_VISIBLE_DEVICES=$seed   python3 relextract.py --src_lang $lang --data SMILER --tgt_lang  $lang --mode train --mtl 1.0 --batch_size 6 --dep 0 --el 0 --seed $seed
        CUDA_VISIBLE_DEVICES=$seed   python3 relextract.py --src_lang $lang --data SMILER --tgt_lang  $lang --mode train --mtl 1.0 --batch_size 6 --dep 1 --el 0 --seed $seed --gnn_depth 2
        #   echo -n "Do training for $lang on $seed "
    done &
  echo "" #### print the new line ###
done

# CUDA_VISIBLE_DEVICES=3   python3 relextract.py --src_lang fr --data SMILER --tgt_lang  fr --mode train --mtl 1.0 --batch_size 6 --dep 1 --el 0 --seed 3 --gnn_depth 2


# CUDA_VISIBLE_DEVICES=2   python3 relextract.py --src_lang en --data IndoRE --tgt_lang  en --mode predict --mtl 0.5 --batch_size 18 --dep 0 --el 0

# CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang ru --tgt_lang ru --data SMILER --mtl 1 --el 0 --dep 0 --batch_size 6 --mode train --seed 0 &
# CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang uk --tgt_lang uk --data SMILER --mtl 1 --el 0 --dep 0 --batch_size 6 --mode train --seed 0 &
# CUDA_VISIBLE_DEVICES=3 python3 relextract.py --src_lang ar --tgt_lang ar --data SMILER --mtl 1 --el 0 --dep 0 --batch_size 6 --mode train --seed 0 &
# CUDA_VISIBLE_DEVICES=4 python3 relextract.py --src_lang fa --tgt_lang fa --data SMILER --mtl 1 --el 0 --dep 0 --batch_size 6 --mode train --seed 0 &

# CUDA_VISIBLE_DEVICES=5 python3 relextract.py --src_lang fa --tgt_lang fa --data SMILER --mtl 1 --el 0 --dep 1 --batch_size 6 --mode train --seed 0 &


# CUDA_VISIBLE_DEVICES=0 python3 relextract.py --src_lang en  --tgt_lang  en --mode train --mtl 0.5 --batch_size 6 --dep 0&
# CUDA_VISIBLE_DEVICES=4 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode train --mtl 0.5 --batch_size 6 --dep 0&
# CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang te  --tgt_lang  te --mode train --mtl 0.5 --batch_size 6 --dep 0&

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  en --mode train --mtl 0.5 --batch_size 6 --dep 1&
# CUDA_VISIBLE_DEVICES=3 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode train --mtl 0.5 --batch_size 6 --dep 1&
# CUDA_VISIBLE_DEVICES=5 python3 relextract.py --src_lang te  --tgt_lang  te --mode train --mtl 0.5 --batch_size 6 --dep 1&


# CUDA_VISIBLE_DEVICES=0 python3 relextract.py --src_lang en  --tgt_lang  en --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 1&
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 1&
# CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang te  --tgt_lang  te --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 1&

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  en --mode train --mtl 0.5 --batch_size 6 --dep 1 --el 1&
# CUDA_VISIBLE_DEVICES=3 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode train --mtl 0.5 --batch_size 6 --dep 1 --el 1&
# CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang te  --tgt_lang  te --mode train --mtl 0.5 --batch_size 6 --dep 1 --el 1&
# CUDA_VISIBLE_DEVICES=0 python3 relextract.py --src_lang te  --tgt_lang  te --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 0&


# CUDA_VISIBLE_DEVICES=5 python3 relextract.py --src_lang en  --tgt_lang  en --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 0&
# CUDA_VISIBLE_DEVICES=3 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 0&
# CUDA_VISIBLE_DEVICES=5 python3 relextract.py --src_lang te  --tgt_lang  te --mode train --mtl 0.5 --batch_size 6 --dep 0 --el 0&

# python3 relextract.py --src_lang en  --tgt_lang  en --mode train --mtl 0.5 --batch_size 6 --dep 1 --el 0 --seed 3