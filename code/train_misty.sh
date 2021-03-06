# !/bin/bash 

# Train

for (( seed = 0 ; seed <= 2; seed++ )) ### Inner for loop ###
do
    for lang in en hi te   ### Outer for loop ###    
    do
        CUDA_VISIBLE_DEVICES=$seed          python3 relextract.py --src_lang $lang  --tgt_lang  $lang --mode train --mtl 1.0 --batch_size 6 --dep 0 --el 0 --seed $seed
        CUDA_VISIBLE_DEVICES=$(($seed+3))   python3 relextract.py --src_lang $lang  --tgt_lang  $lang --mode train --mtl 1.0 --batch_size 6 --dep 1 --el 0 --seed $seed
        #   echo -n "Do training for $lang on $seed "
    done &
  echo "" #### print the new line ###
done

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
