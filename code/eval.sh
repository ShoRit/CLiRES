# Eval

# for src_lang in en hi te   ### Outer for loop ###    
# do
#     for tgt_lang in en hi te 
#     do
#         CUDA_VISIBLE_DEVICES=0  python3 relextract.py --src_lang $src_lang  --tgt_lang  $tgt_lang --mode batch_eval --mtl 0.5 --batch_size 6 --dep 0 --el 0 --seed 0
#         CUDA_VISIBLE_DEVICES=1  python3 relextract.py --src_lang $src_lang  --tgt_lang  $tgt_lang --mode batch_eval --mtl 0.5 --batch_size 6 --dep 1 --el 0 --seed 0
#     done &
#     #   echo -n "Do training for $lang on $seed "
# done
# echo "" #### print the new line ###

for lang in fa uk ru
do 
    CUDA_VISIBLE_DEVICES=0 python3 relextract.py --src_lang ar --tgt_lang $lang --data SMILER --mtl 1 --el 0 --dep 0 --batch_size 6 --mode eval --seed 0
    CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang ar --tgt_lang $lang --data SMILER --mtl 1 --el 0 --dep 1 --batch_size 6 --mode eval --seed 0
    CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang $lang --tgt_lang ar --data SMILER --mtl 1 --el 0 --dep 0 --batch_size 6 --mode eval --seed 0
    CUDA_VISIBLE_DEVICES=3 python3 relextract.py --src_lang $lang --tgt_lang ar --data SMILER --mtl 1 --el 0 --dep 1 --batch_size 6 --mode eval --seed 0
done &



# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 1



# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  en --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  te --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  en --mode predict --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  hi --mode predict --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  te --mode predict --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode predict --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode predict --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode predict --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode predict --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  hi --mode predict --mtl 0.5 --batch_size 6 --dep 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode predict --mtl 0.5 --batch_size 6 --dep 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  hi --mode predict --mtl 0.5 --batch_size 6 --dep 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 1


# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 0
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 0 --el 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 1 --el 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang en  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 1 --el 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 1 --el 1
# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang hi  --tgt_lang  te --mode eval --mtl 0.5 --batch_size 6 --dep 1 --el 1

# CUDA_VISIBLE_DEVICES=1 python3 relextract.py --src_lang te  --tgt_lang  en --mode eval --mtl 0.5 --batch_size 6 --dep 1 --el 1&
# CUDA_VISIBLE_DEVICES=2 python3 relextract.py --src_lang te  --tgt_lang  hi --mode eval --mtl 0.5 --batch_size 6 --dep 1 --el 1&
