# GIFSL
Grafting based Improved Few-Shot Learning

## Stage 1 Training
###Run below two simultaneously
python3 train_grafting_dual_ss4_fc100.py --model_path <save path> --data_root data/ --graftversion 0  --gpu 9
python3 train_grafting_dual_ss4_fc100.py --model_path <save path> --data_root data/ --graftversion 1  --gpu 9

## Stage 2 Training
###Run below two simultaneously
python3 train_grafting_dual_ss4_fc100_kd.py --path_t <path to teacher model> --model_path <save path> --data_root data/ --graftversion 0 --gpu 3
python3 train_grafting_dual_ss4_fc100_kd.py --path_t <path to teacher model> --model_path <save path> --data_root data/ --graftversion 0 --gpu 3
