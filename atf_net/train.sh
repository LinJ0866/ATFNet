
python train.py \
--gpu_id 0 \
--save_path ./output/ \
--dataset_root /home/linj/workspace/vsod/datasets \
--dataset rdvs \
--batchsize 16 \
--lr 2e-4

python train.py \
--gpu_id 0 \
--save_path ./output/ \
--dataset_root /home/linj/workspace/vsod/datasets \
--dataset vidsod_100 \
--batchsize 16 \
--lr 2e-4

python train.py \
--gpu_id 0 \
--save_path ./output/ \
--dataset_root /home/linj/workspace/vsod/datasets \
--dataset dvisal \
--batchsize 16 \
--lr 2e-4