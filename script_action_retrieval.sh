
# Use a Knn classifier on seq-based query encoder features  pretrained via inter-skeleton contrast bewtween seq-based and graph-based representations

 CUDA_VISIBLE_DEVICES=0,1,2,3 python action_retrieval.py \
  --lr 0.1 \
  --batch-size 64 \
  --knn-neighbours 1 \
  --pretrained  ./pretrained_models/ntu_60_cross_view/interskeleton_seq_based_graph_based/checkpoint_0450.pth.tar \
 --finetune-dataset ntu60 --protocol cross_view --pretrain-skeleton-representation seq-based_and_graph-based  --finetune-skeleton-representation seq-based  

 CUDA_VISIBLE_DEVICES=0,1,2,3 python action_retrieval.py \
  --lr 0.1 \
  --batch-size 64 \
  --knn-neighbours 1 \
  --pretrained  ./pretrained_models/ntu_60_cross_subject/interskeleton_seq_based_graph_based/checkpoint_0450.pth.tar \
 --finetune-dataset ntu60 --protocol cross_subject --pretrain-skeleton-representation seq-based_and_graph-based  --finetune-skeleton-representation seq-based  

 CUDA_VISIBLE_DEVICES=0,1,2,3 python action_retrieval.py \
  --lr 0.1 \
  --batch-size 64 \
  --knn-neighbours 1 \
 --pretrained  ./pretrained_models/ntu_120_cross_subject/interskeleton_seq_based_graph_based/checkpoint_0350.pth.tar \
  --finetune-dataset ntu120 --protocol cross_subject --pretrain-skeleton-representation seq-based_and_graph-based  --finetune-skeleton-representation seq-based

 CUDA_VISIBLE_DEVICES=0,1,2,3 python action_retrieval.py \
  --lr 0.1 \
  --batch-size 64 \
  --knn-neighbours 1 \
 --pretrained  ./pretrained_models/ntu_120_cross_setup/interskeleton_seq_based_graph_based/checkpoint_0350.pth.tar \
  --finetune-dataset ntu120 --protocol cross_setup --pretrain-skeleton-representation seq-based_and_graph-based  --finetune-skeleton-representation seq-based
