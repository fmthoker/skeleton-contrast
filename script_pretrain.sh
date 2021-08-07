# Inter-skeleton contrast bewtween seq-based and graph-based representations
# Edit the argument --skeleton-representation to pretrain using different combinations

########################################################################### NTU 60 Cross-View ############################################

CUDA_VISIBLE_DEVICES=0,1,2,3 python  main_moco_inter_skeleton.py \
  --lr 0.01 \
  --batch-size 64 \
  --mlp --moco-t 0.07   --moco-k 16384  --checkpoint-path ./checkpoints/ntu_60_cross_view/interskeleton_seq_based_graph_based  --schedule 351  --epochs 451  --pre-dataset ntu60 --skeleton-representation seq-based_and_graph-based --protocol cross_view

########################################################################### NTU 60 Cross-Subject ############################################

CUDA_VISIBLE_DEVICES=0,1,2,3 python  main_moco_inter_skeleton.py \
  --lr 0.01 \
  --batch-size 64 \
  --mlp --moco-t 0.07   --moco-k 16384  --checkpoint-path ./checkpoints/ntu_60_cross_subject/interskeleton_seq_based_graph_based  --schedule 351  --epochs 451  --pre-dataset ntu60 --skeleton-representation seq-based_and_graph-based --protocol cross_subject


########################################################################### NTU 120 Cross-Setup ############################################

CUDA_VISIBLE_DEVICES=0,1,2,3 python  main_moco_inter_skeleton.py \
  --lr 0.01 \
  --batch-size 64 \
  --mlp --moco-t 0.07   --moco-k 16384  --checkpoint-path ./checkpoints/ntu_120_cross_setup/interskeleton_seq_based_graph_based  --schedule 251 351  --epochs 351  --pre-dataset ntu120 --skeleton-representation seq-based_and_graph-based --protocol cross_setup 

########################################################################### NTU 120 Cross-Subject ############################################

CUDA_VISIBLE_DEVICES=0,1,2,3 python  main_moco_inter_skeleton.py \
  --lr 0.01 \
  --batch-size 64 \
  --mlp --moco-t 0.07   --moco-k 16384  --checkpoint-path ./checkpoints/ntu_120_cross_subject/interskeleton_seq_based_graph_based  --schedule 251 351  --epochs 351  --pre-dataset ntu120 --skeleton-representation seq-based_and_graph-based --protocol cross_subject 
