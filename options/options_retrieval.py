class  opts_ntu_60_cross_view():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 60,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":60
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":60
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy',
     'label_path': './data/NTU-RGB-D-60-AGCN/xview/train_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': './data/NTU-RGB-D-60-AGCN/xview/val_data_joint.npy',
     'label_path': './data/NTU-RGB-D-60-AGCN/xview/val_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-60-AGCN/xview/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }

class  opts_ntu_60_cross_subject():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 60,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":60
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":60
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy',
     'label_path': './data/NTU-RGB-D-60-AGCN/xsub/train_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': './data/NTU-RGB-D-60-AGCN/xsub/val_data_joint.npy',
     'label_path': './data/NTU-RGB-D-60-AGCN/xsub/val_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-60-AGCN/xsub/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }



class  opts_ntu_120_cross_subject():
  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 120,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":120
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":120
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy',
     'label_path': './data/NTU-RGB-D-120-AGCN/xsub/train_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': './data/NTU-RGB-D-120-AGCN/xsub/val_data_joint.npy',
     'label_path': './data/NTU-RGB-D-120-AGCN/xsub/val_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-120-AGCN/xsub/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }

class  opts_ntu_120_cross_setup():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 120,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":120
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":120
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy',
     'label_path': './data/NTU-RGB-D-120-AGCN/xsetup/train_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': './data/NTU-RGB-D-120-AGCN/xsetup/val_data_joint.npy',
     'label_path': './data/NTU-RGB-D-120-AGCN/xsetup/val_label.pkl',
     'num_frame_path': './data/NTU-RGB-D-120-AGCN/xsetup/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
