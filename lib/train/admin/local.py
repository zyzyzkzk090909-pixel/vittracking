class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = 'D:\ORTrack\ORTrack-master'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = 'D:\ORTrack\ORTrack-master\tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = 'D:\ORTrack\ORTrack-master\pretrained_networks'
        self.lasot_dir = '/home/pro-c/文档/zy/ORTrack/ORTrack-master/data/LaSOT'
        self.got10k_dir = '/home/pro-c/文档/zy/ORTrack/ORTrack-master/data/got10k/train'
        self.got10k_val_dir = 'D:\ORTrack\ORTrack-master\data\got10k/val'
        self.lasot_lmdb_dir = 'D:\ORTrack\ORTrack-master\data\lasot_lmdb'
        self.got10k_lmdb_dir = 'D:\ORTrack\ORTrack-master\data\got10k_lmdb'
        self.trackingnet_dir = 'D:\ORTrack\ORTrack-master\data\trackingnet'
        self.trackingnet_lmdb_dir = 'D:\ORTrack\ORTrack-master\data\trackingnet_lmdb'
        self.coco_dir = 'D:\ORTrack\ORTrack-master\data\coco'
        self.coco_lmdb_dir = 'D:\ORTrack\ORTrack-master\data\coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = 'D:\ORTrack\ORTrack-master\data\vid'
        self.imagenet_lmdb_dir = 'D:\ORTrack\ORTrack-master\data\vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
