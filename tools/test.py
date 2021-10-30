from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job, load_backbone
from fashionnets.train_jobs.loader.backbone_loader import load_backbone_info_resnet
from collections import defaultdict
import tensorflow as tf
from fashiondatasets.deepfashion2.DeepFashionCBIR import DeepFashionCBIR
from fashiondatasets.own.helper.mappings import preprocess_image

model_cp = r"F:\workspace\FashNets\34_resnet50_None_triplet0006"

load_backbone(model_cp, input_shape=(224, 224), verbose=True)