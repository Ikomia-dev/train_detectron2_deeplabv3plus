import update_path
from ikomia import core, dataprocess
import copy
from ikomia.dnn import datasetio
from ikomia.dnn import dnntrain
import deeplabutils
from detectron2.config import get_cfg, CfgNode
import os
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import launch
import torch
import json
from subprocess import Popen,PIPE
import sys
import yaml
# Your imports below

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainParam(dataprocess.CDnnTrainProcessParam):

    def __init__(self):
        dataprocess.CDnnTrainProcessParam.__init__(self)
        # Place default value initialization here
        self.inputSize=(800,800)
        self.numClasses=2
        self.maxIter = 1000
        self.warmupFactor = 0.001
        self.warmupIters = 200
        self.polyLRFactor = 0.9
        self.polyLRConstantFactor = 0.0
        self.batch_size = 4
        self.resnetDepth = 50
        self.batchNorm = "BN"
        self.ignoreValue = None
        self.learning_rate = 0.02
        self.trainer= None
        self.weights= None
        self.expertModecfg = ""
        self.evalPeriod = 100
        self.earlyStopping = False
        self.patience = 10
        self.splitTrainTest = 90
        self.numGPU = 1

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.inputSize = tuple([int(u) for u in paramMap["inputSize"].split(' ') ])
        self.maxIter = int(paramMap["maxIter"])
        self.batch_size = int(paramMap["batchSize"])
        self.resnetDepth = int(paramMap["resnetDepth"])
        self.expertModecfg = paramMap["expertModecfg"]
        self.evalPeriod = int(paramMap["evalPeriod"])
        self.earlyStopping = bool(paramMap["earlyStopping"])
        self.patience = int(paramMap["patience"])
        self.splitTrainTest = float(paramMap["splitTrainTest"])
        self.numGPU = int(paramMap["numGPU"])
        self.learning_rate = float(paramMap["learning_rate"])


    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = core.ParamMap()
        paramMap["inputSize"] =str(self.inputSize[0])+" "+str(self.inputSize[1])
        paramMap["maxIter"] = str(self.maxIter)
        paramMap["batchSize"] = str(self.batch_size)
        paramMap["resnetDepth"] = str(self.resnetDepth)
        paramMap["expertModecfg"] = self.expertModecfg
        paramMap["evalPeriod"] = str(self.evalPeriod)
        paramMap["earlyStopping"]=str(self.earlyStopping)
        paramMap["patience"]=str(self.patience)
        paramMap["splitTrainTest"]=str(self.splitTrainTest)
        paramMap["numGPU"] = str(self.numGPU)
        paramMap["learning_rate"] = str(self.learning_rate)
        return paramMap

def training_function(cfg,imgs,meta,train_ratio):
    print("ici3")
    deeplabutils.register_train_test(imgs,meta,train_ratio=train_ratio)
    print("ici3")

    trainer = deeplabutils.MyTrainer(cfg)
    print("ici3")

    trainer.resume_or_load(resume=False)
    print("Starting training job...")
    trainer.train()
    
# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        self.addInput(datasetio.IkDatasetIO(dataprocess.DatasetFormat.OTHER))
        # Create parameters class
        if param is None:
            self.setParam(Detectron2_DeepLabV3Plus_TrainParam())
        else:
            self.setParam(copy.deepcopy(param))


    def run(self):       

        # Core function of your process

        input = self.getInput(0)

        # Get parameters :
        param = self.getParam()
        if len(input.data["images"])>0:
            param.epochs = int(param.maxIter / param.batch_size / len(input.data["images"]))
            param.classes = len(input.data["metadata"]["category_names"])
            param.model_name = "DeepLabV3Plus"
            # Call beginTaskRun for initialization
            self.beginTaskRun()
            self.log_param("Input size",str(param.inputSize))
            if param.expertModecfg == "":
                # Get default config
                cfg = get_cfg()
    
                # Add specific deeplab config
                add_deeplab_config(cfg)
                cfg.merge_from_file(os.path.dirname(os.path.realpath(__file__))+"/model/configs/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")

                # Generic dataset names that will be used
                cfg.DATASETS.TRAIN = ("datasetTrain",)
                cfg.DATASETS.TEST = ("datasetTest",)
                cfg.SOLVER.MAX_ITER = param.maxIter
                cfg.SOLVER.WARMUP_FACTOR = 0.001
                cfg.SOLVER.WARMUP_ITERS = param.maxIter//5
                cfg.SOLVER.POLY_LR_FACTOR = 0.9
                cfg.SOLVER.POLY_LR_CONSTANT_FACTOR = 0.0
                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = param.classes
                cfg.SOLVER.BASE_LR = param.learning_rate
                cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
                cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
                cfg.SOLVER.IMS_PER_BATCH=param.batch_size
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.INPUT_SIZE=param.inputSize
                cfg.TEST.EVAL_PERIOD = param.evalPeriod
                cfg.SPLIT_TRAIN_TEST = param.splitTrainTest
                cfg.SPLIT_TRAIN_TEST_SEED = -1
                cfg.MODEL.BACKBONE.FREEZE_AT=5
                cfg.PATIENCE = -1
                
                ###ATTENTION CODE SIMPLIFIE IL MANQUE DES LIGNES    
                
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)   
                
            userpath = os.path.expanduser("~")
            if sys.platform == "linux" or sys.platform == "darwin":
                pythonpath = sys.executable
            else:
                pythonpath = userpath + "/Ikomia/Python/Python.exe"

            filepath= userpath + "/Ikomia/Plugins/Python/Detectron2_DeepLabV3Plus_Train/sub.py"
            
            images = json.dumps(input.data["images"])
            metadata = json.dumps(input.data["metadata"])
            split = str(param.splitTrainTest/100)
            images_file = os.path.join(userpath,"Ikomia/Plugins/Python/Detectron2_DeepLabV3Plus_Train/images.json")
            cfg_file = os.path.join(userpath,"Ikomia/Plugins/Python/Detectron2_DeepLabV3Plus_Train/cfg_param.yaml")
            
            with open(images_file, 'w') as fp:
                json.dump(input.data, fp)            
            with open(cfg_file,'w') as f:
                f.write(cfg.dump())
                
            args = [cfg_file,images_file,split]
            fout = open(userpath+"/OUT.txt",'w')
            ferr = open(userpath+"/ERR.txt",'w')
            p=Popen([pythonpath,filepath,*args], stdout=fout, stderr=ferr)
                
            
            #while p.poll() is None:
             #   data = p.stdout.readline() 
              #  with open("OUT.txt",'a') as f:
               #     f.write(data)
                #data = p.stderr.readline() 
                #with open("ERR.txt",'a') as f:
                 #   f.write(data)
            p.wait()
            
            
            
        # Call endTaskRun to finalize process
        self.endTaskRun()

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.maxIter
        else:
            return 1

    def stop(self):
        super().stop()
        #self.trainer.run=False

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_DeepLabV3Plus_Train"
        self.info.shortDescription = "Training process for DeepLabv3+ model of Detectron2."
        self.info.description = "Implementation from Detectron2 (Facebook Research). " \
                                "This Ikomia plugin can train model from " \
                                "a given config file and a weight file produced by the Ikomia " \
                                "plugin Detectron2_DeepLabV3Plus_Train."
        self.info.authors = "Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.article = "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
        self.info.journal = "ECCV 2018"
        self.info.year = 2018
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, detectron2, facebook, atrous, convolution, encoder, decoder"

    def create(self, param=None):
        # Create process object
        return Detectron2_DeepLabV3Plus_TrainProcess(self.info.name, param)
