import update_path
from ikomia import core, dataprocess
import copy
from ikomia.dnn import dataset, datasetio
from detectron2.config import get_cfg
from ikomia.dnn import dnntrain
import deeplabutils
import os
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer,DefaultPredictor
import json
import torch
import os
from detectron2.projects.deeplab import add_deeplab_config
import numpy
from detectron2.modeling import build_model

# Your imports below

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainParam(dataprocess.CDnnTrainProcessParam):

    def __init__(self):
        dataprocess.CDnnTrainProcessParam.__init__(self)
        # Place default value initialization here
        self.cfg = get_cfg()
        self.inputSize=(800,800)
        self.numClasses=2
        self.maxIter = 1000
        self.warmupFactor = 0.001
        self.warmupIters = 200
        self.polyLRFactor = 0.9
        self.polyLRConstantFactor = 0.0
        self.batchSize = 4
        self.resnetDepth = 50
        self.freezeAt = 4
        self.batchNorm = "BN"
        self.ignoreValue = None
        self.baseLearningRate = 0.02
        self.trainer= None
        self.weights= None
        self.expertModecfg = ""
        self.evalPeriod = 100
        self.earlyStopping = False
        self.patience = 10
        self.splitTrainTest = 90

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        self.cfg = paramMap["cfg"]
        self.inputSize = paramMap["inputSize"]
        #self.numClasses = paramMap["numClasses"]
        self.maxIter = paramMap["maxIter"]
        #self.warmupFactor = paramMap["warmupFactor"]
        #self.warmupIters = paramMap["warmupIters"]
        #self.polyLRFactor = paramMap["polyLRFactor"]
        #self.polyLRConstantFactor = paramMap["polyLRConstantFactor"]
        self.batchSize = paramMap["batchSize"]
        self.resnetDepth = paramMap["resnetDepth"]
        self.freezeAt = paramMap["freezeAt"]
        #self.batchNorm = paramMap["BN"]
        self.ignoreValue = paramMap["ignoreValue"]
        self.expertModecfg = paramMap["expertModecfg"]
        self.evalPeriod = paramMap["evalPeriod"]
        self.earlyStopping = paramMap["earlyStopping"]
        self.patience = paramMap["patience"]
        self.splitTrainTest = paramMap["splitTrainTest"]


    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = core.ParamMap()
        paramMap["cfg"] = self.cfg
        paramMap["inputSize"] =self.inputSize
        #paramMap["numClasses"] =self.numClasses
        paramMap["maxIter"] = self.maxIter
        #paramMap["warmupFactor"] = self.warmupFactor
        paramMap["warmupIters"] = self.warmupIters
        #paramMap["polyLRFactor"] = self.polyLRFactor
        #paramMap["polyLRConstantFactor"] = self.polyLRConstantFactor
        paramMap["batchSize"] = self.batchSize
        paramMap["resnetDepth"] = self.resnetDepth
        paramMap["freezeAt"] = self.freezeAt
        #paramMap["BN"] = self.batchNorm
        paramMap["ignoreValue"] = self.ignoreValue
        paramMap["expertModecfg"] = self.expertModecfg
        paramMap["evalPeriod"] = self.evalPeriod
        paramMap["earlyStopping"]=self.earlyStopping
        paramMap["patience"]=self.patience
        paramMap["splitTrainTest"]=self.splitTrainTest
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        # Example :  self.addInput(PyDataProcess.CImageProcessIO())
        #           self.addOutput(PyDataProcess.CImageProcessIO())
        self.addInput(datasetio.IkDatasetIO(dataprocess.DatasetFormat.OTHER))
        # Create parameters class
        if param is None:
            self.setParam(Detectron2_DeepLabV3Plus_TrainParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        # Examples :
        # Get input :
        input = self.getInput(0)

        # Get parameters :
        param = self.getParam()
        if len(input.data["images"])>0:
            if param.expertModecfg == "":
                cfg = get_cfg()
                add_deeplab_config(cfg)
                cfg.merge_from_file(os.path.dirname(os.path.realpath(__file__))+"/model/configs/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
                cfg.DATASETS.TRAIN = ("datasetTrain",)
                cfg.DATASETS.TEST = ("datasetTest",)
                cfg.SOLVER.MAX_ITER = param.maxIter
                cfg.SOLVER.WARMUP_FACTOR = 0.001
                cfg.SOLVER.WARMUP_ITERS = param.maxIter//10
                cfg.SOLVER.POLY_LR_FACTOR = 0.9
                cfg.SOLVER.POLY_LR_CONSTANT_FACTOR = 0.0
                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(input.data["metadata"]["category_names"])
                cfg.SOLVER.BASE_LR = param.baseLearningRate
                cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
                cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
                cfg.SOLVER.IMS_PER_BATCH=param.batchSize
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.INPUT_SIZE=param.inputSize
                cfg.TEST.EVAL_PERIOD = param.evalPeriod
                cfg.SPLIT_TRAIN_TEST = param.splitTrainTest
                cfg.SPLIT_TRAIN_TEST_SEED = None
                if param.earlyStopping:
                    cfg.PATIENCE = param.patience
                else:
                    cfg.PATIENCE = -1

                cfg.OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__))+"/output"
            else:
                cfg = None
                with open(param.expertModecfg, 'r') as file:
                    cfg_data = file.read()
                    cfg = CfgNode.load_cfg(cfg_data)
            if cfg is not None:
                deeplabutils.register_train_test(input.data["images"],input.data["metadata"],train_ratio=cfg.SPLIT_TRAIN_TEST/100,seed=cfg.SPLIT_TRAIN_TEST_SEED)

                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

                self.trainer = deeplabutils.MyTrainer(cfg,self)
                self.trainer.resume_or_load(resume=False)
                print("Starting training job...")
                self.trainer.train()
                print("Training job finished.")
                with open(cfg.OUTPUT_DIR+"/Detectron2_DeepLabV3Plus_Train_Config.yaml", 'w') as file:
                    file.write(cfg.dump())
            else :
                print("Error : can't load config file "+param.expertModecfg)
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
        self.trainer.run=False

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_DeepLabV3Plus_Train"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "detectron2, segmentation, semantic"

    def create(self, param=None):
        # Create process object
        return Detectron2_DeepLabV3Plus_TrainProcess(self.info.name, param)
