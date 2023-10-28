# from utils.dataset_tools import maybe_unzip_dataset
import warnings
import torch
from data_loader import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args

warnings.filterwarnings("ignore")
# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
device = torch.device('cuda:1')
model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2,2,40))
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
