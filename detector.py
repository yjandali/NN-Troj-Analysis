import json
import pickle
from collections import OrderedDict
from os import listdir
from os.path import join

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils.flatten import flatten_model, pad_to_target
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)


class Detector:
    model_padding = {
        "MobileNetV2": {"classifier.1.weight": [138, 1280], "classifier.1.bias": [138]},
        "ResNet": {"fc.weight": [138, 2048], "fc.bias": [138]},
        "VisionTransformer": {"head.weight": [138, 768], "head.bias": [138]},
    }

    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        self.model_filepath = join(learned_parameters_dirpath, "data/model.bin")
        self.model_layer_map_filepath = join(
            learned_parameters_dirpath, "data/model_layer_map.bin"
        )
        self.layer_transform_filepath = join(
            learned_parameters_dirpath, "data/layer_transform.bin"
        )

        self.model_skew = {
            "MobileNetV2": metaparameters["infer_model_skew_mobilenetv2"],
            "ResNet": metaparameters["infer_model_skew_resnet"],
            "VisionTransformer": metaparameters["infer_model_skew_visiontransformer"],
        }
        self.normalize = metaparameters["infer_normalize_features"]

        self.input_features = metaparameters["train_input_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_random_std"],
            "scaler": metaparameters["train_weight_table_random_scaler"],
        }
        self.random_forest_kwargs = {
            "n_estimator": metaparameters[
                "train_random_forest_regressor_param_n_estimator"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_sample_leaf": metaparameters[
                "train_random_forest_regressor_param_min_sample_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def _load_model(self, model_path, configure_mode=False):
        model = torch.load(join(model_path, "model.pt"))
        model_class = model.__class__.__name__
        model_repr = OrderedDict(
            {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
        )

        # Load ground truth data if in configure mode
        model_ground_truth = None
        if configure_mode:
            with open(join(model_path, "ground_truth.csv"), "r") as fp:
                model_ground_truth = fp.readlines()[0]

        # Ensure every layer is correctly padded, so that every model has the same
        # number of weights no matter the number of classes
        for (layer, target_padding) in self.model_padding[model_class].items():
            model_repr[layer] = pad_to_target(model_repr[layer], target_padding)

        return model_repr, model_class, model_ground_truth

    def configure(self, models_dirpath, auto_training=True):
        # FIXME add auto-training
        # List all available model and limit to the number provided
        model_path_list = sorted(
            [join(models_dirpath, model) for model in listdir(models_dirpath)]
        )
        print(f"Found {len(model_path_list)} models!")

        model_repr_dict = {}
        model_ground_truth_dict = {}

        for model_path in tqdm(model_path_list):
            model_repr, model_class, model_ground_truth = self._load_model(
                model_path, configure_mode=True
            )

            # Build the list of models
            if model_class not in model_repr_dict.keys():
                model_repr_dict[model_class] = []
                model_ground_truth_dict[model_class] = []

            model_repr_dict[model_class].append(model_repr)
            model_ground_truth_dict[model_class].append(model_ground_truth)

        check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        print("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        print("Generated model layer map!")

        # Flatten model
        print("Flattenning models")
        new_models = {}
        for (model_arch, models) in model_repr_dict.items():
            if model_arch not in new_models.keys():
                new_models[model_arch] = []

            for model in models:
                new_models[model_arch].append(
                    flatten_model(model, model_layer_map[model_arch])
                )

        del model_repr_dict
        print("Model flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(
            new_models, self.weight_table_params, self.input_features
        )
        with open(self.layer_transform_filepath, "wb") as fp:
            pickle.dump(layer_transform, fp)

        print("Model flattened. Fitting feature reduction...")
        X = None
        y = []

        for (model_arch, models) in new_models.items():
            model_index = 0

            for model in models:
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1

                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_arch], model
                )
                if X is None:
                    X = model_feats
                    continue

                X = np.vstack((X, model_feats))

        model = RandomForestRegressor(
            **self.random_forest_kwargs,
            random_state=0,
        )
        model.fit(X, y)

        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        # TODO implement per round inferencing examples
        model_repr, model_class, _ = self._load_model(
            model_filepath, configure_mode=True
        )

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        new_model = flatten_model(model_repr, model_layer_map[model_class])

        with open(self.layer_transform_filepath, "rb") as fp:
            layer_transform = pickle.load(fp)

        X = use_feature_reduction_algorithm(layer_transform[model_class], new_model)

        with open(self.model_filepath, "wb") as fp:
            model: RandomForestRegressor = pickle.load(fp)

        with open(result_filepath, "r") as fp:
            fp.write(model.predict(X))