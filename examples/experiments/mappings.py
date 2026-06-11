from importlib import import_module


class LazyTrainConfig:
    def __init__(self, module_path, class_name="TrainConfig"):
        self.module_path = module_path
        self.class_name = class_name

    def __call__(self, *args, **kwargs):
        module = import_module(self.module_path)
        train_config = getattr(module, self.class_name)
        return train_config(*args, **kwargs)

CONFIG_MAPPING = {
                "ram_insertion": LazyTrainConfig("experiments.ram_insertion.config"),
                "usb_pickup_insertion": LazyTrainConfig("experiments.usb_pickup_insertion.config"),
                "object_handover": LazyTrainConfig("experiments.object_handover.config"),
                "egg_flip": LazyTrainConfig("experiments.egg_flip.config"),
                "robosuite_door": LazyTrainConfig("experiments.robosuite_door.config"),
               }
