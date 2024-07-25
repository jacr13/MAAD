from agents.models.inv_dyn_model.base import InvDynModelBase


def get_idm(type_model: str) -> InvDynModelBase:
    if type_model == "mlp":
        from agents.models.inv_dyn_model.mlp import InvDynModelMLP

        return InvDynModelMLP
    elif type_model == "cnn":
        from agents.models.inv_dyn_model.cnn import InvDynModelCNN

        return InvDynModelCNN
    elif type_model == "mdn":
        from agents.models.inv_dyn_model.mdn import InvDynModelMDN

        return InvDynModelMDN
    else:
        raise ValueError(f"{type_model} is not supported")
