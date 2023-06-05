
from torch.optim import SGD, Adam, AdamW

OPTIMIZERS = [
    'adam',
    'adamw',
    'sgd',
]

SHEDULERS = [
    'cosine'
]


def get_optimizer(model, optim_name, optim_params: dict):
    assert optim_name.lower() in OPTIMIZERS, f"{optim_name} not in f{OPTIMIZERS}"

    match optim_name.lower():
        case 'adam':
            return Adam(
                params=model.parameters(),
                lr=optim_params['learning_rate'],
                weight_decay=optim_params['weight_decay']
            )

        case 'adamw':
            return AdamW(
                params=model.parameters(),
                lr=optim_params['learning_rate'],
                weight_decay=optim_params['weight_decay']
            )

        case 'sgd':
            return SGD(
                params=model.parameters(),
                lr=optim_params['learning_rate'],
                momentum=optim_params['momentum']
            )


# TODO: add scheduler parameters
def get_scheduler(optimizer, scheduler_name, scheduler_params: dict):
    assert scheduler_name.lower() in SHEDULERS, f"{scheduler_name} not in f{SHEDULERS}"

    match scheduler_name.lower():
        case 'cosine':
            return
