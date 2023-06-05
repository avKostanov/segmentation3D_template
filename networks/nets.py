from monai.networks.nets import Unet


def get_architecture(net_name, net_params, num_channels=1, num_classes=1):
    return Unet(
        spatial_dims=3,
        in_channels=num_channels,
        out_channels=num_channels,
        **net_params
    )
