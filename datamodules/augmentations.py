from monai import transforms


def make_augmentation_pipe(prob: float):
    # TODO: validate transform correctness
    interp_mode = {"trilinear", "nearest"}

    return transforms.Compose([
        # intensity transforms
        transforms.OneOf([
            transforms.RandGaussianNoised(keys=['image'], prob=prob, mean=0, std=0.1),
            transforms.RandRicianNoised(keys=['image'], prob=prob, mean=0, std=0.1),
        ]),
        transforms.OneOf([
            transforms.RandGaussianSmoothd(keys=['image'], prob=prob, sigma_x=(0.25, 1.5),
                                           sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
            transforms.RandGaussianSharpend(keys=['image'], prob=prob, mean=0, std=0.1),
        ]),
        transforms.RandAdjustContrastd(keys=['image'], prob=prob, gamma=(0.5, 4.5)),

        # spatial transforms
        transforms.RandFlipd(keys=['image', 'mask'], prob=prob, spatial_axis=0),

        transforms.RandRotated(keys=['image', 'mask'], prob=prob, range_x=[0.4, 0.4], mode=interp_mode),
        transforms.RandAffined(keys=['image', 'mask'], prob=prob, mode=interp_mode),
        transforms.OneOf(
            [
                transforms.Rand3DElasticd(keys=['image', 'mask'], prob=prob, mode=interp_mode),
                transforms.RandGridDistortiond(keys=['image', 'mask'], prob=prob, prob=prob, mode=interp_mode)
            ]
        )
    ])
