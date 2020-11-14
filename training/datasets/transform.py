from torchvision import transforms


def create_transform(model_cfg):
    input_size = model_cfg['input_size']
    mean = model_cfg['mean']
    std = model_cfg['std']
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((19, 19)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return transform, mask_transform
