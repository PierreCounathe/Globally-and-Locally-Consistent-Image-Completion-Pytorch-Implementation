from models import *
from trainers import *
from utils import *
from torchvision import transforms

if __name__ == "__main__":

    # Initialization.
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    lr_d = 1.0
    lr_c = 1.0
    model_c = Completion().to(device)
    model_d = Discriminator().to(device)
    opt_d = torch.optim.Adadelta(model_d.parameters(), lr=lr_d)
    opt_c = torch.optim.Adadelta(model_c.parameters(), lr=lr_c)

    # Set paths for loading.
    path_completion = "weights/model_c_checkpoint_eot9.pth.tar"
    path_discriminator = "weights/model_d_checkpoint_eot9.pth.tar"

    # Loading presaved models.
    load_checkpoint(model_c, opt_c, path_completion, device=device)
    load_checkpoint(model_d, opt_d, path_discriminator, device=device)

    # Creating test_set and train_set.
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((256, 256), scale=(0.6666666666667, 1.0), ratio=(1.0, 1.0)),
            transforms.ToTensor(),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(root="./data", transform=transform, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./data", transform=transform, train=False, download=True)
    dataset_with_labels = True
    test_loader = torch.utils.data.DataLoader(test_set)

    # Generate recompleted images.
    test_and_compare(
        model_c=model_c,
        model_d=model_d,
        test_loader=test_loader,
        number_of_pictures=5,
        h_range_mask=(48, 50),
        w_range_mask=(48, 50),
        num_holes=2,
        p=0.01,
        dataset_with_labels=dataset_with_labels,
        device=device,
        pixel=(130, 107, 95),
    )
