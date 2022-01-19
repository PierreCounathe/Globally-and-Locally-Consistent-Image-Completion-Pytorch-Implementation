from models import *
from trainers import *
from utils import *

if __name__ == "__main__":

    # Initialization.
    use_cuda = torch.cuda.is_available()
    lr_d = 1.0
    lr_c = 1.0
    model_c = Completion().cuda() if use_cuda else Completion()
    model_d = Discriminator().cuda() if use_cuda else Discriminator()
    opt_d = torch.optim.Adadelta(model_d.parameters(), lr=lr_d)
    opt_c = torch.optim.Adadelta(model_c.parameters(), lr=lr_c)

    # Loading presaved models.
    load_checkpoint(model_c, opt_c, "model_c_checkpoint_eot9.pth.tar", use_cuda=use_cuda)
    load_checkpoint(model_d, opt_d, "model_d_checkpoint_eot9.pth.tar", use_cuda=use_cuda)

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
        use_cuda=use_cuda,
        pixel=(130, 107, 95),
    )
