from utils import *
from models import *
import torch
from tqdm import tqdm


def C_loss(
    input, output, mask
):  # The loss for the completion network is calculed only in the completed region !
    return torch.nn.functional.mse_loss(output * mask, input * mask)


def train_C(
    model_c,
    optimizer,
    train_set,
    train_acc_period=100,
    n_epoch=5,
    save_period=10000,
    batch_size=8,
    num_samples=1000,
    use_cuda=True,
    dataset_with_labels=False,
    pixel=(130, 107, 95),
):  # On pourra éventuellement rajouter un test sur une image à intervalle réguliers

    model_c.train()
    i = 0
    for epoch in tqdm(range(n_epoch)):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=6,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=num_samples),
            pin_memory=True,
            drop_last=True,
        )  # On en parlera de ces paramètres là !
        running_loss = 0.0
        for data in tqdm(train_loader):
            if dataset_with_labels:
                x, y = data
            else:
                x = data

            if use_cuda:
                x = x.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            m, _ = mask(batch_size, use_cuda=use_cuda)
            inputs = apply_mask(x, m, pixel, use_cuda=use_cuda)
            outputs = model_c(inputs)

            loss = C_loss(x, outputs, m)
            loss.backward()
            optimizer.step()

            # printing some statistics
            running_loss = 0.33 * loss.item() / batch_size + 0.66 * running_loss
            if i % train_acc_period == train_acc_period - 1:
                print("[%d, %5d] loss: %f" % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            i += 1
            del x  # Je sais pas si c'est utile ! mon but c'est de désalouer la mémoire pour être sûr de pas dépasser la capacité de CUDA
        if (epoch % save_period == save_period - 1) or (epoch == n_epoch - 1):
            filename = "model_c_save/model_c_checkpoint_c_training_epoch{:d}.pth.tar".format(epoch + 1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_c.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=filename,
            )
    print("Finished Training C")


def train_D(
    model_d,
    model_c,
    optimizer,
    train_set,
    train_acc_period=100,
    n_epoch=5,
    save_period=10000,
    batch_size=8,
    num_samples=1000,
    use_cuda=True,
    dataset_with_labels=False,
    pixel=(130, 107, 95),
):

    model_c.eval()
    model_d.train()
    i = 0
    criterion = torch.nn.BCELoss()
    for epoch in tqdm(range(n_epoch)):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=6,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=num_samples),
            pin_memory=True,
            drop_last=True,
        )  # On en parlera de ces paramètres là !
        running_loss = 0.0
        for data in tqdm(train_loader):
            if dataset_with_labels:
                x, _ = data
            else:
                x = data

            if use_cuda:
                x = x.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Feeding the network with completed images

            mask_c, centers_c = mask(batch_size, use_cuda=use_cuda)
            input_c = apply_mask(x, mask_c, pixel, use_cuda=use_cuda)
            output_c = model_c(input_c)
            false = torch.zeros(
                (batch_size, 1)
            )  # The context discriminator should label all images as fake
            if use_cuda:
                false = false.cuda()
            input_gd_fake = (
                output_c.detach()
            )  # We don't need the gradients for this batch, as the Generator is not trained here
            input_ld_fake = hole_cropping(input_gd_fake, centers_c, use_cuda=use_cuda)
            output_fake = model_d((input_ld_fake, input_gd_fake))
            loss_fake = criterion(output_fake, false)  # Loss if labelled as true

            # Feeding the network with real images
            centers_d = mask(
                batch_size, use_cuda=use_cuda, generate_mask=False
            )  # Only generate the hole centers, but not the tensors as it is expensive !
            input_gd_real = x
            input_ld_real = hole_cropping(input_gd_real, centers_d, use_cuda=use_cuda)
            output_real = model_d((input_ld_real, input_gd_real))
            real = torch.ones(
                (batch_size, 1)
            )  # Tensor of ones indicating that images are labelled as true (1)
            if use_cuda:
                real = real.cuda()
            loss_real = criterion(output_real, real)  # Loss if labelled as false

            # Recombining the losses
            loss = (
                loss_real + loss_fake
            ) / 2.0  # The coefficient of each kind of losses could be a way to personalize the net !
            loss.backward()
            optimizer.step()

            # printing some statistics
            running_loss = 0.33 * loss.item() / batch_size + 0.66 * running_loss
            if i % train_acc_period == train_acc_period - 1:
                print("[%d, %5d] loss: %f" % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            i += 1
            del x  # Je sais pas si c'est utile ! mon but c'est de désalouer la mémoire pour être sûr de pas dépasser la capacité de CUDA
        if (epoch % save_period == save_period - 1) or (epoch == n_epoch - 1):
            filename = "model_d_save/model_d_checkpoint_d_training_epoch{:d}.pth.tar".format(epoch + 1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_d.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=filename,
            )
    print("Finished Training D")


def train_C_and_D(
    model_c,
    model_d,
    alpha,
    optimizer_c,
    optimizer_d,
    train_set,
    train_acc_period=100,
    n_epoch=5,
    save_period=10000,
    batch_size=8,
    num_samples=1000,
    use_cuda=True,
    dataset_with_labels=False,
    pixel=(130, 107, 95),
):

    running_loss_c = 0
    running_loss_d = 0
    model_c.train()
    model_d.train()
    i = 0
    criterion = torch.nn.BCELoss()
    for epoch in tqdm(range(n_epoch)):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=6,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=num_samples),
            pin_memory=True,
            drop_last=True,
        )
        for data in tqdm(
            train_loader
        ):  # Normalement on prendra un dataset unsupervised donc sans labels, mais on laisse le choix
            if dataset_with_labels:
                x, _ = data
            else:
                x = data

            if use_cuda:
                x = x.cuda()

            # zero the parameter gradients

            optimizer_c.zero_grad()
            optimizer_d.zero_grad()

            # Training the completion network

            mask_c, centers_c = mask(batch_size, use_cuda=use_cuda)
            input_c = apply_mask(x, mask_c, pixel, use_cuda=use_cuda)
            output_c = model_c(input_c)
            loss_c_1 = C_loss(x, output_c, mask_c)

            # Feeding the discrimination network with completed images

            false = torch.zeros(
                (batch_size, 1)
            )  # The context discriminator should label all images as fake
            if use_cuda:
                false = false.cuda()
            input_gd_fake = (
                output_c.detach()
            )  # We detach here so that the gradients are not used, only doing a backward pass for D
            input_ld_fake = hole_cropping(input_gd_fake, centers_c, use_cuda=use_cuda)
            output_fake = model_d((input_ld_fake, input_gd_fake))
            loss_fake = criterion(output_fake, false)  # Loss if labelled as true

            # Feeding the discrimination network with real images

            centers_d = mask(batch_size, use_cuda=use_cuda, generate_mask=False)
            input_gd_real = x
            input_ld_real = hole_cropping(
                input_gd_real, centers_d, use_cuda=use_cuda
            )  # I used the same mask than in the false feeding, i think it's okay and faster.
            output_real = model_d((input_ld_real, input_gd_real))
            real = torch.ones(
                (batch_size, 1)
            )  # Tensor of ones indicating that images are labelled as true (1)
            if use_cuda:
                real = real.cuda()
            loss_real = criterion(output_real, real)  # Loss if labelled as false

            # Updating the Discriminator model

            loss_d = alpha * (
                loss_real + loss_fake
            )  # The coefficient of each kind of losses could be a way to personalize the net !
            loss_d.backward()
            optimizer_d.step()

            # Computing the second loss for the completion network

            input_gd_fake = output_c  # Not detaching now because we will use the gradients
            input_ld_fake = hole_cropping(input_gd_fake, centers_c, use_cuda=use_cuda)
            output_fake = model_d((input_ld_fake, input_gd_fake))
            loss_c_2 = criterion(
                output_fake, real
            )  # Indeed, the completion network must lure the discriminatoir, hence this loss !

            # Updating the Completion model

            loss_c = loss_c_1 + alpha * loss_c_2
            loss_c.backward()
            optimizer_c.step()

            # printing some statistics

            running_loss_c = 0.33 * loss_c.item() / batch_size + 0.66 * running_loss_c
            running_loss_d = 0.33 * loss_d.item() / batch_size + 0.66 * running_loss_d
            if i % train_acc_period == train_acc_period - 1:
                print(
                    "[%d, %5d] loss for the completion network : %f" % (epoch + 1, i + 1, running_loss_c)
                )
                print(
                    "[%d, %5d] loss for the discrimination network : %f"
                    % (epoch + 1, i + 1, running_loss_d)
                )
                running_loss_c, running_loss_d = 0, 0
            i += 1
            del x  # Je sais pas si c'est utile ! mon but c'est de désalouer la mémoire pour être sûr de pas dépasser la capacité de CUDA
        if (epoch % save_period == save_period - 1) or (epoch == n_epoch - 1):
            filename1 = "model_c_save/model_c_checkpoint_c_and_d_training_epoch{:d}.pth.tar".format(
                epoch + 1
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_c.state_dict(),
                    "optimizer": optimizer_c.state_dict(),
                },
                filename=filename1,
            )
            filename2 = "model_d_save/model_d_checkpoint_c_and_d_training_epoch{:d}.pth.tar".format(
                epoch + 1
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_d.state_dict(),
                    "optimizer": optimizer_d.state_dict(),
                },
                filename=filename2,
            )
    print("Finished Training C&D simultaneously")


if __name__ == "__main__":

    # Override this parameter if you want to use CPU (not recommended).
    use_cuda = torch.cuda.is_available()

    # Initializing ADADELTA models and optimizers.
    model_c = Completion().cuda() if use_cuda else Completion()
    model_d = Discriminator().cuda() if use_cuda else Discriminator()
    lr_d = 1.0
    lr_c = 1.0
    opt_d = torch.optim.Adadelta(model_d.parameters(), lr=lr_d)
    opt_c = torch.optim.Adadelta(model_c.parameters(), lr=lr_c)

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

    # Computing mean pixel value of the dataset.
    # NOTE: this mean value will be reused at inference time, make sure to save it somewhere.
    # For CIFAR10 : mean_pixel = (124.9266, 121.8598, 112.7152)
    mean_pixel = pixel_moyen(train_set, dataset_with_labels=dataset_with_labels)

    # Training only the Completion network (Phase 1).
    train_C(
        model_c,
        opt_c,
        train_set,
        train_acc_period=100,
        n_epoch=5,
        save_period=1,
        batch_size=2,
        num_samples=1000,
        use_cuda=use_cuda,
        dataset_with_labels=dataset_with_labels,
        pixel=mean_pixel,
    )

    # Training only the Discriminator network (Phase 2).
    train_D(
        model_d,
        model_c,
        opt_c,
        train_set,
        train_acc_period=100,
        n_epoch=5,
        save_period=1,
        batch_size=2,
        num_samples=1000,
        use_cuda=use_cuda,
        dataset_with_labels=dataset_with_labels,
        pixel=(130, 107, 95),
    )

    # Training both models jointly (Phase 3).

    # This parameter defines the balance between both
    # discriminator loss and L2 loss for the completion network.
    alpha = 4e-4

    train_C_and_D(
        model_c,
        model_d,
        alpha,
        opt_c,
        opt_d,
        train_set,
        train_acc_period=100,
        n_epoch=5,
        save_period=1,
        batch_size=2,
        num_samples=1000,
        use_cuda=use_cuda,
        dataset_with_labels=dataset_with_labels,
        pixel=(130, 107, 95),
    )
