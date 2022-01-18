from torch import cat, nn


class Completion(nn.Module):
    def __init__(self):
        super(Completion, self).__init__()
        # ----------------
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2
        )
        self.norm1 = nn.BatchNorm2d(64)
        # ---------------- CHANGE IN THE NUMBER OF OUTPUTS
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm3 = nn.BatchNorm2d(128)
        # ---------------- CHANGE IN THE NUMBER OF OUTPUTS
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, dilation=1, padding=1
        )
        self.norm4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm6 = nn.BatchNorm2d(256)

        self.dilatedconv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=2, padding=2
        )
        self.norm7 = nn.BatchNorm2d(256)

        self.dilatedconv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=4, padding=4
        )
        self.norm8 = nn.BatchNorm2d(256)

        self.dilatedconv9 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=8, padding=8
        )
        self.norm9 = nn.BatchNorm2d(256)

        self.dilatedconv10 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=16, padding=16
        )
        self.norm10 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm11 = nn.BatchNorm2d(256)

        self.conv12 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm12 = nn.BatchNorm2d(256)
        # ----------------- CHANGE IN THE NUMBER OF OUTPUTS
        self.deconv13 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, dilation=1, padding=1
        )  # Convtranspose stride = 2 is the 1/2 stride of the paper
        self.norm13 = nn.BatchNorm2d(128)

        self.conv14 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm14 = nn.BatchNorm2d(128)
        # ----------------- CHANGE IN THE NUMBER OF OUTPUTS
        self.deconv15 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, dilation=1, padding=1
        )
        self.norm15 = nn.BatchNorm2d(64)

        self.conv16 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1
        )
        self.norm16 = nn.BatchNorm2d(32)

        self.output = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, stride=1, dilation=1, padding=1
        )

        self.inside_activation = nn.ReLU(True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.norm1(self.inside_activation(self.conv1(x)))
        out = self.norm2(self.inside_activation(self.conv2(out)))
        out = self.norm3(self.inside_activation(self.conv3(out)))
        out = self.norm4(self.inside_activation(self.conv4(out)))
        out = self.norm5(self.inside_activation(self.conv5(out)))
        out = self.norm6(self.inside_activation(self.conv6(out)))
        out = self.norm7(self.inside_activation(self.dilatedconv7(out)))
        out = self.norm8(self.inside_activation(self.dilatedconv8(out)))
        out = self.norm9(self.inside_activation(self.dilatedconv9(out)))
        out = self.norm10(self.inside_activation(self.dilatedconv10(out)))
        out = self.norm11(self.inside_activation(self.conv11(out)))
        out = self.norm12(self.inside_activation(self.conv12(out)))
        out = self.norm13(self.inside_activation(self.deconv13(out)))
        out = self.norm14(self.inside_activation(self.conv14(out)))
        out = self.norm15(self.inside_activation(self.deconv15(out)))
        out = self.norm16(self.inside_activation(self.conv16(out)))
        return self.output_activation(self.output(out))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # LOCAL DISCRIMINATOR
        # input shape : [batch_size, 3, 128, 128] --> 3 x 128 x 128

        # ----------------
        self.Local_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.Local_norm1 = nn.BatchNorm2d(64)
        # shape : [batch_size, 128, 64, 64] --> 64 x 64 x 64
        self.Local_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.Local_norm2 = nn.BatchNorm2d(128)
        # shape : [batch_size, 256, 32, 32] --> 128 x 32 x 32

        self.Local_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.Local_norm3 = nn.BatchNorm2d(256)
        # shape : [batch_size, 512, 16, 16] --> 256 x 16 x 16

        self.Local_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.Local_norm4 = nn.BatchNorm2d(512)
        # shape : [batch_size, 512, 8, 8] --> 512 x 8 x 8

        self.Local_conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.Local_norm5 = nn.BatchNorm2d(512)
        # shape : [batch_size, 512, 4, 4] --> 512 x 4 x 4 = 8 192 features

        # GLOBAL DISCRIMINATOR
        # input shape : [batch_size, 3, 256, 256] --> 3 x 256 x 256

        # ----------------
        self.Global_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.Global_norm1 = nn.BatchNorm2d(64)
        # shape : [batch_size, 64, 128, 128] --> 64 x 128 x 128

        self.Global_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.Global_norm2 = nn.BatchNorm2d(128)
        # shape : [batch_size, 128, 64, 64] --> 128 x 64 x 64

        self.Global_conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2
        )
        self.Global_norm3 = nn.BatchNorm2d(256)
        # shape : [batch_size, 256, 32, 32] --> 256 x 32 x 32

        self.Global_conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2
        )
        self.Global_norm4 = nn.BatchNorm2d(512)
        # shape : [batch_size, 512, 16, 16] --> 512 x 16 x 16

        self.Global_conv5 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2
        )
        self.Global_norm5 = nn.BatchNorm2d(512)
        # shape : [batch_size, 512, 8, 8] --> 512 x 8 x 8

        self.Global_conv6 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2
        )
        self.Global_norm6 = nn.BatchNorm2d(512)
        # shape : [batch_size, 512, 4, 4] --> 512 x 4 x 4 = 8 192 features

        # ----------------

        self.out_features = 8192
        self.fc = nn.Linear(self.out_features, 1024)

        # ---------------- COMBINATION AND CONCATENATION

        self.final_fc = nn.Linear(2048, 1)
        self.inside_activation = nn.ReLU(True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        local_x, global_x = x  # En entrée on passera un tuple avec d'abord l'image locale puis globale !

        Y = self.Local_norm1(self.inside_activation(self.Local_conv1(local_x)))
        Y = self.Local_norm2(self.inside_activation(self.Local_conv2(Y)))
        Y = self.Local_norm3(self.inside_activation(self.Local_conv3(Y)))
        Y = self.Local_norm4(self.inside_activation(self.Local_conv4(Y)))
        Y = self.Local_norm5(self.inside_activation(self.Local_conv5(Y)))
        Y = self.inside_activation(self.fc(Y.view(-1, self.out_features)))

        Z = self.Global_norm1(self.inside_activation(self.Global_conv1(global_x)))
        Z = self.Global_norm2(self.inside_activation(self.Global_conv2(Z)))
        Z = self.Global_norm3(self.inside_activation(self.Global_conv3(Z)))
        Z = self.Global_norm4(self.inside_activation(self.Global_conv4(Z)))
        Z = self.Global_norm5(self.inside_activation(self.Global_conv5(Z)))
        Z = self.Global_norm6(self.inside_activation(self.Global_conv6(Z)))
        Z = self.inside_activation(self.fc(Z.view(-1, self.out_features)))

        out = cat((Y, Z), dim=1)
        out = self.final_fc(out)
        return self.output_activation(out)
