"""
    Generator class for the GAN.
"""

# Modules
import torch
import torch.nn as nn

class G(nn.Module):

    def __init__(self):
        """ Constructor """

        # Extends torch.nn
        super(G, self).__init__()

        # Generator modules
        self.main = nn.Sequential(
                    # 1st Inverse convolution
                    nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),

                    # 2nd inverse convolution
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),

                    # 3rd inverse convolution
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),

                    # 4th inverse convolution
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),

                    # 5th inverse convolution
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                    nn.Tanh()
                    )

    def forward(self, input):
        """
            Forward propagation function,
            which will forward propagate
            the noise through the neural
            networ.

            Arguments:
                param1: Random noise (input of G)
         """
         # Compute output
         output = self.main(input)

         # Return generator output
         return output
