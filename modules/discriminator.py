"""
    Discriminator class for the GAN.
"""

# Modules
import torch
import torch.nn as nn

class D(nn.Module):

    def __init__(self):
        """ Constructor """

        # Extends torch.nn
        super(D, self).__init__()

        # Discriminator modules
        self.main = nn.Sequential(
                    # 1st convolution
                    nn.Conv2d(3, 64, 4, 2, 1, bias = False),
                    nn.LeakyReLu(0.2, inplace = True),

                    # 2nd convolution
                    nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLu(0.2, inplace = True),

                    # 3rd convolution
                    nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLu(0.2, inplace = True),

                    # 4th convolution
                    nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLu(0.2, inplace = True),

                    # 4th convolution
                    nn.Conv2d(512, 1, 4, 1, 0, bias = False),
                    nn.Sigmoid()
                    )

    def forward(self, input):
        """
            Forward propagation function,
            which will return the discrimination
            value.

            Arguments:
                param1: Image crate by the generator
         """
         # Compute output
         output = self.main(input)

         # Return discriminator output
         return output.view(-1)
