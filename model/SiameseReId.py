import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


class SNN_classifier(nn.Module):
    def __init__(self):
        super(SNN_classifier, self).__init__()

        # Resnet18 is way more smaller than densenet121
        self.resnet18 = torchvision.models.resnet18(weights=None)
        # remove linear layer to use the fully connected NN
        self.resnet18 = torch.nn.Sequential(
            *(list(self.resnet18.children())[:-1]))

        # Resnet18 Linear layer input shape is 512
        self.fc = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward_resnet18(self, x):
        y = self.resnet18(x)
        y = y.view(y.shape[0], -1)  # reshape to have shape(1,512)
        return y

    def forward(self, img1, img2_fv):
        # extract features vector for img1
        # img2_fv is a pre extracted features vector
        img1_fv = self.forward_resnet18(img1)

        x = torch.cat((img1_fv, img2_fv), axis=1)
        x = self.fc(x)

        y = self.sigmoid(x)  # get probability

        return y


class SiameseReId():
    def __init__(self, weights=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SNN_classifier().to(self.device)

        if weights is not None:
            self.model.load_state_dict(torch.load(
                weights, map_location=self.device))

        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor()
        ])

    def fv_encoding(self, img):
        self.model.eval()
        return self.model.forward_resnet18(self.preprocess(img).to(self.device).unsqueeze(0)).detach()

    def similarity(self, img_det, tid_fv):
        self.model.eval()
        img_det_fv, tid_fv = self.preprocess(img_det).to(
            self.device).unsqueeze(0), tid_fv.to(self.device)
        return self.model(img_det_fv, tid_fv)
