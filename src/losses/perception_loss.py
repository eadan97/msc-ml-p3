import torch
from torch import nn
from torchvision.transforms import transforms

from src.utils.utils import calculate_gram_matrix
from src.utils.vgg16_extractor import Vgg16Extractor


class PerceptionLoss(nn.Module):

    def __init__(self):
        super(PerceptionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.netVgg = Vgg16Extractor(requires_grad=False)
        self.features_target = None
        self.features_source = None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def normalize_batch(self, batch):
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std

    def set_target_image(self, img):
        img = self.normalize(img)
        self.features_target = self.netVgg(img.detach())

    def set_source_image(self, img):
        img = self.normalize(img)
        self.features_source = self.netVgg(img)

    def get_style_loss(self):
        """
        This shit is unstable in fp 16... gotta look later why and how to fix
        TODO: Read this and fix it
        """
        style_loss = \
            self.mse_loss(
                calculate_gram_matrix(self.features_target['relu1_2']),
                calculate_gram_matrix(self.features_source['relu1_2'])) \
            + self.mse_loss(
                calculate_gram_matrix(self.features_target['relu2_2']),
                calculate_gram_matrix(self.features_source['relu2_2'])) \
            + self.mse_loss(
                calculate_gram_matrix(self.features_target['relu3_3']),
                calculate_gram_matrix(self.features_source['relu3_3'])) \
            + self.mse_loss(
                calculate_gram_matrix(self.features_target['relu4_3']),
                calculate_gram_matrix(self.features_source['relu4_3']))
        return style_loss

    def get_feature_loss(self):
        feature_loss = self.mse_loss(
            self.features_target['relu2_2'],
            self.features_source['relu2_2'])

        return feature_loss
