import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# TODO: Your implementation here

class GraphCutCore:
    def __init__(self, image, scribbles):
        self.image = image
        self.scribbles = scribbles  # annotated image with fg, bg labeles
        self.height, self.width = image.shape[:2]
        self.fg_model = None  # foreground color model
        self.bg_model = None  # background color model

    def load_scribbles(self, path):
        """Load scribbles from a given path .
        Create a mask where foreground pixels are labeled as 1 and background pixels as 2.

        Args:
            path (str): Path to the scribbles annotated image.
        """
        self.scribbles = cv2.imread(path)
        mask = np.zeros(self.scribbles.shape[:2], dtype=np.uint8)
        white_pixels = (self.scribbles[:,:,0] > 240) & (self.scribbles[:,:,1] > 240) & (self.scribbles[:,:,2] > 240)
        mask[white_pixels] = 1  #foreground labeled as 1

        red_pixels = (self.scribbles[:,:,2] > 240) & (self.scribbles[:,:,0] < 10) & (self.scribbles[:,:,1] < 10)
        mask[red_pixels] = 2  #background labeled as 2
        return mask


    def build_color_models_histogram(self, bins=16):
        """_summary_


        Args:
            bins (int, optional): _description_. Defaults to 16.
        """

        fg_pixels = self.image[self.scribbles == 1]
        bg_pixels = self.image[self.scribbles == 2]

        hist_fg = np.histogramdd(
            fg_pixels,
            bins=[bins, bins, bins],
            range=[[0, 256], [0, 256], [0, 256]]
        )

        hist_bg = np.histogramdd(
            bg_pixels,
            bins=[bins, bins, bins],
            range=[[0, 256], [0, 256], [0, 256]]
        )

        hist_fg += 1e-6
        hist_bg += 1e-6

        # Normalize to probability distributions
        hist_fg /= hist_fg.sum()
        hist_bg /= hist_bg.sum()

        self.fg_model = hist_fg
        self.bg_model = hist_bg
        self.hist_bins = bins

    def build_color_models_gmm(self, n_components=5):
        """_summary_

        Args:
            n_components (int, optional): _description_. Defaults to 5.

        Raises:
            ValueError: _description_
        """
        fg_pixels = self.image[self.scribbles == 1]
        bg_pixels = self.image[self.scribbles == 2]

        if len(fg_pixels) < n_components or len(bg_pixels) < n_components:
            raise ValueError("Not enough FG/BG pixels to fit GMMs!")

        fg_pixels = fg_pixels.astype(np.float64)
        bg_pixels = bg_pixels.astype(np.float64)

        self.fg_model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=0
        )
        self.bg_model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=0
        )

        self.fg_model.fit(fg_pixels)
        self.bg_model.fit(bg_pixels)

    def compute_unary_costs(self, use_gmm=False):
        """_summary_

        Args:
            use_gmm (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        H, W = self.height, self.width
        epsilon = 1e-10

        if use_gmm:
            pixels = self.image.reshape(-1, 3).astype(np.float64)

            log_prob_bg = self.bg_model.score_samples(pixels)
            log_prob_fg = self.fg_model.score_samples(pixels)

            cost_bg = -log_prob_bg.reshape(H, W)
            cost_fg = -log_prob_fg.reshape(H, W)

        else:
            pixels = self.image.reshape(-1, 3)

            bin_indices = (pixels * self.hist_bins / 256.0).astype(int)
            bin_indices = np.clip(bin_indices, 0, self.hist_bins - 1)

            prob_fg = self.fg_model[
                bin_indices[:, 0], bin_indices[:, 1], bin_indices[:, 2]
            ]
            prob_bg = self.bg_model[
                bin_indices[:, 0], bin_indices[:, 1], bin_indices[:, 2]
            ]

            cost_fg = -np.log(prob_fg + epsilon).reshape(H, W)
            cost_bg = -np.log(prob_bg + epsilon).reshape(H, W)

        # ----------- HARD SCRIBBLE CONSTRAINTS -----------
        large_cost = 1e10

        # Foreground scribbles (label = 1) → force FG
        cost_fg[self.scribbles == 1] = 0
        cost_bg[self.scribbles == 1] = large_cost

        # Background scribbles (label = 2) → force BG
        cost_bg[self.scribbles == 2] = 0
        cost_fg[self.scribbles == 2] = large_cost

        return cost_bg, cost_fg