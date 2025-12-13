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
        # cv2.imshow("Scribbles", self.scribbles)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # converted to rgb as i realised that the colors were flipped
        # and this was messing up my labeling
        self.scribbles = cv2.cvtColor(self.scribbles, cv2.COLOR_BGR2RGB)

        mask = np.zeros(self.scribbles.shape[:2], dtype=np.uint8)
        white_pixels = (self.scribbles[:,:,0] > 200) & (self.scribbles[:,:,1] > 200) & (self.scribbles[:,:,2] > 200)
        mask[white_pixels] = 1  #foreground labeled as 1

        red_pixels = (self.scribbles[:,:,0] > 200) & (self.scribbles[:,:,1] < 50) & (self.scribbles[:,:,2] < 50)
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

        large_cost = 1e10

        # Foreground scribbles must stay foreground
        cost_fg[self.scribbles == 1] = 0
        cost_bg[self.scribbles == 1] = large_cost

        # Background scribbles must stay background
        cost_bg[self.scribbles == 2] = 0
        cost_fg[self.scribbles == 2] = large_cost

        return cost_bg, cost_fg
    
    def compute_pairwise_weight(self, p, q, beta=0.01, lam=50):
        """
        Compute pairwise edge weight between two pixels.

        Args:
            p, q: pixel coordinates (i, j)
            beta: controls sensitivity to color difference
            lam: smoothness strength

        Returns:
            float: pairwise cost
        """
        diff = self.image[p] - self.image[q]
        dist2 = np.sum(diff * diff)
        return lam * np.exp(-beta * dist2)

    def construct_graph(self, cost_bg, cost_fg):
        """
        Build the graph for max-flow/min-cut.

        Args:
            cost_bg: unary cost for background
            cost_fg: unary cost for foreground

        Returns:
            graph, node_ids
        """
        H, W = self.height, self.width

        graph = maxflow.Graph[float]()
        node_ids = graph.add_nodes(H * W)

        def node_index(i, j):
            return i * W + j

        # Add unary (t-link) costs
        for i in range(H):
            for j in range(W):
                idx = node_index(i, j)
                graph.add_tedge(
                    node_ids[idx],
                    cost_fg[i, j],  # source → node (FG)
                    cost_bg[i, j]   # node → sink (BG)
                )

        # Add pairwise (n-link) costs (4-neighborhood)
        for i in range(H):
            for j in range(W):
                p = (i, j)
                p_idx = node_index(i, j)

                if i + 1 < H:
                    q = (i + 1, j)
                    q_idx = node_index(i + 1, j)
                    w = self.compute_pairwise_weight(p, q)
                    graph.add_edge(node_ids[p_idx], node_ids[q_idx], w, w)

                if j + 1 < W:
                    q = (i, j + 1)
                    q_idx = node_index(i, j + 1)
                    w = self.compute_pairwise_weight(p, q)
                    graph.add_edge(node_ids[p_idx], node_ids[q_idx], w, w)

        return graph, node_ids

    def graph_cut(self, use_gmm=False):
        """
        Run graph cut segmentation.

        Args:
            use_gmm (bool): whether to use GMM or histogram

        Returns:
            segmentation mask (H x W), 1 = FG, 0 = BG
        """
        # 1. Build color models
        if use_gmm:
            self.build_color_models_gmm()
        else:
            self.build_color_models_histogram()

        # 2. Compute unary costs
        cost_bg, cost_fg = self.compute_unary_costs(use_gmm=use_gmm)

        # 3. Build graph
        graph, node_ids = self.construct_graph(cost_bg, cost_fg)

        # 4. Max-flow
        graph.maxflow()

        # 5. Extract segmentation
        H, W = self.height, self.width
        mask = np.zeros((H, W), dtype=np.uint8)

        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if graph.get_segment(node_ids[idx]) == 0:
                    mask[i, j] = 0  # foreground
                else:
                    mask[i, j] = 1  # background

        return mask
    

def compute_iou(pred, gt):
    """
    Compute Intersection over Union.

    Args:
        pred: predicted binary mask (0/1)
        gt: ground truth binary mask (0/1)

    Returns:
        float: IoU score
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


# Load data
image = cv2.imread("dataset/images/scissors.jpg")
scribble_img = cv2.imread("dataset/images-labels/scissors-anno.png")
gt_mask = cv2.imread("dataset/images-gt/scissors.png", 0)
gt_mask = (gt_mask > 0).astype(np.uint8)

# Initialize
gc = GraphCutCore(image, None)
scribble_mask = gc.load_scribbles("dataset/images-labels/scissors-anno.png")
gc.scribbles = scribble_mask

# debugging the labelling
print("Scribbles loaded. FG pixels:", np.sum(scribble_mask == 1), "BG pixels:", np.sum(scribble_mask == 2))

# Run segmentation
pred_mask = gc.graph_cut(use_gmm=True)

# Evaluate
iou = compute_iou(pred_mask, gt_mask)
print("IoU:", iou)


# Visualize results
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(pred_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the predicted mask
output_path = "predicted_mask.png"
cv2.imwrite(output_path, pred_mask * 255)  # multiply by 255 to save as visible binary image
print(f"Predicted mask saved to {output_path}")
