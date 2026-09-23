"""Prediction-guided strip sampling with the existing box residual head."""

from .gt_strip_head import GTStripHead


class PredictionStripHead(GTStripHead):
    """Use each predicted box as both the sampling frame and refinement base."""

    def __init__(
        self,
        in_channels,
        query_dim,
        hidden_dim=64,
        num_points=5,
        kernel_size=3,
        offset_pixels=1.0,
        max_log_scale=2.0,
        sampling_mode="box",
    ):
        if sampling_mode != "box":
            raise ValueError("P1 supports only sampling_mode='box'.")
        super().__init__(
            in_channels=in_channels,
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_points=num_points,
            kernel_size=kernel_size,
            offset_pixels=offset_pixels,
            max_log_scale=max_log_scale,
            feature_mode="real",
        )
        self.sampling_mode = sampling_mode

    def forward(self, feature_map, queries, pred_boxes, batch_indices, image_size):
        """Return refined boxes and eligibility without accepting GT inputs."""
        return super().forward(
            feature_map, queries, pred_boxes, pred_boxes.detach(), batch_indices,
            image_size,
        )
