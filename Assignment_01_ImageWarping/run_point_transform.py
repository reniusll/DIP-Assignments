import cv2
import numpy as np
import gradio as gr


def sample_bilinear(image, x, y, fill_value=255):
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    output = np.full((x.shape[0], channels), fill_value, dtype=np.float32)

    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    if not np.any(valid):
        return output

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = (xv - x0).reshape(-1, 1)
    dy = (yv - y0).reshape(-1, 1)

    working_image = image[..., None] if image.ndim == 2 else image
    top_left = working_image[y0, x0].astype(np.float32)
    top_right = working_image[y0, x1].astype(np.float32)
    bottom_left = working_image[y1, x0].astype(np.float32)
    bottom_right = working_image[y1, x1].astype(np.float32)

    top = top_left * (1.0 - dx) + top_right * dx
    bottom = bottom_left * (1.0 - dx) + bottom_right * dx
    output[valid] = top * (1.0 - dy) + bottom * dy
    return output

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    if image is None:
        return None

    warped_image = np.array(image)

    num_pairs = min(len(source_pts), len(target_pts))
    if num_pairs == 0:
        return warped_image

    source_pts = np.asarray(source_pts[:num_pairs], dtype=np.float32)
    target_pts = np.asarray(target_pts[:num_pairs], dtype=np.float32)

    h, w = warped_image.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    vertices = np.stack([xx.ravel(), yy.ravel()], axis=1)

    displacements = vertices[:, None, :] - target_pts[None, :, :]
    dist_sq = np.sum(displacements * displacements, axis=2)
    exact_match = dist_sq < eps

    safe_dist_sq = np.maximum(dist_sq, eps)
    weights = 1.0 / np.power(safe_dist_sq, alpha)
    weight_sum = np.sum(weights, axis=1, keepdims=True)

    p_star = (weights[:, :, None] * target_pts[None, :, :]).sum(axis=1) / weight_sum
    q_star = (weights[:, :, None] * source_pts[None, :, :]).sum(axis=1) / weight_sum

    p_hat = target_pts[None, :, :] - p_star[:, None, :]
    q_hat = source_pts[None, :, :] - q_star[:, None, :]

    weighted_p_hat = weights[:, :, None] * p_hat
    mu = np.einsum("nki,nkj->nij", weighted_p_hat, p_hat)
    cross = np.einsum("nki,nkj->nij", weights[:, :, None] * q_hat, p_hat)

    det = mu[:, 0, 0] * mu[:, 1, 1] - mu[:, 0, 1] * mu[:, 1, 0]
    invertible = np.abs(det) > eps

    mapped = vertices.copy()
    if np.any(invertible):
        mu_inv = np.zeros_like(mu)
        mu_inv[invertible, 0, 0] = mu[invertible, 1, 1] / det[invertible]
        mu_inv[invertible, 1, 1] = mu[invertible, 0, 0] / det[invertible]
        mu_inv[invertible, 0, 1] = -mu[invertible, 0, 1] / det[invertible]
        mu_inv[invertible, 1, 0] = -mu[invertible, 1, 0] / det[invertible]

        affine = cross @ mu_inv
        local_coords = (vertices - p_star)[:, :, None]
        mapped[invertible] = (
            (affine[invertible] @ local_coords[invertible]).squeeze(axis=2)
            + q_star[invertible]
        )

    if np.any(exact_match):
        matched_vertices, matched_points = np.where(exact_match)
        mapped[matched_vertices] = source_pts[matched_points]

    sampled = sample_bilinear(warped_image, mapped[:, 0], mapped[:, 1], fill_value=255)
    warped_image = np.clip(sampled.reshape(h, w, -1), 0, 255).astype(np.uint8)
    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
