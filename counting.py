import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


BASE_SIZE = (3096, 2185)
OCCUPANCY_THRESHOLD = 0.08
EDGE_DENSITY_THRESHOLD = 0.025

# ROI dibuat dari posisi slot parkir pada gambar input. Nilai ini tetap diskalakan
# otomatis jika ukuran gambar berubah.
BASE_ROIS = np.array(
    [
        [130, 0, 620, 65], [700, 0, 1110, 95],
        [75, 130, 560, 335], [650, 135, 1160, 345],
        [30, 500, 505, 695], [705, 480, 1170, 685],
        [45, 835, 555, 1035], [675, 845, 1170, 1060],
        [35, 1085, 620, 1285], [735, 1110, 1190, 1320],
        [80, 1360, 590, 1605], [710, 1355, 1180, 1620],
        [85, 1645, 585, 1905], [690, 1660, 1165, 1905],
        [115, 1935, 625, 2180], [715, 1945, 1155, 2180],
        [1440, 835, 1635, 1325],
        [1935, 55, 2350, 290], [2500, 40, 3090, 275],
        [1920, 420, 2405, 625], [2510, 400, 3020, 610],
        [1930, 770, 2420, 995], [2520, 780, 3060, 980],
        [1920, 1090, 2405, 1325], [2570, 1100, 3060, 1305],
        [1920, 1360, 2405, 1605], [2520, 1320, 3035, 1585],
        [1920, 1640, 2420, 1905], [2505, 1600, 3060, 1885],
        [1900, 1930, 2410, 2180], [2500, 1930, 3065, 2180],
    ],
    dtype=np.float32,
)


def make_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "steps"), exist_ok=True)


def scaled_rois(image):
    height, width = image.shape[:2]
    scale = np.array([width / BASE_SIZE[0], height / BASE_SIZE[1]] * 2)
    rois = np.rint(BASE_ROIS * scale).astype(int)
    rois[:, [0, 2]] = np.clip(rois[:, [0, 2]], 0, width)
    rois[:, [1, 3]] = np.clip(rois[:, [1, 3]], 0, height)
    return rois


def create_masks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bright = ((v > 150) & (s < 105)).astype(np.uint8) * 255
    dark = ((v < 95) & (s < 150)).astype(np.uint8) * 255
    colored = ((s > 45) & (v > 55)).astype(np.uint8) * 255
    red = (((h < 12) | (h > 165)) & (s > 80) & (v > 80)).astype(np.uint8) * 255

    raw_mask = cv2.bitwise_or(cv2.bitwise_or(bright, dark), cv2.bitwise_or(colored, red))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)
    edges = cv2.Canny(gray, 60, 160)

    return gray, raw_mask, clean_mask, edges


def detect_cars(rois, mask, edges):
    detections = []

    for i, (x1, y1, x2, y2) in enumerate(rois, start=1):
        roi_mask = mask[y1:y2, x1:x2]
        roi_edges = edges[y1:y2, x1:x2]
        area = max(1, roi_mask.size)
        mask_ratio = cv2.countNonZero(roi_mask) / area
        edge_density = cv2.countNonZero(roi_edges) / area

        if mask_ratio >= OCCUPANCY_THRESHOLD or edge_density >= EDGE_DENSITY_THRESHOLD:
            detections.append((i, x1, y1, x2, y2, mask_ratio, edge_density))

    return detections


def draw_rois(image, rois):
    result = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(rois, start=1):
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(result, str(i), (x1 + 6, max(22, y1 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
    return result


def draw_result(image, detections):
    result = image.copy()

    for number, (_, x1, y1, x2, y2, _, _) in enumerate(detections, start=1):
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(result, str(number), (x1 + 8, max(28, y1 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    label = f"Jumlah mobil: {len(detections)}"
    origin = (1230, 80)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(result, (origin[0] - 18, origin[1] - th - 18),
                  (origin[0] + tw + 18, origin[1] + 14), (30, 30, 30), -1)
    cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 3, cv2.LINE_AA)
    return result


def save_overview(paths, output_path):
    titles = ["Original", "Grayscale", "Raw mask", "Morphology",
              "Edges", "ROI candidates", "Result"]

    plt.figure(figsize=(15, 9))
    for i, (title, path) in enumerate(zip(titles, paths), start=1):
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        plt.subplot(2, 4, i)
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_count(output_dir, detections):
    with open(os.path.join(output_dir, "count.txt"), "w", encoding="utf-8") as file:
        file.write(f"Jumlah mobil terdeteksi: {len(detections)}\n")
        for roi_id, x1, y1, x2, y2, mask_ratio, edge_density in detections:
            file.write(
                f"ROI {roi_id}: bbox=({x1}, {y1}, {x2}, {y2}), "
                f"mask_ratio={mask_ratio:.3f}, edge_density={edge_density:.3f}\n"
            )


def run(input_path, output_dir):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {input_path}")

    make_dirs(output_dir)
    steps_dir = os.path.join(output_dir, "steps")

    gray, raw_mask, clean_mask, edges = create_masks(image)
    rois = scaled_rois(image)
    detections = detect_cars(rois, clean_mask, edges)
    roi_preview = draw_rois(image, rois)
    result = draw_result(image, detections)

    paths = [
        os.path.join(steps_dir, "01_original.png"),
        os.path.join(steps_dir, "02_grayscale.png"),
        os.path.join(steps_dir, "03_raw_mask.png"),
        os.path.join(steps_dir, "04_morphology.png"),
        os.path.join(steps_dir, "05_edges.png"),
        os.path.join(steps_dir, "06_roi_candidates.png"),
        os.path.join(output_dir, "result.png"),
    ]

    for path, image_step in zip(paths, [image, gray, raw_mask, clean_mask, edges, roi_preview, result]):
        cv2.imwrite(path, image_step)

    save_overview(paths, os.path.join(steps_dir, "pipeline_overview.png"))
    write_count(output_dir, detections)
    return len(detections), paths[-1], os.path.join(steps_dir, "pipeline_overview.png")


def main():
    parser = argparse.ArgumentParser(description="Hitung jumlah mobil pada citra parkiran.")
    parser.add_argument("--input", default="input/parking.jpg")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    count, result_path, overview_path = run(args.input, args.output)
    print(f"Jumlah mobil terdeteksi: {count}")
    print(f"Hasil deteksi disimpan di: {result_path}")
    print(f"Visualisasi pipeline disimpan di: {overview_path}")


if __name__ == "__main__":
    main()
