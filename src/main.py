import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from extract_defect_mask import (
    create_PCD_aligned,
    calc_gradient,
    filter_defect_points_by_roi,
    extract_largest_defect_cluster,
    expand_defect_with_convex_hull,
    create_defect_mask_image,
    postprocess_mask,
)

main_data_path = "YOUR DATA PATH"
depth_path = main_data_path + "depths/"
json_list = os.listdir(depth_path)

# Hyperparams
KNN = 30
ROI_RATIO = 0.25
PERCENTILE = 80
EPS = 0.05
MIN_SAMPLES = 10


for json_name in tqdm(json_list):
    json_name_split = json_name.split(".")

    json_path = f"{depth_path}{json_name}"
    # pc_json = read_json(json_path)

    # 이미지 로드
    rgb_img = plt.imread(f"{main_data_path}images/{json_name_split[0]}.jpg")
    resize_rgb = cv2.resize(rgb_img, (512, 512))

    # 메인 결함 포인트 추출
    try:
        pcd = create_PCD_aligned(json_path)
        points = np.asarray(pcd.points)

        gradient_magnitude, _ = calc_gradient(pcd, KNN)
        defect_points, _ = filter_defect_points_by_roi(
            points, gradient_magnitude, percentile=PERCENTILE, roi_ratio=ROI_RATIO
        )
        largest_defect_points = extract_largest_defect_cluster(
            defect_points, eps=EPS, min_samples=MIN_SAMPLES
        )

        expanded = expand_defect_with_convex_hull(points, largest_defect_points)

        # 마스크 생성
        mask_img = create_defect_mask_image(points, expanded, image_size=(256, 192))
        smooth_mask = postprocess_mask(mask_img)
        resize_mask = cv2.resize(smooth_mask, (512, 512))

    except:
        resize_mask = np.zeros((512, 512))

    # Mask 저장
    # save_mask_name = f"{main_data_path}masks/{json_name_split[0]}.jpg"
    # plt.imsave(save_mask_name, resize_mask, cmap="gray")

    # Overlay 저장
    plt.figure(figsize=(8, 6))
    plt.imshow(resize_rgb, alpha=1.0)
    plt.imshow(resize_mask, alpha=0.5)
    plt.axis("off")

    # 저장 (예: overlay_result.png)
    save_overlay_name = (
        f"{main_data_path}overlay_percentile80_e005/{json_name_split[0]}.jpg"
    )
    plt.savefig(save_overlay_name, bbox_inches="tight", pad_inches=0)
    plt.close()  # 메모리 정리, 창 닫기
