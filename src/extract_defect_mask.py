import json
import numpy as np
import open3d as o3d
import cv2

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R


def read_json(fpath):
    """reads .json file and return

    Args:
        fpath (path): .json file path

    Returns:
        dict: .json file dictionary
    """
    with open(fpath, "r", encoding="utf-8") as f:
        j = json.load(f)

    return j


def create_PCD_aligned(pc_path, rgb_w=1440, rgb_h=1920, depth_w=192, depth_h=256):

    pc_json = read_json(pc_path)
    depth_map = np.array(pc_json["Depth"]).reshape(depth_w, depth_h)

    # -1~1 범위로 정규화된 좌표계 사용
    X, Y = np.meshgrid(np.linspace(-1, 1, depth_h), np.linspace(-1, 1, depth_w))
    Z = depth_map
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # 4x4 변환 행렬 (역행렬로 카메라 위치 보정)
    matrix = np.array(
        [
            [pc_json["m00"], pc_json["m01"], pc_json["m02"], pc_json["m03"]],
            [pc_json["m10"], pc_json["m11"], pc_json["m12"], pc_json["m13"]],
            [pc_json["m20"], pc_json["m21"], pc_json["m22"], pc_json["m23"]],
            [pc_json["m30"], pc_json["m31"], pc_json["m32"], pc_json["m33"]],
        ]
    )
    inv_matrix = np.linalg.inv(matrix)

    # 동차 좌표 적용
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = (inv_matrix @ points_hom.T).T[:, :3]

    # ⬇ XY 평면과 수평 정렬만 추가 ⬇
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(transformed)
    plane_model, _ = pcd_temp.segment_plane(0.003, 3, 1000)
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)

    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal, z_axis)
    angle = np.arccos(np.clip(np.dot(normal, z_axis), -1, 1))

    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        R_mat = R.from_rotvec(angle * axis).as_matrix()
        transformed = (R_mat @ transformed.T).T

    # 최종 PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed)
    return pcd


# 입력된 point cloud에 대해 KNN 기반 normal 벡터 추정
# 연속된 normal 벡터 간 차이를 이용해 gradient 벡터 및 크기 계산
def calc_gradient(pcd, knn=30):

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))

    normals = np.asarray(pcd.normals)

    gradient_vectors = normals[1:] - normals[:-1]
    gradient_magnitude = np.linalg.norm(gradient_vectors, axis=1)

    return gradient_magnitude, gradient_vectors


# 가장 강한 gradient를 가진 점을 기준으로 주변 점들을 찾아서 gradient 벡터의 z축 방향이 음수인 점만 추출
def filter_defect_points_by_roi(
    points, gradient_magnitude, percentile=98, roi_ratio=0.8
):
    """
    Gradient 상위 percentile의 포인트 중 중심 ROI(예: 80%) 안에 위치한 점만 결함 포인트로 반환

    Args:
        points (np.ndarray): Nx3 포인트 클라우드
        gradient_magnitude (np.ndarray): N-1 길이의 gradient 값
        percentile (float): 상위 몇 %의 gradient를 결함으로 볼지
        roi_ratio (float): 중심 ROI의 크기 비율 (0~1)

    Returns:
        np.ndarray: 필터링된 결함 포인트
    """
    threshold = np.percentile(gradient_magnitude, percentile)
    defect_mask = gradient_magnitude > threshold

    x, y = points[:, 0], points[:, 1]
    x_range, y_range = x.max() - x.min(), y.max() - y.min()
    x_min, x_max = (
        x.min() + x_range * (1 - roi_ratio) / 2,
        x.max() - x_range * (1 - roi_ratio) / 2,
    )
    y_min, y_max = (
        y.min() + y_range * (1 - roi_ratio) / 2,
        y.max() - y_range * (1 - roi_ratio) / 2,
    )

    roi_mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    valid_mask = (
        defect_mask & roi_mask[:-1]
    )  # gradient_magnitude는 N-1 길이이므로 맞춰줌

    return points[:-1][valid_mask], valid_mask


# 검출 영역 중 넓은 영역만을 필터링
def extract_largest_defect_cluster(points, eps=0.01, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        return np.empty((0, 3))
    main_label = unique_labels[np.argmax(counts)]
    return points[labels == main_label]


# 가장 큰 영역의 내부를 채움
def expand_defect_with_convex_hull(all_points, seed_points):
    hull = ConvexHull(seed_points[:, :2])
    delaunay = Delaunay(seed_points[:, :2][hull.vertices])
    mask = delaunay.find_simplex(all_points[:, :2]) >= 0
    return all_points[mask]


# 3D PCL을 2D Mask로 시각화
def create_defect_mask_image(all_points, defect_points, image_size=(256, 192)):
    """
    3D 포인트 클라우드를 2D 이미지(Depth Map) 형태로 변환하여
    결함 영역만 흰색(255), 나머지는 검정색(0)으로 마스크 생성
    """
    mask = np.zeros(image_size, dtype=np.uint8)

    # Normalize X,Y to pixel coordinates
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    def normalize_to_pixel(x, y):
        px = ((x - x_min) / (x_max - x_min) * (image_size[1] - 1)).astype(int)
        py = ((y - y_min) / (y_max - y_min) * (image_size[0] - 1)).astype(int)
        return px, py

    px, py = normalize_to_pixel(defect_points[:, 0], defect_points[:, 1])

    # 결함 포인트는 흰색으로 표시
    for x, y in zip(px, py):
        mask[y, x] = 255

    return mask


# 2D 마스크 이미지 스무딩
def postprocess_mask(mask_img):
    # 1. 작은 구멍 메우기 (Closing)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)

    # 2. 외곽 부드럽게 (Gaussian Blur)
    blurred = cv2.GaussianBlur(closed, (7, 7), sigmaX=0)

    # 3. Threshold 다시 적용 (흐릿한 부분 다시 명확하게)
    _, binary_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    return binary_mask
