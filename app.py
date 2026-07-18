
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="보호구역 자동 추천 시스템 5.0",
    page_icon="🌿",
    layout="wide",
)

HABITAT_LABELS = {
    0: "개발지역·도로",
    1: "산림",
    2: "습지·하천",
    3: "초지·농경지",
}

HABITAT_COLORS = {
    0: (150, 150, 150, 125),
    1: (25, 155, 70, 135),
    2: (30, 120, 225, 140),
    3: (235, 190, 35, 135),
}

DEFAULT_PARAMS = {
    1: {"c": 15.0, "k": 0.22},
    2: {"c": 12.0, "k": 0.35},
    3: {"c": 8.0, "k": 0.28},
}


# =========================================================
# 수학적 최적화
# =========================================================
def species_value(area: int, c: float, k: float) -> float:
    if area <= 0:
        return 0.0
    return float(c * (area ** k))


def optimize_allocation(
    matrix: np.ndarray,
    budget: int,
    params: Dict[int, Dict[str, float]],
) -> Tuple[Dict[int, int], float]:
    available = {h: int(np.sum(matrix == h)) for h in (1, 2, 3)}
    budget = min(max(int(budget), 0), sum(available.values()))

    best_value = -1.0
    best_alloc = {1: 0, 2: 0, 3: 0}

    for forest in range(min(available[1], budget) + 1):
        remain = budget - forest
        for wetland in range(min(available[2], remain) + 1):
            grass = remain - wetland

            if grass < 0 or grass > available[3]:
                continue

            value = (
                species_value(forest, params[1]["c"], params[1]["k"])
                + species_value(wetland, params[2]["c"], params[2]["k"])
                + species_value(grass, params[3]["c"], params[3]["k"])
            )

            if value > best_value:
                best_value = value
                best_alloc = {1: forest, 2: wetland, 3: grass}

    return best_alloc, best_value


def equal_allocation(
    matrix: np.ndarray,
    budget: int,
    params: Dict[int, Dict[str, float]],
) -> Tuple[Dict[int, int], float]:
    available = {h: int(np.sum(matrix == h)) for h in (1, 2, 3)}
    allocation = {1: 0, 2: 0, 3: 0}
    remaining = budget

    while remaining > 0:
        changed = False
        for h in (1, 2, 3):
            if remaining > 0 and allocation[h] < available[h]:
                allocation[h] += 1
                remaining -= 1
                changed = True
        if not changed:
            break

    value = sum(
        species_value(allocation[h], params[h]["c"], params[h]["k"])
        for h in (1, 2, 3)
    )
    return allocation, value


# =========================================================
# 색상 및 질감 특징 추출
# =========================================================
def rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """0~255 RGB 배열을 0~1 HSV로 변환."""
    rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]

    maxc = np.max(rgb01, axis=1)
    minc = np.min(rgb01, axis=1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-8

    mask = nonzero & (maxc == r)
    hue[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6

    mask = nonzero & (maxc == g)
    hue[mask] = (b[mask] - r[mask]) / delta[mask] + 2

    mask = nonzero & (maxc == b)
    hue[mask] = (r[mask] - g[mask]) / delta[mask] + 4

    hue /= 6.0
    saturation = np.where(maxc > 1e-8, delta / maxc, 0.0)
    value = maxc

    return np.column_stack([hue, saturation, value])


def extract_cell_features(
    image: Image.Image,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    셀마다 색상, 식생, 수분, 회색도, 질감, 에지 정보를 추출한다.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = arr.shape

    features = []
    positions = []
    mean_rgbs = []

    for row in range(rows):
        y0 = int(row * height / rows)
        y1 = int((row + 1) * height / rows)

        for col in range(cols):
            x0 = int(col * width / cols)
            x1 = int((col + 1) * width / cols)

            patch = arr[y0:y1, x0:x1]
            if patch.size == 0:
                patch = np.zeros((1, 1, 3), dtype=np.float32)

            flat = patch.reshape(-1, 3)
            mean_rgb = flat.mean(axis=0)
            std_rgb = flat.std(axis=0)
            median_rgb = np.median(flat, axis=0)

            r, g, b = mean_rgb
            brightness = float(mean_rgb.mean())
            channel_spread = float(mean_rgb.max() - mean_rgb.min())

            excess_green = float(2 * g - r - b)
            normalized_green = float(
                (g - (r + b) / 2.0) / (r + g + b + 1e-6)
            )
            blue_dominance = float(
                (b - (r + g) / 2.0) / (r + g + b + 1e-6)
            )
            brown_index = float((r - b) + 0.35 * (r - g))

            # 밝기 질감
            gray = (
                0.299 * patch[:, :, 0]
                + 0.587 * patch[:, :, 1]
                + 0.114 * patch[:, :, 2]
            )
            gray_std = float(gray.std())

            # 인접 픽셀 차이로 단순 에지 강도 계산
            edge_x = (
                np.abs(np.diff(gray, axis=1)).mean()
                if gray.shape[1] > 1 else 0.0
            )
            edge_y = (
                np.abs(np.diff(gray, axis=0)).mean()
                if gray.shape[0] > 1 else 0.0
            )
            edge_strength = float((edge_x + edge_y) / 2.0)

            features.append([
                *mean_rgb,
                *std_rgb,
                *median_rgb,
                brightness,
                channel_spread,
                excess_green,
                normalized_green,
                blue_dominance,
                brown_index,
                gray_std,
                edge_strength,
            ])
            positions.append((row, col))
            mean_rgbs.append(mean_rgb)

    features_array = np.asarray(features, dtype=np.float32)
    hsv = rgb_to_hsv_array(np.asarray(mean_rgbs, dtype=np.float32))
    features_array = np.column_stack([features_array, hsv])

    return features_array, positions, np.asarray(mean_rgbs)


# =========================================================
# 자동 군집화 및 지형 추정
# =========================================================
def detailed_clustering(
    features: np.ndarray,
    rows: int,
    cols: int,
    n_clusters: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    cluster_count = min(n_clusters, len(features))
    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=30,
    )
    labels = model.fit_predict(scaled)

    return labels.reshape(rows, cols), labels


def heuristic_cell_scores(features: np.ndarray) -> np.ndarray:
    """
    각 셀에 대해 개발지, 산림, 습지, 초지 점수를 계산한다.
    입력 특징 배열의 열 순서는 extract_cell_features와 동일하다.
    """
    r, g, b = features[:, 0], features[:, 1], features[:, 2]
    brightness = features[:, 9]
    spread = features[:, 10]
    exg = features[:, 11]
    ng = features[:, 12]
    blue = features[:, 13]
    brown = features[:, 14]
    texture = features[:, 15]
    edge = features[:, 16]
    hue, sat, val = features[:, 17], features[:, 18], features[:, 19]

    # 정규화 보조 함수
    def z(x):
        std = np.std(x)
        if std < 1e-8:
            return np.zeros_like(x)
        return (x - np.mean(x)) / std

    forest = (
        1.7 * z(exg)
        + 1.3 * z(ng)
        + 0.6 * z(sat)
        + 0.5 * z(texture)
        - 0.35 * z(brightness)
    )

    wetland = (
        2.0 * z(blue)
        + 0.9 * z(b)
        - 0.5 * z(exg)
        - 0.35 * z(brightness)
        - 0.25 * z(texture)
    )

    grass = (
        1.0 * z(exg)
        + 0.7 * z(ng)
        + 0.45 * z(brightness)
        + 0.35 * z(brown)
        - 0.55 * z(texture)
    )

    developed = (
        1.0 * z(brightness)
        - 0.9 * z(sat)
        - 0.9 * z(exg)
        + 0.8 * z(edge)
        - 0.35 * z(spread)
    )

    return np.column_stack([developed, forest, wetland, grass])


def map_clusters_to_habitats(
    cluster_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    cell_scores: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    세부 군집별 평균 지형 점수를 계산한 뒤 가장 높은 지형으로 통합한다.
    """
    habitat = np.zeros_like(cluster_matrix, dtype=int)
    summary_rows = []

    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        mean_scores = cell_scores[mask].mean(axis=0)
        code = int(np.argmax(mean_scores))
        habitat[cluster_matrix == cluster] = code

        summary_rows.append({
            "세부 군집": int(cluster),
            "셀 수": int(mask.sum()),
            "자동 지형": HABITAT_LABELS[code],
            "개발지 점수": round(float(mean_scores[0]), 2),
            "산림 점수": round(float(mean_scores[1]), 2),
            "습지 점수": round(float(mean_scores[2]), 2),
            "초지 점수": round(float(mean_scores[3]), 2),
        })

    return habitat, pd.DataFrame(summary_rows)


def spatial_regularization(
    matrix: np.ndarray,
    confidence_scores: np.ndarray,
    passes: int = 2,
) -> np.ndarray:
    """
    주변 8개 셀의 지형과 분류 점수를 함께 고려해 고립 셀을 보정한다.
    가느다란 물길이 전부 사라지지 않도록 습지는 보수적으로 유지한다.
    """
    result = matrix.copy()
    rows, cols = result.shape
    score_grid = confidence_scores.reshape(rows, cols, 4)

    for _ in range(passes):
        updated = result.copy()

        for r in range(rows):
            for c in range(cols):
                neighbor_codes = []

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor_codes.append(int(result[nr, nc]))

                if not neighbor_codes:
                    continue

                counts = np.bincount(neighbor_codes, minlength=4)
                current = int(result[r, c])
                majority = int(np.argmax(counts))
                current_margin = (
                    score_grid[r, c, current]
                    - np.partition(score_grid[r, c], -2)[-2]
                )

                # 현재 판단이 약하고 주변 8칸 중 5칸 이상이 같은 경우만 변경
                if (
                    majority != current
                    and counts[majority] >= 5
                    and current_margin < 0.75
                ):
                    # 물은 선형 지형일 수 있어 더 강한 조건에서만 변경
                    if current == 2 and counts[majority] < 7:
                        continue
                    updated[r, c] = majority

        result = updated

    return result


# =========================================================
# 대표 셀 기반 사진별 재학습
# =========================================================
def parse_cell_list(text: str, rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    '1,2; 3,4' 형식. 화면상 행/열 번호는 1부터 시작.
    """
    cells = []
    text = text.strip()

    if not text:
        return cells

    for item in text.split(";"):
        parts = item.strip().split(",")
        if len(parts) != 2:
            continue

        try:
            row = int(parts[0].strip()) - 1
            col = int(parts[1].strip()) - 1
        except ValueError:
            continue

        if 0 <= row < rows and 0 <= col < cols:
            cells.append((row, col))

    return cells


def calibrate_classifier(
    features: np.ndarray,
    initial_matrix: np.ndarray,
    samples: Dict[int, List[Tuple[int, int]]],
) -> np.ndarray:
    rows, cols = initial_matrix.shape
    train_x = []
    train_y = []

    for code, cells in samples.items():
        for r, c in cells:
            index = r * cols + c
            train_x.append(features[index])
            train_y.append(code)

    # 대표 셀이 너무 적으면 초기 결과 유지
    if len(set(train_y)) < 2 or len(train_y) < 4:
        return initial_matrix.copy()

    classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(np.asarray(train_x), np.asarray(train_y))
    prediction = classifier.predict(features)

    return prediction.reshape(rows, cols)


# =========================================================
# 공간 선택 및 시각화
# =========================================================
def neighbors4(r: int, c: int, rows: int, cols: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def select_connected_cells(
    matrix: np.ndarray,
    habitat: int,
    count: int,
) -> List[Tuple[int, int]]:
    candidates = [(int(r), int(c)) for r, c in zip(*np.where(matrix == habitat))]

    if count <= 0:
        return []
    if count >= len(candidates):
        return candidates

    rows, cols = matrix.shape

    def cell_score(cell, selected):
        r, c = cell
        same_neighbors = sum(
            matrix[nr, nc] == habitat
            for nr, nc in neighbors4(r, c, rows, cols)
        )
        selected_neighbors = sum(
            (nr, nc) in selected
            for nr, nc in neighbors4(r, c, rows, cols)
        )
        return 4 * selected_neighbors + same_neighbors

    seed = max(candidates, key=lambda x: cell_score(x, set()))
    selected = {seed}

    while len(selected) < count:
        remaining = [cell for cell in candidates if cell not in selected]
        selected.add(max(remaining, key=lambda x: cell_score(x, selected)))

    return sorted(selected)


def build_selection_matrix(
    matrix: np.ndarray,
    allocation: Dict[int, int],
) -> np.ndarray:
    selected = np.zeros_like(matrix, dtype=int)

    for habitat in (1, 2, 3):
        for r, c in select_connected_cells(matrix, habitat, allocation[habitat]):
            selected[r, c] = 1

    return selected


def draw_overlay(
    image: Image.Image,
    matrix: np.ndarray,
    selected: np.ndarray | None = None,
    show_numbers: bool = False,
) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    rows, cols = matrix.shape
    width, height = img.size

    for r in range(rows):
        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)
            y0 = int(r * height / rows)
            y1 = int((r + 1) * height / rows)

            if selected is None:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=HABITAT_COLORS[int(matrix[r, c])],
                )
            elif selected[r, c] == 1:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=HABITAT_COLORS[int(matrix[r, c])],
                )
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline=(0, 0, 0, 235),
                    width=3,
                )

            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(255, 255, 255, 165),
                width=1,
            )

            if show_numbers and (x1 - x0) >= 35 and (y1 - y0) >= 25:
                label = f"{r+1},{c+1}"
                draw.text(
                    (x0 + 3, y0 + 2),
                    label,
                    fill=(255, 255, 255, 235),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0, 190),
                )

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_plain_grid(
    image: Image.Image,
    rows: int,
    cols: int,
    show_numbers: bool = False,
) -> Image.Image:
    dummy = np.zeros((rows, cols), dtype=int)
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    width, height = img.size

    for r in range(rows):
        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)
            y0 = int(r * height / rows)
            y1 = int((r + 1) * height / rows)

            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(255, 255, 255, 200),
                width=1,
            )

            if show_numbers and (x1 - x0) >= 35 and (y1 - y0) >= 25:
                draw.text(
                    (x0 + 3, y0 + 2),
                    f"{r+1},{c+1}",
                    fill=(255, 255, 255, 240),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0, 190),
                )

    return Image.alpha_composite(img, overlay).convert("RGB")


def matrix_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        columns=[f"C{i+1}" for i in range(matrix.shape[1])],
        index=[f"R{i+1}" for i in range(matrix.shape[0])],
    )


def dataframe_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return np.clip(numeric.to_numpy(), 0, 3)


def plot_budget_curve(
    matrix: np.ndarray,
    params: Dict[int, Dict[str, float]],
):
    usable = int(np.sum(matrix != 0))
    budgets = list(range(1, usable + 1))
    optimal_values = []
    equal_values = []

    for budget in budgets:
        _, opt = optimize_allocation(matrix, budget, params)
        _, eq = equal_allocation(matrix, budget, params)
        optimal_values.append(opt)
        equal_values.append(eq)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(budgets, optimal_values, label="Optimal allocation")
    ax.plot(budgets, equal_values, linestyle="--", label="Equal allocation")
    ax.set_xlabel("Protected area (cells)")
    ax.set_ylabel("Expected species")
    ax.set_title("Protected Area and Expected Species")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# =========================================================
# Streamlit 화면
# =========================================================
st.title("🌿 보호구역 자동 추천 시스템 5.0")
st.write(
    "색상·식생·수분·질감 특징과 공간 보정을 결합하여 지형을 분류하고, "
    "종-면적 관계식에 따라 최적 보호구역을 추천합니다."
)

st.header("1. 지형 사진 입력")

uploaded = st.file_uploader(
    "항공사진 또는 드론 사진을 업로드하세요.",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("사진을 업로드하면 분석이 시작됩니다.")
    st.stop()

source_image = Image.open(uploaded).convert("RGB")

grid_size = st.slider(
    "격자 크기",
    min_value=6,
    max_value=16,
    value=10,
    step=1,
)

detail_clusters = st.slider(
    "세부 색상 군집 수",
    min_value=6,
    max_value=12,
    value=8,
    step=1,
)

spatial_passes = st.slider(
    "공간 보정 강도",
    min_value=0,
    max_value=3,
    value=1,
    step=1,
)

signature = (
    uploaded.name,
    uploaded.size,
    grid_size,
    detail_clusters,
    spatial_passes,
)

if st.session_state.get("signature") != signature:
    features, positions, mean_rgbs = extract_cell_features(
        source_image,
        grid_size,
        grid_size,
    )

    cluster_matrix, cluster_labels = detailed_clustering(
        features,
        grid_size,
        grid_size,
        detail_clusters,
    )

    scores = heuristic_cell_scores(features)

    automatic_matrix, cluster_summary = map_clusters_to_habitats(
        cluster_matrix,
        cluster_labels,
        scores,
    )

    automatic_matrix = spatial_regularization(
        automatic_matrix,
        scores,
        spatial_passes,
    )

    st.session_state["signature"] = signature
    st.session_state["features"] = features
    st.session_state["auto_matrix"] = automatic_matrix
    st.session_state["cluster_summary"] = cluster_summary
    st.session_state.pop("selection", None)

features = st.session_state["features"]
auto_matrix = st.session_state["auto_matrix"]
cluster_summary = st.session_state["cluster_summary"]

st.header("2. 자동 지형 분류")

preview_cols = st.columns(3)

with preview_cols[0]:
    st.subheader("원본 사진")
    st.image(source_image, use_container_width=True)

with preview_cols[1]:
    st.subheader("격자 번호")
    st.image(
        draw_plain_grid(
            source_image,
            grid_size,
            grid_size,
            show_numbers=True,
        ),
        use_container_width=True,
    )

with preview_cols[2]:
    st.subheader("자동 분류")
    st.image(
        draw_overlay(source_image, auto_matrix),
        use_container_width=True,
    )

st.caption(
    "회색=개발지역·도로 / 초록=산림 / 파랑=습지·하천 / 노랑=초지·농경지"
)

with st.expander("세부 군집 분석 보기"):
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

st.header("3. 사진별 빠른 보정")

use_calibration = st.checkbox(
    "대표 셀을 지정해 이 사진에 맞게 자동 재학습",
    value=False,
)

calibrated_matrix = auto_matrix.copy()

if use_calibration:
    st.write(
        "격자 번호 그림을 보고 각 지형을 대표하는 셀을 입력하세요. "
        "형식 예: `1,2; 2,2; 3,4`"
    )

    sample_columns = st.columns(4)
    samples = {}

    for column, code in zip(sample_columns, range(4)):
        with column:
            text = st.text_input(
                HABITAT_LABELS[code],
                key=f"sample_{signature}_{code}",
                placeholder="예: 1,2; 2,2",
            )
            samples[code] = parse_cell_list(
                text,
                grid_size,
                grid_size,
            )

    total_samples = sum(len(v) for v in samples.values())

    if total_samples >= 4:
        calibrated_matrix = calibrate_classifier(
            features,
            auto_matrix,
            samples,
        )

        # 대표 셀은 반드시 입력한 지형으로 고정
        for code, cells in samples.items():
            for r, c in cells:
                calibrated_matrix[r, c] = code

        st.subheader("사진별 재학습 결과")
        st.image(
            draw_overlay(source_image, calibrated_matrix),
            use_container_width=True,
        )
    else:
        st.info("서로 다른 지형에서 대표 셀을 합계 4개 이상 입력하면 재학습됩니다.")

st.header("4. 최종 지형행렬 확인")

st.code("0=개발지역·도로, 1=산림, 2=습지·하천, 3=초지·농경지")

edited_df = st.data_editor(
    matrix_dataframe(calibrated_matrix),
    use_container_width=True,
    num_rows="fixed",
    key=f"matrix_editor_{signature}_{use_calibration}",
)

matrix = dataframe_matrix(edited_df)

counts = {code: int(np.sum(matrix == code)) for code in range(4)}
count_columns = st.columns(4)

for column, code in zip(count_columns, range(4)):
    column.metric(HABITAT_LABELS[code], f"{counts[code]}칸")

st.header("5. 종-면적 관계식과 최적화")

params = {
    code: DEFAULT_PARAMS[code].copy()
    for code in (1, 2, 3)
}

st.latex(r"\max S=15A_F^{0.22}+12A_W^{0.35}+8A_G^{0.28}")
st.latex(r"A_F+A_W+A_G=A_{\mathrm{total}}")

with st.expander("c, k 값 변경"):
    parameter_columns = st.columns(3)

    for column, code in zip(parameter_columns, (1, 2, 3)):
        with column:
            st.subheader(HABITAT_LABELS[code])
            params[code]["c"] = st.number_input(
                "c",
                min_value=0.1,
                max_value=100.0,
                value=float(DEFAULT_PARAMS[code]["c"]),
                step=0.1,
                key=f"c_{code}",
            )
            params[code]["k"] = st.number_input(
                "k",
                min_value=0.01,
                max_value=0.99,
                value=float(DEFAULT_PARAMS[code]["k"]),
                step=0.01,
                key=f"k_{code}",
            )

usable = counts[1] + counts[2] + counts[3]

if usable == 0:
    st.error("보호 가능한 자연 지형 셀이 없습니다.")
    st.stop()

budget = st.slider(
    "총 보호 가능 면적",
    min_value=1,
    max_value=usable,
    value=min(16, usable),
    step=1,
    format="%d km²",
)

if st.button(
    "최적 보호구역 계산",
    type="primary",
    use_container_width=True,
):
    allocation, objective = optimize_allocation(
        matrix,
        budget,
        params,
    )
    equal_alloc, equal_value = equal_allocation(
        matrix,
        budget,
        params,
    )
    selection = build_selection_matrix(
        matrix,
        allocation,
    )

    st.session_state["selection"] = selection
    st.session_state["allocation"] = allocation
    st.session_state["objective"] = objective
    st.session_state["equal_value"] = equal_value
    st.session_state["result_matrix"] = matrix.copy()
    st.session_state["result_image"] = source_image.copy()
    st.session_state["result_params"] = params.copy()

if "selection" in st.session_state:
    st.divider()
    st.header("6. 최적 보호구역 추천 결과")

    selection = st.session_state["selection"]
    allocation = st.session_state["allocation"]
    objective = st.session_state["objective"]
    equal_value = st.session_state["equal_value"]
    result_matrix = st.session_state["result_matrix"]
    result_image = st.session_state["result_image"]
    result_params = st.session_state["result_params"]

    recommendation = draw_overlay(
        result_image,
        result_matrix,
        selected=selection,
    )

    gain = objective - equal_value
    gain_pct = 0 if equal_value <= 0 else gain / equal_value * 100

    metric_columns = st.columns(4)
    metric_columns[0].metric("총 보호면적", f"{int(selection.sum())} km²")
    metric_columns[1].metric("예상 보존 종수", f"{objective:.2f}종")
    metric_columns[2].metric("균등 배분 대비 증가", f"{gain:.2f}종")
    metric_columns[3].metric("개선율", f"{gain_pct:.1f}%")

    result_columns = st.columns(3)

    with result_columns[0]:
        st.subheader("원본")
        st.image(result_image, use_container_width=True)

    with result_columns[1]:
        st.subheader("최종 지형 분류")
        st.image(
            draw_overlay(result_image, result_matrix),
            use_container_width=True,
        )

    with result_columns[2]:
        st.subheader("추천 보호구역")
        st.image(recommendation, use_container_width=True)

    rows = []

    for code in (1, 2, 3):
        area = allocation[code]
        value = species_value(
            area,
            result_params[code]["c"],
            result_params[code]["k"],
        )
        rows.append({
            "지형": HABITAT_LABELS[code],
            "전체 셀": int(np.sum(result_matrix == code)),
            "선택 면적": area,
            "예상 종수": round(value, 2),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    graph_column, matrix_column = st.columns([1.2, 1])

    with graph_column:
        fig = plot_budget_curve(
            result_matrix,
            result_params,
        )
        st.pyplot(fig)
        plt.close(fig)

    with matrix_column:
        st.subheader("선택행렬")
        st.dataframe(
            pd.DataFrame(selection),
            use_container_width=True,
            hide_index=True,
        )

    largest = max(allocation, key=allocation.get)

    st.success(
        f"총 {budget} km²를 보호할 때 예상 보존 종수는 약 "
        f"{objective:.2f}종입니다. 가장 많은 면적이 배분된 지형은 "
        f"{HABITAT_LABELS[largest]}이며, 균등 배분보다 약 "
        f"{gain_pct:.1f}% 높은 결과가 계산되었습니다."
    )

    buffer = BytesIO()
    recommendation.save(buffer, format="PNG")

    st.download_button(
        "추천 결과 이미지 다운로드",
        data=buffer.getvalue(),
        file_name="protected_area_recommendation_v5.png",
        mime="image/png",
    )
