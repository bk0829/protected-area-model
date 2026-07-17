from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="보호구역 자동 추천 시스템 4.0",
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
    0: (155, 155, 155, 125),
    1: (35, 155, 75, 135),
    2: (35, 125, 225, 140),
    3: (235, 190, 45, 135),
}

CLUSTER_COLORS = [
    (230, 70, 70, 130),
    (70, 120, 230, 130),
    (70, 190, 100, 130),
    (235, 185, 55, 130),
]

DEFAULT_PARAMS = {
    1: {"c": 15.0, "k": 0.22},
    2: {"c": 12.0, "k": 0.35},
    3: {"c": 8.0, "k": 0.28},
}


def species_value(area: int, c: float, k: float) -> float:
    if area <= 0:
        return 0.0
    return float(c * (area ** k))


def optimize_allocation(
    matrix: np.ndarray,
    budget_cells: int,
    params: Dict[int, Dict[str, float]],
) -> Tuple[Dict[int, int], float]:
    available = {h: int(np.sum(matrix == h)) for h in (1, 2, 3)}
    usable = sum(available.values())
    budget_cells = min(max(int(budget_cells), 0), usable)

    best_value = -1.0
    best_alloc = {1: 0, 2: 0, 3: 0}

    for a1 in range(min(available[1], budget_cells) + 1):
        remain1 = budget_cells - a1
        for a2 in range(min(available[2], remain1) + 1):
            a3 = remain1 - a2
            if a3 < 0 or a3 > available[3]:
                continue

            value = (
                species_value(a1, params[1]["c"], params[1]["k"])
                + species_value(a2, params[2]["c"], params[2]["k"])
                + species_value(a3, params[3]["c"], params[3]["k"])
            )

            if value > best_value:
                best_value = value
                best_alloc = {1: a1, 2: a2, 3: a3}

    return best_alloc, best_value


def neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


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

    def score(cell: Tuple[int, int], selected: set[Tuple[int, int]]) -> float:
        r, c = cell
        same = sum(matrix[nr, nc] == habitat for nr, nc in neighbors(r, c, rows, cols))
        touching = sum((nr, nc) in selected for nr, nc in neighbors(r, c, rows, cols))
        return 3 * touching + same

    seed = max(candidates, key=lambda cell: score(cell, set()))
    selected = {seed}

    while len(selected) < count:
        remaining = [cell for cell in candidates if cell not in selected]
        if not remaining:
            break
        selected.add(max(remaining, key=lambda cell: score(cell, selected)))

    return sorted(selected)


def build_selection_matrix(matrix: np.ndarray, allocation: Dict[int, int]) -> np.ndarray:
    selection = np.zeros_like(matrix, dtype=int)
    for habitat in (1, 2, 3):
        for r, c in select_connected_cells(matrix, habitat, allocation[habitat]):
            selection[r, c] = 1
    return selection


def rgb_to_features(rgb: np.ndarray) -> np.ndarray:
    """
    RGB + 밝기 + 채널 차이 + 식생지수 + 청색우세도를 특징으로 사용.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    brightness = (r + g + b) / 3.0
    vegetation = 2.0 * g - r - b
    blue_dominance = b - (r + g) / 2.0
    grayness = np.max(rgb, axis=1) - np.min(rgb, axis=1)

    return np.column_stack([
        r, g, b,
        brightness,
        vegetation,
        blue_dominance,
        grayness,
    ])


def cluster_image(
    image: Image.Image,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    img = np.asarray(image.convert("RGB"))
    height, width, _ = img.shape

    cell_rgbs = []
    positions = []

    for r in range(rows):
        y0 = int(r * height / rows)
        y1 = int((r + 1) * height / rows)

        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)

            patch = img[y0:y1, x0:x1]
            mean_rgb = patch.mean(axis=(0, 1))
            cell_rgbs.append(mean_rgb)
            positions.append((r, c))

    rgb_array = np.asarray(cell_rgbs)
    features = rgb_to_features(rgb_array)
    scaled = StandardScaler().fit_transform(features)

    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)

    cluster_matrix = np.zeros((rows, cols), dtype=int)
    for (r, c), label in zip(positions, labels):
        cluster_matrix[r, c] = int(label)

    rows_summary = []
    for cluster in range(4):
        mask = labels == cluster
        mean_rgb = rgb_array[mask].mean(axis=0)
        r, g, b = mean_rgb
        brightness = (r + g + b) / 3
        vegetation = 2 * g - r - b
        blue_dom = b - (r + g) / 2

        rows_summary.append({
            "군집": cluster,
            "셀 수": int(mask.sum()),
            "평균 R": round(float(r), 1),
            "평균 G": round(float(g), 1),
            "평균 B": round(float(b), 1),
            "밝기": round(float(brightness), 1),
            "식생지수": round(float(vegetation), 1),
            "청색우세도": round(float(blue_dom), 1),
        })

    return cluster_matrix, pd.DataFrame(rows_summary)


def suggest_cluster_mapping(summary: pd.DataFrame) -> Dict[int, int]:
    """
    군집 통계로 초기 지형 이름을 제안한다.
    이후 사용자가 드롭다운으로 쉽게 수정할 수 있다.
    """
    mapping = {}
    unused = {0, 1, 2, 3}

    # 물: 청색 우세도가 가장 높은 군집
    water_cluster = int(summary.loc[summary["청색우세도"].idxmax(), "군집"])
    mapping[water_cluster] = 2
    unused.remove(water_cluster)

    remaining = summary[summary["군집"].isin(unused)]

    # 산림: 남은 군집 중 식생지수가 높고 상대적으로 어두운 군집
    forest_score = remaining["식생지수"] - 0.2 * remaining["밝기"]
    forest_cluster = int(remaining.loc[forest_score.idxmax(), "군집"])
    mapping[forest_cluster] = 1
    unused.remove(forest_cluster)

    remaining = summary[summary["군집"].isin(unused)]

    # 개발지: 남은 군집 중 밝고 식생지수가 낮은 군집
    developed_score = remaining["밝기"] - 0.5 * remaining["식생지수"]
    developed_cluster = int(remaining.loc[developed_score.idxmax(), "군집"])
    mapping[developed_cluster] = 0
    unused.remove(developed_cluster)

    # 마지막 군집은 초지
    last_cluster = unused.pop()
    mapping[last_cluster] = 3

    return mapping


def apply_cluster_mapping(
    cluster_matrix: np.ndarray,
    mapping: Dict[int, int],
) -> np.ndarray:
    habitat = np.zeros_like(cluster_matrix)
    for cluster, code in mapping.items():
        habitat[cluster_matrix == cluster] = code
    return habitat


def draw_overlay(
    image: Image.Image,
    matrix: np.ndarray,
    colors: Dict[int, Tuple[int, int, int, int]] | List[Tuple[int, int, int, int]],
    selected: np.ndarray | None = None,
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
                draw.rectangle([x0, y0, x1, y1], fill=colors[int(matrix[r, c])])
            elif selected[r, c] == 1:
                draw.rectangle([x0, y0, x1, y1], fill=colors[int(matrix[r, c])])
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 235), width=3)

            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 150), width=1)

    return Image.alpha_composite(img, overlay).convert("RGB")


def matrix_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        columns=[f"C{i+1}" for i in range(matrix.shape[1])],
    )


def dataframe_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return np.clip(numeric.to_numpy(), 0, 3)


def plot_budget_curve(matrix: np.ndarray, params: Dict[int, Dict[str, float]]):
    usable = int(np.sum(matrix != 0))
    budgets = list(range(1, usable + 1))
    values = []

    for budget in budgets:
        _, value = optimize_allocation(matrix, budget, params)
        values.append(value)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(budgets, values)
    ax.set_xlabel("Protected area (cells)")
    ax.set_ylabel("Maximum expected species")
    ax.set_title("Protected Area and Expected Species")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


st.title("🌿 보호구역 자동 추천 시스템 4.0")
st.write(
    "사진 내부의 색을 네 개 군집으로 자동 분류한 뒤, "
    "각 군집을 산림·습지·초지·개발지역에 대응시켜 지형행렬을 생성합니다."
)

st.header("1. 지형 사진 입력")
uploaded = st.file_uploader(
    "항공사진 또는 드론 사진을 업로드하세요.",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("사진을 업로드하면 분석을 시작합니다.")
    st.stop()

source_image = Image.open(uploaded).convert("RGB")

grid_size = st.slider(
    "격자 크기",
    min_value=6,
    max_value=16,
    value=10,
    step=1,
)

signature = (uploaded.name, uploaded.size, grid_size)

if st.session_state.get("signature") != signature:
    cluster_matrix, cluster_summary = cluster_image(
        source_image,
        grid_size,
        grid_size,
    )
    suggested_mapping = suggest_cluster_mapping(cluster_summary)

    st.session_state["signature"] = signature
    st.session_state["cluster_matrix"] = cluster_matrix
    st.session_state["cluster_summary"] = cluster_summary
    st.session_state["suggested_mapping"] = suggested_mapping
    st.session_state.pop("selection", None)

cluster_matrix = st.session_state["cluster_matrix"]
cluster_summary = st.session_state["cluster_summary"]
suggested_mapping = st.session_state["suggested_mapping"]

st.header("2. 색상 군집 자동 분석")

preview_cols = st.columns(2)

with preview_cols[0]:
    st.subheader("원본 사진")
    st.image(source_image, use_container_width=True)

with preview_cols[1]:
    st.subheader("4개 색상 군집")
    st.image(
        draw_overlay(source_image, cluster_matrix, CLUSTER_COLORS),
        use_container_width=True,
    )

st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

st.header("3. 군집별 지형 지정")
st.write("자동 제안이 맞으면 그대로 두고, 틀린 군집만 바꾸면 됩니다.")

mapping = {}
mapping_cols = st.columns(4)
label_options = list(HABITAT_LABELS.values())

for col, cluster in zip(mapping_cols, range(4)):
    with col:
        suggested_code = suggested_mapping[cluster]
        selected_name = st.selectbox(
            f"군집 {cluster}",
            options=label_options,
            index=label_options.index(HABITAT_LABELS[suggested_code]),
            key=f"mapping_{signature}_{cluster}",
        )
        mapping[cluster] = next(
            code for code, name in HABITAT_LABELS.items()
            if name == selected_name
        )

if len(set(mapping.values())) < 4:
    st.warning("가능하면 네 군집을 서로 다른 지형으로 지정하는 것이 좋습니다.")

habitat_matrix = apply_cluster_mapping(cluster_matrix, mapping)

st.subheader("지형 분류 결과")
st.image(
    draw_overlay(source_image, habitat_matrix, HABITAT_COLORS),
    use_container_width=True,
)

st.header("4. 지형행렬 세부 보정")
st.caption("필요한 경우에만 숫자를 수정하세요.")
st.code("0=개발지역·도로, 1=산림, 2=습지·하천, 3=초지·농경지")

edited_df = st.data_editor(
    matrix_dataframe(habitat_matrix),
    use_container_width=True,
    num_rows="fixed",
    key=f"editor_{signature}_{tuple(mapping.items())}",
)
matrix = dataframe_matrix(edited_df)

counts = {code: int(np.sum(matrix == code)) for code in range(4)}
metric_cols = st.columns(4)
for col, code in zip(metric_cols, range(4)):
    col.metric(HABITAT_LABELS[code], f"{counts[code]}칸")

st.header("5. 종-면적 함수와 최적화")

params = {
    code: DEFAULT_PARAMS[code].copy()
    for code in (1, 2, 3)
}

st.latex(r"\max S=15A_F^{0.22}+12A_W^{0.35}+8A_G^{0.28}")
st.latex(r"A_F+A_W+A_G=A_{\mathrm{total}}")

with st.expander("c, k 값 변경"):
    pcols = st.columns(3)
    for col, code in zip(pcols, (1, 2, 3)):
        with col:
            st.subheader(HABITAT_LABELS[code])
            params[code]["c"] = st.number_input(
                "c", value=float(DEFAULT_PARAMS[code]["c"]),
                min_value=0.1, max_value=100.0, step=0.1,
                key=f"c_{code}",
            )
            params[code]["k"] = st.number_input(
                "k", value=float(DEFAULT_PARAMS[code]["k"]),
                min_value=0.01, max_value=0.99, step=0.01,
                key=f"k_{code}",
            )

usable = counts[1] + counts[2] + counts[3]

if usable == 0:
    st.error("보호 가능한 자연 지형이 없습니다.")
    st.stop()

budget = st.slider(
    "총 보호 가능 면적",
    min_value=1,
    max_value=usable,
    value=min(16, usable),
    step=1,
    format="%d km²",
)

if st.button("최적 보호구역 계산", type="primary", use_container_width=True):
    allocation, objective = optimize_allocation(matrix, budget, params)
    selection = build_selection_matrix(matrix, allocation)

    st.session_state["selection"] = selection
    st.session_state["allocation"] = allocation
    st.session_state["objective"] = objective
    st.session_state["result_matrix"] = matrix.copy()
    st.session_state["result_image"] = source_image.copy()
    st.session_state["result_params"] = params.copy()

if "selection" in st.session_state:
    st.divider()
    st.header("6. 추천 결과")

    selection = st.session_state["selection"]
    allocation = st.session_state["allocation"]
    objective = st.session_state["objective"]
    result_matrix = st.session_state["result_matrix"]
    result_image = st.session_state["result_image"]
    result_params = st.session_state["result_params"]

    recommendation = draw_overlay(
        result_image,
        result_matrix,
        HABITAT_COLORS,
        selected=selection,
    )

    result_cols = st.columns(3)
    with result_cols[0]:
        st.subheader("원본")
        st.image(result_image, use_container_width=True)
    with result_cols[1]:
        st.subheader("최종 지형 분류")
        st.image(
            draw_overlay(result_image, result_matrix, HABITAT_COLORS),
            use_container_width=True,
        )
    with result_cols[2]:
        st.subheader("추천 보호구역")
        st.image(recommendation, use_container_width=True)

    metrics = st.columns(4)
    metrics[0].metric("총 보호면적", f"{int(selection.sum())} km²")
    metrics[1].metric("예상 보존 종수", f"{objective:.2f}종")
    metrics[2].metric("산림 배분", f"{allocation[1]} km²")
    metrics[3].metric("습지 배분", f"{allocation[2]} km²")

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
            "선택 면적": area,
            "예상 종수": round(value, 2),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    graph_col, matrix_col = st.columns([1.2, 1])
    with graph_col:
        fig = plot_budget_curve(result_matrix, result_params)
        st.pyplot(fig)
        plt.close(fig)
    with matrix_col:
        st.subheader("선택행렬")
        st.dataframe(pd.DataFrame(selection), use_container_width=True, hide_index=True)

    buffer = BytesIO()
    recommendation.save(buffer, format="PNG")

    st.download_button(
        "추천 결과 이미지 다운로드",
        data=buffer.getvalue(),
        file_name="protected_area_recommendation_v4.png",
        mime="image/png",
    )
