
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(
    page_title="보호구역 자동 추천 시스템 3.0",
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

DEFAULT_PARAMS = {
    1: {"c": 15.0, "k": 0.22},
    2: {"c": 12.0, "k": 0.35},
    3: {"c": 8.0, "k": 0.28},
}

EXAMPLE_MATRIX = np.array([
    [1,1,1,1,1,1,2,0,3,3],
    [1,1,1,1,1,1,2,0,3,3],
    [1,1,1,1,1,3,2,0,3,3],
    [1,1,1,1,3,3,2,0,3,1],
    [1,1,1,3,3,3,2,0,1,1],
    [1,1,3,3,3,3,2,0,1,1],
    [1,1,1,3,3,1,2,0,1,3],
    [1,1,1,1,3,1,2,0,1,3],
    [1,1,1,1,1,1,2,0,1,1],
    [1,1,1,1,1,1,2,0,1,1],
], dtype=int)


# =========================================================
# 수학 모델
# =========================================================
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


def equal_allocation(
    matrix: np.ndarray,
    budget_cells: int,
    params: Dict[int, Dict[str, float]],
) -> Tuple[Dict[int, int], float]:
    available = {h: int(np.sum(matrix == h)) for h in (1, 2, 3)}
    alloc = {1: 0, 2: 0, 3: 0}
    remaining = budget_cells

    while remaining > 0:
        changed = False
        for h in (1, 2, 3):
            if remaining > 0 and alloc[h] < available[h]:
                alloc[h] += 1
                remaining -= 1
                changed = True
        if not changed:
            break

    value = sum(
        species_value(alloc[h], params[h]["c"], params[h]["k"])
        for h in (1, 2, 3)
    )
    return alloc, value


# =========================================================
# 공간 연결성
# =========================================================
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
        same_neighbors = sum(
            matrix[nr, nc] == habitat
            for nr, nc in neighbors(r, c, rows, cols)
        )
        selected_neighbors = sum(
            (nr, nc) in selected
            for nr, nc in neighbors(r, c, rows, cols)
        )
        return 3.0 * selected_neighbors + same_neighbors

    seed = max(candidates, key=lambda cell: score(cell, set()))
    selected = {seed}

    while len(selected) < count:
        remaining = [cell for cell in candidates if cell not in selected]
        if not remaining:
            break
        next_cell = max(remaining, key=lambda cell: score(cell, selected))
        selected.add(next_cell)

    return sorted(selected)


def build_selection_matrix(
    matrix: np.ndarray,
    allocation: Dict[int, int],
) -> np.ndarray:
    selection = np.zeros_like(matrix, dtype=int)
    for habitat in (1, 2, 3):
        for r, c in select_connected_cells(matrix, habitat, allocation[habitat]):
            selection[r, c] = 1
    return selection


# =========================================================
# 이미지 분류
# =========================================================
def auto_classify_image(
    image: Image.Image,
    rows: int,
    cols: int,
) -> np.ndarray:
    img = image.convert("RGB")
    arr = np.asarray(img)
    height, width, _ = arr.shape
    matrix = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        y0 = int(r * height / rows)
        y1 = int((r + 1) * height / rows)

        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)
            patch = arr[y0:y1, x0:x1]

            if patch.size == 0:
                continue

            red, green, blue = patch.mean(axis=(0, 1))
            brightness = (red + green + blue) / 3
            spread = max(red, green, blue) - min(red, green, blue)

            water_score = (
                max(0.0, blue - green) * 1.5
                + max(0.0, blue - red)
                + max(0.0, 115 - brightness) * 0.15
            )
            forest_score = (
                max(0.0, green - red) * 1.3
                + max(0.0, green - blue)
                + max(0.0, 165 - brightness) * 0.15
            )
            grass_score = (
                max(0.0, green - blue * 0.8)
                + max(0.0, red - blue) * 0.35
                + max(0.0, brightness - 95) * 0.18
            )
            developed_score = (
                max(0.0, brightness - 130) * 0.35
                + max(0.0, 32 - spread) * 0.65
                + max(0.0, red - green) * 0.15
            )

            scores = {
                0: developed_score,
                1: forest_score,
                2: water_score,
                3: grass_score,
            }
            matrix[r, c] = max(scores, key=scores.get)

    return matrix


def smooth_matrix(matrix: np.ndarray, passes: int = 1) -> np.ndarray:
    """
    주변 8개 셀의 최빈값을 이용해 고립된 오분류를 완화한다.
    개발지(0)는 너무 쉽게 다른 지형으로 바뀌지 않도록 보수적으로 처리한다.
    """
    result = matrix.copy()
    rows, cols = matrix.shape

    for _ in range(passes):
        updated = result.copy()

        for r in range(rows):
            for c in range(cols):
                values = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            values.append(int(result[nr, nc]))

                if not values:
                    continue

                counts = np.bincount(values, minlength=4)
                majority = int(np.argmax(counts))

                if counts[majority] >= 5 and result[r, c] != 0:
                    updated[r, c] = majority

        result = updated

    return result


# =========================================================
# 시각화
# =========================================================
def draw_grid(
    image: Image.Image,
    matrix: np.ndarray,
    classification: bool = False,
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

            if classification:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=HABITAT_COLORS[int(matrix[r, c])],
                )

            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(255, 255, 255, 180),
                width=1,
            )

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_recommendation(
    image: Image.Image,
    matrix: np.ndarray,
    selection: np.ndarray,
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

            if selection[r, c] == 1:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=HABITAT_COLORS[int(matrix[r, c])],
                )
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline=(0, 0, 0, 235),
                    width=3,
                )
            else:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline=(255, 255, 255, 130),
                    width=1,
                )

    return Image.alpha_composite(img, overlay).convert("RGB")


def matrix_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        columns=[f"C{i+1}" for i in range(matrix.shape[1])],
    )


def dataframe_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return np.clip(numeric.to_numpy(), 0, 3)


def make_example_image(matrix: np.ndarray) -> Image.Image:
    rows, cols = matrix.shape
    size = 720
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    base_colors = {
        0: (205, 205, 205),
        1: (85, 145, 95),
        2: (85, 145, 215),
        3: (205, 185, 105),
    }

    for r in range(rows):
        for c in range(cols):
            x0 = int(c * size / cols)
            x1 = int((c + 1) * size / cols)
            y0 = int(r * size / rows)
            y1 = int((r + 1) * size / rows)
            draw.rectangle(
                [x0, y0, x1, y1],
                fill=base_colors[int(matrix[r, c])],
            )

    return img


def plot_budget_curve(
    matrix: np.ndarray,
    params: Dict[int, Dict[str, float]],
):
    usable = int(np.sum(matrix != 0))
    budgets = list(range(1, usable + 1))
    optimum_values = []
    equal_values = []

    for budget in budgets:
        _, opt_value = optimize_allocation(matrix, budget, params)
        _, eq_value = equal_allocation(matrix, budget, params)
        optimum_values.append(opt_value)
        equal_values.append(eq_value)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(budgets, optimum_values, label="Optimal allocation")
    ax.plot(budgets, equal_values, linestyle="--", label="Equal allocation")
    ax.set_xlabel("Protected area (cells)")
    ax.set_ylabel("Expected species")
    ax.set_title("Protected Area and Maximum Expected Species")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# =========================================================
# UI
# =========================================================
st.title("🌿 보호구역 자동 추천 시스템 3.0")
st.write(
    "지형 사진을 격자로 나누어 지형행렬로 변환하고, "
    "종-면적 관계식의 합이 최대가 되는 면적 배분과 위치를 추천합니다."
)

st.warning(
    "자동 지형 분류는 평균 색상과 주변 셀 보정을 이용한 탐구용 모델입니다. "
    "실제 보호구역 지정에는 현장 조사와 전문 토지피복 자료가 필요합니다."
)

st.header("1. 지형 사진 입력")

input_mode = st.radio(
    "입력 방법",
    ["사진 업로드", "발표용 예시 사용"],
    horizontal=True,
)

if input_mode == "사진 업로드":
    uploaded = st.file_uploader(
        "항공사진 또는 드론 사진을 업로드하세요.",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is None:
        st.info("사진을 업로드하면 분석을 시작합니다.")
        st.stop()
    source_image = Image.open(uploaded).convert("RGB")
else:
    uploaded = None
    source_image = make_example_image(EXAMPLE_MATRIX)

grid_size = st.slider(
    "격자 크기",
    min_value=5,
    max_value=15,
    value=10,
    step=1,
)

smoothing = st.slider(
    "자동 분류 보정 강도",
    min_value=0,
    max_value=3,
    value=1,
    step=1,
    help="값이 클수록 주변 셀과 다른 고립된 분류를 더 많이 보정합니다.",
)

signature = (
    uploaded.name if uploaded is not None else "example",
    grid_size,
    smoothing,
    source_image.size,
)

if st.session_state.get("signature") != signature:
    st.session_state["signature"] = signature

    if input_mode == "발표용 예시 사용" and grid_size == 10:
        initial = EXAMPLE_MATRIX.copy()
    else:
        initial = auto_classify_image(source_image, grid_size, grid_size)
        initial = smooth_matrix(initial, smoothing)

    st.session_state["terrain_matrix"] = initial
    st.session_state.pop("selection", None)

matrix = st.session_state["terrain_matrix"]

st.header("2. 자동 격자 및 지형 분류")

cols = st.columns(3)

with cols[0]:
    st.subheader("원본 사진")
    st.image(source_image, use_container_width=True)

with cols[1]:
    st.subheader("자동 격자")
    st.image(draw_grid(source_image, matrix), use_container_width=True)

with cols[2]:
    st.subheader("자동 지형 분류")
    st.image(
        draw_grid(source_image, matrix, classification=True),
        use_container_width=True,
    )

st.caption("회색=개발지역·도로 / 초록=산림 / 파랑=습지·하천 / 노랑=초지·농경지")

st.header("3. 지형행렬 보정")
st.write("자동 분류가 잘못된 셀의 숫자만 수정하세요.")
st.code("0=개발지역·도로, 1=산림, 2=습지·하천, 3=초지·농경지")

edited = st.data_editor(
    matrix_dataframe(matrix),
    use_container_width=True,
    num_rows="fixed",
    key=f"editor_{signature}",
)

matrix = dataframe_matrix(edited)
st.session_state["terrain_matrix"] = matrix

counts = {code: int(np.sum(matrix == code)) for code in range(4)}
metric_columns = st.columns(4)

for column, code in zip(metric_columns, range(4)):
    column.metric(HABITAT_LABELS[code], f"{counts[code]}칸")

st.header("4. 종-면적 함수 및 제약조건")

st.latex(r"\max S=15A_F^{0.22}+12A_W^{0.35}+8A_G^{0.28}")
st.latex(r"A_F+A_W+A_G=A_{\mathrm{total}}")

params = {
    code: DEFAULT_PARAMS[code].copy()
    for code in (1, 2, 3)
}

with st.expander("c, k 값 변경"):
    param_columns = st.columns(3)

    for column, code in zip(param_columns, (1, 2, 3)):
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

st.header("5. 보호 가능 면적 설정")

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
    equal_alloc, equal_value = equal_allocation(matrix, budget, params)
    selection = build_selection_matrix(matrix, allocation)

    st.session_state["selection"] = selection
    st.session_state["allocation"] = allocation
    st.session_state["objective"] = objective
    st.session_state["equal_alloc"] = equal_alloc
    st.session_state["equal_value"] = equal_value
    st.session_state["result_matrix"] = matrix.copy()
    st.session_state["result_image"] = source_image.copy()
    st.session_state["result_params"] = params.copy()

if "selection" in st.session_state:
    st.divider()
    st.header("6. 분석 결과")

    selection = st.session_state["selection"]
    allocation = st.session_state["allocation"]
    objective = st.session_state["objective"]
    equal_value = st.session_state["equal_value"]
    result_matrix = st.session_state["result_matrix"]
    result_image = st.session_state["result_image"]
    result_params = st.session_state["result_params"]

    gain = objective - equal_value
    gain_pct = 0.0 if equal_value == 0 else gain / equal_value * 100

    metrics = st.columns(4)
    metrics[0].metric("총 보호면적", f"{int(selection.sum())} km²")
    metrics[1].metric("최적 예상 종수", f"{objective:.2f}종")
    metrics[2].metric("균등 배분 대비 증가", f"{gain:.2f}종")
    metrics[3].metric("개선율", f"{gain_pct:.1f}%")

    result_columns = st.columns(3)

    with result_columns[0]:
        st.subheader("원본")
        st.image(result_image, use_container_width=True)

    with result_columns[1]:
        st.subheader("지형 분류")
        st.image(
            draw_grid(result_image, result_matrix, classification=True),
            use_container_width=True,
        )

    with result_columns[2]:
        st.subheader("추천 보호구역")
        recommendation = draw_recommendation(
            result_image,
            result_matrix,
            selection,
        )
        st.image(recommendation, use_container_width=True)

    rows = []
    for code in (1, 2, 3):
        area = allocation[code]
        c = result_params[code]["c"]
        k = result_params[code]["k"]
        value = species_value(area, c, k)

        rows.append({
            "지형": HABITAT_LABELS[code],
            "전체 셀": int(np.sum(result_matrix == code)),
            "선택 셀": area,
            "c": c,
            "k": k,
            "예상 종수": round(value, 2),
        })

    st.subheader("지형별 최적 배분")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    graph_col, matrix_col = st.columns([1.25, 1])

    with graph_col:
        st.subheader("보호면적에 따른 예상 종수")
        fig = plot_budget_curve(result_matrix, result_params)
        st.pyplot(fig)
        plt.close(fig)

    with matrix_col:
        st.subheader("선택행렬")
        st.dataframe(
            pd.DataFrame(selection),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("자동 결과 해석")

    largest_habitat = max(allocation, key=allocation.get)
    largest_name = HABITAT_LABELS[largest_habitat]

    st.success(
        f"총 {budget} km²를 보호할 때 예상 종수는 약 {objective:.2f}종으로 계산되었습니다. "
        f"균등 배분보다 약 {gain:.2f}종, {gain_pct:.1f}% 높은 결과입니다. "
        f"가장 많은 면적이 배분된 지형은 {largest_name}이며, "
        f"해당 지형의 c·k 값과 실제 존재 면적이 최적 배분에 영향을 주었습니다."
    )

    f_value = species_value(
        allocation[1],
        result_params[1]["c"],
        result_params[1]["k"],
    )
    w_value = species_value(
        allocation[2],
        result_params[2]["c"],
        result_params[2]["k"],
    )
    g_value = species_value(
        allocation[3],
        result_params[3]["c"],
        result_params[3]["k"],
    )

    st.latex(
        rf"S\approx {f_value:.2f}+{w_value:.2f}+{g_value:.2f}"
        rf"={objective:.2f}"
    )

    image_buffer = BytesIO()
    recommendation.save(image_buffer, format="PNG")

    st.download_button(
        "추천 결과 이미지 다운로드",
        data=image_buffer.getvalue(),
        file_name="protected_area_recommendation_v3.png",
        mime="image/png",
    )

    with st.expander("모델의 한계"):
        st.markdown(
            """
            - 자동 지형 분류는 색상과 주변 셀 보정에 기반한 단순 모델입니다.
            - 실제 \(c\), \(k\) 값은 지역별 생태조사 자료로 추정해야 합니다.
            - 서로 다른 지형에 동일한 종이 서식하면 단순 합산 종수는 중복될 수 있습니다.
            - 실제 보호구역 설계에는 토지 가격, 멸종위기종, 서식지 연결성,
              도로 접근성 및 주민 생활권 등을 추가해야 합니다.
            """
        )
