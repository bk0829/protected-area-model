
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(
    page_title="보호구역 자동 추천 시스템 2.0",
    page_icon="🌿",
    layout="wide",
)

# =========================================================
# 기본 설정
# =========================================================
HABITAT_LABELS = {
    0: "개발지역·도로",
    1: "산림",
    2: "습지·하천",
    3: "초지·농경지",
}

HABITAT_COLORS = {
    0: (165, 165, 165, 125),
    1: (30, 155, 75, 135),
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
    """
    목적함수:
        max Σ c_h A_h^k_h

    제약조건:
        Σ A_h = budget_cells
        0 <= A_h <= available_h

    산림, 습지, 초지의 모든 가능한 정수 면적 배분을 비교한다.
    """
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


# =========================================================
# 공간 선택
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

    def same_neighbor_score(cell: Tuple[int, int]) -> int:
        r, c = cell
        return sum(
            matrix[nr, nc] == habitat
            for nr, nc in neighbors(r, c, rows, cols)
        )

    seed = max(candidates, key=same_neighbor_score)
    selected = {seed}
    frontier = {
        n for n in neighbors(seed[0], seed[1], rows, cols)
        if matrix[n[0], n[1]] == habitat
    }

    while len(selected) < count:
        if frontier:
            next_cell = max(
                frontier,
                key=lambda cell: sum(
                    n in selected
                    for n in neighbors(cell[0], cell[1], rows, cols)
                )
            )
            frontier.remove(next_cell)
        else:
            remaining = [cell for cell in candidates if cell not in selected]
            if not remaining:
                break
            next_cell = max(remaining, key=same_neighbor_score)

        selected.add(next_cell)

        for n in neighbors(next_cell[0], next_cell[1], rows, cols):
            if matrix[n[0], n[1]] == habitat and n not in selected:
                frontier.add(n)

    return sorted(selected)


def build_selection_matrix(
    matrix: np.ndarray,
    allocation: Dict[int, int],
) -> np.ndarray:
    selection = np.zeros_like(matrix, dtype=int)

    for habitat in (1, 2, 3):
        cells = select_connected_cells(
            matrix,
            habitat,
            allocation.get(habitat, 0),
        )
        for r, c in cells:
            selection[r, c] = 1

    return selection


# =========================================================
# 이미지 분석
# =========================================================
def auto_classify_image(
    image: Image.Image,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    각 셀의 평균 RGB를 이용한 탐구용 단순 분류.
    실제 AI 위성영상 분류 모델은 아니다.
    """
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

            # 물: 청색 성분 우세
            if blue > green * 1.06 and blue > red * 1.12:
                code = 2

            # 산림: 녹색 우세 + 지나치게 밝지 않음
            elif green > red * 1.07 and green > blue * 1.03 and brightness < 175:
                code = 1

            # 초지·농경지: 비교적 밝은 녹색 또는 황록색
            elif green >= blue and brightness >= 100 and (red + green) > blue * 1.65:
                code = 3

            # 개발지·도로: 밝은 회색 또는 색상 차이가 적음
            elif brightness > 155 or spread < 24:
                code = 0

            else:
                code = 1

            matrix[r, c] = code

    return matrix


def draw_grid(
    image: Image.Image,
    matrix: np.ndarray,
    show_classification: bool = False,
) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    rows, cols = matrix.shape
    width, height = img.size

    for r in range(rows):
        y0 = int(r * height / rows)
        y1 = int((r + 1) * height / rows)

        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)

            if show_classification:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=HABITAT_COLORS[int(matrix[r, c])],
                )

            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(255, 255, 255, 190),
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
        y0 = int(r * height / rows)
        y1 = int((r + 1) * height / rows)

        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)

            if selection[r, c] == 1:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=HABITAT_COLORS[int(matrix[r, c])],
                )
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline=(0, 0, 0, 230),
                    width=3,
                )
            else:
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline=(255, 255, 255, 140),
                    width=1,
                )

    return Image.alpha_composite(img, overlay).convert("RGB")


def matrix_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        columns=[f"C{index + 1}" for index in range(matrix.shape[1])],
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
        0: (200, 200, 200),
        1: (80, 145, 90),
        2: (80, 145, 210),
        3: (205, 185, 100),
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


# =========================================================
# 화면 구성
# =========================================================
st.title("🌿 보호구역 자동 추천 시스템 2.0")
st.write(
    "지형 사진을 업로드하면 프로그램이 자동으로 격자를 생성하고 "
    "산림·습지·초지·개발지역을 임시 분류한 뒤, "
    "종-면적 관계식의 합이 최대가 되는 보호구역을 추천합니다."
)

st.info(
    "자동 분류는 평균 색상 기반의 탐구용 보조 기능입니다. "
    "분류 결과가 틀린 셀은 지형행렬에서 직접 수정할 수 있습니다."
)

# 1단계: 메인 화면에서 사진 입력
st.header("1. 지형 사진 입력")

input_mode = st.radio(
    "입력 방법",
    ["사진 업로드", "발표용 예시 사용"],
    horizontal=True,
)

if input_mode == "사진 업로드":
    uploaded = st.file_uploader(
        "항공사진 또는 드론 사진을 올려 주세요.",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is None:
        st.warning("사진을 먼저 업로드해 주세요.")
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
    help="10을 선택하면 사진을 10×10, 총 100개 셀로 나눕니다.",
)

# 입력이 변경되었는지 확인하기 위한 키
file_signature = (
    uploaded.name if uploaded is not None else "example",
    grid_size,
    source_image.size,
)

if st.session_state.get("file_signature") != file_signature:
    st.session_state["file_signature"] = file_signature

    if input_mode == "발표용 예시 사용" and grid_size == 10:
        initial_matrix = EXAMPLE_MATRIX.copy()
    else:
        initial_matrix = auto_classify_image(
            source_image,
            grid_size,
            grid_size,
        )

    st.session_state["terrain_matrix"] = initial_matrix
    st.session_state.pop("selection", None)

# 2단계: 자동 격자와 자동 분류
st.header("2. 자동 격자 및 지형 분류")

current_matrix = st.session_state["terrain_matrix"]

preview1, preview2 = st.columns(2)

with preview1:
    st.subheader("원본 사진 + 자동 격자")
    st.image(
        draw_grid(source_image, current_matrix, show_classification=False),
        use_container_width=True,
    )

with preview2:
    st.subheader("자동 지형 분류 결과")
    st.image(
        draw_grid(source_image, current_matrix, show_classification=True),
        use_container_width=True,
    )
    st.caption(
        "회색=개발지역·도로, 초록=산림, 파랑=습지·하천, 노랑=초지·농경지"
    )

# 3단계: 행렬 수정
st.header("3. 지형행렬 확인 및 수정")
st.write("잘못 분류된 셀의 숫자만 고치면 됩니다.")
st.code("0 = 개발지역·도로 / 1 = 산림 / 2 = 습지·하천 / 3 = 초지·농경지")

edited_df = st.data_editor(
    matrix_dataframe(current_matrix),
    use_container_width=True,
    num_rows="fixed",
    key=f"terrain_editor_{file_signature}",
)

matrix = dataframe_matrix(edited_df)
st.session_state["terrain_matrix"] = matrix

counts = {
    code: int(np.sum(matrix == code))
    for code in range(4)
}

count_cols = st.columns(4)
for col, code in zip(count_cols, range(4)):
    col.metric(HABITAT_LABELS[code], f"{counts[code]}칸")

# 4단계: 함수 설정
st.header("4. 종-면적 함수 설정")

st.latex(
    r"S=15A_F^{0.22}+12A_W^{0.35}+8A_G^{0.28}"
)

with st.expander("c, k 값을 직접 바꾸기", expanded=False):
    params = {}

    parameter_cols = st.columns(3)

    for column, code in zip(parameter_cols, (1, 2, 3)):
        with column:
            st.subheader(HABITAT_LABELS[code])

            c_value = st.number_input(
                "c 값",
                min_value=0.1,
                max_value=100.0,
                value=float(DEFAULT_PARAMS[code]["c"]),
                step=0.1,
                key=f"c_{code}",
            )

            k_value = st.number_input(
                "k 값",
                min_value=0.01,
                max_value=0.99,
                value=float(DEFAULT_PARAMS[code]["k"]),
                step=0.01,
                key=f"k_{code}",
            )

            params[code] = {
                "c": float(c_value),
                "k": float(k_value),
            }

if "params" not in locals():
    params = {
        code: DEFAULT_PARAMS[code].copy()
        for code in (1, 2, 3)
    }

# 5단계: 최적화
st.header("5. 보호 가능 면적 설정 및 최적화")

usable_cells = counts[1] + counts[2] + counts[3]

if usable_cells == 0:
    st.error("산림, 습지 또는 초지로 분류된 셀이 없습니다.")
    st.stop()

budget = st.slider(
    "총 보호 가능 면적",
    min_value=1,
    max_value=usable_cells,
    value=min(16, usable_cells),
    step=1,
    format="%d km²",
    help="셀 1칸을 1 km²로 가정합니다.",
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

    st.session_state["allocation"] = allocation
    st.session_state["objective"] = objective
    st.session_state["equal_alloc"] = equal_alloc
    st.session_state["equal_value"] = equal_value
    st.session_state["selection"] = selection
    st.session_state["result_matrix"] = matrix.copy()
    st.session_state["result_image"] = source_image.copy()
    st.session_state["result_params"] = params

# 6단계: 결과
if "selection" in st.session_state:
    st.divider()
    st.header("6. 최적 보호구역 추천 결과")

    allocation = st.session_state["allocation"]
    objective = st.session_state["objective"]
    equal_alloc = st.session_state["equal_alloc"]
    equal_value = st.session_state["equal_value"]
    selection = st.session_state["selection"]

    improvement = objective - equal_value
    improvement_rate = (
        0.0 if equal_value <= 0
        else improvement / equal_value * 100
    )
    result_matrix = st.session_state["result_matrix"]
    result_image = st.session_state["result_image"]
    result_params = st.session_state["result_params"]

    recommendation = draw_recommendation(
        result_image,
        result_matrix,
        selection,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("총 보호면적", f"{int(selection.sum())} km²")
    metric_cols[1].metric("최적 배분 예상 종수", f"{objective:.2f}종")
    metric_cols[2].metric(
        "균등 배분 대비 증가",
        f"{improvement:.2f}종",
        delta=f"{improvement_rate:.1f}%",
    )
    metric_cols[3].metric("균등 배분 예상 종수", f"{equal_value:.2f}종")

    comparison_cols = st.columns(3)

    with comparison_cols[0]:
        st.subheader("원본 사진")
        st.image(result_image, use_container_width=True)

    with comparison_cols[1]:
        st.subheader("자동 지형 분류")
        st.image(
            draw_grid(result_image, result_matrix, show_classification=True),
            use_container_width=True,
        )

    with comparison_cols[2]:
        st.subheader("추천 보호구역")
        st.image(recommendation, use_container_width=True)

    result_rows = []

    for code in (1, 2, 3):
        area = allocation[code]
        c_value = result_params[code]["c"]
        k_value = result_params[code]["k"]
        species = species_value(area, c_value, k_value)

        result_rows.append({
            "지형": HABITAT_LABELS[code],
            "존재 셀 수": int(np.sum(result_matrix == code)),
            "선택 셀 수": area,
            "보호면적(km²)": area,
            "c": c_value,
            "k": k_value,
            "예상 종수": round(species, 2),
        })

    st.subheader("지형별 최적 배분")
    st.dataframe(
        pd.DataFrame(result_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("최적 배분과 균등 배분 비교")
    allocation_comparison = pd.DataFrame([
        {
            "지형": HABITAT_LABELS[1],
            "최적 배분(km²)": allocation[1],
            "균등 배분(km²)": equal_alloc[1],
        },
        {
            "지형": HABITAT_LABELS[2],
            "최적 배분(km²)": allocation[2],
            "균등 배분(km²)": equal_alloc[2],
        },
        {
            "지형": HABITAT_LABELS[3],
            "최적 배분(km²)": allocation[3],
            "균등 배분(km²)": equal_alloc[3],
        },
    ])
    st.dataframe(
        allocation_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"같은 총 보호면적을 균등하게 배분했을 때는 약 "
        f"{equal_value:.2f}종, 최적화했을 때는 약 {objective:.2f}종으로 "
        f"{improvement:.2f}종({improvement_rate:.1f}%) 더 많이 보존하는 것으로 계산되었습니다."
    )

    result_left, result_right = st.columns(2)

    with result_left:
        st.subheader("선택행렬")
        st.caption("1 = 보호구역으로 선택, 0 = 선택하지 않음")
        st.dataframe(
            pd.DataFrame(selection),
            use_container_width=True,
            hide_index=True,
        )

    with result_right:
        st.subheader("목적함수 계산")

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
            rf"A_F={allocation[1]},\quad "
            rf"A_W={allocation[2]},\quad "
            rf"A_G={allocation[3]}"
        )
        st.latex(
            rf"S_F\approx {f_value:.2f},\quad "
            rf"S_W\approx {w_value:.2f},\quad "
            rf"S_G\approx {g_value:.2f}"
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
        file_name="protected_area_recommendation.png",
        mime="image/png",
    )

    with st.expander("모델 해석 및 한계"):
        st.markdown(
            """
            **모델 해석**

            - 프로그램은 산림·습지·초지에 배분할 셀 수의 모든 가능한 정수 조합을 비교합니다.
            - 종-면적 함수의 합이 가장 큰 배분을 최적해로 선택합니다.
            - 실제 위치를 표시할 때는 같은 지형의 셀이 최대한 연결되도록 선택합니다.

            **한계**

            - 색상 기반 자동 분류는 실제 토지피복 분류 AI보다 정확도가 낮습니다.
            - 실제 지역의 \(c\), \(k\) 값은 생태조사 자료로 보정해야 합니다.
            - 서로 다른 지형에 같은 종이 존재하면 예상 종수를 단순 합산할 때 중복될 수 있습니다.
            - 실제 정책에서는 토지 비용, 멸종위기종, 생태통로, 서식지 연결성 등을 추가해야 합니다.
            """
        )
