
from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(
    page_title="보호구역 자동 추천 시스템",
    page_icon="🌿",
    layout="wide",
)

# -----------------------------
# 기본 상수
# -----------------------------
HABITAT_LABELS = {
    0: "개발지역·도로",
    1: "산림",
    2: "습지·하천",
    3: "초지·농경지",
}

HABITAT_COLORS = {
    0: (150, 150, 150, 120),
    1: (30, 160, 70, 130),
    2: (35, 120, 230, 135),
    3: (235, 190, 40, 130),
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


# -----------------------------
# 계산 함수
# -----------------------------
def species_value(area: int, c: float, k: float) -> float:
    """S = cA^k. A=0이면 해당 지형을 보호하지 않은 것으로 보아 0."""
    if area <= 0:
        return 0.0
    return float(c * (area ** k))


def optimize_allocation(
    matrix: np.ndarray,
    budget_cells: int,
    params: Dict[int, Dict[str, float]],
) -> Tuple[Dict[int, int], float]:
    """
    산림(1), 습지(2), 초지(3)에 배분할 셀 수를 완전탐색한다.
    목적함수: sum_h c_h * A_h ^ k_h
    제약조건: sum_h A_h = budget_cells, 0 <= A_h <= available_h
    """
    available = {h: int(np.sum(matrix == h)) for h in (1, 2, 3)}
    usable = sum(available.values())
    budget_cells = min(max(int(budget_cells), 0), usable)

    best_value = -1.0
    best_alloc = {1: 0, 2: 0, 3: 0}

    for a1 in range(min(available[1], budget_cells) + 1):
        remain_after_1 = budget_cells - a1
        for a2 in range(min(available[2], remain_after_1) + 1):
            a3 = remain_after_1 - a2
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
    """
    같은 지형 셀 중 서로 인접한 영역을 우선 선택한다.
    최적 '면적 배분'과 공간적 '연결성'을 함께 보여주기 위한 단순 휴리스틱이다.
    """
    candidates = list(zip(*np.where(matrix == habitat)))
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return [(int(r), int(c)) for r, c in candidates]

    rows, cols = matrix.shape

    # 시작점: 같은 지형 이웃 수가 가장 많은 셀
    def same_neighbor_score(cell):
        r, c = cell
        return sum(matrix[nr, nc] == habitat for nr, nc in neighbors(r, c, rows, cols))

    seed = max(candidates, key=same_neighbor_score)
    selected = {seed}
    frontier = set(
        n for n in neighbors(seed[0], seed[1], rows, cols)
        if matrix[n[0], n[1]] == habitat
    )

    while len(selected) < count:
        if frontier:
            # 이미 선택된 셀과 많이 맞닿은 후보를 우선
            next_cell = max(
                frontier,
                key=lambda cell: sum(n in selected for n in neighbors(cell[0], cell[1], rows, cols))
            )
            frontier.remove(next_cell)
        else:
            remaining = [cell for cell in candidates if cell not in selected]
            if not remaining:
                break
            # 떨어진 군집이 필요하면 가장 조밀한 셀에서 새로 시작
            next_cell = max(remaining, key=same_neighbor_score)

        selected.add(next_cell)
        for n in neighbors(next_cell[0], next_cell[1], rows, cols):
            if matrix[n[0], n[1]] == habitat and n not in selected:
                frontier.add(n)

    return sorted((int(r), int(c)) for r, c in selected)


def build_selection_matrix(
    matrix: np.ndarray,
    allocation: Dict[int, int],
) -> np.ndarray:
    selected = np.zeros_like(matrix, dtype=int)
    for habitat in (1, 2, 3):
        cells = select_connected_cells(matrix, habitat, allocation.get(habitat, 0))
        for r, c in cells:
            selected[r, c] = 1
    return selected


def auto_classify_image(image: Image.Image, rows: int, cols: int) -> np.ndarray:
    """
    셀 평균 RGB를 활용한 단순 분류.
    실제 AI 분류가 아니라 탐구용 초기값 생성 기능이다.
    """
    img = image.convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    matrix = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
        for c in range(cols):
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            patch = arr[y0:y1, x0:x1]
            if patch.size == 0:
                matrix[r, c] = 0
                continue

            red, green, blue = patch.mean(axis=(0, 1))
            brightness = (red + green + blue) / 3
            spread = max(red, green, blue) - min(red, green, blue)

            # 매우 단순한 색상 기반 규칙
            if blue > green * 1.08 and blue > red * 1.12:
                code = 2
            elif green > red * 1.08 and green > blue * 1.05 and brightness < 170:
                code = 1
            elif green >= blue and brightness >= 105 and (green + red) > 1.7 * blue:
                code = 3
            elif brightness > 155 or spread < 25:
                code = 0
            else:
                code = 1

            matrix[r, c] = code

    return matrix


def overlay_grid_and_selection(
    image: Image.Image,
    matrix: np.ndarray,
    selection: np.ndarray,
) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rows, cols = matrix.shape
    w, h = img.size

    for r in range(rows):
        y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
        for c in range(cols):
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            if selection[r, c] == 1:
                color = HABITAT_COLORS[int(matrix[r, c])]
                draw.rectangle([x0, y0, x1, y1], fill=color)
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 220), width=2)
            else:
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 130), width=1)

    return Image.alpha_composite(img, overlay).convert("RGB")


def matrix_to_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=[f"C{c+1}" for c in range(matrix.shape[1])])


def dataframe_to_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    arr = numeric.to_numpy()
    return np.clip(arr, 0, 3)


def make_blank_map(rows: int, cols: int) -> Image.Image:
    size = 700
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            x0, x1 = int(c * size / cols), int((c + 1) * size / cols)
            y0, y1 = int(r * size / rows), int((r + 1) * size / rows)
            code = int(EXAMPLE_MATRIX[r % 10, c % 10])
            base = {
                0: (210, 210, 210),
                1: (95, 155, 95),
                2: (95, 155, 220),
                3: (205, 190, 105),
            }[code]
            draw.rectangle([x0, y0, x1, y1], fill=base)
    return img


# -----------------------------
# 화면
# -----------------------------
st.title("🌿 종-면적 관계식 기반 보호구역 자동 추천 시스템")
st.markdown(
    """
    업로드한 지형 사진을 격자로 나누고 각 셀을 지형행렬로 변환한 뒤,
    다음 목적함수의 값이 최대가 되도록 보호 셀을 배분합니다.

    \[
    \max\left(15A_F^{0.22}+12A_W^{0.35}+8A_G^{0.28}\right)
    \]

    단, \(A_F+A_W+A_G=A_{total}\)이며 개발지역·도로는 보호 후보에서 제외합니다.
    """
)

st.warning(
    "색상 자동 분류는 탐구용 보조 기능입니다. 실제 사진에서는 아래 지형행렬을 직접 수정해야 정확합니다."
)

with st.sidebar:
    st.header("1. 입력 설정")
    uploaded = st.file_uploader("지형 사진 업로드", type=["png", "jpg", "jpeg"])
    grid_size = st.slider("격자 크기", min_value=5, max_value=15, value=10, step=1)
    use_example = st.checkbox("발표 자료의 10×10 예시행렬 사용", value=uploaded is None)

    st.header("2. 종-면적 함수")
    params = {}
    for code in (1, 2, 3):
        st.subheader(HABITAT_LABELS[code])
        c_val = st.number_input(
            f"{HABITAT_LABELS[code]} c",
            min_value=0.1,
            max_value=100.0,
            value=float(DEFAULT_PARAMS[code]["c"]),
            step=0.1,
            key=f"c_{code}",
        )
        k_val = st.number_input(
            f"{HABITAT_LABELS[code]} k",
            min_value=0.01,
            max_value=0.99,
            value=float(DEFAULT_PARAMS[code]["k"]),
            step=0.01,
            key=f"k_{code}",
        )
        params[code] = {"c": float(c_val), "k": float(k_val)}

# 이미지 및 초기 행렬
if uploaded is not None:
    source_image = Image.open(uploaded).convert("RGB")
else:
    source_image = make_blank_map(10, 10)

if use_example:
    initial_matrix = EXAMPLE_MATRIX.copy()
    source_image = make_blank_map(10, 10)
else:
    initial_matrix = auto_classify_image(source_image, grid_size, grid_size)

if "matrix_key" not in st.session_state:
    st.session_state.matrix_key = 0

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("① 입력 지형 사진")
    st.image(source_image, use_container_width=True)

with col2:
    st.subheader("② 지형행렬 확인·수정")
    st.caption("0=개발지역·도로, 1=산림, 2=습지·하천, 3=초지·농경지")
    edited_df = st.data_editor(
        matrix_to_dataframe(initial_matrix),
        use_container_width=True,
        num_rows="fixed",
        key=f"terrain_editor_{st.session_state.matrix_key}",
    )
    matrix = dataframe_to_matrix(edited_df)

available_counts = {code: int(np.sum(matrix == code)) for code in range(4)}
usable_cells = available_counts[1] + available_counts[2] + available_counts[3]

st.subheader("③ 총 보호 가능 면적 설정")
budget_cells = st.slider(
    "선택할 보호 셀 수(셀 1칸 = 1 km²로 가정)",
    min_value=1,
    max_value=max(1, usable_cells),
    value=min(16, max(1, usable_cells)),
    step=1,
)

if st.button("🔍 최적 보호구역 계산", type="primary", use_container_width=True):
    allocation, objective_value = optimize_allocation(matrix, budget_cells, params)
    selection = build_selection_matrix(matrix, allocation)

    st.session_state["allocation"] = allocation
    st.session_state["objective_value"] = objective_value
    st.session_state["selection"] = selection
    st.session_state["matrix_result"] = matrix
    st.session_state["source_result"] = source_image

if "selection" in st.session_state:
    allocation = st.session_state["allocation"]
    objective_value = st.session_state["objective_value"]
    selection = st.session_state["selection"]
    matrix_result = st.session_state["matrix_result"]
    result_image = overlay_grid_and_selection(
        st.session_state["source_result"], matrix_result, selection
    )

    st.divider()
    st.header("최적화 결과")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 보호면적", f"{int(selection.sum())} km²")
    m2.metric("예상 보존 종수", f"{objective_value:.2f}")
    m3.metric("습지 배분", f"{allocation[2]} km²")
    m4.metric("선택 비율", f"{selection.sum() / matrix_result.size * 100:.1f}%")

    result_rows = []
    for code in (1, 2, 3):
        area = allocation[code]
        value = species_value(area, params[code]["c"], params[code]["k"])
        result_rows.append({
            "지형": HABITAT_LABELS[code],
            "존재 셀 수": int(np.sum(matrix_result == code)),
            "선택 셀 수": area,
            "보호면적(km²)": area,
            "c": params[code]["c"],
            "k": params[code]["k"],
            "예상 종수": round(value, 2),
        })

    st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

    out1, out2 = st.columns([1.25, 1])
    with out1:
        st.subheader("④ 사진 위 추천 보호구역")
        st.image(result_image, use_container_width=True)
        st.caption("색으로 덮인 셀이 추천 보호구역이며, 같은 지형끼리 인접하도록 우선 선택했습니다.")

    with out2:
        st.subheader("⑤ 선택행렬")
        st.caption("1=보호구역으로 선택, 0=선택하지 않음")
        st.dataframe(
            pd.DataFrame(selection),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("⑥ 목적함수 계산")
        formula_parts = []
        value_parts = []
        for code, symbol in ((1, "F"), (2, "W"), (3, "G")):
            c = params[code]["c"]
            k = params[code]["k"]
            area = allocation[code]
            val = species_value(area, c, k)
            formula_parts.append(f"{c:g}A_{symbol}^{{{k:g}}}")
            value_parts.append(f"{val:.2f}")

        st.latex(r"S=" + "+".join(formula_parts))
        st.latex(
            rf"A_F={allocation[1]},\quad A_W={allocation[2]},\quad A_G={allocation[3]}"
        )
        st.latex(r"S\approx " + "+".join(value_parts) + rf"={objective_value:.2f}")

    # 이미지 다운로드
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    st.download_button(
        "추천 결과 이미지 저장",
        data=buffer.getvalue(),
        file_name="protected_area_recommendation.png",
        mime="image/png",
    )

    with st.expander("모델 해석 및 한계"):
        st.markdown(
            """
            - **수학적 최적화:** 각 지형에 배분할 정수 셀 수의 모든 가능한 조합을 비교하여
              종-면적 함수 합이 가장 큰 배분을 선택했습니다.
            - **공간 선택:** 최적 배분 수만큼 실제 셀을 고를 때 같은 지형끼리 인접한 영역을
              우선하는 단순 연결성 알고리즘을 사용했습니다.
            - **중요한 한계:** 사진 색상만으로 실제 생태적 가치를 판단할 수 없으며,
              \(c\), \(k\)도 지역 생태조사 자료로 보정해야 합니다.
            - 서로 다른 지형의 종 목록이 겹치면 단순 합산한 종수는 실제 고유 종수를
              과대평가할 수 있습니다.
            """
        )
else:
    st.info("지형행렬과 총 보호면적을 확인한 뒤 ‘최적 보호구역 계산’을 눌러 주세요.")
