import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(
    page_title="WHO 기대수명 예측 시스템",
    layout="wide"
)

st.title("🌍 WHO 기대수명 예측 머신러닝 웹 서비스")
st.write("Linear, Polynomial, Ridge 회귀 모델을 비교하고 기대수명을 실시간으로 예측합니다.")

# 저장된 파일 불러오기
payload = joblib.load("life_expectancy_models.pkl")
models = payload["models"]
performance_df = payload["performance"]
features = payload["features"]
X_test = payload["X_test"]

# ------------------------------------------------------------
# [조건 3] 모델 성능 비교 화면
# ------------------------------------------------------------
st.header("📊 모델 성능 비교")

# 1. 성능 평가지표 테이블 출력
st.dataframe(
    performance_df,
    use_container_width=True,
    hide_index=True
)

# 2. Test R2 점수 비교 막대그래프 시각화
st.subheader("📈 Test R² 점수 비교")
fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#abcdef', '#ff9999', '#99ff99']
bars = ax.bar(performance_df["Model"], performance_df["Test R²"], color=colors, edgecolor='black')

ax.set_ylabel("Test R²")
ax.set_title("Model Comparison (Test R²)", fontsize=14)
ax.set_ylim(min(performance_df["Test R²"].min() - 0.2, 0), 1.1)

# 그래프 위에 수치 표시
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

st.pyplot(fig)

# ------------------------------------------------------------
# [조건 4] 사이드바 UI 및 실시간 예측 구성
# ------------------------------------------------------------
st.sidebar.header("🔧 독립변수(Features) 입력값 설정")

input_values = {}
for feature in features:
    min_val = float(X_test[feature].min())
    max_val = float(X_test[feature].max())
    mean_val = float(X_test[feature].mean())

    input_values[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

st.header("🎯 실시간 기대수명 예측")
selected_model = st.selectbox(
    "예측에 사용할 머신러닝 모델을 선택하세요:",
    ["Linear", "Poly", "Ridge"]
)

# 예측 수행
input_df = pd.DataFrame([input_values])
model = models[selected_model]
prediction = model.predict(input_df)[0]

# 결과 큰 글씨 출력
st.metric(
    label=f"[{selected_model} 모델] 예측 기대수명",
    value=f"{prediction:.2f} 세"
)

# 입력값 요약 제공
st.subheader("💡 현재 입력된 데이터 상스 정보")
st.table(input_df)
