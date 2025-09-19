# Seq2Seq Share Forecasting with Attention Mechanism

## 📖 Overview

This project is an advanced time series forecasting system for **Brand Market Share Prediction**. It combines Granger causality analysis, forward-shift exogenous variable processing, and Attention-based Seq2Seq models to provide accurate future predictions.

### 🎯 Key Features
- **Automatic Schema Recognition**: Automatically detects date, share target, and exogenous variable columns
- **Granger Causality Analysis**: Statistically validates significant lag effects of exogenous variables
- **Forward-shift + Tail-ARIMA**: Generates lag variables considering future impacts
- **Dual Architecture**: Compares Softmax vs Sum-Normalization regularization approaches
- **Attention Seq2Seq**: LSTM-based encoder-decoder with Additive Attention mechanism
- **Future Scenario Forecasting**: Extends monthly predictions up to December 2026

## 🏗️ System Architecture

### 1. Data Processing Pipeline
```
Raw Data → Schema Detection → Logit Transform → Granger Test → Lag Generation → Sequence Building
```

### 2. Model Structure
```
Encoder (LSTM) → Context Vector → Decoder (LSTM) → Attention → Dense Layers → Output
```

## 🔧 Installation & Requirements

### Required Libraries
```bash
pip install tensorflow pandas numpy matplotlib statsmodels tqdm openpyxl
```

### Environment Requirements
- Python 3.7+
- TensorFlow 2.x
- Google Colab or local environment

## 📊 Data Format

### Input Data Structure
The Excel file should follow this format:

```
| Year | Month | Brand1_Share | Brand2_Share | ExogVar1 | ExogVar2 |
|------|-------|--------------|--------------|----------|----------|
| 2019 | 1     | 0.35         | 0.25         | 1000     | 50       |
| 2019 | 2     | 0.37         | 0.23         | 1050     | 52       |
```

**Alternative Date Format:**
```
| Date       | Brand1_Share | Brand2_Share | ExogVar1 | ExogVar2 |
|------------|--------------|--------------|----------|----------|
| 2019-01-01 | 0.35         | 0.25         | 1000     | 50       |
| 2019-02-01 | 0.37         | 0.23         | 1050     | 52       |
```

### Column Requirements
- **Date Columns**: `Year/Month` or `Date` (case-insensitive)
- **Target Columns**: Column names ending with "Share" (case-insensitive)
- **Exogenous Variables**: Any numeric columns excluding date and share columns

## 🚀 Usage

### 1. Setup Paths
```python
IN_XLSX   = "/path/to/your/data.xlsx"     # Input Excel file
OUT_DIR   = "/path/to/output/directory"   # Output directory
```

### 2. Run the Complete Pipeline
```python
# Share Forecast (Seq2Seq + Attention)

## 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## 실행
python seq2seq_share_forecast.py --input "/절대/경로/sampledata.xlsx" --out_dir "./out" --gif

옵션:
--sheet 0              # 엑셀 시트 인덱스
--test_start 2023-01-01
--forecast_end 2026-12-01
--K 6 --H 3            # 시퀀스 길이
--epochs 30            # 조기종료 포함
--gif                  # 브랜드별 Actual vs Pred GIF 생성
```

### 3. Key Parameters
```python
# Time periods
TRAIN_START = pd.Timestamp("2019-01-01")  # Training start
TEST_START  = pd.Timestamp("2023-01-01")  # Test/comparison start
FORECAST_END = pd.Timestamp("2026-12-01") # Forecast end

# Model hyperparameters
K, H = 6, 3  # Encoder length (6 months) / Decoder horizon (3 months)
MAX_LAG = 6  # Maximum lag for Granger causality test
P_THRESH = 0.05  # P-value threshold for significance
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Logit Transformation**: Converts share ratios (0-1) to unbounded space
- **Log Transformation**: Applied to exogenous variables for stability
- **Clipping**: Ensures share values stay within [ε, 1-ε] bounds

### 2. Granger Causality Testing
```python
# Tests if exogenous variables Granger-cause share variables
# Finds optimal lag (1-6) with minimum p-value < 0.05
grangercausalitytests(data, lag, verbose=False)
```

### 3. Forward-Shift Lag Variables
- Creates future-looking lag variables: `X(t+lag) → X_lag(t)`
- Uses ARIMA(1,0,1) to fill the tail gap for the last `lag` periods
- Accounts for forward-looking market influences

### 4. Sequence Generation
```python
# Encoder input: [exog_vars + share_logit] for K timesteps
# Decoder input: [prev_share + prev_exog] for H timesteps  
# Target output: [future_share] for H timesteps
```

### 5. Model Architecture

#### Encoder-Decoder with Attention
```python
# Encoder: LSTM(units) → hidden states
# Decoder: LSTM(units) + initial_state from encoder
# Attention: AdditiveAttention between decoder and encoder states
# Output: TimeDistributed Dense layers
```

#### Dual Normalization Approaches
- **Softmax**: `softmax(logits)` with Categorical Crossentropy loss
- **Sum-Normalization**: `ReLU(logits) / sum(ReLU(logits))` with MSE loss

### 6. Grid Search Optimization
```python
GRID = [
    {"units": 32, "batch_size": 16, "learning_rate": 5e-4},
    {"units": 64, "batch_size": 16, "learning_rate": 5e-4},
    {"units": 32, "batch_size": 32, "learning_rate": 5e-4},
    {"units": 64, "batch_size": 32, "learning_rate": 5e-4},
]
```

### 7. Future Scenario Generation
- Simulates exogenous variables using SARIMAX(1,d,1)(0,1,1,12)
- Applies forward-shift transformations to simulated values
- Performs step-by-step decoder inference for extended forecasting

## 📈 Output Files

### Excel Sheets Generated
1. **`granger_lag`**: Granger causality test results with optimal lags
2. **`forecast_soft`**: Future predictions using Softmax normalization
3. **`forecast_sumnorm`**: Future predictions using Sum-normalization
4. **`compare_2023_plus`**: Comparison of predictions vs actual values (2023+)
5. **`metrics`**: Accuracy metrics (RMSE, MAE, MAPE, sMAPE, MASE) for both models

### Visualization
- **`error_panel.png`**: Monthly average absolute error comparison chart

## 📊 Evaluation Metrics

### Accuracy Measures
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error (using seasonal naive as baseline)

### Share-Specific Metrics
- **JS Divergence**: Jensen-Shannon divergence for probability distribution comparison
- **Brand-wise RMSE**: Individual RMSE for each brand

## 🛠️ Technical Implementation Details

### Memory Management
- Uses `float32` precision throughout to optimize memory usage
- Implements `dtype(object)` removal to prevent mixed-type issues
- Applies efficient numpy array operations

### Robustness Features
- **Exception Handling**: Graceful fallback for ARIMA fitting failures
- **Data Validation**: Automatic detection and correction of data inconsistencies  
- **Progress Tracking**: tqdm progress bars for long-running operations
- **Early Stopping**: Prevents overfitting with validation-based callbacks

### Scalability Considerations
- Configurable sequence lengths (K, H)
- Flexible exogenous variable handling (automatic pass if none available)
- Grid search with early termination for efficiency

## 🔍 Model Selection Logic

### Exogenous Variable Usage Priority
1. **Significant Lag Variables**: If Granger causality tests find significant lags
2. **Original Exogenous Variables**: If no significant lags but exogenous data exists
3. **Share-Only Model**: If no exogenous variables available

### Architecture Comparison
- **Softmax**: Better for cases where shares represent true probabilities
- **Sum-Normalization**: More robust for noisy data with potential negative influences

## ⚠️ Important Notes

### Data Requirements
- Minimum 8 observations for ARIMA tail-filling
- Monthly frequency data expected
- Share values should be in [0, 1] range
- Missing values are forward-filled automatically

### Limitations
- Forward-shift assumption may not hold for all markets
- LSTM performance depends on sufficient training data
- Seasonal patterns require at least 2+ years of historical data

# 참고) Colab 실행 시
# Colab Quickstart (git clone)

이 노트북 가이드는 Google Colab에서 **git clone → 실행 → 결과 확인**까지 한 번에 진행함

## 0) 런타임 준비
- (선택) 런타임 → 하드웨어 가속기: **GPU**
- 새 노트북에서 아래 셀들을 **순서대로** 실행

## 1) 필수 패키지 설치

```python
!pip -q install numpy pandas matplotlib tqdm statsmodels openpyxl pillow tensorflow
import tensorflow as tf, pandas as pd, os, pathlib
print("TensorFlow:", tf.__version__)
```

## 2) Google Drive 마운트 & 데이터 경로 설정

```python
from google.colab import drive
drive.mount('/content/drive')

# 입력 엑셀 경로 (예시)
DATA = "/content/drive/MyDrive/sampledata.xlsx"  # ← 본인 파일 경로로 변경 가능
print("DATA exists? =>", os.path.exists(DATA))
assert os.path.exists(DATA), "❌ 입력 엑셀 경로를 확인하세요."
```

### 데이터 요건
- **날짜**: `date`(YYYY-MM) 또는 `Year`+`Month` 중 하나
- **타깃**: 컬럼명이 `Share`로 끝나는 모든 열 자동 인식 (합 100/1/임의 스케일 자동 정규화)
- **외생변수**: 선택 (없으면 Granger 단계 자동 패스)

## 3) 깃허브에서 스크립트 받기 (git clone)

```python
!rm -rf seq2seq_softmax
!git clone https://github.com/jay-lay-down/seq2seq_softmax.git
%cd seq2seq_softmax
!ls -al
```

## 4) 실행

```python
!python seq2seq_share_forecast.py \
  --input "$DATA" \
  --out_dir "./out" \
  --test_start "2023-01-01" \
  --forecast_end "2026-12-01" \
  --K 6 --H 3 --epochs 20 \
  --gif
```

- `--gif`를 빼면 GIF 미생성 (학습·리포트는 동일)
- 에폭 로그가 **콘솔에 바로 표시**됨 (EarlyStopping 포함)

## 5) 결과 확인

```python
# 산출물 목록
!ls -al out
!ls -al out/figures_dual_bar

# 엑셀 요약 확인
xls_path = "out/seq2seq_dual_bar.xlsx"
assert os.path.exists(xls_path), "❌ out/seq2seq_dual_bar.xlsx 가 생성되지 않았습니다."

xls = pd.ExcelFile(xls_path)
print("sheets:", xls.sheet_names)

# 지표 및 비교표 미리보기
display(pd.read_excel(xls, "metrics").head())
display(pd.read_excel(xls, "compare_2023_plus").head())
```

### 오차 패널 PNG

```python
from IPython.display import Image, display
display(Image(filename="out/figures_dual_bar/error_panel.png"))
```

### 브랜드별 GIF (옵션 `--gif` 사용 시)

```python
!ls -al out/figures_dual_bar | grep '.gif' || echo "No GIF (옵션 --gif를 사용하세요)"
```

## 옵션/파라미터

- `--sheet 0`: 엑셀 시트 인덱스
- `--K 6`, `--H 3`: 인코더/디코더 윈도 길이(월 단위)
- `--test_start 2023-01-01`: 평가 시작 기준월 (학습 윈도 고려해 자동 조정)
- `--forecast_end 2026-12-01`: 예측 종료월
- `--epochs 30`: 최종 학습 epoch (EarlyStopping 포함)
- `--gif`: 실제 vs 예측 GIF 생성

## 산출물 설명

### `out/seq2seq_dual_bar.xlsx`
- `compare_2023_plus`: **2023-01 이후** 실제 vs 예측(softmax / sumnorm)
- `forecast_soft`, `forecast_sumnorm`: 데이터 마지막월 이후 **2026-12**까지 예측
- `metrics`: RMSE/MAE/MAPE/sMAPE/MASE (softmax vs sumnorm)
- `granger_lag`: 유의미한 Granger 결과 또는 `no_exog_or_no_significant_granger`

### `out/figures_dual_bar/`
- `error_panel.png`: 월별 평균 절대오차(브랜드 평균)
- `*.gif`: 브랜드별 실제 vs 예측 타임라인(옵션)

## 트러블슈팅

- **`DATA exists? => False`**: Drive 경로가 틀림. 실제 파일 위치로 수정 필요
- **타깃 없음 오류**: 타깃 열 이름이 반드시 `...Share`로 끝나야 자동 인식됨
- **날짜 인식 실패**: `date`(YYYY-MM) 또는 `Year`+`Month` 중 하나가 필요
- **TF retracing 경고**: 기능상 영향 거의 없음. 무시해도 됨
- **메모리 부족**: `--epochs`를 줄이거나 `--K/--H`를 축소

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Jihee Cho**  
Email: chubbyfinger1010@gmail.com

---

**Last Updated**: September 2025  
**Model Version**: Seq2Seq-Attention-v1.0
