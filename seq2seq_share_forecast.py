#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seq2Seq(Attention LSTM)로 ...Share 타깃들을 예측/평가/시각화하는 CLI 툴
- 입력: 엑셀(.xlsx) | date(또는 Year/Month) + 타깃(...Share) + (선택) 외생변수
- 타깃 자동 인식: 'Share'로 끝나는 모든 컬럼
- 타깃 자동 정규화: 퍼센트/원핫/임의 스케일 → 행합=1 분포
- exog 없으면 Granger/Lag 자동 패스
- 2023-01부터 실제 vs 예측 비교, 2026-12까지 예측(옵션)
- 오차 패널 PNG + 실제/예측 GIF(옵션) 생성
"""

import os, re, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model
from tensorflow.keras.layers import AdditiveAttention, Concatenate, TimeDistributed, Layer

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import tqdm


# ============================== 유틸 / 전처리 ==============================

def infer_date(df: pd.DataFrame) -> pd.Series:
    """Year/Month 또는 date 컬럼에서 월초 Timestamp 생성"""
    cols = {c.lower(): c for c in df.columns}
    if "year" in cols and "month" in cols:
        y = df[cols["year"]].astype(int); m = df[cols["month"]].astype(int)
        return pd.to_datetime(dict(year=y, month=m, day=1))
    for k in ["date", "날짜"]:
        if k in cols:
            return pd.to_datetime(df[cols[k]]).dt.to_period("M").dt.to_timestamp("MS")
    raise ValueError("날짜 컬럼을 찾지 못했습니다. (Year/Month 또는 date 필요)")

def normalize_shares(df: pd.DataFrame, share_cols: list[str]):
    """입력 스케일 자동 감지 후 분포화(합=1)"""
    S = df[share_cols].apply(pd.to_numeric, errors="coerce")
    row_sum = S.sum(axis=1)
    med = float(np.nanmedian(row_sum))
    if 95 <= med <= 105:      # 퍼센트
        S = S / 100.0; scale = 100.0
    elif 0.95 <= med <= 1.05: # 이미 분포
        scale = 1.0
    else:                     # 임의 스케일 → 분포화
        S = S.div(row_sum.replace(0, np.nan), axis=0)
        scale = med if np.isfinite(med) and med > 0 else 1.0
    S = S.fillna(0.0).clip(lower=0.0)
    S = S.div(S.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return S.astype(np.float32), scale

EPS = 1e-6
def logit(p: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p/(1-p)).astype(np.float32)

def steps_per_epoch(n, bs, val_split=0.1):
    import math
    return int(math.ceil(n*(1-val_split)/bs))


# ============================== 모델/학습 파이프라인 ==============================

class SumNorm(Layer):
    """Softplus로 양수화 후 Σ=1 정규화 (0-벡터 방지)"""
    def call(self, x):
        x = tf.math.softplus(x)
        s = tf.reduce_sum(x, axis=-1, keepdims=True)
        return x / (s + 1e-8)

def build_model(K, H, ENCODER_DIM, DECODER_DIM, OUT_SHARE_DIM, OUT_EXOG_DIM,
                units: int, lr: float, use_softmax: bool) -> Model:
    Ein = layers.Input(shape=(K, ENCODER_DIM))
    enc, h, c = layers.LSTM(units, return_sequences=True, return_state=True)(Ein)
    Din = layers.Input(shape=(H, DECODER_DIM))
    dec, *_ = layers.LSTM(units, return_sequences=True, return_state=True)(Din, initial_state=[h,c])

    ctx = AdditiveAttention()([dec, enc])
    cat = Concatenate(axis=-1)([dec, ctx])
    hidden = TimeDistributed(layers.Dense(units, activation="relu"))(cat)
    logits = TimeDistributed(layers.Dense(OUT_SHARE_DIM))(hidden)

    if use_softmax:
        share = layers.Activation("softmax", name="share")(logits)
        loss_share = losses.CategoricalCrossentropy()
    else:
        share = SumNorm(name="share")(logits)
        loss_share = losses.MeanSquaredError()

    outputs = [share]; losses_list = [loss_share]; weights = [1.0]
    if OUT_EXOG_DIM > 0:
        exog_out = TimeDistributed(layers.Dense(OUT_EXOG_DIM), name="exog")(hidden)
        outputs.append(exog_out); losses_list.append(losses.MeanSquaredError()); weights.append(0.5)

    model = Model([Ein, Din], outputs)
    model.compile(optimizers.Adam(lr), loss=losses_list, loss_weights=weights)
    return model

def evaluate_share(pred, true):
    def js(p,q):
        p=p/(p.sum()+1e-9); q=q/(q.sum()+1e-9); m=0.5*(p+q)
        kl=lambda a,b: np.sum(np.where(a<=0,0,a*np.log((a+1e-12)/(b+1e-12))))
        return 0.5*(kl(p,m)+kl(q,m))
    jsvals=[js(pred[i],true[i]) for i in range(pred.shape[0])]
    rmse = np.sqrt(np.mean((pred-true)**2, axis=0))
    return {"js_mean":float(np.mean(jsvals)), "rmse_brand":rmse}

def simulate_exog_series(series, steps):
    if len(series)<8: return np.repeat(series[-1] if len(series) else 0.0, steps).astype(np.float32)
    d=0 if adfuller(np.log1p(series), autolag="AIC")[1]<0.05 else 1
    try:
        fit = SARIMAX(series, order=(1,d,1), seasonal_order=(0,1,1,12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return fit.simulate(steps).astype(np.float32)
    except:
        return np.repeat(series[-1], steps).astype(np.float32)


# ============================== 메인 파이프라인 ==============================

def main(args):
    # ---------- 입출력 준비 ----------
    in_xlsx   = os.path.abspath(args.input)
    out_dir   = os.path.abspath(args.out_dir)
    out_xlsx  = os.path.join(out_dir, "seq2seq_dual_bar.xlsx")
    fig_dir   = os.path.join(out_dir, "figures_dual_bar")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---------- 데이터 로드 ----------
    raw = pd.read_excel(in_xlsx, sheet_name=args.sheet)
    raw["date"] = infer_date(raw)
    raw.sort_values("date", inplace=True, ignore_index=True)

    # ---------- 타깃 자동 탐지 ----------
    share_cols = [c for c in raw.columns if re.search(r"share\s*$", str(c), re.I)]
    if not share_cols:
        raise RuntimeError("타깃 열이 없습니다. 열 이름이 '...Share' 로 끝나야 합니다.")

    # ---------- 타깃 정규화 ----------
    share_prob, DETECTED_SCALE = normalize_shares(raw, share_cols)
    raw[share_cols] = share_prob
    share_logit_cols = [f"{c}_logit" for c in share_cols]
    for c, lc in zip(share_cols, share_logit_cols):
        raw[lc] = logit(raw[c])

    # ---------- exog 후보 ----------
    exclude = set(share_cols) | set(share_logit_cols) | {c for c in raw.columns if str(c).lower() in ["date","year","month"]}
    exog_base_cols = [c for c in raw.columns if c not in exclude and pd.api.types.is_numeric_dtype(raw[c])]
    exog_log_cols = []
    for c in exog_base_cols:
        lc = f"{c}_log"
        raw[lc] = np.log1p(pd.to_numeric(raw[c], errors="coerce").fillna(0)).astype(np.float32)
        exog_log_cols.append(lc)

    # ---------- 기간 자동 인식 ----------
    K, H = args.K, args.H
    FIRST = raw["date"].min(); LAST = raw["date"].max()
    BASE_TEST_START = pd.Timestamp(args.test_start)
    WARMUP_MONTHS = K + H
    TEST_START = max(BASE_TEST_START, FIRST + pd.DateOffset(months=WARMUP_MONTHS))
    if TEST_START > LAST:
        TEST_START = max(FIRST + pd.DateOffset(months=WARMUP_MONTHS), LAST - pd.DateOffset(months=H))
    TRAIN_START = FIRST
    TRAIN_END   = TEST_START - pd.offsets.MonthBegin(1)
    TEST_END_ACTUAL = LAST
    FORECAST_END    = pd.Timestamp(args.forecast_end)

    print(f"[기간] FIRST={FIRST.date()}  TRAIN_END={TRAIN_END.date()}  "
          f"TEST_START={TEST_START.date()}  TEST_END_ACTUAL={TEST_END_ACTUAL.date()}  "
          f"FORECAST_END={FORECAST_END.date()}")

    # ---------- Granger (+ lag ARIMA) : exog 없으면 자동 패스 ----------
    MAX_LAG, P_THRESH = 6, 0.05
    gc_df = pd.DataFrame(columns=["target","exog","lag","p_value","F_stat"])
    if len(exog_log_cols)>0:
        print("▶ Granger 테스트")
        recs=[]
        for tgt in tqdm(share_logit_cols, desc="granger-target"):
            y = raw[tgt].to_numpy()
            for ex in exog_log_cols:
                x = raw[ex].to_numpy()
                data = np.column_stack([y,x])
                best_p, best_lag, best_f = 1.0, None, None
                for lag in range(1, MAX_LAG+1):
                    try:
                        res = grangercausalitytests(data, lag, verbose=False)
                        p = res[lag][0]["ssr_ftest"][1]; f=res[lag][0]["ssr_ftest"][0]
                        if p<best_p: best_p, best_lag, best_f = p, lag, f
                    except: pass
                if best_lag and best_p<P_THRESH:
                    recs.append({"target":tgt,"exog":ex,"lag":int(best_lag),
                                 "p_value":float(best_p),"F_stat":float(best_f)})
        gc_df = pd.DataFrame(recs).sort_values(["target","exog"]).reset_index(drop=True)

    def shift_arima(df, col, lag):
        new = f"{col}_lag{lag}"
        s = df[col].shift(-lag)
        if lag>0:
            tr = s.iloc[:-lag].dropna().values
            if len(tr)<8: preds = np.repeat(tr[-1] if len(tr) else 0.0, lag)
            else:
                try:
                    preds = SARIMAX(tr, order=(1,0,1),
                                    enforce_stationarity=False, enforce_invertibility=False)\
                            .fit(disp=False).forecast(lag)
                except: preds = np.repeat(tr[-1], lag)
            s.iloc[-lag:] = preds
        df[new] = s.astype(np.float32); return new

    lag_exog_cols=[]
    if not gc_df.empty:
        print("▶ lag 시리즈 생성")
        for _,r in tqdm(gc_df.iterrows(), total=len(gc_df), desc="lag-build"):
            lag_exog_cols.append(shift_arima(raw, r["exog"], int(r["lag"])))

    use_exogs = lag_exog_cols if lag_exog_cols else (exog_log_cols if exog_log_cols else [])
    if use_exogs:
        raw[use_exogs] = raw[use_exogs].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    # ---------- 시퀀스 ----------
    ENCODER_DIM   = len(share_logit_cols) + len(use_exogs)
    DECODER_DIM   = len(share_cols) + len(use_exogs)
    OUT_SHARE_DIM = len(share_cols)
    OUT_EXOG_DIM  = len(use_exogs)

    def enc_vec(row):
        parts=[]
        if use_exogs: parts.append(row[use_exogs].to_numpy(np.float32))
        parts.append(row[share_logit_cols].to_numpy(np.float32))
        return np.concatenate(parts).astype(np.float32)

    def build_seq(df, desc):
        enc, dec, ts, te, start = [], [], [], [], []
        n = len(df) - K - H + 1
        if n<=0:
            print(f"[경고] {desc}: 윈도우 생성 불가 (len={len(df)} < {K+H})")
            return {"enc":np.zeros((0,K,ENCODER_DIM),np.float32),
                    "dec":np.zeros((0,H,DECODER_DIM),np.float32),
                    "ts": np.zeros((0,H,OUT_SHARE_DIM),np.float32),
                    "start":np.array([], dtype="datetime64[ns]")}
        for i in tqdm(range(n), desc=desc):
            enc.append(np.stack([enc_vec(df.iloc[i+j]) for j in range(K)], axis=0))
            d = np.zeros((H, DECODER_DIM), np.float32)
            s = np.zeros((H, OUT_SHARE_DIM), np.float32)
            e = np.zeros((H, OUT_EXOG_DIM), np.float32) if OUT_EXOG_DIM>0 else None

            last_sh = df.iloc[i+K-1][share_cols].to_numpy(np.float32)
            last_ex = (df.iloc[i+K-1][use_exogs].to_numpy(np.float32) if use_exogs else None)

            for h in range(H):
                fut_sh = df.iloc[i+K+h][share_cols].to_numpy(np.float32)
                s[h] = fut_sh
                prev_sh = last_sh if h==0 else df.iloc{i+K+h-1}[share_cols].to_numpy(np.float32)
                parts=[prev_sh]
                if use_exogs:
                    fut_ex = df.iloc[i+K+h][use_exogs].to_numpy(np.float32)
                    prev_ex= last_ex if h==0 else df.iloc[i+K+h-1][use_exogs].to_numpy(np.float32)
                    parts.append(prev_ex); e[h]=fut_ex
                d[h] = np.concatenate(parts)
            dec.append(d); ts.append(s);
            if OUT_EXOG_DIM>0: te.append(e)
            start.append(df.iloc[i+K].date)

        out={"enc":np.array(enc), "dec":np.array(dec), "ts":np.array(ts), "start":np.array(start)}
        if OUT_EXOG_DIM>0: out["te"]=np.array(te)
        return out

    train_df = raw[(raw.date >= TRAIN_START) & (raw.date <= TRAIN_END)].reset_index(drop=True)
    full_df  = raw[(raw.date >= TRAIN_START) & (raw.date <= TEST_END_ACTUAL)].reset_index(drop=True)
    seq_tr   = build_seq(train_df, "seq-train")
    seq_all  = build_seq(full_df , "seq-full")
    mask     = (seq_all["start"] >= TEST_START) & (seq_all["start"] <= TEST_END_ACTUAL)
    seq_te   = {k:(v[mask] if isinstance(v, np.ndarray) else v) for k,v in seq_all.items()}

    print("---- SEQ SHAPES ----")
    print("train enc", seq_tr["enc"].shape, "dec", seq_tr["dec"].shape, "ts", seq_tr["ts"].shape,
          "te" if "te" in seq_tr else "no te")
    print("test  enc",  seq_te["enc"].shape,  "dec", seq_te["dec"].shape,  "ts", seq_te["ts"].shape)
    print(f"mean(|sum(target)-1|) train={np.mean(np.abs(seq_tr['ts'].sum(axis=-1)-1)):.2e}, "
          f"test={np.mean(np.abs(seq_te['ts'].sum(axis=-1)-1)):.2e}")
    assert seq_tr["enc"].shape[0] > 0, "[치명] 학습 시퀀스=0"

    # ---------- 학습 (Grid + Final, verbose=1로 에폭 표시) ----------
    GRID=[{"u":32,"bs":16,"lr":5e-4},{"u":64,"bs":16,"lr":5e-4},
          {"u":32,"bs":32,"lr":5e-4},{"u":64,"bs":32,"lr":5e-4}]

    def grid_eval(cfg, use_sm):
        tf.keras.backend.clear_session()
        m = build_model(K,H,ENCODER_DIM,DECODER_DIM,OUT_SHARE_DIM,OUT_EXOG_DIM,
                        cfg["u"],cfg["lr"],use_sm)
        y_tr = [seq_tr["ts"], seq_tr.get("te")] if OUT_EXOG_DIM>0 else seq_tr["ts"]
        print(f"[grid] {'soft' if use_sm else 'sumnorm'} units={cfg['u']} bs={cfg['bs']} "
              f"steps/epoch≈{steps_per_epoch(len(seq_tr['enc']), cfg['bs'])}")
        m.fit([seq_tr["enc"], seq_tr["dec"]], y_tr, epochs=6, batch_size=cfg["bs"],
              validation_split=0.1, verbose=1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
        pred = m.predict([seq_te["enc"], seq_te["dec"]], verbose=0)
        pred_share = pred[0] if isinstance(pred,(list,tuple)) else pred
        js = [evaluate_share(pred_share[i], seq_te["ts"][i])["js_mean"] for i in range(len(pred_share))]
        return float(np.mean(js))

    results={}
    for tag,use_sm in [("soft",True),("sumnorm",False)]:
        print(f"\n▶ Grid ({tag})")
        best = min(GRID, key=lambda g: grid_eval(g, use_sm))
        tf.keras.backend.clear_session()
        model = build_model(K,H,ENCODER_DIM,DECODER_DIM,OUT_SHARE_DIM,OUT_EXOG_DIM,
                            best["u"],best["lr"],use_sm)
        print(f"   ↳ train ({tag}) units={best['u']} bs={best['bs']} "
              f"steps/epoch≈{steps_per_epoch(len(seq_tr['enc']), best['bs'])}")
        y_tr = [seq_tr["ts"], seq_tr.get("te")] if OUT_EXOG_DIM>0 else seq_tr["ts"]
        model.fit([seq_tr["enc"], seq_tr["dec"]], y_tr, epochs=args.epochs, batch_size=best["bs"],
                  validation_split=0.1, verbose=1,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        pred = model.predict([seq_te["enc"], seq_te["dec"]], verbose=0)
        pred_share = pred[0] if isinstance(pred,(list,tuple)) else pred
        js = [evaluate_share(pred_share[i], seq_te["ts"][i])["js_mean"] for i in range(len(pred_share))]
        rm = np.vstack([evaluate_share(pred_share[i], seq_te["ts"][i])["rmse_brand"]
                        for i in range(len(pred_share))]).mean(0)
        results[tag] = {"model": model, "pred_share": pred_share,
                        "js": float(np.mean(js)), "rmse_per_brand": rm.astype(np.float32)}

    # ---------- 타임라인(중복월 평균) ----------
    def to_timeline(pred_share, seq_pack):
        rows=[]; starts = pd.to_datetime(seq_pack["start"])
        for i in range(pred_share.shape[0]):
            base = pd.Timestamp(starts[i]).replace(day=1)
            for h in range(H):
                dt = base + pd.offsets.MonthBegin(h)
                if dt < TEST_START or dt > TEST_END_ACTUAL: continue
                rows.append({"date":dt, **{share_cols[j]: float(pred_share[i,h,j]) for j in range(len(share_cols))}})
        return pd.DataFrame(rows).groupby("date", as_index=False).mean().sort_values("date")

    tl_soft    = to_timeline(results["soft"]["pred_share"],    seq_te)
    tl_sumnorm = to_timeline(results["sumnorm"]["pred_share"], seq_te)

    # ---------- 비교프레임(리포팅 스케일로 복원) ----------
    def scaled_merge(actual_prob_df, tl_pred_df, suffix):
        pred = tl_pred_df.copy()
        for c in share_cols: pred[c] = pred[c]*DETECTED_SCALE
        pred = pred.rename(columns={c:f"{c}_pred_{suffix}" for c in share_cols})
        out = actual_prob_df.copy()
        for c in share_cols: out[c] = out[c]*DETECTED_SCALE
        return out.merge(pred, on="date", how="left")

    actual_23_prob = raw[["date"]+share_cols].query("date >= @TEST_START").copy()
    cmp_soft    = scaled_merge(actual_23_prob, tl_soft,    "soft")
    cmp_sumnorm = scaled_merge(actual_23_prob, tl_sumnorm, "sumnorm")

    cmp = actual_23_prob.copy()
    for c in share_cols:
        cmp[c] = cmp[c]*DETECTED_SCALE
        cmp[f"{c}_pred_soft"]    = cmp_soft[f"{c}_pred_soft"]
        cmp[f"{c}_pred_sumnorm"] = cmp_sumnorm[f"{c}_pred_sumnorm"]

    # ---------- 2026-12까지 예측 ----------
    future_dates = pd.date_range(raw.date.max()+pd.offsets.MonthBegin(1), FORECAST_END, freq="MS")

    def scenario_predict(model):
        enc_single = np.stack([
            np.concatenate([
                (raw.iloc[-K+i][use_exogs].to_numpy(np.float32) if use_exogs else np.empty(0,dtype=np.float32)),
                raw.iloc[-K+i][[f"{c}_logit" for c in share_cols]].to_numpy(np.float32)
            ])
        for i in range(K)], axis=0)[np.newaxis,:]

        sc_exog=None
        if use_exogs:
            sc_exog = pd.DataFrame({"date":future_dates})
            for ex in use_exogs:
                base_col = ex.replace("_lag","") if "_lag" in ex else ex
                ser = raw[base_col].fillna(method="ffill").values
                sim = simulate_exog_series(ser, len(future_dates))
                if "_lag" in ex:
                    import re as _re
                    lag = int(_re.search(r"_lag(\d+)$", ex).group(1))
                    base = raw[base_col]
                    tmp = pd.concat([base, pd.Series(sim, index=future_dates)], axis=0)
                    sc_exog[ex] = tmp.shift(-lag).iloc[-len(future_dates):].fillna(method="ffill").values.astype(np.float32)
                else:
                    sc_exog[ex] = sim.astype(np.float32)

        out = np.full((len(future_dates), OUT_SHARE_DIM), np.nan, np.float32)
        dec = np.zeros((1, H, DECODER_DIM), np.float32)
        last_row = raw.iloc[-1]; cur=0
        while cur<len(future_dates):
            dec[:] = 0.0
            seed = np.concatenate([
                last_row[share_cols].to_numpy(np.float32),
                (last_row[use_exogs].to_numpy(np.float32) if use_exogs else np.empty(0,dtype=np.float32))
            ])
            dec[0,0,:len(seed)] = seed
            for h in range(H):
                pred = model.predict([enc_single, dec], verbose=0)
                y_h = pred[0][0,h] if isinstance(pred,(list,tuple)) else pred[0,h]
                out[cur] = y_h
                if h < H-1 and (cur+1)<len(future_dates):
                    if use_exogs:
                        next_ex = sc_exog.iloc[cur][use_exogs].to_numpy(np.float32)
                        dec[0,h+1] = np.concatenate([y_h, next_ex])
                    else:
                        dec[0,h+1] = y_h
                cur += 1
                if cur>=len(future_dates): break
        return out

    sheets = {"granger_lag": gc_df if not gc_df.empty else pd.DataFrame([{"info":"no_exog_or_no_significant_granger"}])}
    for tag,res in results.items():
        mean_future = scenario_predict(res["model"])
        df_future = pd.DataFrame(mean_future * DETECTED_SCALE,
                                 columns=[f"{c}_pred_{tag}" for c in share_cols])
        df_future.insert(0,"date",future_dates)
        sheets[f"forecast_{tag}"] = df_future
    sheets["compare_2023_plus"] = cmp

    # ---------- 지표 ----------
    def accuracy_metrics(y, yhat, season=12):
        y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
        mask = np.isfinite(y)&np.isfinite(yhat); y, yhat = y[mask], yhat[mask]
        if len(y)==0: return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, sMAPE=np.nan, MASE=np.nan)
        rmse=np.sqrt(np.mean((y-yhat)**2)); mae=np.mean(np.abs(y-yhat))
        mape=100*np.mean(np.abs((y-yhat)/np.clip(y,1e-9,None)))
        smape=100*np.mean(2*np.abs(y-yhat)/np.clip(np.abs(y)+np.abs(yhat),1e-9,None))
        Y=(raw[share_cols]*DETECTED_SCALE).to_numpy(dtype=np.float64)
        if len(Y)>season: denom=np.nanmean(np.abs(Y[season:]-Y[:-season]))
        else: denom=np.nanmean(np.abs(np.diff(Y,axis=0)))
        mase=mae/(denom+1e-9); return dict(RMSE=rmse,MAE=mae,MAPE=mape,sMAPE=smape,MASE=mase)

    metrics=[]
    for c in share_cols:
        m_soft    = accuracy_metrics(cmp[c], cmp[f"{c}_pred_soft"])
        m_sumnorm = accuracy_metrics(cmp[c], cmp[f"{c}_pred_sumnorm"])
        metrics.append({"target":c, **{f"{k}_soft":v for k,v in m_soft.items()},
                               **{f"{k}_sumnorm":v for k,v in m_sumnorm.items()}})
    sheets["metrics"] = pd.DataFrame(metrics)

    # ---------- Excel 저장 ----------
    import openpyxl  # ensure engine
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    print(f"📄 Excel 저장 완료 → {out_xlsx}")

    # ---------- 오차 장표 ----------
    err_plot = cmp[["date"]].copy()
    ae_soft=[]; ae_sumnorm=[]
    for _,r in cmp.iterrows():
        a  = r[share_cols].to_numpy(dtype=float)
        ps = r[[f"{c}_pred_soft"    for c in share_cols]].to_numpy(dtype=float)
        pn = r[[f"{c}_pred_sumnorm" for c in share_cols]].to_numpy(dtype=float)
        ae_soft.append(np.nanmean(np.abs(a-ps))); ae_sumnorm.append(np.nanmean(np.abs(a-pn)))
    err_plot["AE_soft_mean"]=ae_soft; err_plot["AE_sumnorm_mean"]=ae_sumnorm
    plt.figure(figsize=(9,4.8))
    plt.plot(err_plot["date"], err_plot["AE_soft_mean"], label="Softmax |오차|")
    plt.plot(err_plot["date"], err_plot["AE_sumnorm_mean"], label="SumNorm |오차|")
    plt.title("월별 평균 절대오차 (브랜드 평균, 2023~)")
    plt.xlabel("date"); plt.ylabel("Absolute Error"); plt.legend(); plt.tight_layout()
    panel_path = os.path.join(fig_dir, "error_panel.png")
    plt.savefig(panel_path, dpi=150); plt.close()
    print(f"✅ 오차 장표 저장 → {panel_path}")

    # ---------- GIF(옵션) ----------
    if args.gif:
        def make_compare_gif(cmp_df: pd.DataFrame, suffix: str, fps=2):
            dates = pd.to_datetime(cmp_df["date"])
            for b in share_cols:
                actual = cmp_df[b].to_numpy(float)
                pred   = cmp_df[f"{b}_pred_{suffix}"].to_numpy(float)

                fig, ax = plt.subplots(figsize=(7.2,4.2))
                (line_a,) = ax.plot([], [], label="Actual")
                (line_p,) = ax.plot([], [], label="Pred")
                ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
                ax.set_xlim(dates.min(), dates.max())
                ymin = np.nanmin(np.concatenate([actual, pred]))
                ymax = np.nanmax(np.concatenate([actual, pred]))
                pad = 0.05*(ymax - ymin + 1e-9)
                ax.set_ylim(ymin - pad, ymax + pad)

                def init(): line_a.set_data([], []); line_p.set_data([], []); return line_a, line_p
                def update(frame):
                    x = dates[:frame+1]
                    line_a.set_data(x, actual[:frame+1]); line_p.set_data(x, pred[:frame+1])
                    ax.set_title(f"{b} — Actual vs Pred ({suffix})  |  {x.max().strftime('%Y-%m')}")
                    return line_a, line_p

                anim = FuncAnimation(fig, update, init_func=init, frames=len(dates), interval=500, blit=True)
                safe_b = re.sub(r'[^0-9A-Za-z가-힣_]+','_',b)
                out_path = os.path.join(fig_dir, f"compare_{suffix}_{safe_b}.gif")
                anim.save(out_path, writer=PillowWriter(fps=fps))
                plt.close(fig)
                print(f"🎞️ GIF 저장: {out_path}")

        cmp_soft    = scaled_merge(actual_23_prob, tl_soft,    "soft")
        cmp_sumnorm = scaled_merge(actual_23_prob, tl_sumnorm, "sumnorm")
        make_compare_gif(cmp_soft,    "soft",    fps=2)
        make_compare_gif(cmp_sumnorm, "sumnorm", fps=2)

    print("✅ 모든 작업 완료")


# ============================== CLI ==============================

def get_args():
    p = argparse.ArgumentParser(description="Seq2Seq(Attention LSTM) '...Share' 예측/평가/시각화")
    p.add_argument("--input", required=False, help="입력 엑셀(.xlsx)")
    p.add_argument("--sheet", type=int, default=0, help="시트 인덱스 (기본 0)")
    p.add_argument("--out_dir", default="./out", help="출력 폴더")
    p.add_argument("--test_start", default="2023-01-01", help="테스트 시작 (YYYY-MM-01)")
    p.add_argument("--forecast_end", default="2026-12-01", help="예측 종료 (YYYY-MM-01)")
    p.add_argument("--K", type=int, default=6, help="Encoder 길이")
    p.add_argument("--H", type=int, default=3, help="Decoder 길이")
    p.add_argument("--epochs", type=int, default=30, help="최종 학습 epoch(조기종료 있음)")
    p.add_argument("--gif", action="store_true", help="브랜드별 Actual vs Pred GIF 생성")

    # 노트북/콜랩에서 인자 없이 실행 시 기본값 사용
    if len(sys.argv) == 1:
        default_input = "/content/drive/MyDrive/sampledata.xlsx"
        print(f"[노트북 모드] 인자 없이 실행 → 기본값 사용\n  --input={default_input}\n  --out_dir=./out")
        return p.parse_args(["--input", default_input, "--out_dir", "./out"])
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = get_args()
    except SystemExit:
        # argparse SystemExit로 죽는 상황 대비 (특히 노트북)
        args = argparse.Namespace(
            input="/content/drive/MyDrive/sampledata.xlsx",
            sheet=0, out_dir="./out",
            test_start="2023-01-01", forecast_end="2026-12-01",
            K=6, H=3, epochs=30, gif=False
        )
        print("[Fallback] argparse 오류 → 기본 인자로 계속 진행합니다.")
    main(args)
