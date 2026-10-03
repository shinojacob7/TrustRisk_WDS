import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_dir = os.path.join(base_dir, 'data', 'raw')
v2_dir = os.path.join(base_dir, 'experiments', 'v2')
res_dir = os.path.join(base_dir, 'results', 'v2')

sensors = [
    'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
    'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3',
    'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6',
    'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9',
    'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
    'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422'
]

def build_features(X_raw, lags=1):
    T = len(X_raw)
    Y = X_raw[lags:]
    lag_feats = [X_raw[lags-l : T-l] for l in range(1, lags + 1)]
    lag_matrix = np.hstack(lag_feats) if lags > 0 else np.empty((T, 0))
    return Y, lag_matrix, X_raw[lags:]

def run():
    df4 = pd.read_csv(os.path.join(raw_dir, 'BATADAL_dataset04.csv'), skipinitialspace=True)
    attack_idx = df4[df4['ATT_FLAG'] == 1].index.values
    events = np.split(attack_idx, np.where(np.diff(attack_idx) != 1)[0] + 1)
    
    margin = 48
    all_margins = set()
    for ev in events:
        for idx in range(max(0, ev[0] - margin), min(len(df4), ev[-1] + margin + 1)):
            all_margins.add(idx)
            
    normal_idx_ds4 = np.array(sorted(set(range(len(df4))) - all_margins))
    ds4_blocks = np.array_split(normal_idx_ds4, 5)
    
    # 3. LEAKAGE AUDIT (Reference check)
    print("=== TASK 3: LEAKAGE AUDIT & REFERENCE REPRODUCTION ===")
    ref_alarm_hours = []
    
    for fold in range(5):
        val_norm = ds4_blocks[fold]
        train_norm = np.setdiff1d(normal_idx_ds4, val_norm); train_norm = train_norm[np.abs(train_norm[:, None] - val_norm[None, :]).min(axis=1) >= 24]
        
        gap = np.min(np.abs(val_norm[:, None] - train_norm[None, :]))
        assert gap >= 24, f"Gap is {gap}, expected >= 24"
        assert len(np.intersect1d(train_norm, val_norm)) == 0
        
        print(f"Fold {fold}: Train Range={train_norm[0]}-{train_norm[-1]} ({len(train_norm)}), Val Range={val_norm[0]}-{val_norm[-1]} ({len(val_norm)}), Gap={gap}")
        
        X_tr = df4.loc[train_norm, sensors].values
        Y_tr, lag_tr, X_curr_tr = build_features(X_tr, 1)
        
        models, scalers = [], []
        train_res = np.zeros_like(Y_tr)
        for i in range(len(sensors)):
            X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr_feat)
            mod = Ridge(alpha=1.0)
            mod.fit(X_tr_sc, Y_tr[:, i])
            models.append(mod)
            scalers.append(sc)
            train_res[:, i] = Y_tr[:, i] - mod.predict(X_tr_sc)
            
        res_mu, res_std = np.mean(train_res, axis=0), np.std(train_res, axis=0) + 1e-8
        train_top3 = np.mean(np.sort(np.abs(train_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
        thr = np.percentile(train_top3, 99.9)
        
        eval_idxs = np.concatenate([[max(0, val_norm[0]-1)], val_norm])
        X_val = df4.loc[eval_idxs, sensors].values
        Y_val, lag_val, X_curr_val = build_features(X_val, 1)
        val_res = np.zeros_like(Y_val)
        for i in range(len(sensors)):
            X_val_sc = scalers[i].transform(np.hstack([lag_val, np.delete(X_curr_val, i, axis=1)]))
            val_res[:, i] = Y_val[:, i] - models[i].predict(X_val_sc)
            
        val_top3 = np.mean(np.sort(np.abs(val_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
        ref_alarm_hours.append(int(np.sum(val_top3 > thr)))
        
    print(f"Reference Alarm Hours per fold (In-sample thr, Top-3, q99.9): {ref_alarm_hours}\n")
    
    # 2. INNER CV FOR CUSUM & EVALUATION
    results = []
    
    grid_k = [3, 4, 5, 6]
    grid_h = [15, 20, 25, 30]
    
    for fold in range(5):
        val_norm = ds4_blocks[fold]
        train_norm = np.setdiff1d(normal_idx_ds4, val_norm); train_norm = train_norm[np.abs(train_norm[:, None] - val_norm[None, :]).min(axis=1) >= 24]
        val_event = events[fold]
        train_events = [ev for i, ev in enumerate(events) if i != fold]
        
        # Outer Model
        X_tr = df4.loc[train_norm, sensors].values
        Y_tr, lag_tr, X_curr_tr = build_features(X_tr, 1)
        models, scalers = [], []
        train_res = np.zeros_like(Y_tr)
        for i in range(len(sensors)):
            X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr_feat)
            mod = Ridge(alpha=1.0)
            mod.fit(X_tr_sc, Y_tr[:, i])
            models.append(mod)
            scalers.append(sc)
            train_res[:, i] = Y_tr[:, i] - mod.predict(X_tr_sc)
        res_mu, res_std = np.mean(train_res, axis=0), np.std(train_res, axis=0) + 1e-8
        
        # Inner CV for Threshold
        # To strictly use OOF scores, we use 3-fold inner on train_norm
        kf = KFold(n_splits=3, shuffle=False)
        inner_oof_res = np.zeros_like(Y_tr)
        for tr_idx, va_idx in kf.split(Y_tr):
            # Inner fit
            for i in range(len(sensors)):
                X_tr_feat = np.hstack([lag_tr, np.delete(X_curr_tr, i, axis=1)])
                X_in_tr, X_in_va = X_tr_feat[tr_idx], X_tr_feat[va_idx]
                sc_inner = StandardScaler()
                X_in_tr_sc = sc_inner.fit_transform(X_in_tr)
                mod_inner = Ridge(alpha=1.0)
                mod_inner.fit(X_in_tr_sc, Y_tr[tr_idx, i])
                inner_oof_res[va_idx, i] = Y_tr[va_idx, i] - mod_inner.predict(sc_inner.transform(X_in_va))
                
        inner_res_mu, inner_res_std = np.mean(inner_oof_res, axis=0), np.std(inner_oof_res, axis=0) + 1e-8
        inner_top3 = np.mean(np.sort(np.abs(inner_oof_res - inner_res_mu)/inner_res_std, axis=1)[:, -3:], axis=1)
        
        q999 = np.percentile(inner_top3, 99.9)
        
        # CUSUM Inner CV Tuning
        best_k, best_h, best_rec = None, None, -1
        # Quick OOF for the 4 inner events
        inner_ev_scores = []
        for ev in train_events:
            ev_idx = np.concatenate([[ev[0]-1], ev])
            X_ev = df4.loc[ev_idx, sensors].values
            Y_ev, lag_ev, X_curr_ev = build_features(X_ev, 1)
            ev_res = np.zeros_like(Y_ev)
            for i in range(len(sensors)):
                ev_res[:, i] = Y_ev[:, i] - models[i].predict(scalers[i].transform(np.hstack([lag_ev, np.delete(X_curr_ev, i, axis=1)])))
            ev_top3 = np.mean(np.sort(np.abs(ev_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
            inner_ev_scores.append(ev_top3)
            
        train_weeks = len(inner_top3) / 168.0
        
        for k in grid_k:
            for h in grid_h:
                def cusum(s):
                    c = np.zeros_like(s)
                    for idx in range(1, len(s)): c[idx] = max(0, c[idx-1] + s[idx] - k)
                    return c
                c_alarms = (cusum(inner_top3) > h).astype(int)
                clusters = np.split(np.where(c_alarms == 1)[0], np.where(np.diff(np.where(c_alarms == 1)[0]) != 1)[0] + 1)
                fa_events = len([c for c in clusters if len(c) > 0])
                if fa_events / train_weeks <= 1.0:
                    recs = [np.sum((cusum(s) > h)) for s in inner_ev_scores]
                    tot_rec = np.sum(recs)
                    if tot_rec > best_rec:
                        best_rec = tot_rec
                        best_k, best_h = k, h
                        
        if best_k is None: best_k, best_h = grid_k[-1], grid_h[-1]
        
        # Outer Eval
        val_eval_idx = np.concatenate([[max(0, val_norm[0]-1)], val_norm])
        X_val = df4.loc[val_eval_idx, sensors].values
        Y_val, lag_val, X_curr_val = build_features(X_val, 1)
        val_res = np.zeros_like(Y_val)
        for i in range(len(sensors)):
            val_res[:, i] = Y_val[:, i] - models[i].predict(scalers[i].transform(np.hstack([lag_val, np.delete(X_curr_val, i, axis=1)])))
        val_top3 = np.mean(np.sort(np.abs(val_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
        
        ev_eval_idx = np.concatenate([[val_event[0]-1], val_event])
        X_ev = df4.loc[ev_eval_idx, sensors].values
        Y_ev, lag_ev, X_curr_ev = build_features(X_ev, 1)
        ev_res = np.zeros_like(Y_ev)
        for i in range(len(sensors)):
            ev_res[:, i] = Y_ev[:, i] - models[i].predict(scalers[i].transform(np.hstack([lag_ev, np.delete(X_curr_ev, i, axis=1)])))
        ev_top3 = np.mean(np.sort(np.abs(ev_res - res_mu)/res_std, axis=1)[:, -3:], axis=1)
        
        # 1. Base (q99.9)
        val_base = (val_top3 > q999).astype(int)
        ev_base = (ev_top3 > q999).astype(int)
        
        # 2. Pers (3/6)
        val_pers = (pd.Series(val_base).rolling(6, min_periods=1).sum() >= 3).astype(int).values
        ev_pers = (pd.Series(ev_base).rolling(6, min_periods=1).sum() >= 3).astype(int).values
        
        # 3. CUSUM
        def run_c(s):
            c = np.zeros_like(s)
            for idx in range(1, len(s)): c[idx] = max(0, c[idx-1] + s[idx] - best_k)
            return (c > best_h).astype(int)
        val_c = run_c(val_top3)
        ev_c = run_c(ev_top3)
        
        val_weeks = len(val_top3) / 168.0
        
        def process_rule(name, v_al, e_al, k=None, h=None):
            clusters = np.split(np.where(v_al == 1)[0], np.where(np.diff(np.where(v_al == 1)[0]) != 1)[0] + 1)
            fa_evts = len([c for c in clusters if len(c) > 0])
            
            # FPR(b) Optimistic Masking
            v_al_opt = v_al.copy()
            for c in clusters:
                if len(c) >= 4:
                    v_al_opt[c] = 0
            
            fp = np.sum(v_al)
            tn = len(v_al) - fp
            tp = np.sum(e_al)
            fn = len(e_al) - tp
            
            tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
            
            proj_tp = tpr * 0.195
            proj_fp = fpr * 0.805
            proj_p = proj_tp / (proj_tp + proj_fp) if (proj_tp + proj_fp) > 0 else 0
            proj_f1 = 2 * proj_p * tpr / (proj_p + tpr) if (proj_p + tpr) > 0 else 0
            
            return {
                "fold": fold, "rule": name, "k": k, "h": h,
                "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
                "alarm_hours": int(fp), "FA_events": fa_evts, "weeks": val_weeks,
                "TPR": tpr, "FPR": fpr, "Precision": proj_p, "Projected_F1": proj_f1
            }
            
        results.append(process_rule("Pers(3/6)", val_pers, ev_pers))
        results.append(process_rule("CUSUM", val_c, ev_c, best_k, best_h))
        
    df_res = pd.DataFrame(results)
    
    # Pool
    pooled = []
    for r in ["Pers(3/6)", "CUSUM"]:
        sub = df_res[df_res['rule'] == r]
        tp, fn, fp, tn = sub['TP'].sum(), sub['FN'].sum(), sub['FP'].sum(), sub['TN'].sum()
        weeks = sub['weeks'].sum()
        fa_evts = sub['FA_events'].sum()
        
        tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        
        proj_tp = tpr * 0.195
        proj_fp = fpr * 0.805
        proj_p = proj_tp / (proj_tp + proj_fp) if (proj_tp + proj_fp) > 0 else 0
        proj_f1 = 2 * proj_p * tpr / (proj_p + tpr) if (proj_p + tpr) > 0 else 0
        
        pooled.append({
            "fold": "pooled", "rule": r, "k": None, "h": None,
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "alarm_hours": int(fp), "FA_events": int(fa_evts), "weeks": weeks,
            "TPR": tpr, "FPR": fpr, "Precision": proj_p, "Projected_F1": proj_f1
        })
        
    df_res = pd.concat([df_res, pd.DataFrame(pooled)])
    df_res.to_csv(os.path.join(res_dir, 'cv_report.csv'), index=False)
    df_res.to_json(os.path.join(res_dir, 'cv_report.json'), orient='records', indent=4)
    
    with open(os.path.join(res_dir, 'cv_report.json'), 'r') as f:
        print("\n=== cv_report.json VERBATIM ===")
        print(f.read())
        
    # Freeze
    config = {
        "regime": "DS04-999-only",
        "score": "top3_mean_z",
        "lags": 1,
        "quantile": 99.9,
        "rule": "CUSUM",
        "k": "inner_cv",
        "h": "inner_cv",
        "span": "N/A",
        "weights": "none",
        "threshold_procedure": "Inner OOF scores of training normals",
        "final_training_set": "All DS04 -999 rows"
    }
    with open(os.path.join(v2_dir, 'frozen_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
run()
