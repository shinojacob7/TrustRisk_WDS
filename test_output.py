import pandas as pd

df = pd.read_csv('experiments/final/trust_pipeline.csv')
normal_frame = df[df['true_label'] == 0].iloc[100]
attack_frames = df[df['true_label'] == 1]

if len(attack_frames) > 0:
    # Pick a frame where an attack is actually happening and detected
    # We sort by p_attack to find a clear detection
    detected_attack = attack_frames.sort_values(by='p_attack', ascending=False).iloc[0]
else:
    print('No attack frames found.')
    exit()

baseline_risk = 0.85 # Assume a high-criticality asset (e.g. Pump)

def print_stats(frame, name):
    print(f'--- {name} STATE (Frame {int(frame["frame_idx"])}) ---')
    print(f'True Label: {frame["true_label"]}')
    print(f'Raw Attack Prob: {frame["p_attack"]:.4f}')
    print(f'Confidence (C): {frame["C"]:.4f}')
    print(f'Uncertainty (U_norm): {frame["U_norm"]:.4f}')
    print(f'Explanation (ECP): {frame["ECP"]:.4f}')
    print(f'Trust Score (TS): {frame["Trust_Score"]:.4f}')
    tdcri = (frame["p_attack"] * frame["Trust_Score"]) * baseline_risk
    print(f'-> Final TDCRI: {tdcri:.4f}\n')

print_stats(normal_frame, "NORMAL")
print_stats(detected_attack, "ATTACK")
