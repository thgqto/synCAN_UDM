# CANet_train_FINAL_SPACE.py
import zipfile, os, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn, torch.optim as optim

# ------------------------------------------------------------------
# 1. LOAD DATA – SPACE-SEPARATED
# ------------------------------------------------------------------
def load_data_from_zip(zip_files):
    all_rows = []
    header = None
    for i, zip_path in enumerate(zip_files):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if not name.lower().endswith('.csv'):
                    continue
                with z.open(name) as f:
                    lines = [l.decode('utf-8').strip() for l in f.readlines()]
                    lines = [l for l in lines if l]
                    if i == 0 and lines:
                        header = [h.strip() for h in lines[0].split(',') if h.strip()]
                        data_lines = lines[1:]
                    else:
                        data_lines = lines
                    for line in data_lines:
                        values = line.split()  # SPACE-SEPARATED
                        if len(values) > len(header):
                            values = values[:len(header)]
                        elif len(values) < len(header):
                            values += [''] * (len(header) - len(values))
                        all_rows.append(values)
    df = pd.DataFrame(all_rows, columns=header)
    # Keep Label as string
    for col in df.columns:
        if col not in {'ID', 'Label'}:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ------------------------------------------------------------------
# 2. PREPROCESS – FINAL
# ------------------------------------------------------------------
def preprocess(df):
    df['Label'] = df['Label'].astype(str).str.strip()
    df['Label'] = df['Label'].replace({'0.0': '0', '1.0': '1'})

    df_normal = df[df['Label'] == '0'].copy()
    if len(df_normal) == 0:
        # DEBUG: show what labels exist
        print("ERROR: No '0' labels. Found:", sorted(df['Label'].unique()))
        raise ValueError("No normal samples.")

    df_normal = df_normal.drop(columns=['Time'], errors='ignore')
    signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
    for col in signal_cols:
        if col not in df_normal.columns:
            df_normal[col] = 0.0
        df_normal[col] = pd.to_numeric(df_normal[col], errors='coerce').fillna(0.0)

    ids = sorted(df_normal['ID'].astype(str).unique())
    signals_dict = {cid: signal_cols for cid in ids}
    N = 4 * len(ids)

    scalers = {}
    for col in signal_cols:
        valid = df_normal[col].values.reshape(-1, 1)
        scalers[col] = MinMaxScaler().fit(valid if valid.ptp() > 0 else [[0], [1]])
        df_normal[col] = scalers[col].transform(valid)

    freq = df_normal['ID'].value_counts(normalize=True)
    scale_factors = {cid: 1.0 / freq.get(cid, 1e-6) for cid in ids}

    df_normal = df_normal[['ID'] + signal_cols].reset_index(drop=True)
    return df_normal, ids, signals_dict, N, scalers, scale_factors

# ------------------------------------------------------------------
# 3. MODEL
# ------------------------------------------------------------------
class CANet(nn.Module):
    def __init__(self, ids, h_scale=10):
        super().__init__()
        self.ids = ids
        self.h_scale = h_scale
        self.N = 4 * len(ids)
        self.lstms = nn.ModuleDict({cid: nn.LSTM(4, 4 * h_scale, batch_first=True) for cid in ids})
        self.decoder = nn.Sequential(
            nn.Linear(self.N * h_scale, self.N * h_scale // 2), nn.ELU(),
            nn.Linear(self.N * h_scale // 2, self.N - 1), nn.ELU(),
            nn.Linear(self.N - 1, self.N), nn.ELU()
        )
        self.slices = {cid: slice(i*4, (i+1)*4) for i, cid in enumerate(ids)}
    def forward(self, x, cur_id, hidden):
        x = x.unsqueeze(1)
        out, (h,c) = self.lstms[cur_id](x, hidden[cur_id])
        hidden[cur_id] = (h,c)
        lat = torch.cat([hidden[cid][0].squeeze(0) for cid in self.ids], dim=1)
        rec = self.decoder(lat)
        return rec[:, self.slices[cur_id]], hidden

# ------------------------------------------------------------------
# 4. TRAIN
# ------------------------------------------------------------------
def train_canet(df_normal, ids, scale_factors, h_scale=10, epochs=3, batch_size=2, seq_len=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CANet(ids, h_scale).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    n_seq = len(df_normal) - seq_len
    signal_cols = ['Signal1','Signal2','Signal3','Signal4']
    for _ in range(epochs):
        for _ in range(batch_size):
            start = np.random.randint(0, n_seq)
            seq = df_normal.iloc[start:start + seq_len]
            hidden = {cid: (torch.zeros(1,1,4*h_scale, device=device), torch.zeros(1,1,4*h_scale, device=device)) for cid in ids}
            loss = 0.0
            for _, r in seq.iterrows():
                cid = str(r['ID'])
                inp = torch.tensor(r[signal_cols].values.astype(np.float32), device=device).unsqueeze(0)
                rec, hidden = model(inp, cid, hidden)
                loss += nn.functional.mse_loss(rec, inp) * scale_factors[cid]
            loss /= seq_len
            opt.zero_grad()
            loss.backward()
            opt.step()
    # Thresholds
    model.eval()
    errs = {col: [] for col in signal_cols}
    sample = df_normal.sample(min(10000, len(df_normal)))
    hidden = {cid: (torch.zeros(1,1,4*h_scale, device=device), torch.zeros(1,1,4*h_scale, device=device)) for cid in ids}
    with torch.no_grad():
        for _, r in sample.iterrows():
            cid = str(r['ID'])
            inp = torch.tensor(r[signal_cols].values.astype(np.float32), device=device).unsqueeze(0)
            rec, hidden = model(inp, cid, hidden)
            err = (rec - inp) ** 2
            for i, col in enumerate(signal_cols):
                errs[col].append(err[0,i].item())
    thresholds = {col: np.percentile(errs[col], 99.99) for col in errs}
    return model, thresholds

# ------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------
if __name__ == '__main__':
    zip_files = ['train_1.zip', 'train_2.zip', 'train_3.zip', 'train_4.zip']
    df = load_data_from_zip(zip_files)
    df_normal, ids, signals_dict, N, scalers, scale_factors = preprocess(df)
    model, thresholds = train_canet(df_normal, ids, scale_factors)

    torch.save(model.state_dict(), 'canet_model.pt')
    pickle.dump(scalers, open('scalers.pkl', 'wb'))
    pickle.dump(signals_dict, open('signals_dict.pkl', 'wb'))
    pickle.dump(model.slices, open('slices.pkl', 'wb'))
    pickle.dump(thresholds, open('thresholds.pkl', 'wb'))
    pickle.dump(ids, open('ids.pkl', 'wb'))

    print("CANet training completed successfully. Model and artifacts saved.")