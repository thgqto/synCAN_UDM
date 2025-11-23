# evaluate.py
import zipfile, pickle, torch, torch.nn as nn, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def load_test(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith('.csv'))
        lines = z.open(name).read().decode('utf-8').strip().splitlines()
        header = [h.strip() for h in lines[0].split(',') if h.strip()]
        data = []
        for line in lines[1:]:
            if not line.strip(): continue
            values = line.split()
            if len(values) > len(header): values = values[:len(header)]
            if len(values) < len(header): values += [''] * (len(header) - len(values))
            data.append(values)
        df = pd.DataFrame(data, columns=header)
        for c in df.columns:
            if c not in {'ID','Label'}: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

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

# Load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scalers = pickle.load(open('scalers.pkl','rb'))
ids = pickle.load(open('ids.pkl','rb'))
model = CANet(ids)
model.load_state_dict(torch.load('canet_model.pt', map_location=device))
model.slices = pickle.load(open('slices.pkl','rb'))
model.to(device)
thresholds = pickle.load(open('thresholds.pkl','rb'))

# Test
df = load_test('test_flooding.zip')
for c in ['Signal1','Signal2','Signal3','Signal4']:
    df[c] = scalers[c].transform(df[c].values.reshape(-1,1))
df['Label'] = df['Label'].astype(str).str.strip().replace({'0.0': '0', '1.0': '1'})
labels = (df['Label'] == '1').astype(int).values
df = df[['ID'] + ['Signal1','Signal2','Signal3','Signal4']]

model.eval()
hidden = {cid: (torch.zeros(1,1,40, device=device), torch.zeros(1,1,40, device=device)) for cid in ids}
preds = []
with torch.no_grad():
    for _, r in df.iterrows():
        cid = str(r['ID'])
        if cid not in ids: 
            preds.append(0); continue
        inp = torch.tensor(r[['Signal1','Signal2','Signal3','Signal4']].values.astype(np.float32), device=device).unsqueeze(0)
        rec, hidden = model(inp, cid, hidden)
        err = (rec - inp) ** 2
        anomaly = any(err[0,i].item() > thresholds[f'Signal{i+1}'] for i in range(4))
        preds.append(1 if anomaly else 0)

print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
print(f"F1: {f1_score(labels, preds):.4f}")