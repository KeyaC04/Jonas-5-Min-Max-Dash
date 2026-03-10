import os, random, base64, io, re, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback, DiskcacheManager
import plotly.graph_objects as go

# Background callbacks
import diskcache
cache = diskcache.Cache("./.cache_jonas")
background_callback_manager = DiskcacheManager(cache)

# Try PyTorch (graceful fallback if not installed)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ==============================
# DETERMINISM
# ==============================
def set_global_determinism(seed: int = 42):
    import os, random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

# ==============================
# CONFIG
# ==============================
JONAS_RECENT_DAYS = 20   # Jonas export provides CDay/CAmt 1..20
JONAS_FC_WEEKS    = 12   # weekly forecast buckets 1..12

# STRICT NN ONLY: learn Min/Max purely from Jonas Min.Stock/Max.Stock (no formulae / seatbelts)
STRICT_NN_ONLY    = True

# NN hyperparameters
NN_EPOCHS = 40
NN_BATCH  = 128
NN_LR     = 9.01e-3

# ==============================
# HELPERS
# ==============================
def drop_dup_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()

def safe_float(x, default=0.0):
    try:
        v = float(x)
        if not np.isfinite(v): return float(default)
        return v
    except Exception:
        return float(default)

def normalize_material_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(" ", "", regex=False)
    s = s.replace({"nan": "", "None": "", "<NA>": ""})
    return s

def _norm_mat_key(m: str) -> str:
    """Canonical material key: remove all non-alphanumeric, uppercase."""
    m = (m or "")
    return re.sub(r"[^A-Za-z0-9]", "", m).upper()

def decode_upload(contents: str) -> bytes:
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)

# Regexes for Jonas headers (tolerant patterns)
_JONAS_CDAY_RE = re.compile(r"^(?:C\s*Day|CDay)[-_\s]*(\d+)$", re.I)
_JONAS_CAMT_RE = re.compile(r"^(?:C\s*Amt|CAmt)[-_\s]*(\d+)$", re.I)
_JONAS_WK_RE   = re.compile(r"^(?:Fc\s*Week|FcWeek)[-_\s]*(\d+)$", re.I)
_JONAS_REQ_RE  = re.compile(r"^(?:Fc\s*Req|FcReq)[-_\s]*(\d+)$", re.I)

def _normalize_header(h: str) -> str:
    s = str(h).replace("\u00A0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _jonas_find_pairs(df: pd.DataFrame, day_prefix=True):
    """
    Robust finder for (CDay i, CAmt i) or (FcWeek i, FcReq i),
    tolerant to spacing/hyphen/underscore differences.

    Weekly: when multiple 'FcReq i' exist, pick the one positionally closest
    to 'FcWeek i', preferably after it and before the next 'FcWeek'.
    """
    norm_to_orig = {}
    norm_to_pos  = {}
    for idx, c in enumerate(df.columns):
        norm = _normalize_header(c)
        norm_to_orig[norm] = c
        norm_to_pos[norm]  = idx

    if day_prefix:
        key_re = _JONAS_CDAY_RE
        val_re = _JONAS_CAMT_RE
        idx_to_keycol, idx_to_valcol = {}, {}
        for norm in list(norm_to_orig.keys()):
            m = key_re.match(norm)
            if m:
                idx_to_keycol[int(m.group(1))] = norm_to_orig[norm]
            m = val_re.match(norm)
            if m:
                idx_to_valcol[int(m.group(1))] = norm_to_orig[norm]
        idxs = sorted(set(idx_to_keycol).intersection(idx_to_valcol))
        return {i: idx_to_keycol[i] for i in idxs}, {i: idx_to_valcol[i] for i in idxs}

    week_pos, req_pos_list = {}, {}
    for norm in list(norm_to_orig.keys()):
        m = _JONAS_WK_RE.match(norm)
        if m:
            i = int(m.group(1))
            week_pos[i] = (norm, norm_to_pos[norm])
        m = _JONAS_REQ_RE.match(norm)
        if m:
            i = int(m.group(1))
            req_pos_list.setdefault(i, []).append((norm, norm_to_pos[norm]))

    idx_to_keycol, idx_to_valcol = {}, {}
    for i, (wk_norm, wk_p) in week_pos.items():
        candidates = req_pos_list.get(i, [])
        if not candidates: continue
        next_wk_p = week_pos[i + 1][1] if (i + 1) in week_pos else None

        def ok(p):
            after = (p >= wk_p)
            before_next = (True if next_wk_p is None else p < next_wk_p)
            return after and before_next

        preferred = [c for c in candidates if ok(c[1])]
        chosen = min(preferred or candidates, key=lambda x: abs(x[1] - wk_p))
        idx_to_keycol[i] = norm_to_orig[wk_norm]
        idx_to_valcol[i] = norm_to_orig[chosen[0]]
    idxs = sorted(set(idx_to_keycol).intersection(idx_to_valcol))
    return {i: idx_to_keycol[i] for i in idxs}, {i: idx_to_valcol[i] for i in idxs}

def vendor_transit_days(vendor: str) -> int:
    v = (vendor or "").strip().lower()
    if "ampson" in v:       # Ampson Engineering Pvt. Ltd.
        return 1
    if "sri lakshmi" in v:  # Sri Lakshmi Tools
        return 7
    if "aviza" in v:
        return 1
    if "sahil" in v:
        return 1
    if "superpacks" in v:
        return 1
    if "sustainable" in v:
        return 1
    
    return 7                # default for others

def log1p_clip(arr):
    arr = np.asarray(arr, dtype=float)
    return np.log1p(np.clip(arr, 0, None))

def expm1_clip(arr):
    arr = np.asarray(arr, dtype=float)
    return np.clip(np.expm1(arr), 0, None)

# -------- LCM utilities (for your Max rule) --------
def _lcm_pair(a: int, b: int) -> int:
    """LCM for two non-negative ints; lcm(0,x)=x, lcm(0,0)=0."""
    a, b = int(max(0, a)), int(max(0, b))
    if a == 0 or b == 0:
        return max(a, b)
    return abs(a*b) // math.gcd(a, b)

def lcm_array(moq: np.ndarray, minlot: np.ndarray) -> np.ndarray:
    a = np.clip(np.nan_to_num(moq, nan=0.0), 0, None).astype(np.int64)
    b = np.clip(np.nan_to_num(minlot, nan=0.0), 0, None).astype(np.int64)
    out = np.zeros_like(a, dtype=np.int64)
    for i in range(len(a)):
        out[i] = _lcm_pair(a[i], b[i])
    return out.astype(float)

# ==============================
# RULES LOADER (config_rules.xlsx)
# ==============================
def parse_rules_xlsx(rules_bytes: bytes) -> pd.DataFrame:
    """
    Returns canonical material rules with optional columns:
    material_id, rule_leadtime_days (info only), moq, min_lot, trolley_size,
    total_trolleys, kanban_bin_size
    """
    xls = pd.ExcelFile(io.BytesIO(rules_bytes), engine="openpyxl")
    sheet = "material_rules" if "material_rules" in xls.sheet_names else xls.sheet_names[0]
    raw = pd.read_excel(xls, sheet_name=sheet, dtype=str, engine="openpyxl")

    cols_norm = {str(c).strip().lower(): c for c in raw.columns}
    def pick(*keys):
        for k in cols_norm:
            if all(sub in k for sub in keys):
                return cols_norm[k]
        return None

    col_mat   = pick("material") or pick("mat") or pick("material_id")
    col_lt    = pick("transit","day")
    col_moq   = pick("moq")
    col_lot   = pick("min","lot")
    col_tsize = pick("qty","per","trolley") or pick("trolley","size")
    col_troll = pick("trolley","total") or pick("trolley","constraint") or pick("total","trolley")
    col_bin   = pick("kanban","bin") or pick("kanban","size")

    can = pd.DataFrame()
    if col_mat:   can["material_id"]        = raw[col_mat].map(_norm_mat_key)
    if col_lt:    can["rule_leadtime_days"] = pd.to_numeric(raw[col_lt], errors="coerce")
    if col_moq:   can["moq"]                = pd.to_numeric(raw[col_moq], errors="coerce")
    if col_lot:   can["min_lot"]            = pd.to_numeric(raw[col_lot], errors="coerce")
    if col_tsize: can["trolley_size"]       = pd.to_numeric(raw[col_tsize], errors="coerce")
    if col_troll: can["total_trolleys"]     = pd.to_numeric(raw[col_troll], errors="coerce")
    if col_bin:   can["kanban_bin_size"]    = pd.to_numeric(raw[col_bin], errors="coerce")

    if "material_id" in can.columns:
        can = can[can["material_id"].astype(str).str.len() > 0].copy()

    return can.reset_index(drop=True)

# ==============================
# DATASET FROM JONAS EXPORT (STRICT NN labels from Min/Max)
# ==============================
def build_dataset_jonas(jonas_bytes: bytes) -> pd.DataFrame:
    df_raw = pd.read_excel(io.BytesIO(jonas_bytes), engine="openpyxl", dtype=str)
    df_raw = drop_dup_cols(df_raw)

    df_raw["material_id"] = normalize_material_id(df_raw.get("Mat.Nb.", "").fillna(""))
    df_raw["vendor"]      = df_raw.get("Supplier", "UNKNOWN").fillna("UNKNOWN").astype(str).str.strip()

    # Stocks
    cur_stock = pd.to_numeric(df_raw.get("Cur.Stock"), errors="coerce").fillna(0.0).clip(lower=0.0)
    min_stock = pd.to_numeric(df_raw.get("Min.Stock"), errors="coerce").fillna(0.0).clip(lower=0.0)
    max_stock = pd.to_numeric(df_raw.get("Max.Stock"), errors="coerce").fillna(0.0).clip(lower=0.0)

    # Find daily & weekly column pairs
    day_keycols, day_valcols = _jonas_find_pairs(df_raw, day_prefix=True)
    wk_keycols,  wk_valcols  = _jonas_find_pairs(df_raw, day_prefix=False)

    rows = []
    for ridx, r in df_raw.iterrows():
        mat = r["material_id"]
        if not mat:
            continue

        # ---- Daily consumption (latest 20 by date, safest ordering) ----
        daily_pairs = []
        for i in sorted(day_keycols):
            day_str = str(r.get(day_keycols[i], "")).strip()
            qty     = safe_float(r.get(day_valcols[i], 0.0), 0.0)
            if not np.isfinite(qty) or qty < 0: qty = 0.0
            daily_pairs.append((i, day_str, qty))

        def _trydate(s):
            try: return pd.to_datetime(s, errors="coerce")
            except: return pd.NaT

        daily_pairs = sorted(
            daily_pairs,
            key=lambda x: ((_trydate(x[1]) if pd.notna(_trydate(x[1])) else pd.Timestamp(1970,1,1)), x[0])
        )
        daily_qtys = [q for (_, _, q) in daily_pairs][-JONAS_RECENT_DAYS:]
        if len(daily_qtys) < JONAS_RECENT_DAYS:
            daily_qtys = [0.0] * (JONAS_RECENT_DAYS - len(daily_qtys)) + daily_qtys

        # ---- Weekly forecast (keep weekly; NO weekly->daily) ----
        wk_qtys = []
        for i in range(1, JONAS_FC_WEEKS + 1):
            q = safe_float(r.get(wk_valcols.get(i)), 0.0) if i in wk_valcols else 0.0
            if q < 0: q = -q
            wk_qtys.append(q)

        vendor = r["vendor"]
        lt_days  = vendor_transit_days(vendor)
        lt_weeks = lt_days / 7.0

        rows.append({
            "material_id":   mat,
            "vendor":        vendor,
            "safety_stock":  float(min_stock.iloc[ridx]) if ridx < len(min_stock) else 0.0,
            "init_stock":    float(cur_stock.iloc[ridx]) if ridx < len(cur_stock) else 0.0,
            "min_lot":       1,
            "leadtime_days": int(lt_days),
            "leadtime_weeks": lt_weeks,

            # STRICT inputs for NN
            "Cons_Recent": list(daily_qtys),   # daily (len=20)
            "FC_WEEKLY":   list(wk_qtys),      # weekly (len=12)

            # Optional x-axis helpers
            "Recent_T": list(range(len(daily_qtys))),
            "Future_W": list(range(1, len(wk_qtys)+1)),

            # STRICT labels
            "Label_Min": float(min_stock.iloc[ridx]) if ridx < len(min_stock) else 0.0,
            "Label_Max": float(max_stock.iloc[ridx]) if ridx < len(max_stock) else 0.0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No materials parsed from Jonas export.")

    df["logistics_type"] = "STANDARD"
    df["category"]       = "DAILY"
    return df

# ==============================
# NN (features: daily 20 + weekly 12)
# ==============================
class MinMaxDS(Dataset if TORCH_AVAILABLE else object):
    def __init__(self, X_seq, X_num, v_idx, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.v_idx = torch.tensor(v_idx, dtype=torch.long)
        self.y     = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X_seq[i], self.X_num[i], self.v_idx[i], self.y[i]

class Net(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, seq_dim, num_dim, n_vendor):
        super().__init__()
        self.emb_vendor = nn.Embedding(max(1, n_vendor), 16)
        self.seq_enc = nn.Sequential(
            nn.Linear(seq_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + num_dim + 16, 128), nn.ReLU(),
            nn.Identity(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)  # [log1p(Min), log1p(Max)]
        )
    def forward(self, x_seq, x_num, v_idx):
        e = self.emb_vendor(v_idx)
        s = self.seq_enc(x_seq)
        return self.head(torch.cat([s, x_num, e], dim=1))

def train_nn_and_predict(df: pd.DataFrame,
                         epochs=NN_EPOCHS, batch=NN_BATCH, lr=NN_LR,
                         seed=42, device="cpu") -> pd.DataFrame:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Please `pip install torch`")

    # INPUTS: 20 daily + 12 weekly (no weekly->daily conversion)
    X_cons = np.array(df["Cons_Recent"].to_list(), dtype=float)   # (N, 20)
    X_wk   = np.array(df["FC_WEEKLY"].to_list(),   dtype=float)   # (N, 12)
    X_seq  = log1p_clip(np.hstack([X_cons, X_wk]))                # (N, 32)

    # Lightweight numeric features (kept; no formula-based use)
    num_cols = ["leadtime_weeks","safety_stock","init_stock","min_lot","leadtime_days"]
    X_num = pd.to_numeric(df[num_cols].stack(), errors="coerce").unstack().fillna(0.0).to_numpy(dtype=float)
    mu, sd = X_num.mean(axis=0, keepdims=True), X_num.std(axis=0, keepdims=True) + 1e-6
    X_num_scaled = (X_num - mu) / sd

    # Vendor embedding
    def make_vocab(series: pd.Series) -> Dict[str,int]:
        vals = sorted(series.fillna("__UNKNOWN__").astype(str).unique().tolist())
        if "__UNKNOWN__" not in vals: vals.append("__UNKNOWN__")
        return {v: i for i, v in enumerate(vals)}
    def to_index_safe(values: pd.Series, vocab: Dict[str, int], unk_token: str="__UNKNOWN__") -> np.ndarray:
        if unk_token not in vocab:
            vocab[unk_token] = len(vocab)
        mapped = values.fillna(unk_token).astype(str).map(lambda v: vocab.get(v, vocab[unk_token]))
        mapped = pd.to_numeric(mapped, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(vocab[unk_token])
        return mapped.astype("int64").to_numpy()

    vendor_vocab = make_vocab(df["vendor"])
    v_idx = to_index_safe(df["vendor"], vendor_vocab, "__UNKNOWN__")

    # TARGETS: Jonas Min/Max only (STRICT)
    y_min = pd.to_numeric(df["Label_Min"], errors="coerce").fillna(0.0).to_numpy()
    y_max = pd.to_numeric(df["Label_Max"], errors="coerce").fillna(0.0).to_numpy()
    Y = log1p_clip(np.vstack([y_min, y_max]).T)

    # Deterministic split
    order = np.argsort(df["material_id"].astype(str).to_numpy())
    N = len(df); split = max(1, int(0.8 * N))
    tr_idx, va_idx = order[:split], order[split:]

    train_ds = MinMaxDS(X_seq[tr_idx], X_num_scaled[tr_idx], v_idx[tr_idx], Y[tr_idx])
    val_ds   = MinMaxDS(X_seq[va_idx], X_num_scaled[va_idx], v_idx[va_idx], Y[va_idx])

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False, num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0, generator=g)

    torch.manual_seed(seed)

    model = Net(seq_dim=X_seq.shape[1], num_dim=X_num_scaled.shape[1], n_vendor=len(vendor_vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.SmoothL1Loss()

    def eval_loss():
        model.eval(); losses=[]
        with torch.no_grad():
            for xs, xn, v, y in val_loader:
                xs, xn, v, y = xs.to(device), xn.to(device), v.to(device), y.to(device)
                losses.append(loss_fn(model(xs, xn, v), y).item())
        return float(np.mean(losses)) if losses else 0.0

    best_val, best_state = float("inf"), None
    for ep in range(1, int(max(1,epochs))+1):
        model.train()
        for xs, xn, v, y in train_loader:
            xs, xn, v, y = xs.to(device), xn.to(device), v.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(xs, xn, v), y); loss.backward(); opt.step()
        vloss = eval_loss()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: w.detach().cpu().clone() for k, w in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # PREDICT (STRICT): raw NN, no clamps/seatbelts/rounding
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_seq, dtype=torch.float32).to(device),
                      torch.tensor(X_num_scaled, dtype=torch.float32).to(device),
                      torch.tensor(v_idx, dtype=torch.long).to(device)).cpu().numpy()
    raw = expm1_clip(preds)  # non-negative
    min_pred = raw[:,0]
    max_pred = raw[:,1]

    df2 = df.copy()
    df2["Pred_Min_NN"] = min_pred
    df2["Pred_Max_NN"] = max_pred
    # Default final = NN (will be updated post-processing for constrained materials)
    df2["Pred_Min"] = df2["Pred_Min_NN"]
    df2["Pred_Max"] = df2["Pred_Max_NN"]

    df2.attrs["nn_meta"] = {
        "val_loss": float(best_val if np.isfinite(best_val) else 0.0),
        "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
        "strict_nn_only": bool(STRICT_NN_ONLY)
    }
    return df2

# ---------- POST-PROCESSING OVERLAY (apply only to listed materials) ----------
def apply_constraints_post(df_with_nn: pd.DataFrame, rules_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Apply constraints ONLY to materials present in rules_df.
    Others remain exactly at NN outputs.
    """
    df = df_with_nn.copy()

    # Initialize rule columns so UI can display zeros if missing
    for c in ["rule_leadtime_days","moq","min_lot","trolley_size","total_trolleys","kanban_bin_size"]:
        if c not in df.columns:
            df[c] = 0.0

    if rules_df is None or rules_df.empty or ("material_id" not in rules_df.columns):
        return df

    # Map material_id_key -> rule row
    rules_map = {str(r["material_id"]): r for _, r in rules_df.iterrows() if pd.notna(r.get("material_id"))}

    # Prepare arrays
    min_final = df["Pred_Min_NN"].to_numpy(float)
    max_final = df["Pred_Max_NN"].to_numpy(float)

    for idx, row in df.iterrows():
        key = _norm_mat_key(row["material_id"])
        rr = rules_map.get(key)
        if rr is None:
            continue  # leave NN values unchanged

        # Copy rule fields for UI
        for src, dst in [
            ("rule_leadtime_days","rule_leadtime_days"),
            ("moq","moq"),
            ("min_lot","min_lot"),
            ("trolley_size","trolley_size"),
            ("total_trolleys","total_trolleys"),
            ("kanban_bin_size","kanban_bin_size"),
        ]:
            if src in rr.index and pd.notna(rr[src]):
                try:
                    df.at[idx, dst] = float(rr[src])
                except:
                    pass

        # Derived: warehouse trolleys = ceil(total/3)
        tot = df.at[idx, "total_trolleys"]
        df.at[idx, "warehouse_trolleys"] = math.ceil(tot/3.0) if tot and tot > 0 else 0.0

        # NOW: constraint application
        nn_min = float(df.at[idx, "Pred_Min_NN"])
        nn_max = float(df.at[idx, "Pred_Max_NN"])
        kanban = float(df.at[idx, "kanban_bin_size"] or 0.0)

        if kanban > 0:
            # Kanban precedence
            min_final[idx] = min(nn_min, kanban)
            max_final[idx] = min(nn_max, 2.0 * kanban)
        else:
            # No Kanban -> LCM(MOQ, MinLotSize)
            moq  = int(max(0.0, float(df.at[idx, "moq"] or 0.0)))
            mlot = int(max(0.0, float(df.at[idx, "min_lot"] or 0.0)))
            # Compute LCM
            if moq == 0 and mlot == 0:
                # no constraints → leave NN as-is
                pass
            else:
                if moq == 0: lot = mlot
                elif mlot == 0: lot = moq
                else: lot = abs(moq*mlot) // math.gcd(moq, mlot)
                # Max_final = max(Max_NN, Min_NN + 2*lot)
                max_final[idx] = max(nn_max, nn_min + 2.0 * float(lot))

    df["Pred_Min"] = min_final
    df["Pred_Max"] = max_final
    return df

# ==============================
# DASH APP
# ==============================
app = Dash(__name__, background_callback_manager=background_callback_manager)
app.title = "Jonas — Min/Max"
app.server.max_content_length = 128 * 1024 * 1024

def kpi_item(label, value):
    return html.Div(
        style={"border":"1px solid #e5e5e5","borderRadius":"10px","padding":"10px 12px",
               "minWidth":"200px","wordBreak":"break-word","overflowWrap":"anywhere"},
        children=[html.Div(label, style={"fontSize":"12px","color":"#666"}),
                  html.Div(value,  style={"fontSize":"22px","fontWeight":600})]
    )

app.layout = html.Div(
    style={"fontFamily":"Segoe UI, Arial","margin":"16px","maxWidth":"1650px"},
    children=[
        html.H2("Min/Max — Jonas export"),
        html.Br(),
        html.Div(
            style={"display":"flex","gap":"16px","flexWrap":"wrap","alignItems":"flex-start"},
            children=[
                html.Div(
                    style={"minWidth":"520px","maxWidth":"740px","flex":"0 1 740px",
                           "border":"1px solid #ddd","borderRadius":"12px","padding":"14px"},
                    children=[
                        html.H4("1) Upload Jonas export (.xlsx)"),
                        dcc.Upload(
                            id="upload-jonas",
                            children=html.Div(["Drag & drop or ", html.B("select jonasexport.xlsx")]),
                            style={"border":"1px dashed #aaa","borderRadius":"8px","padding":"14px","textAlign":"center"},
                            multiple=False
                        ),
                        html.Br(),

                        html.H4("2) Upload Config Rules (.xlsx)"),
                        dcc.Upload(
                            id="upload-rules",
                            children=html.Div(["Drag & drop or ", html.B("select config_rules.xlsx")]),
                            style={"border":"1px dashed #aaa","borderRadius":"8px","padding":"14px","textAlign":"center"},
                            multiple=False
                        ),
                        html.Div(id="rules-status", style={"color":"#444","marginTop":"8px"}),

                        html.Hr(),
                        html.H4("Status"),
                        dcc.Loading(html.Div(id="status", style={"color":"#444"}), type="dot"),
                        html.Div(id="train-status", style={"marginTop":"4px","color":"#444"}),

                        html.Hr(),
                        html.H4("3) Select vendor"),
                        dcc.Dropdown(id="vendor-select", placeholder="Select a vendor…",
                                     options=[], value=None, style={"width":"100%"}),

                        html.Hr(),
                        html.H4("4) Select material (filtered by vendor)"),
                        dcc.Dropdown(id="material-select", placeholder="Select a material…",
                                     options=[], value=None, style={"width":"100%"}),
                        html.Br(),
                        html.Div(id="material-details",
                                 style={"border":"1px solid #ddd","borderRadius":"12px","padding":"16px"})
                    ]
                ),
                html.Div(style={"flex":"1","minWidth":"720px"},
                    children=[
                        # MATERIAL ONLY
                        dcc.Loading(dcc.Graph(id="material-graph"), type="cube"),
                        html.Div(id="kpis", style={"display":"flex","gap":"12px","flexWrap":"wrap","marginTop":"8px"})
                    ])
            ]
        ),
        dcc.Store(id="store-df"),
        dcc.Store(id="store-df-nn"),
        dcc.Store(id="store-rules"),
        dcc.Store(id="store-df-final"),
    ]
)

def empty_fig(msg):
    fig = go.Figure()
    fig.update_layout(title=msg, xaxis={"visible": False}, yaxis={"visible": False},
                      annotations=[{"text": msg, "xref":"paper","yref":"paper","showarrow": False,"font":{"size":14}}])
    return fig

# ====== RULES UPLOAD ======
@callback(
    Output("store-rules", "data"),
    Output("rules-status", "children"),
    Input("upload-rules", "contents"),
    prevent_initial_call=True
)
def on_upload_rules(rules_contents):
    if not rules_contents:
        return None, "No rules file uploaded yet."
    try:
        header, content_string = rules_contents.split(",", 1)
        rules_bytes = base64.b64decode(content_string)
        rules_df = parse_rules_xlsx(rules_bytes)
        msg = f"Rules loaded ✓  materials with constraints: {len(rules_df):,}"
        return rules_df.to_json(orient="split"), msg
    except Exception as e:
        return None, f"Rules error: {type(e).__name__}: {e}"

# ====== JONAS UPLOAD ======
@callback(
    Output("store-df","data"),
    Output("status","children"),
    inputs=[Input("upload-jonas","contents")],
    progress=[Output("status","children")],
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager
)
def on_upload_jonas(set_progress, jonas_contents):
    if not jonas_contents:
        return None, "Please upload jonasexport.xlsx"
    try:
        set_progress(("Decoding file…",))
        jonas_bytes = decode_upload(jonas_contents)

        set_global_determinism(42)
        set_progress(("Reading & parsing Jonas export (daily + weekly)…",))
        df = build_dataset_jonas(jonas_bytes)

        status = (f"Loaded ✓ Materials: {len(df):,}. "
                  f"Daily={len(df.iloc[0]['Cons_Recent'])} (expected 20); "
                  f"Weekly forecast={len(df.iloc[0]['FC_WEEKLY'])} (expected 12). "
                  f"NN will train automatically.")
        set_progress((status,))
        return df.to_json(orient="split"), status
    except Exception as e:
        msg = f"Error: {type(e).__name__}: {e}"
        set_progress((msg,))
        return None, msg

# ====== TRAIN NN ======
@callback(
    Output("store-df-nn","data"),
    Output("train-status","children"),
    inputs=[Input("store-df","data")],
    progress=[Output("train-status","children")],
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager
)
def on_train(set_progress, data_json):
    if not data_json:
        return None, "Waiting for upload…"
    try:
        set_progress(("Preparing training data…",))
        df = pd.read_json(io.StringIO(data_json), orient="split")
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        set_progress((f"Training STRICT NN (device={device}) [epochs={NN_EPOCHS}, batch={NN_BATCH}, lr={NN_LR}] …",))
        df_nn = train_nn_and_predict(df, epochs=NN_EPOCHS, batch=NN_BATCH, lr=NN_LR,
                                     seed=42, device=device)
        meta = df_nn.attrs.get("nn_meta", {})
        msg = (f"STRICT NN trained ✓  val_loss={meta.get('val_loss', 0):.4f}  "
               f"(epochs={meta.get('epochs')}, batch={meta.get('batch')}, lr={meta.get('lr')})")
        set_progress((msg,))
        return df_nn.to_json(orient="split"), msg
    except Exception as e:
        msg = f"Training error: {type(e).__name__}: {e}"
        set_progress((msg,))
        return None, msg

# ====== APPLY CONSTRAINTS OVERLAY ======
@callback(
    Output("store-df-final","data"),
    Input("store-df-nn","data"),
    State("store-rules","data"),
    prevent_initial_call=True
)
def overlay_constraints(df_nn_json, rules_json):
    if not df_nn_json:
        return None
    df_nn = pd.read_json(io.StringIO(df_nn_json), orient="split")
    rules_df = pd.read_json(io.StringIO(rules_json), orient="split") if rules_json else None
    df_final = apply_constraints_post(df_nn, rules_df)
    return df_final.to_json(orient="split")

# ====== VENDOR & MATERIAL DROPDOWNS ======
@app.callback(
    Output("vendor-select","options"),
    Output("vendor-select","value"),
    Input("store-df-final","data"),
    State("vendor-select","value"),
)
def populate_vendors(data_final_json, current_vendor):
    if not data_final_json: return [], None
    try:
        df = pd.read_json(io.StringIO(data_final_json), orient="split")
    except Exception:
        return [], None
    vendors = sorted(df["vendor"].astype(str).unique().tolist())
    options = [{"label": v, "value": v} for v in vendors]
    value = current_vendor if current_vendor in vendors else (vendors[0] if vendors else None)
    return options, value

@app.callback(
    Output("material-select","options"),
    Output("material-select","value"),
    Input("store-df-final","data"),
    Input("vendor-select","value"),
    State("material-select","value"),
)
def populate_materials_by_vendor(data_final_json, vendor_value, current_material):
    if not data_final_json or not vendor_value: return [], None
    try:
        df = pd.read_json(io.StringIO(data_final_json), orient="split")
    except Exception:
        return [], None
    mats = df.loc[df["vendor"].astype(str) == str(vendor_value), "material_id"].astype(str).tolist()
    options = [{"label": m, "value": m} for m in mats]
    value = current_material if current_material in mats else (mats[0] if mats else None)
    return options, value

# ====== MATERIAL DRILLDOWN (material-only chart; weekly forecast shown) ======
@app.callback(
    Output("material-details","children"),
    Output("material-graph","figure"),
    Output("kpis","children"),
    Input("store-df-final","data"),
    Input("material-select","value"),
)
def render_material_drilldown(data_final_json, material_id):
    if not data_final_json:
        return "Upload the Jonas export to begin.", empty_fig("Waiting for NN training…"), []
    try:
        df = pd.read_json(io.StringIO(data_final_json), orient="split")
    except Exception as e:
        return f"Failed to load results: {type(e).__name__}: {e}", empty_fig("Error"), []

    if not material_id:
        return "Select a material.", empty_fig("Select a material"), []

    row = df[df["material_id"].astype(str) == str(material_id)]
    if row.empty:
        return "Selected material not found.", empty_fig("Not found"), []
    r = row.iloc[0]

    def _sf(x, d=0.0):
        try:
            v = float(x)
            if not np.isfinite(v): return float(d)
            return v
        except:
            return float(d)

    def _tolist(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [float(v) if pd.notna(v) else 0.0 for v in x]
        try:
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                return [float(arr)]
            return [float(v) if np.isfinite(v) else 0.0 for v in arr.tolist()]
        except Exception:
            return []

    cons = _tolist(r.get("Cons_Recent"))
    wk   = _tolist(r.get("FC_WEEKLY"))

    # Use final values (which equal NN values for unconstrained materials)
    minlvl  = _sf(r.get("Pred_Min", r.get("Pred_Min_NN", 0.0)), 0.0)
    maxlvl  = _sf(r.get("Pred_Max", r.get("Pred_Max_NN", minlvl)), minlvl)
    initlvl = _sf(r.get("init_stock", 0.0), 0.0)

    details = html.Div(
        style={"border":"1px solid #ddd","borderRadius":"12px","padding":"16px","width":"100%","boxSizing":"border-box","overflow":"hidden"},
        children=[
            html.H5(f"Material: {r['material_id']}", style={"marginTop":"0","marginBottom":"12px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit, minmax(240px, 1fr))",
                            "columnGap":"16px","rowGap":"10px"},
                     children=[
                         html.Div([html.Small("Vendor", style={"color":"#666"}), html.Div(str(r.get("vendor","UNKNOWN")))]),
                         html.Div([html.Small("Transit Time (days)", style={"color":"#666"}), html.Div(f"{int(r.get('leadtime_days',7))}")]),
                         html.Div([html.Small("Safety Stock (Jonas Minimum Stock)", style={"color":"#666"}), html.Div(f"{_sf(r.get('safety_stock',0.0),0.0):,.0f}")]),
                         html.Div([html.Small("Initial Stock", style={"color":"#666"}), html.Div(f"{initlvl:,.0f}")]),
                         html.Div([html.Small("Predicted Min. Stock", style={"color":"#666"}), html.Div(f"{minlvl:,.0f}")]),
                         html.Div([html.Small("Predicted Max. Stock", style={"color":"#666"}), html.Div(f"{maxlvl:,.0f}")]),
                         html.Div([html.Small("MOQ", style={"color":"#666"}), html.Div(f"{int(_sf(r.get('moq',0),0))}")]),
                         html.Div([html.Small("MinLotSize", style={"color":"#666"}), html.Div(f"{int(_sf(r.get('min_lot',0),0))}")]),
                         html.Div([html.Small("Kanban Bin Size", style={"color":"#666"}), html.Div(f"{int(_sf(r.get('kanban_bin_size',0),0))}")]),
                         html.Div([html.Small("Qty. per Trolley", style={"color":"#666"}), html.Div(f"{int(_sf(r.get('trolley_size',0),0))}")]),
                         html.Div([html.Small("Total Trolleys", style={"color":"#666"}), html.Div(f"{int(_sf(r.get('total_trolleys',0),0))}")]),
                         html.Div([html.Small("Warehouse Trolleys", style={"color":"#666"}), html.Div(f"{int(_sf(r.get('warehouse_trolleys',0),0))}")]),
                     ]),
            html.Div(style={"marginTop":"6px","color":"#666","fontSize":"12px"},
                     children="Details used are as provided by the user in the Excel files.")
        ]
    )

    # Build axis labels
    x_daily  = [f"D{i+1}" for i in range(len(cons))]
    x_weekly = [f"W{i+1}" for i in range(len(wk))]
    if not x_weekly:
        fw = r.get("Future_W") or []
        x_weekly = [f"W{int(i)}" for i in fw] if fw else []
    x_all = x_daily + x_weekly

    fig = go.Figure()

    # Daily line
    if cons and x_daily and (len(cons) == len(x_daily)):
        fig.add_trace(
            go.Scatter(
                x=x_daily, y=cons, mode="lines+markers",
                name="Consumption (daily)", line=dict(color="#2E86AB", width=2), marker=dict(size=6)
            )
        )

    # Weekly forecast as LINE (not bars)
    if wk and x_weekly and (len(wk) == len(x_weekly)):
        fig.add_trace(
            go.Scatter(
                x=x_weekly, y=wk, mode="lines+markers",
                name="Forecast (weekly)", line=dict(color="#F39C12", dash="dash", width=3), marker=dict(size=6)
            )
        )

    # Min/Max & Initial as flat lines
    if x_all:
        fig.add_trace(go.Scatter(x=x_all, y=[minlvl]*len(x_all), mode="lines",
                                 name="Min (final)", line=dict(color="#27AE60", dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=x_all, y=[maxlvl]*len(x_all), mode="lines",
                                 name="Max (final)", line=dict(color="#C0392B", dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=x_all, y=[initlvl]*len(x_all), mode="lines",
                                 name="Initial Stock", line=dict(color="#7D3C98", dash="longdash", width=2), opacity=0.9))

    fig.update_layout(
        title=f"STRICT NN Min/Max — {r['material_id']}",
        xaxis_title="Timeline (D1..D20 = recent days, W1..W12 = next weeks)",
        yaxis_title="Quantity",
        legend_title="Series",
        hovermode="x unified",
        xaxis=dict(type="category", categoryorder="array", categoryarray=x_all),
        margin=dict(l=10, r=10, t=50, b=10),
        height=560)

    kpis = [
        kpi_item("Predicted Min. Stock", f"{minlvl:,.0f}"),
        kpi_item("Predicted Max. Stock", f"{maxlvl:,.0f}"),
        kpi_item("Transit Time (days)", f"{int(r.get('leadtime_days',7))}"),
        kpi_item("Safety Stock (Jonas Minimum Stock)", f"{_sf(r.get('safety_stock',0.0), 0.0):,.0f}"),
        kpi_item("Initial Stock", f"{initlvl:,.0f}"),
        kpi_item("MOQ", f"{int(_sf(r.get('moq',0),0))}"),
        kpi_item("MinLotSize", f"{int(_sf(r.get('min_lot',0),0))}"),
        kpi_item("Warehouse Trolleys", f"{int(_sf(r.get('warehouse_trolleys',0),0))}"),
    ]
    return details, fig, kpis

if __name__ == "__main__":
    app.run(debug=True, port=8055)
