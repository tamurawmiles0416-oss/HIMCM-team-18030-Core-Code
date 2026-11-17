import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
 
CITY = pd.read_csv("data44/cities.csv"); SPEC = pd.read_csv("data44/indicator_spec.csv")
INDS = list(SPEC.indicator)
DIM = dict(zip(SPEC.indicator, SPEC.dimension)); DIR = dict(zip(SPEC.indicator, SPEC.direction))
DIMS = ["Academics","Safety","Wellbeing","Built environment","Mobility","Economy",
        "Cultural","Peer density","Equity","Future"]
NC = len(CITY)
 
def adaptive_bounds(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    lo, hi = min(lo, x.min()), max(hi, x.max()); span = hi - lo
    return min(x.min() - .10*span, lo - .10*span), max(x.max() + .10*span, hi + .10*span)
 
def normalise(df):
    Z = pd.DataFrame(index=df.index)
    for k in INDS:
        x = df[k].values.astype(float); lo, hi = adaptive_bounds(x)
        z = (x - lo)/(hi - lo) if hi > lo else np.zeros_like(x)
        Z[k] = np.clip(z if DIR[k] > 0 else 1 - z, 0, 1)
    return Z
 
def entropy_weights(Z):
    P = (Z + 1e-9)/(Z + 1e-9).sum(axis=0)
    e = -(P*np.log(P)).sum(axis=0)/np.log(len(Z))
    return pd.Series((1 - e)/(1 - e).sum(), index=Z.columns)
 
def critic_weights(Z):
    c = Z.std(ddof=1).values * (1 - Z.corr().values).sum(axis=0)
    return pd.Series(c/c.sum(), index=Z.columns)
 
PAIRS = {
 ("Academics","Safety"):2.0000, ("Academics","Wellbeing"):2.2134, ("Academics","Built environment"):3.0000,
 ("Academics","Mobility"):3.6169, ("Academics","Economy"):2.6031, ("Academics","Cultural"):2.5491,
 ("Academics","Peer density"):3.3178, ("Academics","Equity"):2.7108, ("Academics","Future"):2.3522,
 ("Safety","Wellbeing"):1.5157, ("Safety","Built environment"):2.5508, ("Safety","Mobility"):2.5508,
 ("Safety","Economy"):2.0000, ("Safety","Cultural"):2.8211, ("Safety","Peer density"):2.8211,
 ("Safety","Equity"):2.3050, ("Safety","Future"):2.6031, ("Wellbeing","Built environment"):2.0000,
 ("Wellbeing","Mobility"):2.2134, ("Wellbeing","Economy"):1.5692, ("Wellbeing","Cultural"):1.7468,
 ("Wellbeing","Peer density"):2.3050, ("Wellbeing","Equity"):2.0000, ("Wellbeing","Future"):2.2134,
 ("Built environment","Mobility"):1.0000, ("Built environment","Economy"):0.7846, ("Built environment","Cultural"):0.9659,
 ("Built environment","Peer density"):2.0000, ("Built environment","Equity"):1.5692, ("Built environment","Future"):2.2134,
 ("Mobility","Economy"):0.7846, ("Mobility","Cultural"):0.9659, ("Mobility","Peer density"):1.2746,
 ("Mobility","Equity"):1.0000, ("Mobility","Future"):2.2134, ("Economy","Cultural"):1.1132,
 ("Economy","Peer density"):1.1132, ("Economy","Equity"):1.2746, ("Economy","Future"):1.0000,
 ("Cultural","Peer density"):0.6444, ("Cultural","Equity"):0.5056, ("Cultural","Future"):2.5508,
 ("Peer density","Equity"):0.7846, ("Peer density","Future"):1.9332, ("Equity","Future"):1.5157}
 
def ahp_matrix(pairs):
    idx = {d: i for i, d in enumerate(DIMS)}; A = np.ones((10, 10))
    for (a, b), v in pairs.items(): A[idx[a], idx[b]] = v; A[idx[b], idx[a]] = 1/v
    return A
 
A = ahp_matrix(PAIRS)
vals, vecs = np.linalg.eig(A); k = np.argmax(vals.real)
wdim = pd.Series(np.abs(vecs[:, k].real), index=DIMS); wdim = wdim/wdim.sum()
lam = vals.real[k]; CR = (lam - 10)/9/1.49
 
DQ = {"ap":0.85,"read":0.82,"math":0.82,"spend":0.80,"heat":0.85,"green":0.78,"div":0.75,
      "sport":0.85,"cult":0.85,"walk":0.85,"score":0.85,"lib":0.88,"phys":0.80,"sleep":0.80,
      "safety":0.85,"net":0.86,"bike":0.88,"transit":0.90,"asth":0.90,"ment":0.88,"ratio":0.92,
      "pfood":0.85,"commute":0.92,"unemp":0.90,"roaddeath":0.88,"suicide":0.80,"depress":0.75,
      "vegcover":0.78,"teenden":0.75,"religdiv":0.78,"cpi":0.92,"rd":0.90,"gdpg":0.85,
      "sci":0.82,"gini":0.80}
 
def transform(df):
    Z = normalise(df)
    for k, f in DQ.items():
        if k in Z: Z[k] = 0.5 + f*(Z[k] - 0.5)
    return np.sqrt(Z)
 
def dim_to_indicator(wdim):
    w = pd.Series(0.0, index=INDS)
    for d in DIMS:
        mem = [k for k in INDS if DIM[k] == d]
        for k in mem: w[k] = wdim[d]/len(mem)
    return w
 
def combine(Z, alpha=0.50, beta=0.25):
    wa, we, wc = dim_to_indicator(wdim)[Z.columns], entropy_weights(Z), critic_weights(Z)
    w = alpha*wa + beta*we + (1 - alpha - beta)*wc
    return w/w.sum(), wa, we, wc
 
def prospect(Z, w, lam=2.25, a=0.88):
    D = Z.values - 0.5
    V = np.where(D >= 0, D**a, -lam*(-D)**a)
    V = (V - V.min(0))/(V.max(0) - V.min(0) + 1e-12)
    return pd.Series(V @ w.values, index=Z.index)
 
def topsis(Z, w):
    V = Z.values*w.values
    dp = np.sqrt(((V - V.max(0))**2).sum(1)); dn = np.sqrt(((V - V.min(0))**2).sum(1))
    return pd.Series(dn/(dp + dn), index=Z.index)
 
def vikor(Z, w, v=0.5):
    V = Z.values*w.values; f, n = V.max(0), V.min(0); rng = f - n; rng[rng == 0] = 1e-9
    S = (w.values*(f - V)/rng).sum(1); R = (w.values*(f - V)/rng).max(1)
    Q = v*(S - S.min())/(S.max() - S.min() + 1e-12) + (1 - v)*(R - R.min())/(R.max() - R.min() + 1e-12)
    return pd.Series(1 - Q, index=Z.index)
 
def saw(Z, w):
    return pd.Series(Z.values @ w.values, index=Z.index)
 
def rank(s):
    return pd.Series(rankdata(-s.values, method="min").astype(int), index=s.index)
 
Z = transform(CITY)
w, wa, we, wc = combine(Z, 0.50, 0.25)
yhi = prospect(Z, w); yhi = 100*(yhi - yhi.min())/(yhi.max() - yhi.min())
rk = rank(yhi)
ev = pd.DataFrame({"topsis": topsis(Z, w), "vikor": vikor(Z, w), "saw": saw(Z, w)})
borda = pd.Series(100*(NC + 1 - np.column_stack([rank(ev[c]).values for c in ev.columns]).mean(1))/NC, index=Z.index)
 
rng = np.random.default_rng(20261107)
R = np.zeros((3000, NC)); wa_v, we_v, wc_v = wa.values, we.values, wc.values
for b in range(3000):
    a = np.clip(rng.normal(0.50, 0.07), 0.15, 0.80)
    bt = np.clip(rng.normal(0.25, 0.07), 0.05, 0.80 - a - 0.01)
    wb = a*wa_v + bt*we_v + (1 - a - bt)*wc_v
    wb = wb*np.exp(rng.normal(0, 0.15, len(wb))); wb /= wb.sum()
    R[b] = rank(prospect(Z, pd.Series(wb, index=Z.columns))).values
p1 = (R == 1).mean(0); lo = np.percentile(R, 5, 0); hi = np.percentile(R, 95, 0)
 
out = pd.DataFrame({"city": CITY.city, "yhi": yhi.round(2), "rank": rk,
                    "p_rank1": p1.round(4), "rank_lo": lo.astype(int), "rank_hi": hi.astype(int)})
print("AHP CR =", round(CR, 4)); print(out.sort_values("rank").head(10).to_string(index=False))
