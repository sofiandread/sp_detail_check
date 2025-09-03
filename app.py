# app.py
from flask import Flask, request, jsonify
import base64, zlib, struct, math, os

app = Flask(__name__)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# =========================
# ---------- utils --------
# =========================

def truthy(v):
    if v is None: return False
    return str(v).strip().lower() in ("1","true","yes","y","on")

def finite(v, fb=0.0):
    """Return float(v) if finite, else fallback."""
    try:
        n = float(v)
        return n if math.isfinite(n) else fb
    except Exception:
        return fb

def finite_pos(v, fb=300.0):
    """Finite and > 0; else fallback."""
    n = finite(v, fb)
    return n if n > 0 else fb

def safe_int(v, fb=0):
    """Best-effort int conversion for finite numbers."""
    try:
        n = float(v)
        if math.isfinite(n):
            return int(n)
    except Exception:
        pass
    return fb

def safe_round(v, nd=0, fb=0):
    """Round only if finite; else fallback."""
    try:
        n = float(v)
        if math.isfinite(n):
            return round(n, nd)
    except Exception:
        pass
    return fb

def safe_max(*vals, fb=0):
    """max over finite values; ignore non-finite."""
    fs = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return max(fs) if fs else fb

def parse_human_size(raw=''):
    m = str(raw or '').lower().strip()
    if not m: return 0
    try:
        parts = m.split()
        n = float(parts[0])
        if not math.isfinite(n): return 0
        u = parts[1] if len(parts) > 1 else "b"
        u = u.lower()
        if u.startswith('mb'): return n * 1024 * 1024
        if u.startswith('kb'): return n * 1024
        return n
    except Exception:
        return 0

def b64_bytes_len(s=''):
    if not s: return 0
    s = str(s)
    pad = 0
    if s.endswith('=='): pad = 2
    elif s.endswith('='): pad = 1
    try:
        L = len(s)
        return max(0, (L * 3) // 4 - pad)
    except Exception:
        return 0

def percentile(arr, p):
    if not arr: return None
    try:
        arr2 = sorted(arr)
        idx = max(0, int(round((len(arr2)-1) * p)))
        return arr2[idx]
    except Exception:
        return None

def clamp_roi(x0,y0,x1,y1,w,h):
    try:
        x0=max(0,min(x0,w-1)); y0=max(0,min(y0,h-1))
        x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
        if x1<x0: x0,x1=x1,x0
        if y1<y0: y0,y1=y1,y0
        return (x0,y0,x1,y1)
    except Exception:
        return (0,0,-1,-1)

def inset_roi(roi, inset):
    try:
        x0,y0,x1,y1 = roi
        xi, yi = x0+inset, y0+inset
        xa, ya = x1-inset, y1-inset
        if xa<=xi or ya<=yi: return roi
        return (xi,yi,xa,ya)
    except Exception:
        return roi

# =========================
# ----- PNG decoding ------
# =========================

def otsu_threshold(gray):
    if not gray:
        return 128
    hist = [0]*256
    try:
        for g in gray:
            if 0 <= g <= 255:
                hist[g]+=1
        total = len(gray)
        if total <= 0: return 128
        s_all = 0
        for i in range(256):
            s_all += i*hist[i]
        sB = 0; wB = 0; best = -1.0; thr = 128
