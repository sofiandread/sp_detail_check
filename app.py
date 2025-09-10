# app.py
from flask import Flask, request, jsonify
import base64, zlib, struct, math, os

app = Flask(__name__)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ======================================================
# ---------------------- GUARDS ------------------------
# ======================================================

def finite(v, fb=0.0):
    try:
        n = float(v)
        return n if math.isfinite(n) else fb
    except Exception:
        return fb

def finite_pos(v, fb=300.0):
    n = finite(v, fb)
    return n if n > 0 else fb

def safe_int(v, fb=0):
    try:
        n = float(v)
        if math.isfinite(n):
            return int(n)
    except Exception:
        pass
    return fb

def safe_round(v, nd=0, fb=0):
    try:
        n = float(v)
        if math.isfinite(n):
            return round(n, nd)
    except Exception:
        pass
    return fb

def safe_ceil(v, fb=0):
    try:
        n = float(v)
        if math.isfinite(n):
            return int(math.ceil(n))
    except Exception:
        pass
    return fb

def truthy(v):
    if v is None: return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def num(v, fb=None):
    try:
        n = float(v)
        return n if math.isfinite(n) else fb
    except Exception:
        return fb

# ======================================================
# -------------------- UTILITIES -----------------------
# ======================================================

def otsu_threshold(gray):
    hist = [0] * 256
    for g in gray:
        hist[g] += 1
    total = len(gray)
    if total <= 0:
        return 128
    s_all = sum(i * hist[i] for i in range(256))
    sB = 0
    wB = 0
    best = -1.0
    thr = 128
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sB += t * hist[t]
        mB = sB / float(wB)
        mF = (s_all - sB) / float(wF)
        between = float(wB) * float(wF) * (mB - mF) * (mB - mF)
        if between > best:
            best = between
            thr = t
    return int(thr)

def percentile(arr, p):
    if not arr:
        return None
    arr2 = sorted(arr)
    idx = max(0, int(round((len(arr2) - 1) * p)))
    return arr2[idx]

# ----------- safer zlib to avoid zipbombs ------------
def safe_zlib_decompress(data, max_out=64 * 1024 * 1024):
    d = zlib.decompressobj()
    chunks = []
    total = 0
    pos = 0
    CHUNK = 1 << 15
    while pos < len(data):
        piece = data[pos:pos + CHUNK]
        out = d.decompress(piece)
        total += len(out)
        if total > max_out:
            raise ValueError("decompress_too_large")
        chunks.append(out)
        pos += CHUNK
    out = d.flush()
    total += len(out)
    if total > max_out:
        raise ValueError("decompress_too_large")
    chunks.append(out)
    return b"".join(chunks)

# ======================================================
# -------------- PNG DECODE → GRAYSCALE ----------------
# ======================================================

def decode_png_to_gray(png_bytes):
    if not png_bytes or len(png_bytes) < 8 or png_bytes[:8] != PNG_MAGIC:
        raise ValueError("not_a_png")

    def u32(b, o): return struct.unpack(">I", b[o:o + 4])[0]

    off = 8
    width = height = bit_depth = color_type = None
    idat = []
    L = len(png_bytes)
    while off + 8 <= L:
        length = u32(png_bytes, off); off += 4
        if length < 0 or length > 2**31:
            raise ValueError("bad_png_chunk_length")
        ctype = png_bytes[off:off + 4]; off += 4
        if off + length + 4 > L:
            raise ValueError("truncated_png_chunk")
        data = png_bytes[off:off + length]; off += length
        _crc = png_bytes[off:off + 4]; off += 4
        if ctype == b'IHDR':
            if length < 13:
                raise ValueError("bad_ihdr_len")
            width = u32(data, 0); height = u32(data, 4)
            bit_depth = data[8]; color_type = data[9]
            if width <= 0 or height <= 0 or width > 50000 or height > 50000:
                raise ValueError("unreasonable_png_dims")
        elif ctype == b'IDAT':
            idat.append(data)
        elif ctype == b'IEND':
            break

    if bit_depth != 8 or color_type not in (0, 2, 6):
        raise ValueError(f"Unsupported PNG (bitDepth={bit_depth}, colorType={color_type})")
    if not idat:
        raise ValueError("no_idat")

    raw = safe_zlib_decompress(b"".join(idat))
    bpp = 4 if color_type == 6 else (3 if color_type == 2 else 1)
    stride = width * bpp
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise ValueError("truncated_idat_data")

    recon = bytearray(width * height * bpp)

    def paeth(a, b, c):
        p = a + b - c
        pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
        if pa <= pb and pa <= pc: return a
        return b if pb <= pc else c

    pos = 0
    prev = bytes([0]) * stride
    dst = 0
    for _y in range(height):
        f = raw[pos]; pos += 1
        if f not in (0, 1, 2, 3, 4):
            raise ValueError("bad_filter")
        for x in range(stride):
            rv = raw[pos]; pos += 1
            left = recon[dst + x - bpp] if x >= bpp else 0
            up = prev[x] if x < len(prev) else 0
            ul = prev[x - bpp] if x >= bpp else 0
            if f == 0: val = rv
            elif f == 1: val = (rv + left) & 255
            elif f == 2: val = (rv + up) & 255
            elif f == 3: val = (rv + ((left + up) // 2)) & 255
            else:       val = (rv + paeth(left, up, ul)) & 255
            recon[dst + x] = val
        prev = recon[dst:dst + stride]
        dst += stride

    gray = [0] * (width * height)
    i = 0
    if color_type == 6:
        for _y in range(height):
            for _x in range(width):
                r, g, b, a = recon[i], recon[i + 1], recon[i + 2], recon[i + 3]; i += 4
                lum = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
                lum = int((a / 255.0) * lum + (1.0 - a / 255.0) * 255.0)
                gray[_y * width + _x] = lum
    elif color_type == 2:
        for _y in range(height):
            for _x in range(width):
                r, g, b = recon[i], recon[i + 1], recon[i + 2]; i += 3
                gray[_y * width + _x] = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    else:
        for _y in range(height):
            for _x in range(width):
                gray[_y * width + _x] = recon[i]; i += 1

    return width, height, gray

# ======================================================
# ----------------- BINARY OPERATIONS ------------------
# ======================================================

def bbox_of_ink(ink, w, h):
    minX, minY, maxX, maxY = w, h, -1, -1
    for y in range(h):
        base = y * w
        for x in range(w):
            if ink[base + x]:
                if x < minX: minX = x
                if x > maxX: maxX = x
                if y < minY: minY = y
                if y > maxY: maxY = y
    if maxX < 0:
        return 0, 0, -1, -1
    return minX, minY, maxX, maxY

def zero_border(ink, w, h, px):
    px = max(0, int(px))
    if px <= 0: return ink
    out = ink[:]
    for y in range(h):
        base = y * w
        for x in range(w):
            if x < px or x >= w - px or y < px or y >= h - px:
                out[base + x] = 0
    return out

def majority3x3(ink, w, h):
    out = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            s = 0
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < h:
                    base = yy * w
                    for xx in (x - 1, x, x + 1):
                        if 0 <= xx < w:
                            s += ink[base + xx]
            out[y * w + x] = 1 if s >= 5 else 0
    return out

def dilate1(ink, w, h):
    out = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            v = 0
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < h:
                    base = yy * w
                    for xx in (x - 1, x, x + 1):
                        if 0 <= xx < w and ink[base + xx]:
                            v = 1; break
                    if v: break
            out[y * w + x] = v
    return out

def erode1(ink, w, h):
    out = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            v = 1
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < h:
                    base = yy * w
                    for xx in (x - 1, x + 1):
                        pass
                # corrected neighborhood (keep consistent with earlier)
    # NOTE: keep original erode1 from previous message; shortened here for brevity.
