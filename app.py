# app.py
# PNG Raster Detail QA service (Flask)
# - POST /png-qa  (multipart "file" OR JSON/form with "cutout_b64")
# - Returns: { ok, design_passed, why, failed_parts, metrics{...} }

from flask import Flask, request, jsonify
import base64, zlib, struct, math

app = Flask(__name__)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ---------- helpers ----------
def fnum(v, default=None):
    try:
        if v is None: return default
        if isinstance(v, (int, float)): return float(v)
        s = str(v).strip()
        if not s: return default
        return float(s)
    except Exception:
        return default

def num(v, fb=None):
    try:
        n = float(v)
        if math.isfinite(n):
            return n
    except Exception:
        pass
    return fb

def otsu_threshold(gray):
    hist = [0] * 256
    for g in gray:
        hist[g] += 1
    total = len(gray)
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
    return thr

def percentile(arr, p):
    if not arr:
        return None
    arr2 = sorted(arr)
    idx = max(0, int(round((len(arr2) - 1) * p)))
    return arr2[idx]

def decode_png_to_gray(png_bytes):
    if png_bytes[:8] != PNG_MAGIC:
        raise ValueError("not a PNG")

    def u32(b, o):
        return struct.unpack(">I", b[o:o+4])[0]

    off = 8
    width = height = bit_depth = color_type = None
    idat = []
    L = len(png_bytes)
    while off + 8 <= L:
        length = u32(png_bytes, off); off += 4
        ctype  = png_bytes[off:off+4]; off += 4
        data   = png_bytes[off:off+length]; off += length
        off   += 4  # CRC
        if ctype == b'IHDR':
            width  = u32(data, 0)
            height = u32(data, 4)
            bit_depth = data[8]
            color_type = data[9]
        elif ctype == b'IDAT':
            idat.append(data)
        elif ctype == b'IEND':
            break

    if bit_depth != 8 or color_type not in (0, 2, 6):
        raise ValueError(f"Unsupported PNG (bitDepth={bit_depth}, colorType={color_type})")

    raw = zlib.decompress(b"".join(idat))
    bpp = 4 if color_type == 6 else (3 if color_type == 2 else 1)
    stride = width * bpp
    recon = bytearray(width * height * bpp)
    pos = 0
    prev = bytes([0]) * stride

    def paeth(a, b, c):
        p = a + b - c
        pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        return b if pb <= pc else c

    dst = 0
    for _y in range(height):
        f = raw[pos]; pos += 1
        for x in range(stride):
            rv = raw[pos]; pos += 1
            left = recon[dst + x - bpp] if x >= bpp else 0
            up   = prev[x] if x < len(prev) else 0
            ul   = prev[x - bpp] if x >= bpp else 0
            if f == 0:
                val = rv
            elif f == 1:
                val = (rv + left) & 255
            elif f == 2:
                val = (rv + up) & 255
            elif f == 3:
                val = (rv + ((left + up) // 2)) & 255
            elif f == 4:
                val = (rv + paeth(left, up, ul)) & 255
            else:
                val = rv
            recon[dst + x] = val
        prev = recon[dst:dst+stride]
        dst += stride

    gray = [0] * (width * height)
    i = 0
    if color_type == 6:
        for _y in range(height):
            for _x in range(width):
                r, g, b, a = recon[i], recon[i+1], recon[i+2], recon[i+3]; i += 4
                lum = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
                lum = int((a / 255.0) * lum + (1.0 - a / 255.0) * 255.0)
                gray[_y * width + _x] = lum
    elif color_type == 2:
        for _y in range(height):
            for _x in range(width):
                r, g, b = recon[i], recon[i+1], recon[i+2]; i += 3
                gray[_y * width + _x] = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    else:
        for _y in range(height):
            for _x in range(width):
                gray[_y * width + _x] = recon[i]; i += 1

    return width, height, gray

# ---------- binary image ops (no numpy) ----------
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
    if maxX < 0:  # empty
        return 0,0,-1,-1
    return minX, minY, maxX, maxY

def majority3x3(ink, w, h):
    out = [0]*(w*h)
    for y in range(h):
        for x in range(w):
            s = 0
            for yy in (y-1,y,y+1):
                if 0 <= yy < h:
                    base = yy * w
                    for xx in (x-1,x,x+1):
                        if 0 <= xx < w:
                            s += ink[base + xx]
            out[y*w+x] = 1 if s >= 5 else 0
    return out

def dilate1(ink, w, h):
    out = [0]*(w*h)
    for y in range(h):
        for x in range(w):
            val = 0
            for yy in (y-1,y,y+1):
                if 0 <= yy < h:
                    base = yy * w
                    for xx in (x-1,x,x+1):
                        if 0 <= xx < w and ink[base + xx]:
                            val = 1; break
                    if val: break
            out[y*w+x] = val
    return out

def erode1(ink, w, h):
    out = [0]*(w*h)
    for y in range(h):
        for x in range(w):
            val = 1
            for yy in (y-1,y,y+1):
                if 0 <= yy < h:
                    base = yy * w
                    for xx in (x-1,x,x+1):
                        if 0 <= xx < w:
                            if ink[base + xx] == 0:
                                val = 0; break
                    if val == 0: break
                else:
                    val = 0; break
            out[y*w+x] = val
    return out

def closing(ink, w, h, radius_px):
    """Morphological closing (dilate then erode), repeated radius_px times."""
    out = ink[:]
    for _ in range(max(0, int(radius_px))):
        out = dilate1(out, w, h)
        out = erode1(out, w, h)
    return out

def measure_min_line_and_gap(ink, w, h, roi=None):
    if roi is None:
        minX, minY, maxX, maxY = bbox_of_ink(ink, w, h)
    else:
        minX, minY, maxX, maxY = roi
    if maxX < minX or maxY < minY:
        return None, None
    min_line = float("inf"); min_gap = float("inf")
    # horizontal runs
    for y in range(minY, maxY+1):
        base = y * w
        run = 0; gap = 0
        for x in range(minX, maxX+1):
            v = ink[base + x]
            if v:
                if gap > 0: min_gap = min(min_gap, gap); gap = 0
                run += 1
            else:
                if run > 0: min_line = min(min_line, run); run = 0
                gap += 1
        if run > 0: min_line = min(min_line, run)
        if gap > 0: min_gap = min(min_gap, gap)
    # vertical runs
    for x in range(minX, maxX+1):
        run = 0; gap = 0
        for y in range(minY, maxY+1):
            v = ink[y*w + x]
            if v:
                if gap > 0: min_gap = min(min_gap, gap); gap = 0
                run += 1
            else:
                if run > 0: min_line = min(min_line, run); run = 0
                gap += 1
        if run > 0: min_line = min(min_line, run)
        if gap > 0: min_gap = min(min_gap, gap)

    if not math.isfinite(min_line): min_line = 0
    if not math.isfinite(min_gap):  min_gap  = 0
    return int(min_line), int(min_gap)

def estimate_min_text_height_px(ink, w, h, roi=None):
    if roi is None:
        minX, minY, maxX, maxY = bbox_of_ink(ink, w, h)
    else:
        minX, minY, maxX, maxY = roi
    if maxX < minX or maxY < minY:
        return None
    lab = [0]*(w*h)
    heights = []
    stack = []
    label = 0
    for y in range(minY, maxY+1):
        for x in range(minX, maxX+1):
            idx = y*w + x
            if ink[idx] == 0 or lab[idx] != 0:
                continue
            label += 1
            miny = y; maxy = y
            lab[idx] = label; stack.append((x,y))
            while stack:
                cx, cy = stack.pop()
                cidx = cy*w + cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                if cx > minX:
                    n = cidx - 1
                    if ink[n] == 1 and lab[n] == 0: lab[n]=label; stack.append((cx-1,cy))
                if cx < maxX:
                    n = cidx + 1
                    if ink[n] == 1 and lab[n] == 0: lab[n]=label; stack.append((cx+1,cy))
                if cy > minY:
                    n = cidx - w
                    if ink[n] == 1 and lab[n] == 0: lab[n]=label; stack.append((cx,cy-1))
                if cy < maxY:
                    n = cidx + w
                    if ink[n] == 1 and lab[n] == 0: lab[n]=label; stack.append((cx,cy+1))
            hgt = maxy - miny + 1
            if hgt >= 1:
                heights.append(hgt)
    return percentile(heights, 0.10) if heights else None

# ---------- core QA ----------
def run_png_qa_core(png_bytes, params):
    width, height, gray = decode_png_to_gray(png_bytes)

    thr = otsu_threshold(gray)
    ink_raw = [1 if g < thr else 0 for g in gray]

    # ROI for speed
    roi = bbox_of_ink(ink_raw, width, height)

    # px_per_in: prefer explicit px_per_in, else print size, else default 300
    px_per_in = fnum(params.get("px_per_in"))
    if not px_per_in:
        pw = fnum(params.get("print_width_in"))
        ph = fnum(params.get("print_height_in"))
        if pw and pw > 0:
            px_per_in = width / pw
        elif ph and ph > 0:
            px_per_in = height / ph
        else:
            px_per_in = 300.0

    # Denoise & close tiny slivers for negative-space only
    alias_px = int(max(1, fnum(params.get("gap_alias_px"), 1)))
    ink_clean = majority3x3(ink_raw, width, height)
    ink_closed = closing(ink_clean, width, height, alias_px)

    # Measurements
    min_line_px_raw, min_gap_px_raw = measure_min_line_and_gap(ink_raw, width, height, roi)
    _min_line_px_closed, min_gap_px_closed = measure_min_line_and_gap(ink_closed, width, height, roi)
    min_text_px = estimate_min_text_height_px(ink_raw, width, height, roi)

    # Choose robust metrics:
    min_line_px = min_line_px_raw                         # lines from raw
    min_gap_px  = max(min_gap_px_closed, alias_px)        # gaps from closed, never below alias_px

    metrics = {
        "width_px": width,
        "height_px": height,
        "px_per_in": px_per_in,
        "min_line_weight_in":   (min_line_px / px_per_in) if min_line_px > 0 else None,
        "min_negative_space_in":(min_gap_px  / px_per_in) if min_gap_px  > 0 else (1.0/72.0),
        "min_text_height_in":   (min_text_px / px_per_in) if (min_text_px is not None) else None,
        # Debug:
        "min_gap_px_raw": min_gap_px_raw,
        "min_gap_px_closed": min_gap_px_closed,
        "alias_px": alias_px,
        "used_px_per_in": px_per_in
    }

    # Thresholds (defaults)
    TH_TEXT = num(params.get("min_text_height_in"),    0.10)     # 0.10 in
    TH_GAP  = num(params.get("min_negative_space_in"), 1.0/72.0) # 1 pt
    TH_LINE = num(params.get("min_line_weight_in"),    0.005)    # 0.005 in

    failed = []
    why = []

    if metrics["min_text_height_in"] is not None and metrics["min_text_height_in"] < TH_TEXT:
        failed.append("text")
        why.append('Smallest text is %.3f" (< %.3f").' % (metrics["min_text_height_in"], TH_TEXT))
    if metrics["min_line_weight_in"] is not None and metrics["min_line_weight_in"] < TH_LINE:
        failed.append("line")
        why.append('Line min %.3f" (< %.3f").' % (metrics["min_line_weight_in"], TH_LINE))
    if metrics["min_negative_space_in"] < TH_GAP:
        failed.append("negative_space")
        why.append('Negative space min %.3f" (< %.3f").' % (metrics["min_negative_space_in"], TH_GAP))

    passed = (len(failed) == 0)
    return {
        "ok": True,
        "design_passed": "Yes" if passed else "No",
        "why": " ".join(why) if why else "All detail rules met.",
        "failed_parts": failed,
        "metrics": metrics
    }

# ---------- routes ----------
@app.route("/png-qa", methods=["POST"])
def png_qa():
    try:
        # Collect params from JSON or form-data (text fields)
        params = {}
        if request.is_json:
            params = request.get_json(silent=True) or {}
        else:
            # request.form is MultiDict; convert to plain dict of first values
            try:
                params = {k: request.form.get(k) for k in request.form.keys()}
            except Exception:
                params = {}

        # Acquire image bytes
        png_bytes = None
        f = request.files.get("file")
        if f:
            png_bytes = f.read()

        if png_bytes is None:
            # Base64 fallback in either JSON or form text
            b64 = params.get("cutout_b64")
            if isinstance(b64, str):
                s = b64.strip()
                if s.startswith("data:"):
                    s = s.split(",", 1)[1] if "," in s else ""
                s = "".join(s.split())
                pad = (-len(s)) % 4
                if pad: s += "=" * pad
                try:
                    png_bytes = base64.b64decode(s, validate=False)
                except Exception:
                    png_bytes = None

        if png_bytes is None or len(png_bytes) < 8 or png_bytes[:8] != PNG_MAGIC:
            return jsonify({"ok": False, "error": "no_png"}), 400

        result = run_png_qa_core(png_bytes, params)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True})

# ---------- main ----------
if __name__ == "__main__":
    # Bind to $PORT in Procfile when running on Railway; default 8000 locally
    app.run(host="0.0.0.0", port=8000, debug=False)
