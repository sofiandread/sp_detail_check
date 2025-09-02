from flask import Flask, request, jsonify
import base64, zlib, struct, math, os

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
        if math.isfinite(n): return n
    except Exception:
        pass
    return fb

def truthy(v):
    if v is None: return False
    return str(v).strip().lower() in ("1","true","yes","y","on")

def otsu_threshold(gray):
    hist = [0]*256
    for g in gray: hist[g]+=1
    total = len(gray)
    s_all = sum(i*hist[i] for i in range(256))
    sB = 0; wB = 0; best = -1.0; thr = 128
    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sB += t*hist[t]
        mB = sB/float(wB)
        mF = (s_all - sB)/float(wF)
        between = float(wB)*float(wF)*(mB-mF)*(mB-mF)
        if between > best:
            best = between; thr = t
    return thr

def percentile(arr, p):
    if not arr: return None
    arr2 = sorted(arr)
    idx = max(0, int(round((len(arr2)-1)*p)))
    return arr2[idx]

# ---------- PNG → grayscale ----------
def decode_png_to_gray(png_bytes):
    if png_bytes[:8] != PNG_MAGIC:
        raise ValueError("not a PNG")

    def u32(b,o): return struct.unpack(">I", b[o:o+4])[0]

    off=8; width=height=bit_depth=color_type=None; idat=[]; L=len(png_bytes)
    while off+8<=L:
        length=u32(png_bytes,off); off+=4
        ctype=png_bytes[off:off+4]; off+=4
        data=png_bytes[off:off+length]; off+=length
        off+=4
        if ctype==b'IHDR':
            width=u32(data,0); height=u32(data,4)
            bit_depth=data[8]; color_type=data[9]
        elif ctype==b'IDAT':
            idat.append(data)
        elif ctype==b'IEND':
            break

    if bit_depth!=8 or color_type not in (0,2,6):
        raise ValueError(f"Unsupported PNG (bitDepth={bit_depth}, colorType={color_type})")

    raw=zlib.decompress(b"".join(idat))
    bpp=4 if color_type==6 else (3 if color_type==2 else 1)
    stride=width*bpp
    recon=bytearray(width*height*bpp)

    def paeth(a,b,c):
        p=a+b-c
        pa=abs(p-a); pb=abs(p-b); pc=abs(p-c)
        if pa<=pb and pa<=pc: return a
        return b if pb<=pc else c

    pos=0; prev=bytes([0])*stride; dst=0
    for _y in range(height):
        f=raw[pos]; pos+=1
        for x in range(stride):
            rv=raw[pos]; pos+=1
            left=recon[dst+x-bpp] if x>=bpp else 0
            up=prev[x] if x<len(prev) else 0
            ul=prev[x-bpp] if x>=bpp else 0
            if f==0: val=rv
            elif f==1: val=(rv+left)&255
            elif f==2: val=(rv+up)&255
            elif f==3: val=(rv+((left+up)//2))&255
            elif f==4: val=(rv+paeth(left,up,ul))&255
            else: val=rv
            recon[dst+x]=val
        prev=recon[dst:dst+stride]; dst+=stride

    gray=[0]*(width*height); i=0
    if color_type==6:
        for _y in range(height):
            for _x in range(width):
                r,g,b,a=recon[i],recon[i+1],recon[i+2],recon[i+3]; i+=4
                lum=int(0.2126*r+0.7152*g+0.0722*b)
                lum=int((a/255.0)*lum+(1.0-a/255.0)*255.0)
                gray[_y*width+_x]=lum
    elif color_type==2:
        for _y in range(height):
            for _x in range(width):
                r,g,b=recon[i],recon[i+1],recon[i+2]; i+=3
                gray[_y*width+_x]=int(0.2126*r+0.7152*g+0.0722*b)
    else:
        for _y in range(height):
            for _x in range(width):
                gray[_y*width+_x]=recon[i]; i+=1
    return width, height, gray

# ---------- binary ops ----------
def bbox_of_ink(ink,w,h):
    minX,minY,maxX,maxY=w,h,-1,-1
    for y in range(h):
        base=y*w
        for x in range(w):
            if ink[base+x]:
                if x<minX:minX=x
                if x>maxX:maxX=x
                if y<minY:minY=y
                if y>maxY:maxY=y
    if maxX<0: return 0,0,-1,-1
    return minX,minY,maxX,maxY

def zero_border(ink,w,h,px):
    if px<=0: return ink
    out = ink[:]
    for y in range(h):
        base=y*w
        for x in range(w):
            if x<px or x>=w-px or y<px or y>=h-px:
                out[base+x]=0
    return out

def majority3x3(ink,w,h):
    out=[0]*(w*h)
    for y in range(h):
        for x in range(w):
            s=0
            for yy in (y-1,y,y+1):
                if 0<=yy<h:
                    base=yy*w
                    for xx in (x-1,x,x+1):
                        if 0<=xx<w: s+=ink[base+xx]
            out[y*w+x]=1 if s>=5 else 0
    return out

def dilate1(ink,w,h):
    out=[0]*(w*h)
    for y in range(h):
        for x in range(w):
            v=0
            for yy in (y-1,y,y+1):
                if 0<=yy<h:
                    base=yy*w
                    for xx in (x-1,x,x+1):
                        if 0<=xx<w and ink[base+xx]:
                            v=1; break
                    if v: break
            out[y*w+x]=v
    return out

def erode1(ink,w,h):
    out=[0]*(w*h)
    for y in range(h):
        for x in range(w):
            v=1
            for yy in (y-1,y,y+1):
                if 0<=yy<h:
                    base=yy*w
                    for xx in (x-1,x,x+1):
                        if 0<=xx<w and ink[base+xx]==0:
                            v=0; break
                    if v==0: break
                else:
                    v=0; break
            out[y*w+x]=v
    return out

def closing(ink,w,h,radius_px):
    out=ink[:]
    for _ in range(max(0,int(radius_px))):
        out=dilate1(out,w,h)
        out=erode1(out,w,h)
    return out

def clamp_roi(x0,y0,x1,y1,w,h):
    x0=max(0,min(x0,w-1)); y0=max(0,min(y0,h-1))
    x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
    if x1<x0: x0,x1=x1,x0
    if y1<y0: y0,y1=y1,y0
    return (x0,y0,x1,y1)

def inset_roi(roi, inset):
    x0,y0,x1,y1 = roi
    xi, yi = x0+inset, y0+inset
    xa, ya = x1-inset, y1-inset
    if xa<=xi or ya<=yi: return roi
    return (xi,yi,xa,ya)

# ---------- measurements ----------
def measure_min_line_and_gap_pos(ink,w,h,roi=None):
    if roi is None:
        minX,minY,maxX,maxY=bbox_of_ink(ink,w,h)
    else:
        minX,minY,maxX,maxY=roi
    if maxX<minX or maxY<minY:
        return (0,0,None,None)

    min_line=1e9; min_gap=1e9
    pos_line=None; pos_gap=None

    # horizontal
    for y in range(minY,maxY+1):
        base=y*w; run=0; gap=0
        for x in range(minX,maxX+1):
            v=ink[base+x]
            if v:
                if gap>0 and gap<min_gap:
                    min_gap=gap
                    pos_gap={"orient":"H","y":y,"x0":x-gap,"x1":x-1,"len_px":gap}
                gap=0; run+=1
            else:
                if run>0 and run<min_line:
                    min_line=run
                    pos_line={"orient":"H","y":y,"x0":x-run,"x1":x-1,"len_px":run}
                run=0; gap+=1
        if run>0 and run<min_line:
            min_line=run
            pos_line={"orient":"H","y":y,"x0":maxX+1-run,"x1":maxX,"len_px":run}
        if gap>0 and gap<min_gap:
            min_gap=gap
            pos_gap={"orient":"H","y":y,"x0":maxX+1-gap,"x1":maxX,"len_px":gap}

    # vertical
    for x in range(minX,maxX+1):
        run=0; gap=0
        for y in range(minY,maxY+1):
            v=ink[y*w+x]
            if v:
                if gap>0 and gap<min_gap:
                    min_gap=gap
                    pos_gap={"orient":"V","x":x,"y0":y-gap,"y1":y-1,"len_px":gap}
                gap=0; run+=1
            else:
                if run>0 and run<min_line:
                    min_line=run
                    pos_line={"orient":"V","x":x,"y0":y-run,"y1":y-1,"len_px":run}
                run=0; gap+=1
        if run>0 and run<min_line:
            min_line=run
            pos_line={"orient":"V","x":x,"y0":maxY+1-run,"y1":maxY,"len_px":run}
        if gap>0 and gap<min_gap:
            min_gap=gap
            pos_gap={"orient":"V","x":x,"y0":maxY+1-gap,"y1":maxY,"len_px":gap}

    if not math.isfinite(min_line): min_line=0
    if not math.isfinite(min_gap): min_gap=0

    # centers
    if pos_gap:
        if pos_gap.get("orient")=="H":
            pos_gap["cx"]=(pos_gap["x0"]+pos_gap["x1"])//2
            pos_gap["cy"]=pos_gap["y"]
        else:
            pos_gap["cx"]=pos_gap["x"]
            pos_gap["cy"]=(pos_gap["y0"]+pos_gap["y1"])//2
    if pos_line:
        if pos_line.get("orient")=="H":
            pos_line["cx"]=(pos_line["x0"]+pos_line["x1"])//2
            pos_line["cy"]=pos_line["y"]
        else:
            pos_line["cx"]=pos_line["x"]
            pos_line["cy"]=(pos_line["y0"]+pos_line["y1"])//2

    return int(min_line), int(min_gap), pos_line, pos_gap

def estimate_min_text_height_px_filtered(ink,w,h,roi, ppi):
    """Return a robust small text height in px, ignoring specks and edge nicks."""
    x0,y0,x1,y1 = roi
    if x1<x0 or y1<y0:
        return None, 0

    # thresholds
    min_h_px = max(int(round(ppi*0.06)), 6)           # ignore comps shorter than ~0.06" or 6 px
    min_area_px = max(int(round((ppi*ppi)*0.003)), 40) # ignore tiny areas
    heights=[]; kept=0

    W=w; H=h
    lab=[0]*(W*H); stack=[]; label=0
    for y in range(y0,y1+1):
        for x in range(x0,x1+1):
            idx=y*W+x
            if ink[idx]==0 or lab[idx]!=0: continue
            label+=1
            lab[idx]=label; stack.append((x,y))
            minx=maxx=x; miny=maxy=y; area=0
            touch_edge=False
            while stack:
                cx,cy=stack.pop(); cidx=cy*W+cx
                area+=1
                if cx<minx:minx=cx
                if cx>maxx:maxx=cx
                if cy<miny:miny=cy
                if cy>maxy:maxy=cy
                # 4-neighborhood
                if cx>x0 and ink[cidx-1]==1 and lab[cidx-1]==0: lab[cidx-1]=label; stack.append((cx-1,cy))
                if cx<x1 and ink[cidx+1]==1 and lab[cidx+1]==0: lab[cidx+1]=label; stack.append((cx+1,cy))
                if cy>y0 and ink[cidx-W]==1 and lab[cidx-W]==0: lab[cidx-W]=label; stack.append((cx,cy-1))
                if cy<y1 and ink[cidx+W]==1 and lab[cidx+W]==0: lab[cidx+W]=label; stack.append((cx,cy+1))
                # edge contact?
                if cx==x0 or cx==x1 or cy==y0 or cy==y1:
                    touch_edge=True
            hgt=maxy-miny+1
            if (not touch_edge) and hgt>=min_h_px and area>=min_area_px:
                heights.append(hgt); kept+=1

    if not heights: 
        return None, 0
    return percentile(heights, 0.25), kept  # 25th percentile of filtered components

# ---------- core ----------
def run_png_qa_core(png_bytes, params):
    w, h, gray = decode_png_to_gray(png_bytes)

    thr = otsu_threshold(gray)
    ink_raw = [1 if g < thr else 0 for g in gray]

    ignore_border = int(max(0, fnum(params.get("ignore_border_px"), 0)))
    ink_nb = zero_border(ink_raw, w, h, ignore_border)

    roi = bbox_of_ink(ink_nb, w, h)
    roi = clamp_roi(*roi, w, h)
    roi_w = max(0, roi[2]-roi[0]+1); roi_h = max(0, roi[3]-roi[1]+1)

    # ppi
    px_per_in = fnum(params.get("px_per_in"))
    use_roi_ppi = truthy(params.get("use_roi_for_ppi"))
    if not px_per_in:
        pw = fnum(params.get("print_width_in"))
        ph = fnum(params.get("print_height_in"))
        if use_roi_ppi and roi_w>0 and roi_h>0:
            if pw and pw>0: px_per_in = roi_w / pw
            elif ph and ph>0: px_per_in = roi_h / ph
        if not px_per_in:
            if pw and pw>0: px_per_in = w / pw
            elif ph and ph>0: px_per_in = h / ph
            else: px_per_in = 300.0

    # insets & guards
    user_inset_px = int(max(0, fnum(params.get("inset_px"), 0)))
    inset_frac     = fnum(params.get("inset_frac"), 0.02)    # 2% of ROI
    auto_inset_px  = int(round(min(roi_w, roi_h) * inset_frac)) if roi_w and roi_h else 0
    ppi_inset_px   = int(round(px_per_in * 0.01))            # 1% of ppi
    base_inset     = max(user_inset_px, auto_inset_px, ppi_inset_px)

    edge_guard_px  = int(max(0, fnum(params.get("edge_guard_px"), 12)))
    effective_inset = base_inset + edge_guard_px
    roi_interior = inset_roi(roi, effective_inset)

    # cleaning / alias tolerance
    alias_px = int(max(1, fnum(params.get("gap_alias_px"), 2)))
    ink_clean  = majority3x3(ink_nb, w, h)
    ink_closed = closing(ink_clean, w, h, alias_px)

    # measure inside guarded interior
    min_line_px_raw, _mg_unused, pos_line_raw, _pg_unused = \
        measure_min_line_and_gap_pos(ink_nb, w, h, roi_interior)
    _ml_closed, min_gap_px_closed, _pl_unused, pos_gap_closed = \
        measure_min_line_and_gap_pos(ink_closed, w, h, roi_interior)

    # text height with filtering
    min_text_px_filtered, comp_used = estimate_min_text_height_px_filtered(
        ink_nb, w, h, roi_interior, px_per_in
    )

    # span floor for gaps in px: at least the spec gap in px
    TH_TEXT = num(params.get("min_text_height_in"),    0.10)
    TH_GAP  = num(params.get("min_negative_space_in"), 1.0/72.0)
    TH_LINE = num(params.get("min_line_weight_in"),    0.005)

    spec_gap_px = math.ceil(TH_GAP * px_per_in)
    min_gap_span_px = int(max(spec_gap_px, fnum(params.get("min_gap_span_px"), spec_gap_px)))
    if pos_gap_closed and pos_gap_closed.get("len_px", 0) < min_gap_span_px:
        min_gap_px_closed = max(min_gap_px_closed, min_gap_span_px)

    # final px metrics
    min_line_px = min_line_px_raw
    min_gap_px  = max(min_gap_px_closed, alias_px)

    # convert to inches with half-px tolerance
    min_line_in = (min_line_px + 0.5)/px_per_in if min_line_px>0 else None
    min_gap_in  = (min_gap_px  + 0.5)/px_per_in if min_gap_px >0 else (1.0/72.0)
    min_text_in = (min_text_px_filtered/px_per_in) if (min_text_px_filtered is not None) else None

    # pass/fail
    failed=[]; why=[]; margins={}
    if min_text_in is not None:
        margins["text_in"] = min_text_in - TH_TEXT
        if min_text_in < TH_TEXT:
            failed.append("text")
            why.append('Smallest text is %.3f" (< %.3f").' % (min_text_in, TH_TEXT))
    if min_line_in is not None:
        margins["line_in"] = min_line_in - TH_LINE
        if min_line_in < TH_LINE:
            failed.append("line")
            why.append('Line min %.3f" (< %.3f").' % (min_line_in, TH_LINE))
    margins["negative_space_in"] = min_gap_in - TH_GAP
    if min_gap_in < TH_GAP:
        failed.append("negative_space")
        why.append('Negative space min %.3f" (< %.3f").' % (min_gap_in, TH_GAP))

    ratios=[]
    if min_text_in is not None: ratios.append(min_text_in/TH_TEXT)
    if min_line_in is not None: ratios.append(min_line_in/TH_LINE)
    if min_gap_in is not None:  ratios.append(min_gap_in /TH_GAP)
    confidence = max(0.0, min(1.0, min(ratios))) if ratios else 1.0

    return {
        "ok": True,
        "design_passed": "Yes" if not failed else "No",
        "why": " ".join(why) if why else "All detail rules met.",
        "failed_parts": failed,
        "confidence": confidence,
        "metrics": {
            "width_px": w, "height_px": h,
            "roi": {"x0": roi[0], "y0": roi[1], "x1": roi[2], "y1": roi[3], "w": roi_w, "h": roi_h},
            "roi_interior": {
                "x0": roi_interior[0], "y0": roi_interior[1],
                "x1": roi_interior[2], "y1": roi_interior[3],
                "w": max(0, roi_interior[2]-roi_interior[0]+1),
                "h": max(0, roi_interior[3]-roi_interior[1]+1),
            },
            "px_per_in": px_per_in, "used_px_per_in": px_per_in,
            "alias_px": alias_px,
            "edge_guard_px": edge_guard_px,
            "effective_inset_px": effective_inset,
            "spec_gap_px": spec_gap_px,
            "min_gap_span_px": min_gap_span_px,
            "min_line_weight_in":   min_line_in,
            "min_negative_space_in":min_gap_in,
            "min_text_height_in":   min_text_in,
            "text_components_used": comp_used,
            "debug_min_gap":  pos_gap_closed,
            "debug_min_line": pos_line_raw,
            "ignore_border_px": ignore_border,
            "use_roi_for_ppi": use_roi_ppi,
            "margins_in": margins
        }
    }

# ---------- routes ----------
@app.route("/png-qa", methods=["POST"])
def png_qa():
    try:
        params = {}
        if request.is_json:
            params = request.get_json(silent=True) or {}
        else:
            try:
                params = {k: request.form.get(k) for k in request.form.keys()}
            except Exception:
                params = {}

        png_bytes=None
        f = request.files.get("file")
        if f: png_bytes = f.read()
        if png_bytes is None:
            b64 = params.get("cutout_b64")
            if isinstance(b64,str):
                s=b64.strip()
                if s.startswith("data:"): s=s.split(",",1)[1] if "," in s else ""
                s="".join(s.split()); pad=(-len(s))%4
                if pad: s+="="*pad
                try:
                    png_bytes = base64.b64decode(s, validate=False)
                except Exception:
                    png_bytes=None

        if png_bytes is None or len(png_bytes)<8 or png_bytes[:8]!=PNG_MAGIC:
            return jsonify({"ok": False, "error": "no_png"}), 400

        return jsonify(run_png_qa_core(png_bytes, params)), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
