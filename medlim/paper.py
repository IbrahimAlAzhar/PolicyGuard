"""Text views of a paper: what the extractor sees, and safe-length paper text for generation."""
import re

END_MATTER = re.compile(
    r"(?:^|\n)[ \t]*(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?[ \t]*(?:acknowledge?ments?|references|bibliography|funding|"
    r"conflicts?\s+of\s+interests?|competing\s+interests?|author\s*(?:s'|'s|s)?\s+contributions?|"
    r"data\s+availability|supplementary\s+(?:materials?|information)|declarations|appendix|appendices)\b[^\n]{0,60}(?=\n|$)",
    re.I)
DISCUSSION = re.compile(r"(?:^|\n)[ \t]*(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?[ \t]*(?:discussion|general\s+discussion|"
                        r"discussion\s+and\s+conclusions?)\b[^\n]{0,40}(?=\n|$)", re.I)


def body_end(text):
    half = len(text) // 2
    for m in END_MATTER.finditer(text):
        if m.start() >= half:
            return m.start()
    return len(text)


def extraction_view(full_text, lim_start=-1, lim_end=-1, max_chars=18000):
    """Text the extractor reads: whole body if short; otherwise the discussion onwards (last max_chars of the body)
    plus, if the regex found a limitation passage outside that window, a window around it."""
    t = str(full_text or "")
    end = body_end(t)
    body = t[:end]
    if len(body) <= max_chars:
        return body
    d = [m.start() for m in DISCUSSION.finditer(body)]
    start = d[-1] if d and len(body) - d[-1] <= max_chars else max(0, len(body) - max_chars)
    view = body[start:]
    try:
        ls, le = int(float(lim_start)), int(float(lim_end))
    except (TypeError, ValueError):
        ls, le = -1, -1
    if 0 <= ls < start:
        a, b = max(0, ls - 1500), min(start, le + 1500)
        extra = body[a:b]
        budget = max_chars - len(extra) - 20
        view = extra + "\n[...]\n" + (view[-budget:] if budget > 2000 else view[-2000:])
    return view[-max_chars - 4000:]


def generation_text(text, max_chars):
    """paper text for generation: head + tail when too long (methods and discussion both matter)."""
    t = str(text or "")
    if len(t) <= max_chars:
        return t
    head = int(max_chars * 0.55)
    return t[:head] + "\n[...]\n" + t[-(max_chars - head):]
