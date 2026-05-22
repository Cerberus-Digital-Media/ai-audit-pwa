"""
AI Visibility Audit — Vercel Serverless Function (Flask)
POST /api/audit
Body: { client_name, industry, location, website, competitors[], custom_prompts[] }
Returns: { results[], scores, prompts[] }

API keys are read from Vercel Environment Variables:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
"""

from flask import Flask, request, jsonify, make_response
import os, re, concurrent.futures

app = Flask(__name__)

POSITIVE = {
    "recommend","best","excellent","top","great","leading","trusted","expert",
    "reputable","award","renowned","outstanding","premier","exceptional","quality",
    "innovative","proven","strong","preferred","well-regarded","highly","impressive",
    "successful","notable","respected","experienced","established","recognized","solid","reliable",
}
NEGATIVE = {
    "avoid","poor","bad","unreliable","overpriced","questionable","mediocre",
    "disappointing","problematic","issues","concerns","complaints","lawsuit",
    "scam","fraud","misleading","inconsistent","underperform","struggle",
    "criticism","negative","worst",
}

def generate_prompts(data):
    name     = data["client_name"]
    industry = data["industry"]
    loc      = data["location"]
    prompts  = [
        f"What are the best {industry}s in {loc}?",
        f"Who are the top {industry}s in {loc}?",
        f"Can you recommend a {industry} in {loc}?",
        f"I need a {industry} in {loc} — who should I consider?",
        f"What is {name}?",
        f"Tell me about {name} — are they reputable?",
        f"Is {name} a good {industry} to work with?",
        f"Who are the most well-known {industry}s in {loc}?",
    ]
    for comp in data.get("competitors", []):
        if comp.strip():
            prompts.append(f"Compare {name} and {comp} as a {industry}.")
    prompts += [p for p in data.get("custom_prompts", []) if p.strip()]
    return prompts

def analyze(response, client_name, competitors):
    tl   = response.lower()
    nl   = client_name.lower()
    mentioned     = nl in tl
    mention_count = len(re.findall(re.escape(nl), tl))
    snippet = None
    if mentioned:
        idx   = tl.find(nl)
        start = max(0, idx - 100)
        end   = min(len(response), idx + len(client_name) + 220)
        snippet = ("…" if start > 0 else "") + response[start:end].strip() + "…"
    sentiment = "not_mentioned"
    if mentioned:
        idx   = tl.find(nl)
        ctx   = tl[max(0, idx-180):min(len(tl), idx+360)]
        pos   = sum(1 for w in POSITIVE if w in ctx)
        neg   = sum(1 for w in NEGATIVE if w in ctx)
        if neg and pos:  sentiment = "mixed"
        elif pos:        sentiment = "positive"
        elif neg:        sentiment = "negative"
        else:            sentiment = "neutral"
    return {
        "mentioned":             mentioned,
        "mention_count":         mention_count,
        "sentiment":             sentiment,
        "snippet":               snippet,
        "competitors_mentioned": [c for c in competitors if c.lower() in tl],
    }

def query_openai(prompt, data):
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp   = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing honest, balanced recommendations."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=500, temperature=0.7,
        )
        text = resp.choices[0].message.content
        return {"platform": "ChatGPT (GPT-4o)", "prompt": prompt, "response": text,
                "error": None, **analyze(text, data["client_name"], data.get("competitors", []))}
    except Exception as e:
        return {"platform": "ChatGPT (GPT-4o)", "prompt": prompt, "response": "", "error": str(e),
                "mentioned": False, "mention_count": 0, "sentiment": "not_mentioned",
                "snippet": None, "competitors_mentioned": []}

def query_claude(prompt, data):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
        client  = anthropic.Anthropic(api_key=key)
        msg     = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        return {"platform": "Claude (Anthropic)", "prompt": prompt, "response": text,
                "error": None, **analyze(text, data["client_name"], data.get("competitors", []))}
    except Exception as e:
        return {"platform": "Claude (Anthropic)", "prompt": prompt, "response": "", "error": str(e),
                "mentioned": False, "mention_count": 0, "sentiment": "not_mentioned",
                "snippet": None, "competitors_mentioned": []}

def query_gemini(prompt, data):
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model  = genai.GenerativeModel("gemini-1.5-pro")
        resp   = model.generate_content(prompt)
        text   = resp.text
        return {"platform": "Gemini (Google)", "prompt": prompt, "response": text,
                "error": None, **analyze(text, data["client_name"], data.get("competitors", []))}
    except Exception as e:
        return {"platform": "Gemini (Google)", "prompt": prompt, "response": "", "error": str(e),
                "mentioned": False, "mention_count": 0, "sentiment": "not_mentioned",
                "snippet": None, "competitors_mentioned": []}

QUERY_FNS = [query_openai, query_claude, query_gemini]

def compute_scores(results):
    platforms = {}
    for r in results:
        if r is None or r.get("error"):
            continue
        p = platforms.setdefault(r["platform"],
            {"total":0,"mentioned":0,"positive":0,"neutral":0,"mixed":0,"negative":0})
        p["total"] += 1
        if r["mentioned"]:
            p["mentioned"] += 1
        s = r.get("sentiment")
        if s in p:
            p[s] += 1
    platform_scores = {
        pname: round(d["mentioned"] / d["total"] * 100) if d["total"] else 0
        for pname, d in platforms.items()
    }
    overall = round(sum(platform_scores.values()) / len(platform_scores)) if platform_scores else 0
    comp_counts = {}
    for r in results:
        if r:
            for c in r.get("competitors_mentioned", []):
                comp_counts[c] = comp_counts.get(c, 0) + 1
    return {"overall": overall, "platforms": platform_scores,
            "platform_details": platforms, "competitor_counts": comp_counts}

def cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/api/audit", methods=["POST", "OPTIONS"])
def audit():
    if request.method == "OPTIONS":
        return cors(make_response("", 200))
    try:
        data = request.get_json(force=True)
        if not data.get("client_name") or not data.get("industry") or not data.get("location"):
            raise ValueError("client_name, industry, and location are required.")
        prompts = generate_prompts(data)
        tasks   = [(fn, prompt) for prompt in prompts for fn in QUERY_FNS]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(fn, prompt, data) for fn, prompt in tasks]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r is not None:
                    results.append(r)
        scores  = compute_scores(results)
        return cors(jsonify({"results": results, "scores": scores, "prompts": prompts}))
    except Exception as e:
        return cors(jsonify({"error": str(e)})), 400
