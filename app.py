#!/usr/bin/env python3
# FutureWork AI web UI
from pathlib import Path
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from html import escape
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; PROCESSED=ROOT/'processed';RESULTS=ROOT/'results'

@lru_cache(maxsize=1)
def load_data():
    matrix=PROCESSED/'occupation_skill_matrix.csv';titles=PROCESSED/'occupation_lookup.csv'
    missing=[str(x) for x in [matrix,titles] if not x.exists()]
    if missing:raise FileNotFoundError('Upload your Phase 1 files: '+', '.join(missing))
    raw=pd.read_csv(matrix,dtype={'occupation_code':str})
    lookup=pd.read_csv(titles,dtype={'occupation_code':str}).drop_duplicates('occupation_code').set_index('occupation_code')
    x=raw.set_index('occupation_code').apply(pd.to_numeric,errors='coerce')
    x=x.loc[x.index.intersection(lookup.index)]
    eligible=x.notna().mean(axis=1)>=.60
    x=x.loc[eligible];lookup=lookup.loc[x.index]
    fill=x.fillna(x.median()).fillna(1)
    sim=pd.DataFrame(cosine_similarity(np.maximum(fill.to_numpy(dtype=float)-1,0)),index=x.index,columns=x.index)
    mappings=pd.read_csv(DATA/'verified_mapping.csv',dtype=str).fillna('')
    mappings=mappings[mappings.occupation_code.isin(x.index)&mappings.au_code.str.fullmatch(r'\d{6}')].drop_duplicates('occupation_code')
    reviewed=mappings.loc[mappings.reviewed.str.strip().str.lower()=='yes'].set_index('occupation_code')
    au=pd.read_csv(DATA/'au_occupation_titles.csv',dtype={'au_code':str}).dropna(subset=['au_code','au_title']).drop_duplicates('au_code')
    # Ranked mappings are automatically generated, NOT occupation equivalence verification.
    # All 910 have a suggested candidate, but some suggestions are wrong.
    queue=pd.read_csv(DATA/'futurework_910_review_queue.csv',dtype={'occupation_code':str,'au_code':str})
    queue=queue.drop_duplicates('occupation_code').set_index('occupation_code')
    bestcodes=pd.DataFrame(index=x.index)
    bestcodes['candidate_au_code']=queue['au_code'].reindex(x.index)
    bestcodes['candidate_au_title']=queue['au_title'].reindex(x.index)
    bestcodes['title_confidence']=pd.to_numeric(queue['title_similarity'].reindex(x.index),errors='coerce')
    bestcodes['review_priority']=queue['review_priority'].reindex(x.index).fillna('unknown')
    vic=pd.read_csv(DATA/'vic_anzsco4_workforce.csv',dtype={'anzsco4':str})
    vic['anzsco4']=vic.anzsco4.str.zfill(4)
    nums=['employment_2025','employment_growth_2025_2035','new_workers_2025_2028','new_workers_2025_2035']
    for col in nums: vic[col]=pd.to_numeric(vic[col],errors='coerce')
    regions=['Victoria']+sorted(vic.region.dropna().unique().tolist())
    # Optional names for skills
    dic={}
    dp=PROCESSED/'skill_dictionary.csv'
    if dp.exists():
        d=pd.read_csv(dp,dtype=str).fillna('')
        for _,row in d.iterrows():
            vals=row.tolist()
            if len(vals)>=2: dic[vals[0]]=vals[1]
    return x,lookup,sim,reviewed,bestcodes,vic,regions,nums,dic

@lru_cache(maxsize=20)
def region_table(region):
    x,look,sim,rev,best,vic,regs,nums,dic=load_data()
    filtered=vic if region=='Victoria' else vic[vic.region==region]
    table=filtered.groupby('anzsco4',as_index=True)[nums].sum(min_count=1)
    table['growth_pct']=np.where(table.employment_2025>0,100*table.employment_growth_2025_2035/table.employment_2025,np.nan)
    return table

def recommendations(code,region,limit=5,pool_size=80):
    x,lookup,sim,reviewed,best,vic,regions,nums,dic=load_data()
    if code not in sim.index:raise ValueError('Invalid occupation')
    series=sim.loc[code].drop(code).sort_values(ascending=False)
    table=region_table(region)
    rows=[]
    for i,(target,score) in enumerate(series.items()):
        family=target[:2]==code[:2]
        if target in reviewed.index:
            au_code=str(reviewed.loc[target,'au_code']);quality='Reviewed prototype';confidence=1.0
        else:
            entry=best.loc[target]
            au_code=str(entry.candidate_au_code) if pd.notna(entry.candidate_au_code) else ''
            confidence=float(entry.title_confidence) if pd.notna(entry.title_confidence) else 0.0
            quality='Unverified automatic candidate' if au_code and len(au_code)==6 else 'No mapping candidate'
        rec={'code':target,'title':str(lookup.loc[target,'occupation_title']),'similarity':float(score),
             'family':family,'quality':quality,'au_code':au_code,'title_confidence':confidence,
             'growth_pct':np.nan,'new_workers':np.nan,'group':''}
        group=au_code[:4]
        if quality!='No mapping candidate' and group in table.index:
            rec['growth_pct']=float(table.loc[group,'growth_pct']);rec['new_workers']=float(table.loc[group,'new_workers_2025_2028']);rec['group']=group
        rows.append(rec)
        if i>=pool_size and len(rows)>=pool_size+1:break
    top=rows[:limit]
    # Strict occupational-family guardrail: prioritize same major occupational group.
    available=[r for r in rows if r['family'] and np.isfinite(r['growth_pct']) and r['similarity']>=.90 and (r['quality']=='Reviewed prototype' or r['title_confidence']>=.75)]
    if available:
        growths=np.array([r['growth_pct'] for r in available]);lo,hi=np.min(growths),np.max(growths)
        for r in available:
            rank=float((r['growth_pct']-lo)/(hi-lo)) if hi>lo else .5
            r['score']=.8*r['similarity']+.2*rank
        available.sort(key=lambda r:r['score'],reverse=True)
    return top,available[:limit],series

def skill_differences(source,target,n=7):
    x,*_=load_data();src=x.loc[source];tgt=x.loc[target]
    both=pd.DataFrame({'source':src,'target':tgt}).dropna();both['delta']=both.target-both.source
    return both.sort_values('delta',ascending=False).head(n)

def html_page(code=None,region='Victoria',chosen=None):
    x,lookup,sim,rev,best,vic,regions,nums,dictskills=load_data()
    if code not in lookup.index:code=str(lookup.index[0])
    if region not in regions:region='Victoria'
    options=''.join(f'<option value="{escape(str(lookup.loc[c,"occupation_title"])+" | "+str(c))}"></option>' for c in lookup.index)
    region_options=''.join(f'<option value="{escape(r)}" {"selected" if r==region else ""}>{escape(r)}</option>' for r in regions)
    top,market,series=recommendations(code,region)
    def fmt(v,unit=''):
        return f'{v:,.1f}{unit}' if np.isfinite(v) else 'Unavailable'
    def row_html(r,i):
        name=escape(r['title']);badge='Reviewed prototype' if r['quality']=='Reviewed prototype' else ('AUTO—UNVERIFIED' if r['quality']=='Unverified automatic candidate' else 'Unavailable')
        growth=fmt(r['growth_pct'],'%')
        return f'<tr><td>{i+1}</td><td><strong>{name}</strong><span class="muted">{escape(r["code"])}</span></td><td>{r["similarity"]:.3f}</td><td>{growth}</td><td><small>{badge}</small></td></tr>'
    top_table=''.join(row_html(r,i) for i,r in enumerate(top))
    market_table=''.join(row_html(r,i) for i,r in enumerate(market))
    market_note=('<p class="notice"><strong>EXPLORATORY ONLY:</strong> Market-linked careers are restricted to the same broad O*NET occupational family. AUTO—UNVERIFIED means the Australian occupation title was algorithmically selected, NOT confirmed. Its group projections may be wrong for the suggested job; do not use for personal career decisions without checking occupational duties, qualifications and the Australian classification.</p>' if market else '<p class="notice">No adequately mapped, closely related occupations found within the current search pool. We do not substitute unrelated professions.</p>')
    target=chosen if chosen in [r['code'] for r in top] else top[0]['code']
    target_options=''.join(f'<option value="{r["code"]}" {"selected" if target==r["code"] else ""}>{escape(r["title"])}</option>' for r in top)
    gaps=skill_differences(code,target)
    gaprows=''.join(f'<tr><td>{escape(dictskills.get(str(idx),str(idx)))}</td><td>{row.source:.2f}</td><td>{row.target:.2f}</td><td>{row.delta:+.2f}</td></tr>' for idx,row in gaps.iterrows())
    select_value=escape(str(lookup.loc[code,'occupation_title'])+' | '+code)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FutureWork AI — Career Navigator</title>
<style>
:root{{--bg:#0b1220;--panel:#152238;--text:#eaf3ff;--sub:#a9bed8;--cyan:#4be1d4;--line:#30455d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,Segoe UI,sans-serif}}a{{color:var(--cyan)}}header{{padding:25px 5%;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}}main{{max-width:1150px;margin:auto;padding:26px 18px}}.logo{{font-weight:800;font-size:24px}}.logo span{{color:var(--cyan)}}.tag{{color:#a2f6ed;font-size:12px;letter-spacing:1px}}h1{{font-size:34px;margin:7px 0}}h2{{font-size:21px}}.intro{{color:var(--sub);max-width:850px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:20px}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}form{{display:flex;flex-wrap:wrap;gap:12px;align-items:end}}label{{display:block;font-size:13px;color:var(--sub);margin-bottom:6px}}input,select{{width:100%;background:#0a172a;color:white;border:1px solid #476076;padding:12px;border-radius:8px;font-size:15px}}.field{{flex:1;min-width:235px}}button{{background:var(--cyan);color:#10202d;border:0;padding:13px 20px;border-radius:8px;font-weight:750;cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:11px 8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--sub)}}.muted{{display:block;color:var(--sub);font-size:11px;margin-top:3px}}.notice{{background:#21334a;padding:13px;border-left:3px solid var(--cyan);border-radius:4px;line-height:1.45;color:#dbe7f4}}.footer{{color:var(--sub);font-size:12px;margin-top:35px}}.pill{{font-size:12px;padding:7px 10px;background:#143e43;color:var(--cyan);border-radius:999px}}@media(max-width:830px){{.cols{{grid-template-columns:1fr}}h1{{font-size:26px}}}}
</style></head><body><header><div class="logo">Future<span>Work</span> AI</div><div class="tag">IBM LINUXONE • TECH FOR GOOD</div></header><main><span class="pill">Career transition navigator</span><h1>Find your next opportunity.</h1><p class="intro">Explore 910 O*NET skill profiles. See Victorian occupational-group projections attached to automatic Australian mapping candidates; unverified matches are exploratory only.</p>
<section class="card"><form method="get" action="/"><div class="field"><label for="occupation">Current occupation (start typing to search)</label><input id="occupation" name="occupation" list="jobs" required value="{select_value}"><datalist id="jobs">{options}</datalist></div><div class="field"><label for="region">Victorian region</label><select name="region" id="region">{region_options}</select></div><button type="submit">Explore careers →</button></form><p class="muted">Data: O*NET occupational skills; Jobs and Skills Australia mapping context; Victorian Skills Authority projections 2025–2035</p></section>
<div class="cols"><section class="card"><h2>Closest skill matches</h2><p class="intro">Ranked by similarity of occupational skill requirements, not probability of employment.</p><table><thead><tr><th>#</th><th>Occupation</th><th>Skill match</th><th>VIC group growth</th><th>Mapping</th></tr></thead><tbody>{top_table}</tbody></table></section>
<section class="card"><h2>Market-linked career options</h2><p class="intro">Relevant alternatives with mapped Victorian occupation-group projections. Region: <b>{escape(region)}</b>.</p>{market_note}<table><thead><tr><th>#</th><th>Occupation</th><th>Skill match</th><th>Group growth</th><th>Mapping</th></tr></thead><tbody>{market_table}</tbody></table></section></div>
<section class="card"><h2>Skills to explore</h2><form method="get"><input type="hidden" name="occupation" value="{select_value}"><input type="hidden" name="region" value="{escape(region)}"><div class="field"><label>Compare with a top skill match</label><select name="target">{target_options}</select></div><button type="submit">Compare skills</button></form><p class="intro">Skills more important in the target occupation (O*NET importance ratings). These are not personal skill deficiencies.</p><table><thead><tr><th>Skill dimension</th><th>Current job</th><th>Target job</th><th>Difference</th></tr></thead><tbody>{gaprows}</tbody></table></section>
<p class="footer">Decision support only. Similarity doesn't establish qualification, registration, employability, pay, vacancies or hiring chances. Victorian figures refer to broader ANZSCO four-digit occupation groups, NOT specific jobs or vacancies. Automatic O*NET-to-Australian mappings are UNVERIFIED guesses, sometimes incorrect. Only four mappings have received prototype review; no government-approved crosswalk is claimed. © FutureWork AI Datathon prototype.</p></main></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse,parse_qs
        parsed=urlparse(self.path)
        if parsed.path=='/health':
            data=b'OK';self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        if parsed.path not in ('/','/index.html'):
            self.send_error(404);return
        params=parse_qs(parsed.query);label=params.get('occupation',[''])[0];region=params.get('region',['Victoria'])[0]
        code=label.rsplit(' | ',1)[-1]
        target=params.get('target',[None])[0]
        try:
            content=html_page(code,region,target).encode('utf-8');status=200
        except Exception as exc:
            content=(f'<h2>Unable to start FutureWork AI</h2><p>{escape(str(exc))}</p><p>Place Phase 1 CSV files in processed/ and data CSVs in data/.</p>').encode();status=500
        self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
    def log_message(self,format,*args):return

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8899);args=parser.parse_args()
    load_data();print(f'FutureWork AI UI ready on http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
