from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Standalone synthetic smoke test: NO real outcomes are claimed.
codes=['15-1252.00','15-1254.00','15-1211.00','15-1241.00','15-1243.00']
X=pd.DataFrame({'skill_a':[4,4,3.5,3.8,3.9],'skill_b':[3,3.1,4,3.2,3.3],'skill_c':[3,3.5,3,3.2,3.0]},index=codes)
SIM=pd.DataFrame(cosine_similarity(np.maximum(X.values-1,0)),index=codes,columns=codes)
assert (SIM.values>=-1e-9).all() and (SIM.values<=1.0000001).all()
markets=pd.read_csv(Path(__file__).resolve().parent.parent/'data'/'au_market_signals.csv',dtype={'au_code':str})
map_=pd.read_csv(Path(__file__).resolve().parent.parent/'data'/'verified_mapping.csv',dtype={'occupation_code':str,'au_code':str})
approved=map_[map_.reviewed.str.lower()=='yes']
assert len(approved)==4, f'Expected 4 prototype-reviewed mappings; got {len(approved)}'
assert not approved.occupation_code.duplicated().any()
joined=approved.merge(markets,on=['au_code','au_classification'],validate='one_to_one',how='left')
assert joined.growth_pct_10yr.notna().sum()>=3
unknown=set(codes)-set(approved.occupation_code)
assert unknown, 'Synthetic case should include at least one unmapped occupation'
print('PASS — synthetic scoring constraints, 4 reviewed mappings, missing-data separation')
