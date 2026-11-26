from pathlib import Path
import json, joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
ROOT=Path(__file__).resolve().parent
models=joblib.load(ROOT/'models/model.joblib'); metrics=json.loads((ROOT/'models/metrics.json').read_text())
app=FastAPI(title='EV Range Lab',version='1.0',description='Experimental range and battery health estimates from synthetic training data.')
class Inputs(BaseModel):
 model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
 age_years:float=Field(ge=0,le=8)
 cycles:int=Field(ge=0,le=1800)
 fast_charge_pct:float=Field(ge=0,le=100)
 avg_battery_temp_c:float=Field(ge=20,le=45)
 capacity_kwh:float=Field(ge=2,le=5)
 soc_pct:float=Field(ge=0,le=100)
 ambient_temp_c:float=Field(ge=10,le=45)
 speed_kmh:float=Field(ge=15,le=85)
 payload_kg:float=Field(ge=45,le=180)
 traffic:int=Field(ge=0,le=2)
 eco_mode:int=Field(ge=0,le=1)
 trip_km:float=Field(default=0,ge=0,le=500)
@app.get('/',include_in_schema=False)
def home(): return FileResponse(ROOT/'static/index.html')
@app.get('/health')
def health(): return {'status':'ok','model_ready':True}
@app.get('/api/metrics')
def model_metrics(): return metrics
@app.post('/api/predict')
def predict(payload:Inputs):
 d=payload.model_dump(); result={}
 for target,entry in models.items():
  row=pd.DataFrame([d],columns=entry['features'])
  if target=='range_km': row['soc_pct']=max(5,d['soc_pct'])
  pred=float(entry['model'].predict(row)[0]); q=entry['q']; lower=max(0,pred-q); upper=max(0,pred+q)
  if target=='soh_pct': pred=min(100,max(0,pred)); lower=min(100,lower); upper=min(100,upper)
  elif d['soc_pct']==0: pred=lower=upper=0
  elif d['soc_pct']<5: pred*=d['soc_pct']/5; lower=0; upper*=d['soc_pct']/5
  result[target]={'estimate':round(max(0,pred),1),'lower':round(lower,1),'upper':round(upper,1)}
 reserve=round(result['range_km']['lower']*.8,1)
 result['trip']={'distance_km':d['trip_km'],'planning_range_km':reserve,'status':'not_requested' if not d['trip_km'] else ('within_demo_budget' if d['trip_km']<=reserve else 'charging_needed_or_replan')}
 result['warnings']=['Simulation-trained estimates; not validated on real scooters. Use manufacturer/BMS range and an independent charging plan for travel.']
 if d['soc_pct']<5: result['warnings'].append('Below training domain; range extrapolated and interval is not calibrated.')
 return result
