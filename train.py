"""Synthetic scooter data; not real telemetry."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parent
HEALTH=['age_years','cycles','fast_charge_pct','avg_battery_temp_c']
RANGE=HEALTH+['capacity_kwh','soc_pct','ambient_temp_c','speed_kmh','payload_kg','traffic','eco_mode']
def generate(n=12000,seed=42):
 r=np.random.default_rng(seed)
 d=pd.DataFrame(dict(age_years=r.uniform(0,8,n),cycles=r.integers(0,1801,n),fast_charge_pct=r.uniform(0,100,n),avg_battery_temp_c=r.uniform(20,45,n),capacity_kwh=r.uniform(2,5,n),soc_pct=r.uniform(5,100,n),ambient_temp_c=r.uniform(10,45,n),speed_kmh=r.uniform(15,85,n),payload_kg=r.uniform(45,180,n),traffic=r.integers(0,3,n),eco_mode=r.integers(0,2,n)))
 soh=np.clip(100-1.35*d.age_years-.012*d.cycles-.000055*d.cycles*d.fast_charge_pct-.13*np.maximum(d.avg_battery_temp_c-30,0)**1.3+r.normal(0,1.7,n),40,100)
 wh=24+.005*(d.speed_kmh-30)**2+.075*(d.payload_kg-65)+3.4*d.traffic+.20*np.abs(d.ambient_temp_c-26)-3*d.eco_mode+r.normal(0,1.4,n)
 d['soh_pct']=soh; d['range_km']=1000*d.capacity_kwh*d.soc_pct/100*soh/100/np.maximum(wh,15)
 return d.round(4)
def scores(y,p):
 return {k:round(float(v),4) for k,v in dict(mae=mean_absolute_error(y,p),rmse=np.sqrt(mean_squared_error(y,p)),r2=r2_score(y,p)).items()}
def main():
 (ROOT/'data').mkdir(exist_ok=True); (ROOT/'models').mkdir(exist_ok=True)
 d=generate(); d.to_csv(ROOT/'data/synthetic_ev.csv',index=False)
 train,other=train_test_split(d,test_size=.4,random_state=42)
 val,other=train_test_split(other,test_size=.5,random_state=43)
 cal,test=train_test_split(other,test_size=.5,random_state=44)
 out={'dataset':'Synthetic simulation only','rows':len(d),'split':{'train':len(train),'validation':len(val),'calibration':len(cal),'test':len(test)},'seed':42,'targets':{}}; models={}
 for target,features in [('range_km',RANGE),('soh_pct',HEALTH)]:
  candidates={'Linear regression':LinearRegression(),'Random forest':RandomForestRegressor(n_estimators=60,min_samples_leaf=3,n_jobs=2,random_state=42),'Gradient boosting':HistGradientBoostingRegressor(max_iter=180,learning_rate=.09,random_state=42)}
  validation={}
  for name,m in candidates.items():
   m.fit(train[features],train[target]); validation[name]=scores(val[target],m.predict(val[features]))
  winner=min(validation,key=lambda name:validation[name]['mae']); m=candidates[winner]
  combined=pd.concat([train,val]); m.fit(combined[features],combined[target])
  err=np.abs(cal[target]-m.predict(cal[features])); q=float(np.sort(err)[min(len(err)-1,int(np.ceil((len(err)+1)*.9))-1)])
  p=m.predict(test[features]); out['targets'][target]={'selected_model':winner,'validation':validation,'test':scores(test[target],p),'interval_half_width':round(q,3),'test_interval_coverage':round(float(np.mean(np.abs(test[target]-p)<=q)),4)}
  models[target]={'model':m,'features':features,'q':q}
 joblib.dump(models,ROOT/'models/model.joblib',compress=3); (ROOT/'models/metrics.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
