from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
SAMPLE=dict(capacity_kwh=3.7,soc_pct=75,age_years=2,cycles=450,fast_charge_pct=20,avg_battery_temp_c=30,ambient_temp_c=28,speed_kmh=35,payload_kg=80,traffic=1,eco_mode=1,trip_km=30)
def test_home(): assert client.get('/').status_code==200
def test_health(): assert client.get('/health').json()['model_ready']
def test_metrics(): assert client.get('/api/metrics').json()['rows']==12000
def test_prediction():
 r=client.post('/api/predict',json=SAMPLE); assert r.status_code==200
 d=r.json(); assert 0<d['range_km']['estimate'] and 0<d['soh_pct']['estimate']<=100
 assert d['range_km']['lower']<=d['range_km']['estimate']<=d['range_km']['upper']
 assert d['trip']['planning_range_km']<=d['range_km']['estimate']
def test_zero_charge(): assert client.post('/api/predict',json={**SAMPLE,'soc_pct':0}).json()['range_km']['estimate']==0
def test_low_charge(): assert 'extrapolated' in client.post('/api/predict',json={**SAMPLE,'soc_pct':2}).json()['warnings'][1]
def test_invalid(): assert client.post('/api/predict',json={**SAMPLE,'soc_pct':101}).status_code==422
def test_trip_long(): assert client.post('/api/predict',json={**SAMPLE,'trip_km':500}).json()['trip']['status']=='charging_needed_or_replan'
