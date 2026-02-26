from app import app
import numpy as np

with app.test_client() as c:
    r = c.post('/api/analyze', json={'city':'Delhi NCR'})
    data = r.get_json()
    temps = data['historical']['temps']
    dates = data['historical']['dates']
    
    max_t = max(temps)
    idx = temps.index(max_t)
    print("Max historical returned:", max_t)
    print("Date:", dates[idx])
