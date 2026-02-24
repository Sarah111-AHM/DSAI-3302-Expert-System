
def generate_training_data(n_samples=200):
    np.random.seed(42)  
    data = []
    for _ in range(n_samples):
        bp = np.random.uniform(90, 210)
        chol = np.random.uniform(100, 300)
        hr = np.random.uniform(50, 150)
        age = np.random.uniform(20, 90)
        smoking = np.random.uniform(0, 2) 
        diabetes = np.random.uniform(70, 300)  # glucose level      
        risk = 0

        if bp < 120:
            risk += 0.5
        elif bp < 140:
            risk += 1.5
        else:
            risk += 2.5

        if chol < 180:
            risk += 0.5
        elif chol < 240:
            risk += 1.5
        else:
            risk += 2.5

        if hr < 60:
            risk += 0.5
        elif hr < 100:
            risk += 1.0
        else:
            risk += 2.0
            
        if age < 40:
            risk += 0.3
        elif age < 60:
            risk += 1.2
        else:
            risk += 2.0
            
        if smoking < 0.2:
            risk += 0.2
        elif smoking < 1:
            risk += 1.5
        else:
            risk += 2.5

        if diabetes < 100:
            risk += 0.3
        elif diabetes < 126:
            risk += 1.3
        else:
            risk += 2.3

        chd = min(4.0, risk / 2.5)
        data.append([bp, chol, hr, age, smoking, diabetes, chd])
    
    return np.array(data)
