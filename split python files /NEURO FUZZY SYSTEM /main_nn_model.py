
class NeuroFuzzyCHD:
    def __init__(self):
        self.membership_params = {}
        self.rule_weights = {}
        self.trained = False
        
    def initialize_membership_functions(self):
        self.membership_params['bp'] = {'Low': [90, 115, 140],'Medium': [120, 145, 170],'High': [150, 180, 210]}
        
        self.membership_params['chol'] = {'Low': [90, 150, 210],'High': [170, 230, 300]}
        
        self.membership_params['hr'] = {'Slow': [40, 60, 80],'Moderate': [65, 85, 105],'Fast': [90, 140, 180]}
        
        self.membership_params['age'] = {'Young': [0, 25, 40],'Middle': [30, 45, 60],'Old': [50, 70, 100]}
        
        self.membership_params['smoking'] = {'None': [-0.2, 0, 0.2],'Light': [0, 0.3, 0.7],'Heavy': [0.5, 1.5, 3]}

        self.membership_params['diabetes'] = {'No': [60, 80, 100],'Pre': [90, 110, 130],'Yes': [115, 200, 350]}
        
        for i in range(15):
            self.rule_weights[i] = 1.0

    def triangular_mf(self, x, a, b, c):
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

    def fuzzify_input(self, X):
        bp, chol, hr, age, smoking, diabetes = X 
        memberships = []
        # BP memberships
        memberships.append(self.triangular_mf(bp, *self.membership_params['bp']['Low']))
        memberships.append(self.triangular_mf(bp, *self.membership_params['bp']['Medium']))
        memberships.append(self.triangular_mf(bp, *self.membership_params['bp']['High']))
        # Chol memberships
        memberships.append(self.triangular_mf(chol, *self.membership_params['chol']['Low']))
        memberships.append(self.triangular_mf(chol, *self.membership_params['chol']['High']))
        # HR memberships
        memberships.append(self.triangular_mf(hr, *self.membership_params['hr']['Slow']))
        memberships.append(self.triangular_mf(hr, *self.membership_params['hr']['Moderate']))
        memberships.append(self.triangular_mf(hr, *self.membership_params['hr']['Fast']))
        # Age memberships
        memberships.append(self.triangular_mf(age, *self.membership_params['age']['Young']))
        memberships.append(self.triangular_mf(age, *self.membership_params['age']['Middle']))
        memberships.append(self.triangular_mf(age, *self.membership_params['age']['Old']))
        # Smoking memberships
        memberships.append(self.triangular_mf(smoking, *self.membership_params['smoking']['None']))
        memberships.append(self.triangular_mf(smoking, *self.membership_params['smoking']['Light']))
        memberships.append(self.triangular_mf(smoking, *self.membership_params['smoking']['Heavy']))
        # Diabetes memberships
        memberships.append(self.triangular_mf(diabetes, *self.membership_params['diabetes']['No']))
        memberships.append(self.triangular_mf(diabetes, *self.membership_params['diabetes']['Pre']))
        memberships.append(self.triangular_mf(diabetes, *self.membership_params['diabetes']['Yes']))
        return np.array(memberships)
    
    def apply_rules(self, memberships): 
        r1 = min(memberships[0], memberships[3], memberships[5]) * self.rule_weights[0]         # Rule 1: Low BP & Low Chol & Slow HR 
        r2 = min(memberships[0], memberships[3], memberships[6]) * self.rule_weights[1]         # Rule 2: Low BP & Low Chol & Moderate HR 
        r3 = min(memberships[1], memberships[3], memberships[6]) * self.rule_weights[2]         # Rule 3: Medium BP & Low Chol & Moderate HR 
        r4 = min(memberships[1], memberships[4], memberships[5]) * self.rule_weights[3]         # Rule 4: Medium BP & High Chol & Slow HR 
        r5 = min(memberships[2], memberships[3], memberships[6]) * self.rule_weights[4]         # Rule 5: High BP & Low Chol & Moderate HR 
        r6 = min(memberships[2], memberships[4], memberships[7]) * self.rule_weights[5]         # Rule 6: High BP & High Chol & Fast HR 
        r7 = memberships[10] * self.rule_weights[6]                                             # Rule 7: Old Age 
        r8 = memberships[13] * self.rule_weights[7]                                             # Rule 8: Heavy Smoking 
        r9 = memberships[16] * self.rule_weights[8]                                             # Rule 9: Diabetic 
        r10 = min(memberships[9], memberships[12]) * self.rule_weights[9]                       # Rule 10: Middle Age & Light Smoking 
        r11 = min(memberships[10], memberships[15]) * self.rule_weights[10]                     # Rule 11: Old Age & Pre-diabetic
        r12 = min(memberships[12], memberships[15]) * self.rule_weights[11]                     # Rule 12: Light Smoking & Pre-diabetic 
        r13 = min(memberships[10], memberships[13]) * self.rule_weights[12]                     # Rule 13: Old Age & Heavy Smoking
        r14 = min(memberships[10], memberships[16]) * self.rule_weights[13]                     # Rule 14: Old Age & Diabetic
        r15 = min(memberships[13], memberships[16]) * self.rule_weights[14]                     # Rule 15: Heavy Smoking & Diabetic 
        
        healthy_center = 0.75
        middle_center = 2.0
        sick_center = 3.25
        
        healthy_rules = [r1, r2]
        middle_rules = [r3, r4, r10, r11, r12]
        sick_rules = [r5, r6, r7, r8, r9, r13, r14, r15]
        numerator = (sum(healthy_rules) * healthy_center +sum(middle_rules) * middle_center +sum(sick_rules) * sick_center)
        denominator = (sum(healthy_rules) + sum(middle_rules) + sum(sick_rules))

        if denominator == 0:
            return 0
        return numerator / denominator
    
    def predict(self, X):
        if len(X.shape) == 1:
            memberships = self.fuzzify_input(X)
            return self.apply_rules(memberships)
        else:
            results = []
            for i in range(X.shape[0]):
                memberships = self.fuzzify_input(X[i])
                results.append(self.apply_rules(memberships))
            return np.array(results)

    def train_neuro_fuzzy(self, X, y, epochs=50, learning_rate=0.01):

        print("\n Training Neuro-Fuzzy System...")
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(X.shape[0]):
                # Forward 
                memberships = self.fuzzify_input(X[i])
                y_pred = self.apply_rules(memberships)
                # error
                error = y[i] - y_pred
                epoch_loss += error ** 2
                for rule_idx in self.rule_weights:
                    self.rule_weights[rule_idx] += learning_rate * error * 0.01
            total = sum(self.rule_weights.values())
            for rule_idx in self.rule_weights:
                self.rule_weights[rule_idx] /= total
            losses.append(epoch_loss / X.shape[0])
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/X.shape[0]:.4f}")
        self.trained = True
        return losses
