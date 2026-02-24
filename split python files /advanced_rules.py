def apply_advanced_rules(bp_fuzz, chol_fuzz, hr_fuzz, age_fuzz, smoke_fuzz, diabetes_fuzz):
    # Original 6 rules 
    rule1 = min(bp_fuzz['Low'], chol_fuzz['Low'], hr_fuzz['Slow'])
    rule2 = min(bp_fuzz['Low'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule3 = min(bp_fuzz['Medium'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule4 = min(bp_fuzz['Medium'], chol_fuzz['High'], hr_fuzz['Slow'])
    rule5 = min(bp_fuzz['High'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule6 = min(bp_fuzz['High'], chol_fuzz['High'], hr_fuzz['Fast'])
    # additional factors)
    rule7 = age_fuzz['Old'] 
    rule8 = smoke_fuzz['Heavy'] 
    rule9 = diabetes_fuzz['Yes']  
    rule10 = min(age_fuzz['Middle'], smoke_fuzz['Light'])
    rule11 = min(age_fuzz['Old'], diabetes_fuzz['Pre'])
    rule12 = min(smoke_fuzz['Light'], diabetes_fuzz['Pre'])
    rule13 = min(age_fuzz['Old'], smoke_fuzz['Heavy'])
    rule14 = min(age_fuzz['Old'], diabetes_fuzz['Yes'])
    rule15 = min(smoke_fuzz['Heavy'], diabetes_fuzz['Yes'])

    rules = {'Healthy': [rule1, rule2],'Middle': [rule3, rule4, rule10, rule11, rule12],'Sick': [rule5, rule6, rule7, rule8, rule9, rule13, rule14, rule15]}
    return rules
