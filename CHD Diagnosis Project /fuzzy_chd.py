%%writefile fuzzy_chd.py
import numpy as np
import matplotlib.pyplot as plt

def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:
        return 0

def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)
    else:
        return 0

def indeed(mu):
    return mu ** 2

def somewhat(mu):
    return np.sqrt(mu)

def fuzzify_bp(bp):
    return {
        'Low': triangular(bp, 100, 115, 130),
        'Medium': triangular(bp, 120, 145, 170),
        'High': triangular(bp, 160, 180, 200)
    }

def fuzzify_chol(chol):
    return {
        'Low': trapezoidal(chol, 100, 120, 180, 200),
        'High': trapezoidal(chol, 180, 200, 260, 280)
    }

def fuzzify_hr(hr):
    return {
        'Slow': triangular(hr, 50, 65, 80),
        'Moderate': triangular(hr, 70, 85, 100),
        'Fast': triangular(hr, 90, 145, 200)
    }

def apply_rules(bp_fuzz, chol_fuzz, hr_fuzz, use_hedges=False, hedge_type=None, rule_to_modify=None):
    rule1 = min(bp_fuzz['Low'], chol_fuzz['Low'], hr_fuzz['Slow'])
    rule2 = min(bp_fuzz['Low'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule3 = min(bp_fuzz['Medium'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule4 = min(bp_fuzz['Medium'], chol_fuzz['High'], hr_fuzz['Slow'])
    rule5 = min(bp_fuzz['High'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule6 = min(bp_fuzz['High'], chol_fuzz['High'], hr_fuzz['Fast'])
    
    rules = {
        'Healthy': [rule1, rule2],
        'Middle': [rule3, rule4],
        'Sick': [rule5, rule6]
    }
    
    if use_hedges and hedge_type and rule_to_modify:
        rule_map = {
            1: ('Healthy', 0),
            2: ('Healthy', 1),
            3: ('Middle', 0),
            4: ('Middle', 1),
            5: ('Sick', 0),
            6: ('Sick', 1)
        }
        if rule_num in rule_map:
            category, idx = rule_map[rule_num]
            original_value = rules[category][idx]
            if hedge_type == 'indeed':
                rules[category][idx] = indeed(original_value)
            elif hedge_type == 'somewhat':
                rules[category][idx] = somewhat(original_value)
    
    return rules

def aggregate_rules(rules):
    return {
        'Healthy': max(rules['Healthy']),
        'Middle': max(rules['Middle']),
        'Sick': max(rules['Sick'])
    }

HEALTHY_CENTER = 0.75
MIDDLE_CENTER = 2.0
SICK_CENTER = 3.25

def defuzzify_cog(aggregated):
    numerator = (aggregated['Healthy'] * HEALTHY_CENTER +
                 aggregated['Middle'] * MIDDLE_CENTER +
                 aggregated['Sick'] * SICK_CENTER)
    denominator = (aggregated['Healthy'] +
                   aggregated['Middle'] +
                   aggregated['Sick'])
    if denominator == 0:
        return 0
    return numerator / denominator

def defuzzify_sugeno(rules):
    all_rules = []
    all_values = []
    for strength in rules['Healthy']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(HEALTHY_CENTER)
    for strength in rules['Middle']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(MIDDLE_CENTER)
    for strength in rules['Sick']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(SICK_CENTER)
    if not all_rules:
        return 0
    numerator = sum(r * v for r, v in zip(all_rules, all_values))
    denominator = sum(all_rules)
    return numerator / denominator

def diagnose_patient(bp, chol, hr, use_hedges=False, hedge_type=None, rule_to_modify=None, verbose=False):
    bp_fuzz = fuzzify_bp(bp)
    chol_fuzz = fuzzify_chol(chol)
    hr_fuzz = fuzzify_hr(hr)
    rules = apply_rules(bp_fuzz, chol_fuzz, hr_fuzz, use_hedges, hedge_type, rule_to_modify)
    aggregated = aggregate_rules(rules)
    cog_result = defuzzify_cog(aggregated)
    sugeno_result = defuzzify_sugeno(rules)
    details = {
        'bp_fuzz': bp_fuzz,
        'chol_fuzz': chol_fuzz,
        'hr_fuzz': hr_fuzz,
        'rules': rules,
        'aggregated': aggregated
    }
    return cog_result, sugeno_result, details
