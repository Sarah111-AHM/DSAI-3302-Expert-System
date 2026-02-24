def aggregate_rules_mamdani(rules):
    return {'Healthy': max(rules['Healthy']),'Middle': max(rules['Middle']),'Sick': max(rules['Sick'])}

def defuzzify_mamdani_cog(aggregated):
    numerator = (aggregated['Healthy'] * HEALTHY_CENTER + aggregated['Middle'] * MIDDLE_CENTER + aggregated['Sick'] * SICK_CENTER)
    denominator = (aggregated['Healthy'] + aggregated['Middle'] + aggregated['Sick'])

    if denominator == 0:
        return 0
    return numerator / denominator

def defuzzify_sugeno_weighted_average(rules):
    all_rules = []
    all_values = []
    for strength in rules['Healthy']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(HEALTHY_CENTER)
    # Middle rules 
    for strength in rules['Middle']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(MIDDLE_CENTER)
    # Sick rules 
    for strength in rules['Sick']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(SICK_CENTER)
    if not all_rules:
        return 0
    # Weighted average
    numerator = sum(r * v for r, v in zip(all_rules, all_values))
    denominator = sum(all_rules)
    return numerator / denominator
