import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# =============================================================================
# PART 1: ENHANCED MEMBERSHIP FUNCTIONS (with new factors)
# =============================================================================

def triangular(x, a, b, c):
    """Triangular membership function"""
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0

def trapezoidal(x, a, b, c, d):
    """Trapezoidal membership function"""
    if x <= a or x >= d:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)
    return 0

# =============================================================================
# NEW FACTOR 1: AGE Membership Functions
# =============================================================================

def fuzzify_age(age):
    """
    Age categories (medical guidelines):
        Young: < 30
        Middle-aged: 30-55
        Old: > 50
    """
    return {
        'Young': trapezoidal(age, 0, 0, 25, 35),
        'Middle': trapezoidal(age, 30, 40, 50, 60),
        'Old': trapezoidal(age, 50, 65, 100, 120)
    }

# =============================================================================
# NEW FACTOR 2: SMOKING Status
# =============================================================================

def fuzzify_smoking(packs_per_day):
    """
    Smoking intensity (packs per day):
        None: 0 packs
        Light: 0-0.5 packs
        Heavy: >0.5 packs
    """
    return {
        'None': trapezoidal(packs_per_day, -0.2, 0, 0, 0.2),
        'Light': triangular(packs_per_day, 0, 0.3, 0.7),
        'Heavy': trapezoidal(packs_per_day, 0.5, 1, 3, 4)
    }

# =============================================================================
# NEW FACTOR 3: DIABETES Status
# =============================================================================

def fuzzify_diabetes(glucose_level):
    """
    Diabetes based on blood glucose (mg/dL):
        No: < 100
        Pre-diabetic: 100-125
        Diabetic: > 125
    """
    return {
        'No': trapezoidal(glucose_level, 0, 0, 90, 110),
        'Pre': triangular(glucose_level, 100, 112.5, 125),
        'Yes': trapezoidal(glucose_level, 120, 150, 300, 400)
    }

# =============================================================================
# ORIGINAL FACTORS (from Phase 8)
# =============================================================================

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

# Output centers
HEALTHY_CENTER = 0.75
MIDDLE_CENTER = 2.0
SICK_CENTER = 3.25

# =============================================================================
# PART 2: ENHANCED RULE BASE (with new factors)
# =============================================================================

def apply_advanced_rules(bp_fuzz, chol_fuzz, hr_fuzz, age_fuzz, smoke_fuzz, diabetes_fuzz):
    """
    Enhanced rule base with 15 rules (6 original + 9 new)

    NEW RULES (based on medical knowledge):
    7. IF Age is Old THEN CHD is Sick (age risk)
    8. IF Smoking is Heavy THEN CHD is Sick (smoking risk)
    9. IF Diabetes is Yes THEN CHD is Sick (diabetes risk)
    10. IF Age is Middle AND Smoking is Light THEN CHD is Middle
    11. IF Age is Old AND Diabetes is Pre THEN CHD is Middle
    12. IF Smoking is Light AND Diabetes is Pre THEN CHD is Middle
    13. IF Age is Old AND Smoking is Heavy THEN CHD is Sick
    14. IF Age is Old AND Diabetes is Yes THEN CHD is Sick
    15. IF Smoking is Heavy AND Diabetes is Yes THEN CHD is Sick
    """

    # Original 6 rules (from Table 2)
    rule1 = min(bp_fuzz['Low'], chol_fuzz['Low'], hr_fuzz['Slow'])
    rule2 = min(bp_fuzz['Low'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule3 = min(bp_fuzz['Medium'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule4 = min(bp_fuzz['Medium'], chol_fuzz['High'], hr_fuzz['Slow'])
    rule5 = min(bp_fuzz['High'], chol_fuzz['Low'], hr_fuzz['Moderate'])
    rule6 = min(bp_fuzz['High'], chol_fuzz['High'], hr_fuzz['Fast'])

    # NEW RULES (based on additional factors)
    rule7 = age_fuzz['Old']  # Age only
    rule8 = smoke_fuzz['Heavy']  # Smoking only
    rule9 = diabetes_fuzz['Yes']  # Diabetes only

    rule10 = min(age_fuzz['Middle'], smoke_fuzz['Light'])
    rule11 = min(age_fuzz['Old'], diabetes_fuzz['Pre'])
    rule12 = min(smoke_fuzz['Light'], diabetes_fuzz['Pre'])

    rule13 = min(age_fuzz['Old'], smoke_fuzz['Heavy'])
    rule14 = min(age_fuzz['Old'], diabetes_fuzz['Yes'])
    rule15 = min(smoke_fuzz['Heavy'], diabetes_fuzz['Yes'])

    # Organize rules by output category
    rules = {
        'Healthy': [rule1, rule2],
        'Middle': [rule3, rule4, rule10, rule11, rule12],
        'Sick': [rule5, rule6, rule7, rule8, rule9, rule13, rule14, rule15]
    }

    return rules

# =============================================================================
# PART 3: INFERENCE ENGINE (Mamdani vs Sugeno)
# =============================================================================

def aggregate_rules_mamdani(rules):
    """
    Mamdani aggregation: MAX operator on fuzzy sets
    Returns membership degrees for each output category
    """
    return {
        'Healthy': max(rules['Healthy']),
        'Middle': max(rules['Middle']),
        'Sick': max(rules['Sick'])
    }

def defuzzify_mamdani_cog(aggregated):
    """Centroid of Gravity for Mamdani inference"""
    numerator = (aggregated['Healthy'] * HEALTHY_CENTER +
                 aggregated['Middle'] * MIDDLE_CENTER +
                 aggregated['Sick'] * SICK_CENTER)

    denominator = (aggregated['Healthy'] +
                   aggregated['Middle'] +
                   aggregated['Sick'])

    if denominator == 0:
        return 0
    return numerator / denominator

def defuzzify_sugeno_weighted_average(rules):
    """
    Sugeno inference: weighted average of rule outputs
    (uses original firing strengths, not aggregated)
    """
    all_rules = []
    all_values = []

    # Healthy rules (centers = 0.75)
    for strength in rules['Healthy']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(HEALTHY_CENTER)

    # Middle rules (centers = 2.0)
    for strength in rules['Middle']:
        if strength > 0:
            all_rules.append(strength)
            all_values.append(MIDDLE_CENTER)

    # Sick rules (centers = 3.25)
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

# =============================================================================
# PART 4: COMPLETE DIAGNOSIS FUNCTION (with all features)
# =============================================================================

def diagnose_patient_advanced(bp, chol, hr, age, smoking, diabetes, verbose=False):
    """
    Complete diagnosis with all 6 factors
    """
    # Fuzzification
    bp_fuzz = fuzzify_bp(bp)
    chol_fuzz = fuzzify_chol(chol)
    hr_fuzz = fuzzify_hr(hr)
    age_fuzz = fuzzify_age(age)
    smoke_fuzz = fuzzify_smoking(smoking)
    diabetes_fuzz = fuzzify_diabetes(diabetes)

    # Apply rules
    rules = apply_advanced_rules(bp_fuzz, chol_fuzz, hr_fuzz, age_fuzz, smoke_fuzz, diabetes_fuzz)

    # Mamdani inference
    aggregated_mamdani = aggregate_rules_mamdani(rules)
    mamdani_result = defuzzify_mamdani_cog(aggregated_mamdani)

    # Sugeno inference
    sugeno_result = defuzzify_sugeno_weighted_average(rules)

    if verbose:
        print(f"\nPatient: BP={bp}, Chol={chol}, HR={hr}, Age={age}, Smoking={smoking}, Diabetes={diabetes}")
        print("\nFuzzification Results:")
        print(f"  BP: Low={bp_fuzz['Low']:.3f}, Medium={bp_fuzz['Medium']:.3f}, High={bp_fuzz['High']:.3f}")
        print(f"  Chol: Low={chol_fuzz['Low']:.3f}, High={chol_fuzz['High']:.3f}")
        print(f"  HR: Slow={hr_fuzz['Slow']:.3f}, Moderate={hr_fuzz['Moderate']:.3f}, Fast={hr_fuzz['Fast']:.3f}")
        print(f"  Age: Young={age_fuzz['Young']:.3f}, Middle={age_fuzz['Middle']:.3f}, Old={age_fuzz['Old']:.3f}")
        print(f"  Smoking: None={smoke_fuzz['None']:.3f}, Light={smoke_fuzz['Light']:.3f}, Heavy={smoke_fuzz['Heavy']:.3f}")
        print(f"  Diabetes: No={diabetes_fuzz['No']:.3f}, Pre={diabetes_fuzz['Pre']:.3f}, Yes={diabetes_fuzz['Yes']:.3f}")

        print("\nRule Strengths (Sick rules especially):")
        print(f"  Rule5(Sick): {rules['Sick'][0]:.3f}")
        print(f"  Rule6(Sick): {rules['Sick'][1]:.3f}")
        print(f"  Rule7(Sick - Age): {rules['Sick'][2]:.3f}")
        print(f"  Rule8(Sick - Smoking): {rules['Sick'][3]:.3f}")
        print(f"  Rule9(Sick - Diabetes): {rules['Sick'][4]:.3f}")

        print(f"\nMamdani COG Result: {mamdani_result:.3f}")
        print(f"Sugeno Result: {sugeno_result:.3f}")
        print(f"Difference (Mamdani - Sugeno): {mamdani_result - sugeno_result:.3f}")

    return {
        'mamdani': mamdani_result,
        'sugeno': sugeno_result,
        'rules': rules,
        'aggregated': aggregated_mamdani
    }

# =============================================================================
# PART 5: SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(base_patient):
    """
    Analyze how each input affects the CHD output
    base_patient: tuple (bp, chol, hr, age, smoking, diabetes)
    """
    bp, chol, hr, age, smoking, diabetes = base_patient

    # Define variation ranges
    variations = np.linspace(0.5, 1.5, 11)  # 50% to 150% in 10 steps

    # Store results
    results = {
        'BP': [],
        'Chol': [],
        'HR': [],
        'Age': [],
        'Smoking': [],
        'Diabetes': []
    }

    # Analyze each factor
    for factor in results.keys():
        print(f"\nAnalyzing {factor}...")
        for var in variations:
            if factor == 'BP':
                new_val = bp * var
                result_dict = diagnose_patient_advanced(new_val, chol, hr, age, smoking, diabetes)
                sugeno = result_dict['sugeno']
            elif factor == 'Chol':
                new_val = chol * var
                result_dict = diagnose_patient_advanced(bp, new_val, hr, age, smoking, diabetes)
                sugeno = result_dict['sugeno']
            elif factor == 'HR':
                new_val = hr * var
                result_dict = diagnose_patient_advanced(bp, chol, new_val, age, smoking, diabetes)
                sugeno = result_dict['sugeno']
            elif factor == 'Age':
                new_val = age * var
                result_dict = diagnose_patient_advanced(bp, chol, hr, new_val, smoking, diabetes)
                sugeno = result_dict['sugeno']
            elif factor == 'Smoking':
                new_val = smoking * var
                result_dict = diagnose_patient_advanced(bp, chol, hr, age, new_val, diabetes)
                sugeno = result_dict['sugeno']
            elif factor == 'Diabetes':
                new_val = diabetes * var
                result_dict = diagnose_patient_advanced(bp, chol, hr, age, smoking, new_val)
                sugeno = result_dict['sugeno']

            results[factor].append(sugeno)

    return variations, results

def plot_sensitivity_analysis(variations, results):
    """Plot sensitivity analysis results"""
    plt.figure(figsize=(12, 8))

    for factor, values in results.items():
        plt.plot(variations * 100, values, marker='o', linewidth=2, label=factor)

    plt.xlabel('Input Variation (% of base value)', fontsize=12)
    plt.ylabel('CHD Output', fontsize=12)
    plt.title('Sensitivity Analysis: Effect of Each Input on CHD Diagnosis', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.5, label='Sick threshold (2.5)')
    plt.axhline(y=1.5, color='g', linestyle='--', alpha=0.5, label='Healthy threshold (1.5)')

    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Sensitivity plot saved as 'sensitivity_analysis.png'")

# =============================================================================
# PART 6: COMPARISON BETWEEN ORIGINAL AND ADVANCED SYSTEM
# =============================================================================

def compare_systems():
    """Compare original (3-factor) vs advanced (6-factor) system"""
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL (3 factors) vs ADVANCED (6 factors)")
    print("=" * 70)

    # Test patients with different risk profiles
    test_cases = [
        {
            'name': 'Young, healthy',
            'bp': 110, 'chol': 150, 'hr': 65,
            'age': 25, 'smoking': 0, 'diabetes': 80
        },
        {
            'name': 'Middle-aged, smoker',
            'bp': 130, 'chol': 190, 'hr': 75,
            'age': 45, 'smoking': 0.8, 'diabetes': 95
        },
        {
            'name': 'Elderly, diabetic',
            'bp': 150, 'chol': 210, 'hr': 85,
            'age': 70, 'smoking': 0.3, 'diabetes': 150
        }
    ]

    print(f"\n{'Case':<25} {'Original':<12} {'Advanced':<12} {'Difference':<12}")
    print("-" * 65)

    for case in test_cases:
        # Original system (3 factors) - using Phase 8 function
        from fuzzy_chd import diagnose_patient as diagnose_original
        # The original diagnose_patient function likely returns 3 values (cog, sugeno, details)
        # If it returns a dict, this will also need adjustment, but for now assuming it returns 3 values.
        _, sugeno_orig, _ = diagnose_original(case['bp'], case['chol'], case['hr'])

        # Advanced system (6 factors)
        result = diagnose_patient_advanced(
            case['bp'], case['chol'], case['hr'],
            case['age'], case['smoking'], case['diabetes']
        )
        sugeno_adv = result['sugeno']

        print(f"{case['name']:<25} {sugeno_orig:<12.3f} {sugeno_adv:<12.3f} {sugeno_adv - sugeno_orig:<+12.3f}")

# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ADVANCED FUZZY EXPERT SYSTEM FOR CHD DIAGNOSIS")
    print("=" * 70)

    # Test patient with medium risk
    test_patient = (130, 190, 75, 50, 0.5, 110)

    # 1. Run advanced diagnosis
    print("\n1. ADVANCED DIAGNOSIS (with all 6 factors)")
    print("-" * 50)
    result = diagnose_patient_advanced(*test_patient, verbose=True)

    # 2. Mamdani vs Sugeno comparison
    print("\n" + "=" * 70)
    print("2. MAMDANI vs SUGENO COMPARISON")
    print("=" * 70)
    print(f"Mamdani (COG): {result['mamdani']:.3f}")
    print(f"Sugeno (Weighted Average): {result['sugeno']:.3f}")
    print(f"Absolute Difference: {abs(result['mamdani'] - result['sugeno']):.3f}")

    if abs(result['mamdani'] - result['sugeno']) < 0.1:
        print("→ Both methods give very similar results")
    else:
        print("→ Noticeable difference between methods")

    # 3. Sensitivity Analysis
    print("\n" + "=" * 70)
    print("3. SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Running sensitivity analysis (this may take a few seconds)...")

    variations, sensitivity_results = sensitivity_analysis(test_patient)

    # Find most influential factor
    max_ranges = {}
    for factor, values in sensitivity_results.items():
        max_ranges[factor] = max(values) - min(values)

    most_influential = max(max_ranges, key=max_ranges.get)
    print(f"\nMost influential factor: {most_influential}")
    print(f"Range of influence: {max_ranges[most_influential]:.3f}")

    # Plot sensitivity
    plot_sensitivity_analysis(variations, sensitivity_results)

    # 4. Compare original vs advanced
    compare_systems()

    print("\n" + "=" * 70)
    print("ADVANCED CHALLENGE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nDeliverables generated:")
    print("1. Advanced fuzzy system with 6 input factors")
    print("2. Mamdani vs Sugeno comparison")
    print("3. Sensitivity analysis plot (sensitivity_analysis.png)")
    print("4. System comparison table")
