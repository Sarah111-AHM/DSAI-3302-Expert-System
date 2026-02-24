def diagnose_patient_advanced(bp, chol, hr, age, smoking, diabetes, verbose=False):
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
    return {'mamdani': mamdani_result, 'sugeno': sugeno_result, 'rules': rules, 'aggregated': aggregated_mamdani}
