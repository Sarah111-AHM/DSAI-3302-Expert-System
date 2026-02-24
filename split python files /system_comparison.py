def compare_systems():
    # Compare the orginal with advanced 
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL (3 factors) vs ADVANCED (6 factors)")
    print("=" * 70)
    # Test patients with different risk profiles
    test_cases = [
        {'name': 'Young, healthy','bp': 110, 'chol': 150, 'hr': 65,'age': 25, 'smoking': 0, 'diabetes': 80},
        {'name': 'Middle-aged, smoker','bp': 130, 'chol': 190, 'hr': 75,'age': 45, 'smoking': 0.8, 'diabetes': 95},
        {'name': 'Elderly, diabetic','bp': 150, 'chol': 210, 'hr': 85,'age': 70, 'smoking': 0.3, 'diabetes': 150}
    ]

    print(f"\n{'Case':<25} {'Original':<12} {'Advanced':<12} {'Difference':<12}")
    print("-" * 65)

    for case in test_cases:
        from fuzzy_chd import diagnose_patient as diagnose_original
        _, sugeno_orig, _ = diagnose_original(case['bp'], case['chol'], case['hr'])

        result = diagnose_patient_advanced(case['bp'], case['chol'], case['hr'],case['age'], case['smoking'], case['diabetes'])
        sugeno_adv = result['sugeno']

        print(f"{case['name']:<25} {sugeno_orig:<12.3f} {sugeno_adv:<12.3f} {sugeno_adv - sugeno_orig:<+12.3f}")
