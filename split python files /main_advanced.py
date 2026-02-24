
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ADVANCED FUZZY EXPERT SYSTEM FOR CHD DIAGNOSIS")
    print("=" * 70)

    test_patient = (130, 190, 75, 50, 0.5, 110) # Test patient with medium risk

    print("\n1. ADVANCED DIAGNOSIS (with all 6 factors)")
    print("-" * 50)
    result = diagnose_patient_advanced(*test_patient, verbose=True)

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

    print("\n" + "=" * 70)
    print("3. SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Running sensitivity analysis (this may take a few seconds)...")

    variations, sensitivity_results = sensitivity_analysis(test_patient)

    max_ranges = {}     # influential factor
    for factor, values in sensitivity_results.items():
        max_ranges[factor] = max(values) - min(values)
    most_influential = max(max_ranges, key=max_ranges.get)
    print(f"\nMost influential factor: {most_influential}")
    print(f"Range of influence: {max_ranges[most_influential]:.3f}")

    plot_sensitivity_analysis(variations, sensitivity_results) # Plot sensitivity

    compare_systems()

    print("\n" + "=" * 70)
    print("ADVANCED CHALLENGE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nDeliverables generated:")
    print("1. Advanced fuzzy system with 6 input factors")
    print("2. Mamdani vs Sugeno comparison")
    print("3. Sensitivity analysis plot (sensitivity_analysis.png)")
    print("4. System comparison table")
