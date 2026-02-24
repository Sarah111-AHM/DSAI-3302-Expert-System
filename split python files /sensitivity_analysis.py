
def sensitivity_analysis(base_patient):
    bp, chol, hr, age, smoking, diabetes = base_patient
    variations = np.linspace(0.5, 1.5, 11)  # 50% to 150% in 10 steps
    results = {'BP': [],'Chol': [],'HR': [],'Age': [],'Smoking': [],'Diabetes': []}

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
    #sensitivity analysis results
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
