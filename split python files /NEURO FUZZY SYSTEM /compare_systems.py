def compare_all_systems(): 
    print("\n" + "=" * 80)
    print("NEURO-FUZZY SYSTEM FOR CHD DIAGNOSIS")
    print("=" * 80)
    
    print("\nGenerating synthetic patient data...")
    data = generate_training_data(500)
    X = data[:, :6]
    y = data[:, 6]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print("\n" + "-" * 50)
    print("SYSTEM 1: Pure Fuzzy System (Expert Rules)")
    print("-" * 50)
    
    fuzzy_system = NeuroFuzzyCHD()
    fuzzy_system.initialize_membership_functions()
    y_pred_fuzzy = fuzzy_system.predict(X_test)
    mse_fuzzy = mean_squared_error(y_test, y_pred_fuzzy)
    r2_fuzzy = r2_score(y_test, y_pred_fuzzy)

    print(f"MSE: {mse_fuzzy:.4f}")
    print(f"R² Score: {r2_fuzzy:.4f}")
    
    print("\n" + "-" * 50)
    print("SYSTEM 2: Neural Network (Pure Learning)")
    print("-" * 50)
    
    nn = SimpleNeuralNetwork(input_size=6, hidden_size=12, learning_rate=0.01)
    nn_losses = nn.train(X_train, y_train, epochs=100, verbose=False)
    y_pred_nn = nn.forward(X_test).flatten()
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    
    print(f"MSE: {mse_nn:.4f}")
    print(f"R² Score: {r2_nn:.4f}")
    
    print("\n" + "-" * 50)
    print("SYSTEM 3: Neuro-Fuzzy System (Best of Both)")
    print("-" * 50)
    
    nf_system = NeuroFuzzyCHD()
    nf_system.initialize_membership_functions()
    nf_losses = nf_system.train_neuro_fuzzy(X_train, y_train, epochs=50, learning_rate=0.01)
    y_pred_nf = nf_system.predict(X_test)
    mse_nf = mean_squared_error(y_test, y_pred_nf)
    r2_nf = r2_score(y_test, y_pred_nf)
    
    print(f"\nMSE: {mse_nf:.4f}")
    print(f"R² Score: {r2_nf:.4f}")
    print("\n" + "=" * 80)
    print("SYSTEM COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n{'System':<25} {'MSE':<12} {'R² Score':<12} {'Advantage':<20}")
    print("-" * 70)
    print(f"{'Pure Fuzzy':<25} {mse_fuzzy:<12.4f} {r2_fuzzy:<12.4f} {'Uses expert knowledge':<20}")
    print(f"{'Neural Network':<25} {mse_nn:<12.4f} {r2_nn:<12.4f} {'Learns from data':<20}")
    print(f"{'Neuro-Fuzzy (ANFIS)':<25} {mse_nf:<12.4f} {r2_nf:<12.4f} {'Both knowledge + learning':<20}")
    
    plt.figure(figsize=(15, 5))
    # MSE Comparison
    plt.subplot(1, 3, 1)
    systems = ['Pure Fuzzy', 'Neural Network', 'Neuro-Fuzzy']
    mse_values = [mse_fuzzy, mse_nn, mse_nf]
    colors = ['blue', 'green', 'red']
    bars = plt.bar(systems, mse_values, color=colors, alpha=0.7)
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Comparison')
    for bar, val in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    # R² Comparison
    plt.subplot(1, 3, 2)
    r2_values = [r2_fuzzy, r2_nn, r2_nf]
    bars = plt.bar(systems, r2_values, color=colors, alpha=0.7)
    plt.ylabel('R² Score')
    plt.title('R² Comparison (higher is better)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,f'{val:.3f}', ha='center', va='bottom')
    # Learning Curves
    plt.subplot(1, 3, 3)
    plt.plot(nn_losses, label='Neural Network', color='green')
    plt.plot(nf_losses, label='Neuro-Fuzzy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neuro_fuzzy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n Comparison plot saved as 'neuro_fuzzy_comparison.png'")
    
    print("\n" + "=" * 80)
    print("NEURO-FUZZY ADVANTAGES")
    print("=" * 80)
    return {'fuzzy': {'mse': mse_fuzzy, 'r2': r2_fuzzy},'neural': {'mse': mse_nn, 'r2': r2_nn},'neuro_fuzzy': {'mse': mse_nf, 'r2': r2_nf}}

if __name__ == "__main__":
    results = compare_all_systems()
    
