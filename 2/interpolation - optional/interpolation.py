import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

m_values = [0.504032, 0.678161, 0.988142]
# Al20 Al15 Al10

#data_m1 = np.array([[0.0, 0], [0.1, 100], [0.2, 180], [0.3, 240], [0.4, 280], [0.5, 300]]) #Al20
#data_m2 = np.array([[0.0, 0], [0.08, 110], [0.18, 200], [0.28, 270], [0.38, 310], [0.48, 320]]) #Al15
#data_m3 = np.array([[0.0, 0], [0.07, 120], [0.17, 220], [0.27, 290], [0.37, 330], [0.47, 340]]) #Al10
datarem_m1 = pd.read_csv('data_m1.csv').values
datarem_m2 = pd.read_csv('data_m2.csv').values
datarem_m3 = pd.read_csv('data_m3.csv').values

def clean_duplicate_strain(data):
    strain = data[:, 0]
    stress = data[:, 1]
    
    strain_dict = {}
    
    for s, t in zip(strain, stress):
        s_rounded = round(s, 10)  
        if s_rounded in strain_dict:
            strain_dict[s_rounded]['sum'] += t
            strain_dict[s_rounded]['count'] += 1
        else:
            strain_dict[s_rounded] = {'sum': t, 'count': 1}
    
    cleaned_strain = []
    cleaned_stress = []
    
    for s_rounded, values in strain_dict.items():
        cleaned_strain.append(s_rounded)
        cleaned_stress.append(values['sum'] / values['count'])
    
    sorted_indices = np.argsort(cleaned_strain)
    cleaned_data = np.column_stack([
        np.array(cleaned_strain)[sorted_indices],
        np.array(cleaned_stress)[sorted_indices]
    ])
    
    return cleaned_data

data_m1 = clean_duplicate_strain(datarem_m1)
data_m2 = clean_duplicate_strain(datarem_m2)
data_m3 = clean_duplicate_strain(datarem_m3)
all_data = [data_m1, data_m2, data_m3]



def interpolate_stress(strain_data, stress_data, target_strains):
    interp_func = interpolate.interp1d(strain_data, stress_data, 
                                      kind='cubic', 
                                      fill_value='extrapolate',
                                      bounds_error=False)
    return interp_func(target_strains)

def get_common_strain_range(all_data):
    min_strains = []
    max_strains = []
    
    for data in all_data:
        strains = data[:, 0]
        min_strains.append(np.min(strains))
        max_strains.append(np.max(strains))
    
    common_min = np.max(min_strains) 
    common_max = np.min(max_strains) 
    
    return common_min, common_max

def extrapolate_to_m0(m_values, all_data, num_points=50):
    common_min, common_max = get_common_strain_range(all_data)
    print(f"Common strain range: {common_min:.3f} to {common_max:.3f}")
    
    strain_common = np.linspace(common_min, common_max, num_points)
    interpolated_stresses = []
    
    for i, data in enumerate(all_data):
        strains = data[:, 0]
        stresses = data[:, 1]
        
        stress_interp = interpolate_stress(strains, stresses, strain_common)
        interpolated_stresses.append(stress_interp)
        
        print(f"Data m={m_values[i]}: {len(strains)} points → {num_points} interpolated points")
    
    stress_m0 = np.zeros_like(strain_common)
    slopes = np.zeros_like(strain_common)
    intercepts = np.zeros_like(strain_common)
    r_squared = np.zeros_like(strain_common)
    
    for j, strain in enumerate(strain_common):
        stress_at_strain = [interpolated_stresses[i][j] for i in range(len(m_values))]
        slope, intercept, r_value, p_value, std_err = stats.linregress(m_values, stress_at_strain)
        stress_m0[j] = intercept
        slopes[j] = slope
        intercepts[j] = intercept
        r_squared[j] = r_value**2
    
    extrapolation_details = {
        'strain': strain_common,
        'stress_m0': stress_m0,
        'slopes': slopes,
        'intercepts': intercepts,
        'r_squared': r_squared,
        'interpolated_stresses': interpolated_stresses
    }
    
    return strain_common, stress_m0, extrapolation_details

strain_common, stress_m0, details = extrapolate_to_m0(m_values, all_data, num_points=1000)

def plot_results(m_values, all_data, strain_common, stress_m0, details):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1 = axes[0, 0]
    colors = ['b', 'g', 'r']
    markers = ['o', 's', '^']
    
    for i, (data, m) in enumerate(zip(all_data, m_values)):
        ax1.plot(data[:, 0], data[:, 1], 
                marker=markers[i], linestyle='--', 
                color=colors[i], alpha=0.7,
                label=f'm = {m}', markersize=6)
    
    ax1.plot(strain_common, stress_m0, 
            'k-', linewidth=3, 
            label='Extrapolated to m=0 (frictionless)')
    
    ax1.set_xlabel('Engineering Strain', fontsize=12)
    ax1.set_ylabel('Engineering Stress (MPa)', fontsize=12)
    ax1.set_title('Stress-Strain Curves and Extrapolation to m=0', 
                 fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    sample_strains = [0.1, 0.2, 0.3, 0.4]
    colors_sample = ['red', 'blue', 'green', 'purple']
    
    for idx, (strain_sample, color) in enumerate(zip(sample_strains, colors_sample)):
        strain_idx = np.argmin(np.abs(strain_common - strain_sample))
        actual_strain = strain_common[strain_idx]
        
        stress_values = [details['interpolated_stresses'][i][strain_idx] 
                        for i in range(len(m_values))]
        
        m_extended = np.linspace(0, max(m_values)*1.2, 10)
        stress_line = details['slopes'][strain_idx] * m_extended + details['intercepts'][strain_idx]
        
        ax2.plot(m_values, stress_values, 'o', color=color, markersize=8,
                label=f'ε={actual_strain:.2f}, R²={details["r_squared"][strain_idx]:.3f}')
        ax2.plot(m_extended, stress_line, '--', color=color, alpha=0.7)
    
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Diameter-to-height ratio (m = d/h)', fontsize=12)
    ax2.set_ylabel('Stress (MPa)', fontsize=12)
    ax2.set_title('Linear Extrapolation at Different Strains', 
                 fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(strain_common, details['slopes'], 'b-', linewidth=2)
    ax3.set_xlabel('Strain', fontsize=12)
    ax3.set_ylabel('Slope (dσ/dm) (MPa)', fontsize=12)
    ax3.set_title('Variation of Extrapolation Slope with Strain', 
                 fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(strain_common, details['r_squared'], 'g-', linewidth=2)
    ax4.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='R² = 0.95')
    ax4.set_xlabel('Strain', fontsize=12)
    ax4.set_ylabel('Coefficient of Determination (R²)', fontsize=12)
    ax4.set_title('Quality of Linear Extrapolation', 
                 fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("Extrapolation Statistical Summary:")
    print("="*50)
    print(f"Mean R²: {np.mean(details['r_squared']):.4f}")
    print(f"Min R²: {np.min(details['r_squared']):.4f} at strain {strain_common[np.argmin(details['r_squared'])]:.3f}")
    print(f"Max R²: {np.max(details['r_squared']):.4f} at strain {strain_common[np.argmax(details['r_squared'])]:.3f}")
    print(f"Mean slope: {np.mean(details['slopes']):.2f} MPa")
    print(f"Output strain range: {strain_common[0]:.3f} to {strain_common[-1]:.3f}")

plot_results(m_values, all_data, strain_common, stress_m0, details)

def save_results(strain_common, stress_m0, details, filename='friction_corrected_curve.csv'):
    """
    Save results to CSV file
    """
    df = pd.DataFrame({
        'Strain': strain_common,
        'Stress_m0_MPa': stress_m0,
        'Slope_dSigma_dm': details['slopes'],
        'R_squared': details['r_squared']
    })
    
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to '{filename}'.")
    
    print("\nSample of corrected data:")
    print(df.head(10))

save_results(strain_common, stress_m0, details)
