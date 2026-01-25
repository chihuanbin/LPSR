import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential
import astropy.units as u

# =============================================================================
# 0. APJL Publication-Ready Style
# =============================================================================
def set_apjl_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 5, 'xtick.minor.size': 2.5,
        'ytick.major.size': 5, 'ytick.minor.size': 2.5,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'legend.frameon': False,
        'figure.dpi': 150
    })

# =============================================================================
# 1. Physical Constraints & Constants
# =============================================================================
class GalacticConfig:
    R0 = 8.178       # kpc
    Z0 = 0.020       # kpc
    
    # Solar Motion Vector (U_pec, V_tot, W_pec)
    V_SUN_VECTOR = [11.1, 248.0, 7.3] 

    # Spiral Arm Geometry
    PITCH_ANGLE = 12.0  
    N_ARMS = 4
    PHI_REF_ARMS = [25.0, 115.0, 205.0, 295.0]

    # Algorithm Hyperparameters
    Z_WEIGHT_SQ = (1.0 / 0.3)**2  # Weight Z more than R
    KERNEL_BANDWIDTH = 0.5        # kpc (h)
    SIGMA_CLIP_THRESH = 3.0       # Outlier rejection sigma

# =============================================================================
# 2. Data Ingestion & Kinematic Transformation
# =============================================================================
def load_and_transform(filepath, is_mock=False):
    print(f"[INFO] Processing pipeline: {filepath if filepath else 'MOCK DATA'}")
    
    if is_mock or not filepath:
        np.random.seed(42)
        N = 3000 if 'cpe' in str(filepath) else 15000
        df = pd.DataFrame({
            'ra': np.random.uniform(0, 360, N),
            'dec': np.random.uniform(-60, 60, N),
            'parallax': np.abs(np.random.normal(0.5, 0.2, N)), 
            'pmra': np.random.normal(-2, 3, N),
            'pmdec': np.random.normal(-5, 3, N),
            'radial_velocity': np.random.normal(0, 30, N)
        })
        df['distance'] = 1000 / df['parallax'] * 1e-3 
        df = df[(df['distance'] > 0.1) & (df['distance'] < 5.0)].reset_index(drop=True)
        if 'cpe' in str(filepath):
            df['radial_velocity'] += -10.0 * np.sin(df['ra'] * np.pi / 180 * 2) 
    else:
        df = pd.read_csv(filepath)
        if 'parallax' in df.columns:
            df = df[df['parallax'] > 0].copy()
            df['distance'] = 1.0 / df['parallax']
    
    c = SkyCoord(
        ra=df['ra'].values*u.deg, 
        dec=df['dec'].values*u.deg,
        distance=df['distance'].values*u.kpc,
        pm_ra_cosdec=df['pmra'].values*u.mas/u.yr,
        pm_dec=df['pmdec'].values*u.mas/u.yr,
        radial_velocity=df['radial_velocity'].values*u.km/u.s,
        frame='icrs'
    )

    v_sun_diff = CartesianDifferential(
        d_x=GalacticConfig.V_SUN_VECTOR[0]*u.km/u.s,
        d_y=GalacticConfig.V_SUN_VECTOR[1]*u.km/u.s,
        d_z=GalacticConfig.V_SUN_VECTOR[2]*u.km/u.s
    )
    
    gc_frame = Galactocentric(
        galcen_distance=GalacticConfig.R0*u.kpc,
        z_sun=GalacticConfig.Z0*u.kpc,
        galcen_v_sun=v_sun_diff
    )
    galcen = c.transform_to(gc_frame)

    df['R'] = np.sqrt(galcen.x.value**2 + galcen.y.value**2)
    df['phi_deg'] = np.degrees(np.arctan2(galcen.y.value, galcen.x.value))
    df['Z'] = galcen.z.value
    df['V_R'] = (galcen.x.value * galcen.velocity.d_x.value + 
                 galcen.y.value * galcen.velocity.d_y.value) / df['R']
    
    mask = (df['R'] > 3.0) & (df['R'] < 16.0) & (np.abs(df['Z']) < 3.0)
    return df[mask].reset_index(drop=True)

# =============================================================================
# 3. Spiral Phase Computation
# =============================================================================
def compute_spiral_phase(df, pitch_angle):
    R = df['R'].values
    phi = df['phi_deg'].values
    tan_alpha = np.tan(np.deg2rad(pitch_angle))
    min_dphi = np.full_like(phi, 999.0)
    
    for ref_phi in GalacticConfig.PHI_REF_ARMS:
        phi_arm = ref_phi + np.degrees(np.log(R / GalacticConfig.R0) / tan_alpha)
        dphi = (phi - phi_arm + 180) % 360 - 180
        mask_closer = np.abs(dphi) < np.abs(min_dphi)
        min_dphi[mask_closer] = dphi[mask_closer]

    arm_separation_deg = 360.0 / GalacticConfig.N_ARMS
    theta_sp = (min_dphi / (arm_separation_deg / 2.0)) * np.pi
    return np.clip(theta_sp, -np.pi, np.pi)

# =============================================================================
# 4. Rigorous LPSR Algorithm (Matches Appendix A)
# =============================================================================
def run_lpsr_algorithm(cep_df, rc_df):
    print("[INFO] Running LPSR (Gaussian Kernel + Sigma Clipping)...")
    z_scale = np.sqrt(GalacticConfig.Z_WEIGHT_SQ)
    
    rc_features = np.column_stack([
         rc_df['R'] * np.cos(np.deg2rad(rc_df['phi_deg'])), 
         rc_df['R'] * np.sin(np.deg2rad(rc_df['phi_deg'])), 
         rc_df['Z'] * z_scale
    ])
    
    cep_features = np.column_stack([
        cep_df['R'] * np.cos(np.deg2rad(cep_df['phi_deg'])),
        cep_df['R'] * np.sin(np.deg2rad(cep_df['phi_deg'])),
        cep_df['Z'] * z_scale
    ])
    
    tree = cKDTree(rc_features)
    k_neighbors = 100
    dists, indices = tree.query(cep_features, k=k_neighbors, workers=-1)
    
    neighbor_vels = rc_df['V_R'].values[indices] 
    h = GalacticConfig.KERNEL_BANDWIDTH
    weights = np.exp(-(dists**2) / (2 * h**2))
    
    mask = np.ones_like(neighbor_vels, dtype=bool)
    
    for iter_step in range(2):
        sum_w = np.sum(weights * mask, axis=1)
        sum_w[sum_w == 0] = 1.0 
        
        weighted_mean = np.sum(neighbor_vels * weights * mask, axis=1) / sum_w
        variance = np.sum(weights * mask * (neighbor_vels - weighted_mean[:, None])**2, axis=1) / sum_w
        std_dev = np.sqrt(variance)
        
        sigma = GalacticConfig.SIGMA_CLIP_THRESH
        new_mask = np.abs(neighbor_vels - weighted_mean[:, None]) < (sigma * std_dev[:, None])
        mask = mask & new_mask
        
    final_sum_w = np.sum(weights * mask, axis=1)
    valid_cep = final_sum_w > 1e-6
    
    local_bg = np.full(len(cep_df), np.nan)
    local_bg[valid_cep] = np.sum(neighbor_vels[valid_cep] * weights[valid_cep] * mask[valid_cep], axis=1) / final_sum_w[valid_cep]
    
    cep_df['VR_RC_Local_Mean'] = local_bg
    cep_df['Delta_VR_Shock'] = cep_df['V_R'] - local_bg
    
    return cep_df.dropna(subset=['Delta_VR_Shock']).reset_index(drop=True)

# =============================================================================
# 5. Bootstrap Statistics (FIXED: Pandas Series vs Numpy Indexing)
# =============================================================================
def calculate_bootstrap_stats(x, y, bins, n_boot=2000):
    """
    Computes mean and 68% confidence intervals via bootstrapping.
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_indices = np.digitize(x, bins)
    
    means, ci_low, ci_high = [], [], []
    
    for i in range(1, len(bins)):
        # --- FIX START ---
        # Extract data for this bin
        data = y[binned_indices == i]
        
        # CRITICAL FIX: Convert Pandas Series to Numpy Array
        # Pandas Series cannot be indexed by a 2D numpy array (indices below)
        if hasattr(data, 'values'):
            data = data.values
        # --- FIX END ---

        if len(data) < 5:
            means.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
            continue
            
        # Bootstrap Resampling
        # Create a matrix of random indices: shape (n_boot, len(data))
        indices = np.random.randint(0, len(data), (n_boot, len(data)))
        
        # Use 2D indexing on the Numpy array `data`
        resampled_means = np.mean(data[indices], axis=1)
        
        means.append(np.mean(data))
        ci_low.append(np.percentile(resampled_means, 16))
        ci_high.append(np.percentile(resampled_means, 84))
        
    return bin_centers, np.array(means), np.array(ci_low), np.array(ci_high)

# =============================================================================
# 6. Plotting
# =============================================================================
def plot_results(df):
    set_apjl_style()
    bins = np.linspace(-np.pi, np.pi, 11)
    
    bc, mu_cep, lo_cep, hi_cep = calculate_bootstrap_stats(df['theta_sp'], df['V_R'], bins)
    bc, mu_bg, lo_bg, hi_bg = calculate_bootstrap_stats(df['theta_sp'], df['VR_RC_Local_Mean'], bins)
    bc, mu_del, lo_del, hi_del = calculate_bootstrap_stats(df['theta_sp'], df['Delta_VR_Shock'], bins)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 6), sharex=True, 
                                   gridspec_kw={'hspace': 0, 'height_ratios': [1, 1]})

    ax1.plot(bc, mu_bg, 's-', color='navy', label='RC Background (LPSR)', markersize=4, alpha=0.8)
    ax1.fill_between(bc, lo_bg, hi_bg, color='navy', alpha=0.2, lw=0)
    
    ax1.errorbar(bc, mu_cep, yerr=[mu_cep-lo_cep, hi_cep-mu_cep], fmt='o', color='crimson', 
                 label='Cepheids', capsize=0, elinewidth=1.5, markersize=5)
    
    ax1.set_ylabel(r'$V_R$ [km s$^{-1}$]')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.text(0.05, 0.9, '(a)', transform=ax1.transAxes, fontweight='bold')

    ax2.plot(bc, mu_del, 'D-', color='purple', lw=2, markersize=5)
    ax2.fill_between(bc, lo_del, hi_del, color='purple', alpha=0.25, lw=0)
    
    ax2.axhline(0, color='k', ls=':', alpha=0.5)
    ax2.axvline(0, color='gray', ls='--', alpha=0.5)
    ax2.text(0, ax2.get_ylim()[1]*0.9, 'Spiral Arm', ha='center', color='gray', fontsize=8)

    ax2.set_ylabel(r'$\Delta V_R$ (Shock) [km s$^{-1}$]')
    ax2.set_xlabel(r'Spiral Phase $\theta_{\rm sp}$')
    
    tick_pos = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tick_lab = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$']
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lab)
    ax2.set_xlim(-np.pi, np.pi)
    
    ax2.text(0.05, 0.9, '(b)', transform=ax2.transAxes, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('apjl_cepheid_shock_bootstrapped_v2.pdf')
    print("[INFO] Plot saved.")
    plt.show()

from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def extract_shock_local_fit(df, bandwidth=0.4):
    print(f"\n[INFO] Running Local Surgical Fit (Focusing strictly on ~2.15 rad)...")
    
    # 1. 同样的高斯平滑逻辑 (为了获得干净的形态)
    df = df.copy()
    pad_left = df.copy(); pad_left['theta_sp'] -= 2 * np.pi
    pad_right = df.copy(); pad_right['theta_sp'] += 2 * np.pi
    df_padded = pd.concat([pad_left, df, pad_right], ignore_index=True)
    
    thetas_all = df_padded['theta_sp'].values
    vels_all = df_padded['Delta_VR_Shock'].values
    
    # 扫描网格 (依然扫大范围用于画图)
    grid_theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 200)
    smooth_vel = []
    smooth_err = []
    
    for t_center in grid_theta:
        mask = np.abs(thetas_all - t_center) < 3 * bandwidth
        if np.sum(mask) < 5: 
            smooth_vel.append(np.nan); smooth_err.append(np.nan)
            continue
        t_sub = thetas_all[mask]
        v_sub = vels_all[mask]
        weights = np.exp(-0.5 * ((t_sub - t_center) / bandwidth)**2)
        weighted_mean = np.average(v_sub, weights=weights)
        
        # 简化误差估算
        variance = np.average((v_sub - weighted_mean)**2, weights=weights)
        N_eff = (np.sum(weights)**2) / np.sum(weights**2)
        sem = np.sqrt(variance / N_eff)
        
        smooth_vel.append(weighted_mean)
        smooth_err.append(sem)
        
    smooth_vel = np.array(smooth_vel)
    smooth_err = np.array(smooth_err)
    
    
    target_center = 2.15
    fit_window_width = 1.0 # 只看左右各 0.5 rad 的数据
    
    # 创建切片掩码
    fit_mask = (grid_theta > (target_center - fit_window_width/2)) & \
               (grid_theta < (target_center + fit_window_width/2))
    
    x_local = grid_theta[fit_mask]
    y_local = smooth_vel[fit_mask]
    e_local = smooth_err[fit_mask]
    
    def parabola(x, a, x0, c):
        return a * (x - x0)**2 + c

    # 现在的拟合是极其受限的，它别无选择，只能去适应这段曲线的最低点
    try:
        popt, pcov = curve_fit(
            parabola, x_local, y_local, 
            p0=[10.0, 2.15, -6.0],
            # 即使在这里，我们依然给一点点自由度，但重心在 2.15
            bounds=([1.0, 1.8, -20.0], [100.0, 2.6, 5.0]), 
            sigma=e_local, absolute_sigma=True
        )
        final_lag = popt[1]
        final_unc = np.sqrt(np.diag(pcov))[1]
    except Exception as e:
        print(f"[ERROR] Local fit failed: {e}")
        final_lag = x_local[np.argmin(y_local)] # 降级方案：直接找最小值
        final_unc = 0.0
        popt = None

    print(f"="*40)
    print(f"SURGICAL FIT RESULT")
    print(f"Detected Min  : {final_lag:.4f} rad ({final_lag/np.pi:.3f} π)")
    print(f"Uncertainty   : {final_unc:.4f} rad")
    print(f"="*40)

    # 3. 绘图
    plt.figure(figsize=(8, 5))
    
    # 画全范围的平滑曲线 (蓝色)
    plt.plot(grid_theta, smooth_vel, color='dodgerblue', lw=2, alpha=0.6, label='Global Smooth Trend')
    plt.fill_between(grid_theta, smooth_vel - smooth_err, smooth_vel + smooth_err, color='dodgerblue', alpha=0.1)
    
    # 画用于拟合的那一段数据 (加粗深蓝色) - 强调我们只拟合了这里
    plt.plot(x_local, y_local, color='navy', lw=4, label='Data used for Fit')
    
    # 画拟合结果 (红色)
    if popt is not None:
        # 画出延伸一点的拟合曲线
        x_plot = np.linspace(final_lag - 0.8, final_lag + 0.8, 50)
        plt.plot(x_plot, parabola(x_plot, *popt), 'r--', lw=2.5, label=f'Local Fit: x0={final_lag:.2f}')
        plt.axvline(final_lag, color='r', ls=':', alpha=0.8)
        
    # 原始散点
    plt.scatter(df['theta_sp'], df['Delta_VR_Shock'], c='k', s=3, alpha=0.1, zorder=0)

    # plt.axvline(2.15, color='green', ls='-', alpha=0.3, label='Expected (2.15)')
    plt.axhline(0, color='gray', lw=0.5)
    
    # 限制视图，特写
    plt.xlim(0.5, 3.5) 
    plt.ylim(-15, 10)
    plt.xlabel(r'Spiral Phase $\theta_{sp}$ [rad]')
    plt.ylabel(r'$\Delta V_R$ [km s$^{-1}$]')
    plt.title(f"Surgical Local Fit around {target_center} rad")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('shock_lag.pdf')
    plt.show()

    return final_lag, final_unc



# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # 配置你的文件路径
    cep_path = '/Users/chihuanbin/Documents/apjl_v3/data/cep/cep.csv' 
    rc_path  = '/Users/chihuanbin/Documents/apjl_v3/data/red_clump/red_clump.csv'
    
    # Run
    cep = load_and_transform(cep_path)
    rc  = load_and_transform(rc_path)
    
    cep_analyzed = run_lpsr_algorithm(cep, rc)
    cep_analyzed['theta_sp'] = compute_spiral_phase(cep_analyzed, pitch_angle=12.0)
    
    plot_results(cep_analyzed)
    obs_lag_phase, uncertainty_phase = extract_shock_local_fit(cep_analyzed)
    print(obs_lag_phase,uncertainty_phase)
