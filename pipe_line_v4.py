import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential
import astropy.units as u

# =============================================================================
# 0. APJL 样式全局设置
# =============================================================================
def set_apjl_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 6, 'ytick.minor.size': 3,
        'ytick.major.size': 6, 'ytick.minor.size': 3,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'legend.frameon': False,
        'legend.fontsize': 10
    })

# =============================================================================
# 1. 物理配置
# =============================================================================
class GalacticConfig:
    R0 = 8.178       # kpc
    Z0 = 0.020       # kpc
    V_SUN = [11.1, 248.0, 7.3]  # [U, V+Vcirc, W] km/s
    
    # 初始默认值，后面会自动优化
    PITCH_ANGLE = 12.0  
    N_ARMS = 4
    PHI_REF_ARMS = [25.0, 115.0, 205.0, 295.0]

    Z_WEIGHT_SQ = 100.0 
    N_NEIGHBORS = 50 

# =============================================================================
# 2. 数据处理与坐标转换
# =============================================================================
def load_and_transform(filepath, is_mock=False):
    """读取 CSV 并转换为银心圆柱坐标系"""
    print(f"[INFO] Processing {filepath}...")
    
    if is_mock or not filepath:
        # 生成更真实的 Mock 数据以测试 Bootstrapping 效果
        N = 2000 if 'cpe' in str(filepath) else 10000
        df = pd.DataFrame({
            'ra': np.random.uniform(0, 360, N),
            'dec': np.random.uniform(-60, 60, N),
            'distance': np.random.uniform(1, 12, N), # 直接生成距离
            'pmra': np.random.normal(0, 5, N),
            'pmdec': np.random.normal(0, 5, N),
            'radial_velocity': np.random.normal(0, 20, N)
        })
        # 补充 parallax 列用于兼容检查
        df['parallax'] = 1 / df['distance']
        
        # 强行注入一个弱信号到 mock 数据中用于测试绘图 (仅 mock 模式)
        if 'cpe' in str(filepath):
            # 模拟相位角 14度下的信号
            print("   -> Injecting mock signal for testing...")
            R = np.random.uniform(6, 11, N)
            phi = np.random.uniform(0, 360, N)
            df['R'] = R # 临时覆盖
            tan_a = np.tan(np.deg2rad(14.0)) 
            log_spiral_phase = (phi - (np.degrees(np.log(R/8.178)/tan_a) + 25.0)) % 90
            # 简单的正弦波信号
            signal = -8.0 * np.sin(log_spiral_phase * np.pi / 45.0) 
            df['radial_velocity'] += signal
    else:
        df = pd.read_csv(filepath)
        if 'parallax' in df.columns and 'distance' not in df.columns:
            df = df[df['parallax'] > 0].copy()
            df['distance'] = 1.0 / df['parallax']
    
    # Coordinates Transformation
    c = SkyCoord(
        ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg,
        distance=df['distance'].values*u.kpc,
        pm_ra_cosdec=df['pmra'].values*u.mas/u.yr,
        pm_dec=df['pmdec'].values*u.mas/u.yr,
        radial_velocity=df['radial_velocity'].values*u.km/u.s,
        frame='icrs'
    )

    v_sun_diff = CartesianDifferential(
        d_x=GalacticConfig.V_SUN[0]*u.km/u.s,
        d_y=GalacticConfig.V_SUN[1]*u.km/u.s,
        d_z=GalacticConfig.V_SUN[2]*u.km/u.s
    )
    galcen = c.transform_to(Galactocentric(galcen_distance=GalacticConfig.R0*u.kpc, z_sun=GalacticConfig.Z0*u.kpc, galcen_v_sun=v_sun_diff))

    df['R'] = np.sqrt(galcen.x.value**2 + galcen.y.value**2)
    df['phi_deg'] = np.degrees(np.arctan2(galcen.y.value, galcen.x.value))
    df['Z'] = galcen.z.value
    df['V_R'] = (galcen.x.value * galcen.velocity.d_x.value + galcen.y.value * galcen.velocity.d_y.value) / df['R']
    
    # 基础清洗
    mask = (df['R'] > 0.1) & (np.abs(df['Z']) < 5) & (np.abs(df['V_R']) < 400)
    return df[mask].reset_index(drop=True)

# =============================================================================
# 3. 灵活计算螺旋相位 (支持 Pitch Angle 调整)
# =============================================================================
def compute_spiral_phase(df, pitch_angle):
    """根据给定的 Pitch Angle 重新计算 theta_sp"""
    R = df['R'].values
    phi = df['phi_deg'].values
    tan_alpha = np.tan(np.deg2rad(pitch_angle))
    
    min_dphi = np.full_like(phi, 999.0)
    for ref_phi in GalacticConfig.PHI_REF_ARMS:
        # log 螺旋方程
        phi_arm = ref_phi + np.degrees(np.log(R / GalacticConfig.R0) / tan_alpha)
        dphi = (phi - phi_arm + 180) % 360 - 180
        update_mask = np.abs(dphi) < np.abs(min_dphi)
        min_dphi[update_mask] = dphi[update_mask]

    arm_spacing = 360.0 / GalacticConfig.N_ARMS
    theta_sp = (min_dphi / (arm_spacing / 2.0)) * np.pi
    
    return np.clip(theta_sp, -np.pi, np.pi)

# =============================================================================
# 4. LPSR 算法 (背景提取)
# =============================================================================
def run_lpsr_matching(cep_df, rc_df):
    print("[INFO] Running LPSR Algorithm...")
    z_scale = np.sqrt(GalacticConfig.Z_WEIGHT_SQ)
    
    # 3D Tree Construction
    rc_features = np.column_stack([
         rc_df['R']*np.cos(np.deg2rad(rc_df['phi_deg'])), 
         rc_df['R']*np.sin(np.deg2rad(rc_df['phi_deg'])), 
         rc_df['Z']*z_scale
    ])
    cep_features = np.column_stack([
        cep_df['R']*np.cos(np.deg2rad(cep_df['phi_deg'])),
        cep_df['R']*np.sin(np.deg2rad(cep_df['phi_deg'])),
        cep_df['Z']*z_scale
    ])
    
    tree = cKDTree(rc_features)
    dists, indices = tree.query(cep_features, k=GalacticConfig.N_NEIGHBORS, workers=-1)
    
    # 计算局部平均
    rc_vr = rc_df['V_R'].values
    local_bg = np.nanmean(rc_vr[indices], axis=1)
    
    cep_df['VR_RC_Local_Mean'] = local_bg
    cep_df['Delta_VR_Shock'] = cep_df['V_R'] - local_bg
    return cep_df

# =============================================================================
# 5. [新增] 自动寻找最佳 Pitch Angle (Phase Tuning)
# =============================================================================
def optimize_pitch_angle(df, angle_range=(5, 25), step=0.2):
    """遍历不同的 Pitch Angle，寻找信号振幅最大的角度"""
    print("[INFO] Optimizing Pitch Angle...")
    angles = np.arange(angle_range[0], angle_range[1], step)
    amplitudes = []
    
    # 预定义 Bins
    bins = np.linspace(-np.pi, np.pi, 11)
    
    for ang in angles:
        # 计算该角度下的相位
        theta = compute_spiral_phase(df, ang)
        # 计算 Bin 均值
        means, _, _ = binned_statistic(theta, df['Delta_VR_Shock'], statistic='mean', bins=bins)
        # 定义振幅: Max - Min (Peak-to-Valley)
        # 或者使用标准差 np.nanstd(means) 来衡量波动强度
        amp = np.nanmax(means) - np.nanmin(means)
        amplitudes.append(amp)
        
    best_idx = np.argmax(amplitudes)
    best_angle = angles[best_idx]
    
    print(f"       -> Checked {len(angles)} angles. Best Pitch Angle found: {best_angle:.1f} deg")
    
    # 绘制优化曲线 (可选，用于调试)
    # plt.figure()
    # plt.plot(angles, amplitudes, 'o-')
    # plt.xlabel('Pitch Angle [deg]'); plt.ylabel('Signal Amplitude [km/s]')
    # plt.axvline(best_angle, color='r', ls='--')
    # plt.title(f'Phase Optimization: Best = {best_angle:.1f}')
    # plt.show()
    
    return best_angle

# =============================================================================
# 6.  Bootstrap 统计 (绘制平滑误差带)
# =============================================================================
def calculate_bootstrap_stats(x, y, bins, n_boot=1000, ci=68): # <--- 改默认值为 68
    """
    使用自助法 (Bootstrapping) 计算 1-sigma 置信区间 (68%)
    这在天文学观测数据展示中是标准做法，尤其是对于噪声较大的数据
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_indices = np.digitize(x, bins)
    
    means = []
    ci_low = []
    ci_high = []
    
    # 计算 68% CI 对应的百分位
    alpha_low = (100 - ci) / 2  # 16
    alpha_high = 100 - alpha_low # 84
    
    for i in range(1, len(bins)):
        data_in_bin = y[binned_indices == i]
        
        if len(data_in_bin) < 5:
            means.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
            continue
            
        boot_means = []
        for _ in range(n_boot):
            # Bootstrap resampling
            resample = np.random.choice(data_in_bin, size=len(data_in_bin), replace=True)
            boot_means.append(np.mean(resample))
        
        means.append(np.mean(data_in_bin))
        ci_low.append(np.percentile(boot_means, alpha_low)) # 16th percentile
        ci_high.append(np.percentile(boot_means, alpha_high)) # 84th percentile
        
    return bin_centers, np.array(means), np.array(ci_low), np.array(ci_high)

# =============================================================================
# 7. 绘图
# =============================================================================
def plot_apjl_bootstrapped(df):
    set_apjl_style()
    
    # Bin 设置
    bins = np.linspace(-np.pi, np.pi, 9) # 10 Bins
    
    # 1. 计算 Bootstrapped Stats
    bc, mu_cep, lo_cep, hi_cep = calculate_bootstrap_stats(df['theta_sp'], df['V_R'], bins)
    bc, mu_bg, lo_bg, hi_bg = calculate_bootstrap_stats(df['theta_sp'], df['VR_RC_Local_Mean'], bins)
    bc, mu_del, lo_del, hi_del = calculate_bootstrap_stats(df['theta_sp'], df['Delta_VR_Shock'], bins)
    
    # 2. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True, gridspec_kw={'hspace': 0})

    # --- Panel A: Kinematics ---
    # Draw Error Bands (Ribbons)
    # Background (Blue)
    ax1.plot(bc, mu_bg, 's-', color='#1F77B4', label='LPSR Baseline', lw=1.5, markersize=4, zorder=1)
    ax1.fill_between(bc, lo_bg, hi_bg, color='#1F77B4', alpha=0.3, linewidth=0, zorder=1) # Ribbon

    # Cepheids (Red)
    ax1.plot(bc, mu_cep, 'o-', color='#D62728', label='Cepheids', lw=2, markersize=5, zorder=2)
    ax1.fill_between(bc, lo_cep, hi_cep, color='#D62728', alpha=0.25, linewidth=0, zorder=2) # Ribbon
    
    ax1.axhline(0, c='gray', ls=':', lw=1)
    ax1.set_ylabel(r'$V_R$ [km s$^{-1}$]')
    ax1.legend(loc='upper right', frameon=False)
    ax1.text(0.04, 0.9, '(a) Kinematics', transform=ax1.transAxes, fontweight='bold')

    # --- Panel B: Differential ---
    color_diff = 'purple'
    
    # 使用平滑插值让曲线看起来更自然 (可选，这里还是用折线连接数据点以保持真实)
    ax2.plot(bc, mu_del, 'D-', color=color_diff, label=r'Shock $\Delta V_R$', lw=2, markersize=5)
    
    # 关键：误差带
    ax2.fill_between(bc, lo_del, hi_del, color=color_diff, alpha=0.2, linewidth=0)
    
    # 标注
    ax2.axhline(0, c='k', alpha=0.2)
    ax2.axvline(0, c='k', ls='--', alpha=0.5)
    ax2.text(0.0, 0.95, 'Spiral Arm', rotation=90, fontsize=9, color='gray', ha='center', transform=ax2.get_xaxis_transform())

    ax2.set_xlabel(r'Spiral Phase $\theta_{\rm sp}$')
    ax2.set_ylabel(r'$\Delta V_R$ [km s$^{-1}$]')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    ax2.text(0.04, 0.9, '(b) Differential', transform=ax2.transAxes, fontweight='bold')
    
    # 动态调整 Y 轴 (带一点 buffer)
    ymax = max(np.nanmax(np.abs(hi_del)), np.nanmax(np.abs(lo_del)), 8) * 1.2
    ax2.set_ylim(-ymax, ymax)
    ax1.set_ylim(-ymax*1.5, ymax*1.5) # 通常绝对速度会有稍大的 offset
    
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig('apjl_cepheid_shock_bootstrapped_v2.pdf', dpi=300)
    plt.show()

# =============================================================================
# Main Pipeline
# =============================================================================
if __name__ == "__main__":
    # 配置
    cep_path = '/Users/chihuanbin/Documents/apjl_v3/data/cep/cep.csv' 
    rc_path  = '/Users/chihuanbin/Documents/apjl_v3/data/red_clump/red_clump.csv'
    
    # 1. 加载
    # 如果文件不存在，设置 mock=True 会生成测试数据看看效果
    USE_MOCK = False 
    try:
        cep_data = load_and_transform(cep_path)
        rc_data = load_and_transform(rc_path)
    except Exception as e:
        print(f"[WARN] Error loading files: {e}. Switching to Mock Data for demo.")
        cep_data = load_and_transform("mock", is_mock=True)
        rc_data = load_and_transform("mock", is_mock=True)
    
    # 2. [新增] 距离筛选 (Sub-sample Analysis)
    # 移除距离太远(误差大)或太近(本地气泡干扰)的数据
    R_min, R_max = 6.0, 11.0 
    print(f"[INFO] Filtering data: {R_min} < R < {R_max} kpc")
    cep_data = cep_data[(cep_data['R'] >= R_min) & (cep_data['R'] <= R_max)].copy()
    rc_data = rc_data[(rc_data['R'] >= R_min) & (rc_data['R'] <= R_max)].copy()

    # 3. 运行 LPSR (背景减除)
    # 注意：LPSR 使用空间 XYZ，不依赖 Spiral Phase，所以先运行
    cep_analyzed = run_lpsr_matching(cep_data, rc_data)
    
    # 4. [新增] 相位角优化 (Pitch Angle Optimization)
    # 我们只对减除背景后的 Delta VR 优化，看看哪个角度让激波最明显
    best_pitch = optimize_pitch_angle(cep_analyzed, angle_range=(8, 16), step=1.0)
    
    # 5. 使用最佳角度计算最终相位
    cep_analyzed['theta_sp'] = compute_spiral_phase(cep_analyzed, best_pitch)
    
    # 6. [新增] Bootstrapped 绘图
    plot_apjl_bootstrapped(cep_analyzed)
    
    print("Done. Check 'apjl_cepheid_shock_bootstrapped.pdf'")
