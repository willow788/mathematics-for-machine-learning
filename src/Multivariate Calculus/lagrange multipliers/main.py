import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def f(x, y):
    return x**2 +2*y**2 -2*x


#differentuial of f(x, y)
def grad_f(x, y):
    return np.array ([2*x - 2, 4*y])


def g(x, y):
    return  x**2 +y**2 -4

#differential of g(x, y)
def grad_g(x, y):
    return np.array([2*x, 2*y])

def lagrange_multiplier_method(vars):
    x, y, lam = vars
    fx, fy = grad_f(x, y)
    gx , gy = grad_g(x, y)

    eqn1 = fx - lam*gx
    eqn2 = fy - lam*gy
    eqn3 = g(x, y)
    #constraint circle g(x, y)= 0

    return [eqn1, eqn2, eqn3]


#multiple guesses
guesses = [
    (1, 1, 1),
    (2, 0, 1),  
    (0, 2, 1),
    (-2, 0, 1), 
    (0, -2, 1),
    (-1, -1, 1),
    (1, -1, 1),

]


sol = [] # to catch the solutions

for guess in guesses:
    soln = fsolve(lagrange_multiplier_method, guess)
    x, y, lam = soln
    pt = np.round([x, y, lam], 6)
    
    if not any(np.allclose(pt[:2], s[:2]) for s in sol):
        sol.append(pt)

print("Critical points (x, y):")
for s in sol:
    x, y, lam = s
    print(f"(x, y) = ({x:.4f}, {y:.4f}), λ = {lam:.4f}, f(x,y) = {f(x, y):.4f}")

# Visualization - Create contour plot
xv = np.linspace(-3, 3, 600)
yv = np.linspace(-3, 3, 600)
X, Y = np.meshgrid(xv, yv)
Z = f(X, Y)

# Create figure with dark theme and enhanced styling
fig, ax = plt.subplots(figsize=(14, 12), facecolor='#0a0e27')
ax.set_facecolor('#0d1117')

# Multi-layer contour effect with vibrant colors
contourf1 = plt.contourf(X, Y, Z, levels=80, cmap='twilight_shifted', alpha=0.85)
contourf2 = plt.contourf(X, Y, Z, levels=40, cmap='rainbow', alpha=0.3)

# Enhanced colorbar
cbar = plt.colorbar(contourf1, label='f(x, y)', pad=0.02, shrink=0.8)
cbar.ax.tick_params(labelsize=11, colors='cyan')
cbar.set_label('f(x, y)', fontsize=14, weight='bold', rotation=270, labelpad=25, color='cyan')
cbar.outline.set_edgecolor('cyan')
cbar.outline.set_linewidth(2)

# Vibrant contour lines
contours = plt.contour(X, Y, Z, levels=30, cmap='cool', linewidths=1.5, alpha=0.7)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f', colors='cyan')

# Constraint circle with ULTRA glow effect (neon style)
theta = np.linspace(0, 2*np.pi, 600)
circle_x = 2 * np.cos(theta)
circle_y = 2 * np.sin(theta)

# Multiple glow layers for neon effect
for width, alpha in [(18, 0.12), (14, 0.2), (10, 0.3), (6, 0.5)]:
    plt.plot(circle_x, circle_y, color='#00ffff', linewidth=width, alpha=alpha, zorder=3)

# Main constraint line - bright cyan
plt.plot(circle_x, circle_y, color='#00ffff', linewidth=5, zorder=4, 
         label='Constraint: $x^2 + y^2 = 4$', linestyle='-')

# Plot critical points with MEGA enhanced styling
colors = ['#ff00ff', '#ffff00', '#00ff00', '#ff0080']
point_labels = ['Maximum', 'Minimum', 'Saddle 1', 'Saddle 2']

for idx, s in enumerate(sol):
    x, y, lam = s
    color = colors[idx % len(colors)]
    
    # Multi-layer glow effect
    for size, alpha in [(1200, 0.08), (900, 0.12), (650, 0.18), (450, 0.28)]:
        plt.scatter(x, y, c=color, s=size, alpha=alpha, zorder=5, edgecolors='none')
    
    # Main point with thick border
    plt.scatter(x, y, c=color, s=300, alpha=1, zorder=8, 
                edgecolors='white', linewidths=4)
    
    # Inner sparkle
    plt.scatter(x, y, c='white', s=100, alpha=0.95, zorder=9, marker='*')
    
    # Value label with dramatic styling
    value = f(x, y)
    plt.text(x+0.25, y+0.25, f'★ {value:.2f}', fontsize=12, fontweight='heavy',
             color='white', zorder=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.6', facecolor=color, 
                      edgecolor='white', linewidth=3, alpha=0.98))

# Enhanced labels with glow effect
plt.xlabel('x', fontsize=16, weight='heavy', color='#00ffff', style='italic', labelpad=12)
plt.ylabel('y', fontsize=16, weight='heavy', color='#00ffff', style='italic', labelpad=12)
plt.title('⚡ Lagrange Multipliers: Constrained Optimization ⚡', 
          fontsize=18, weight='heavy', pad=20, color='#00ffff', style='italic',
          bbox=dict(boxstyle='round,pad=1', facecolor='#1a1a2e', 
                   edgecolor='#00ffff', linewidth=3, alpha=0.9))

plt.axis('equal')

# Neon-style grid
plt.grid(True, alpha=0.25, linestyle='--', linewidth=1, color='#00ffff')

# Stylized legend with glow
legend = plt.legend(fontsize=13, loc='upper right', framealpha=0.95,
                   fancybox=True, shadow=True, edgecolor='#00ffff', 
                   facecolor='#0d1117', labelcolor='#00ffff')
legend.get_frame().set_linewidth(2)

# Enhanced ticks and spines
ax.tick_params(labelsize=12, colors='#00ffff', width=2, length=7)
for spine in ax.spines.values():
    spine.set_edgecolor('#00ffff')
    spine.set_linewidth(3)

# Add subtle info text
fig.text(0.99, 0.01, 'Constrained Optimization', ha='right', fontsize=11, 
         color='#00ffff', alpha=0.7, style='italic', weight='bold')

plt.subplots_adjust(top=0.92)
plt.tight_layout(rect=[0, 0.02, 1, 0.92])
plt.show()


