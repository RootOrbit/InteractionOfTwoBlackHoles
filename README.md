import numpy as np
from PIL import Image

"""

Made by Cyberft

"""

width, height = 2000, 1200
# x = (m-1000)/500, y = (601-n)/600 pro m=1..2000, n=1..1200
m = np.linspace(1, width, width)    
n = np.linspace(1, height, height)  
# 
x_vals = (m - 1000) / 500.0                   
y_vals = (601 - n) / 600.0                    
x_grid, y_grid = np.meshgrid(x_vals, y_vals)  



def F(x):
    """Funkce F(x) = 255 * e^{-1000 * x^-} / (1 + e^{1000 * x^-}), kde x^- je záporná část x."""
    #x_minus = max(-x, 0) 
    x_minus = np.where(x < 0, -x, 0)
    #např. 50
    t = 1000 * x_minus
    t = np.clip(t, None, 50) 
    return 255.0 / (1.0 + np.exp(t))

def B(x, y):
    """Funkce B(x,y) = \sum_{s=1}^{37} s * cos(s * x). (Součet s váhami, využívá hodnotu x)"""
    
    s = np.arange(1, 38)[:, None, None]  #tvar (37,1,1) 
    # s * x_grid má tvar 
    cos_terms = np.cos(s * x)  # cos(s * x) 
    # \sum_{s=1}^{37} s * cos(s*x)
    B_val = np.sum(s * cos_terms, axis=0)
    return B_val

def K(x, y):
    """Funkce K(x,y) = \sum_{s=1}^{37} cos(s * x)."""
    s = np.arange(1, 38)[:, None, None]
    return np.sum(np.cos(s * x), axis=0)

def T(x, y):
    """Funkce T(x,y) = \sum_{s=1}^{37} sin(s * x)."""
    s = np.arange(1, 38)[:, None, None]
    return np.sum(np.sin(s * x), axis=0)

#Úhel pro rotaci souřadnic (72 stupňů = 2π/5) používá se v součtech přes index u
phi = 2 * np.pi / 5.0

def A(x, y):
    """Funkce A(x,y) = \sum_{u=1}^{5} cos(u * (x*cos(phi) + y*sin(phi)))."""
    u = np.arange(1, 6)[:, None, None]
    # Rotovaná lineární kombinace souřadnic pro index u x*cos(phi) + y*sin(phi)
    U_lin = x * np.cos(phi) + y * np.sin(phi)
    return np.sum(np.cos(u * U_lin), axis=0)

def U_func(x, y):
    """Funkce U(x,y) = \sum_{u=1}^{5} sin(u * (x*cos(phi) + y*sin(phi)))."""
    u = np.arange(1, 6)[:, None, None]
    U_lin = x * np.cos(phi) + y * np.sin(phi)
    return np.sum(np.sin(u * U_lin), axis=0)

def L_val(x, y, u_index, s_index):
    """Funkce L_{u,s}(x,y) = cos(u * (x*cos(phi) + y*sin(phi)) + s * x). 
       (Pomocná funkce závislá na indexech u a s)"""
    return np.cos(u_index * (x * np.cos(phi) + y * np.sin(phi)) + s_index * x)

def P(x, y):
    """Funkce P(x,y) = A(x,y) * K(x,y) - U(x,y) * T(x,y).
       (Ekvivalentní \sum_{u=1}^5 \sum_{s=1}^{37} cos(u * (x*cos(phi)+y*sin(phi)) + s*x))"""
    return A(x, y) * K(x, y) - U_func(x, y) * T(x, y)

def Q(x, y):
    """Funkce Q(x,y) = U(x,y) * K(x,y) + A(x,y) * T(x,y).
       (Ekvivalentní \sum_{u=1}^5 \sum_{s=1}^{37} sin(u * (x*cos(phi)+y*sin(phi)) + s*x))"""
    return U_func(x, y) * K(x, y) + A(x, y) * T(x, y)

def H0(x, y):
    """Funkce H0(x,y) dle obrázku: H0 = 1 - exp(-50 * (y^2 + B(x,y)/5 + 37)) + 20 * K(x,y) / (5 + 37)."""
    return (1.0 - np.exp(-50.0 * (y**2 + B(x, y) / 5.0 + 37.0))) + (20.0 * K(x, y)) / (5.0 + 37.0)

def H1(x, y):
    """Funkce H1(x,y) = P(x,y) (přímé přiřazení výsledku P)."""
    return P(x, y)

def H2(x, y):
    """Funkce H2(x,y) = Q(x,y) (přímé přiřazení výsledku Q)."""
    return Q(x, y)

H0_vals = H0(x_grid, y_grid)
H1_vals = H1(x_grid, y_grid)
H2_vals = H2(x_grid, y_grid)

R = F(H0_vals)
G = F(H1_vals)
B_channel = F(H2_vals)

R_uint8 = np.clip(np.rint(R), 0, 255).astype(np.uint8)
G_uint8 = np.clip(np.rint(G), 0, 255).astype(np.uint8)
B_uint8 = np.clip(np.rint(B_channel), 0, 255).astype(np.uint8)
rgb_array = np.stack([R_uint8, G_uint8, B_uint8], axis=2)  

image = Image.fromarray(rgb_array, mode='RGB')
image.save('interacting_black_holes.png')
print("Obrázek uložen jako interacting_black_holes.png")
