import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt

Datos_semanas = [0, 1, 2]   #0 = semana 1. 1= semana 2. 2= semana 3
Susceptibles = [340000, 340000, 340000]
Infectados = [1111, 11444, 34670]
Muertos = [0, 480, 2500]

def SIR_model(y, t, beta, gamma, mu):                                            # => Presentación matemática del modelo SIR incluyendo el valor de muertes.
    S, I, R, M = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - mu * I
    dRdt = gamma * I
    dMdt = mu * I
    return [dSdt, dIdt, dRdt, dMdt]

S = np.array(Susceptibles)
I = np.array(Infectados)
M = np.array(Muertos)

gamma_estimado = (I[1:] - I[:-1]) / I[:-1]                                       # => Estimación de gamma 
mu_estimado = (M[1:] - M[:-1]) / I[:-1]                                          # => Estimación de beta

Gamma = np.mean(gamma_estimado)
Mu = np.mean(mu_estimado)

R0_est = (S[:-1] / I[:-1]) * ((I[1:] - I[:-1]) + (M[1:] - M[:-1])) / I[:-1]      # => Estimación de R0 
beta = np.mean(R0_est * (Gamma + Mu))

Semanas = np.linspace(0, 10, 100)
y0 = [Susceptibles[0], Infectados[0], 0, Muertos[0]]
sol = odeint(SIR_model, y0, Semanas, args=(beta, Gamma, Mu))

S_sol, I_sol, R_sol, M_sol = sol.T

 
def new_func(Gamma, Mu, beta):
    print(f"Estimación de β: {beta:.4f}")
    print(f"Estimación de γ: {Gamma:.4f}")
    print(f"Estimación de μ: {Mu:.4f}")
    print(f"Estimación de R0: {beta / (Gamma + Mu):.4f}")

new_func(Gamma, Mu, beta)

#Graficar el modelo

plt.figure(figsize=(10,6))
plt.plot(Semanas, S_sol, label='Susceptibles')
plt.plot(Semanas, I_sol, label='Infectados')
plt.plot(Semanas, R_sol, label='Recuperados')
plt.plot(Semanas, M_sol, label='Muertos')
plt.xlabel('Semanas')
plt.ylabel('Número de personas')
plt.legend()
plt.title('Simulación del modelo SIR con mortalidad')
plt.show()