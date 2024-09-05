import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt

Datos_semanas = [0, 1, 2]   #0 = semana 1. 1= semana 2. 2= semana 3
Susceptibles = [340000, 340000, 340000]
Infectados = [1111, 11444, 34670]
Muertos = [0, 480, 2500]

def SIR_model(y, t, beta, gamma, mu):
    S, I, R, M = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - mu * I
    dRdt = gamma * I
    dMdt = mu * I
    return [dSdt, dIdt, dRdt, dMdt]

S = np.array(Susceptibles)
I = np.array(Infectados)
M = np.array(Muertos)

gamma = float(input("Ingrese el valor de gamma (tasa de recuperación): "))       #En vez de estimar gamma y beta, cambiamos esta parte para introducir manualmente ambos valores.
beta = float(input("Ingrese el valor de beta (tasa de transmisión): "))
#mu = np.mean(mu_estimado)
mu = 0.3045                                                                      #Tasa de mortalidad, agregamos este parametro para agregar robustez al modelo.
Semanas = np.linspace(0, 10, 100)

y0= [S[0], I[0], 0, M[0]]

sol = odeint(SIR_model, y0, Semanas, args=(beta, gamma, mu))

S_sol, I_sol, R_sol, M_sol = sol.T
#print(Semanas, R_sol)                                                            #Para saber exactamente el valor de personas S, I, R o muertas agregamos un print().

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
