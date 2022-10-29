import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
kolik_bodu=500
#volné vekotry pro normalizaci
measured_valuesN=[]
real_valuesN=[]
guessN=[]
# charakteristika sondy
# impulse_response=np.linspace(0,3.5,num=5)
impulse_response=[0.1,0.2,0.3,0.4,0.5,0.0001,15,-2,3,-1]
#zavedeni promenne tau
tau=np.linspace(-5,5,num=kolik_bodu)
#nastrel x0 pro optimalizaci
x0=len(tau)*[1]
#abs.funkcni hodnoty sin(tau)
real_values=(abs(np.cos(tau)))
#normalizace sin(tau) hodnot
for i in range (0,len(real_values)):
    real_valuesN.append(
        (real_values[i]-max(real_values))/(min(real_values)-max(real_values))
    )
# normalizace namerených hodnot

measured_values=np.convolve(real_valuesN,impulse_response,"valid")
#measured values dont need to be normalized i think
for i in range (0,len(measured_values)):
    measured_valuesN.append(
        (measured_values[i]-max(measured_values))/(min(measured_values)-max(measured_values))
    )

def to_opt(x):
    return sum((np.convolve(x,impulse_response,"valid")-measured_values)**2)

guess = scipy.optimize.minimize(to_opt,x0)


for i in range (0,len(measured_values)):
    guessN.append(
        (guess.x[i]-max(guess.x[0:kolik_bodu-10]))/(min(guess.x[0:kolik_bodu-10])-max(guess.x[0:kolik_bodu-10]))
        )


fig, axes = plt.subplots(1,1)
axes.plot(real_valuesN, marker="o", label="Opravdové hodnoty", color='tab:blue')
axes.plot(measured_values, marker=".", label="Naměřené hodnoty", color='tab:red')
axes.plot(guess.x[0:kolik_bodu-10], marker=".", label="Odhah po optimalizaci", color='tab:green')
axes.legend(['Opravdové hodnoty', 'Naměřené hodnoty', 'Odhad po optimalizaci'])
# axes.set_yscale('log')
axes.set_xlabel("x")
axes.set_ylabel("y")
fig.suptitle('Srovnání funkcí před a po konvolucí')
plt.show()