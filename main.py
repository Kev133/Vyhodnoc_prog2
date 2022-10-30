import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
kolik_bodu=200
#volné vekotry pro normalizaci
measured_valuesN=[]
real_valuesN=[]
guessN=[]
# charakteristika sondy
impulse_response=np.linspace(0,0.5,num=10)
#impulse_response=[0.0001,15,-2,3,-1,-1,-1,1,-1,1,-1]

#zavedeni promenne tau
tau=np.linspace(0,10,num=kolik_bodu)

#nastrel x0 pro optimalizaci
x0=len(tau)*[0.5]

#abs.funkcni hodnoty sin(tau)
real_values=abs(np.sin(tau))

#normalizace sin(tau) hodnot
for i in range (0,len(real_values)):
    real_valuesN.append(
        (real_values[i]-max(real_values))/(min(real_values)-max(real_values)))

# tvorba namerenych hodnot = konvoluce real_hodnot a impulsni charakteristiky
#otazka jestli pouzivat valid, nebo same, nebo full jako mod convoluce, musim se na to zeptat nekoho
measured_values=np.convolve(real_valuesN,impulse_response,"valid")

#measured values nemusi byt normalizovany, ze sondy uz budou ale to je bonus
#takze tahle cast nic nedela
# for i in range (0,len(measured_values)):
#     measured_valuesN.append(
#         (measured_values[i]-max(measured_values))/(min(measured_values)-max(measured_values)))
#

# funkce pres kterou iteruje optimalize.minimize
def to_opt(x):
    return sum((np.convolve(x,impulse_response,"valid")-measured_values)**2)

#odhad skutecnych hodnot dane funkce, pres optimalizaci
guess = scipy.optimize.minimize(to_opt,x0)

#normalizace odhadu,nemusi se delat, mam tady jen jako pojistku
#NOT USED
# for i in range (0,len(measured_values)):
#     guessN.append(
#         (guess.x[i]-max(guess.x[0:kolik_bodu]))/(min(guess.x[0:kolik_bodu])-max(guess.x[0:kolik_bodu]))
#         )

#vykresleni do grafu

fig, axes = plt.subplots(1,1)
axes.plot(real_valuesN, marker="o", label="Opravdové hodnoty", color='tab:blue')
axes.plot(measured_values, marker=".", label="Naměřené hodnoty", color='tab:red')
axes.plot(guess.x[0:kolik_bodu], marker=".", label="Odhah po optimalizaci", color='tab:green')
axes.legend(['Opravdové hodnoty', 'Naměřené hodnoty', 'Odhad po optimalizaci'])
# axes.set_yscale('log')
axes.set_xlabel("x")
axes.set_ylabel("y")
fig.suptitle('Srovnání funkcí před a po konvolucí')
plt.show()