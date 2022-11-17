import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
import math


#TODO neslo by pridat i kLa ktere se pak teda najde? Bylo by to zajimavy tak nakodovat
# a alspon vic motivace pro me, spis ale spare time vec




kolik_bodu = 200
# volné vekotry pro normalizaci
measured_valuesN = []
real_valuesN = []
guessN = []
impulse_responseN=[]
# zavedeni promenne tau
tau = np.linspace(0, 15, num=150)
tau2 = np.linspace(0, 15, num=150)
# nastrel x0 pro optimalizaci
x0 = len(tau) * [0.5]
# charakteristika sondy
#impulse_response = np.linspace(0, 3, num=kolik_bodu)
impulse_response= np.exp(-1.5*tau2)*(-1.5*tau2)
for i in range (0,len(impulse_response)):
    impulse_responseN.append(
        (impulse_response[i]-max(impulse_response))/(min(impulse_response)-max(impulse_response)))

# abs.funkcni hodnoty sin(tau)
real_values = np.exp(-1.05*tau)
for i in range(0, len(real_values)):
    real_valuesN.append(
        (real_values[i] - max(real_values)) / (min(real_values) - max(real_values)))

# tvorba namerenych hodnot = konvoluce real_hodnot a impulsni charakteristiky
# otazka jestli pouzivat valid, nebo same, nebo full jako mod convoluce, musim se na to zeptat nekoho
measured_values=np.convolve(real_valuesN, impulse_responseN)
print(len(real_valuesN))
print(len(impulse_response))
print(len(measured_values))


# measured values nemusi byt normalizovany, ze sondy uz budou ale to je bonus
# takze tahle cast nic nedela
for i in range (0,len(measured_values)):
     measured_valuesN.append(
        (measured_values[i]-min(measured_values))/(max(measured_values)-min(measured_values)))


# funkce pres kterou iteruje optimalize.minimize
def to_opt(x):
    return sum((np.convolve(x, impulse_responseN) - measured_values) ** 2)


# odhad skutecnych hodnot dane funkce, pres optimalizaci
guess = scipy.optimize.minimize(to_opt, x0)

# normalizace odhadu,nemusi se delat, mam tady jen jako pojistku
# NOT USED
# for i in range (0,len(measured_values)):
#     guessN.append(
#         (guess.x[i]-max(guess.x[0:kolik_bodu]))/(min(guess.x[0:kolik_bodu])-max(guess.x[0:kolik_bodu]))
#         )

# vykresleni do grafu

fig, axes = plt.subplots(1, 1)
axes.plot(real_valuesN, marker="o", label="Opravdové hodnoty", color='tab:blue')
axes.plot(measured_valuesN[0:int(len(measured_valuesN)/2)], marker=".", label="Naměřené hodnoty", color='tab:red')
axes.plot(guess.x, marker=".", label="Odhah po optimalizaci", color='tab:green')
axes.plot(impulse_responseN, marker=".", label="Impulse", color='tab:orange')
axes.legend(['Opravdové hodnoty', 'Naměřené hodnoty', 'Odhad po optimalizaci', "impulzní charakteristika"])
# axes.set_yscale('log')
axes.set_xlabel("x")
axes.set_ylabel("y")
fig.suptitle('Optimalizace')
plt.show()
