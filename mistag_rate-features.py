import matplotlib
import matplotlib.pyplot as plt

n_feat = [30, 22, 20, 18, 16, 14, 12]
AUC = [0.950, 0.949, 0.948, 0.946, 0.945, 0.941, 0.936]
efficiency = [0.60004, 0.59920, 0.60074, 0.60013, 0.60041, 0.60093, 0.60065]
mistag_rate = [0.02361, 0.02421, 0.02458, 0.02528, 0.02579, 0.02771, 0.03012]


fig = plt.figure(figsize=(10, 7))

plt.plot(n_feat, mistag_rate, 'bo')
plt.title("Mistag rate ad efficienza fissata r'$\epsilon$' = 0.600 r'$\pm$' 0.001  in funzione del numero di features")
plt.xlabel("Numero di features")
plt.ylabel("Mistag rate")
plt.ylim(0.022, 0.031)
yticks = [0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030]
plt.yticks(yticks)
plt.grid()
plt.xlim(10, 32)
plt.yticks(yticks)
xticks = [10,12,14,16,18,20,22,24,26,28,30,32]
#plt.show()
plt.savefig('Mistag_rate-features.png')
