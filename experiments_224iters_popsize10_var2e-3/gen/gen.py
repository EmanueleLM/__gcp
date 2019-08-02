"""
   Generate scores and seeds for this batch of experiments
"""
import matlpotlib.pyplot as plt
import numpy as np

seeds = []
for i in range(223):
    file_ = 'seeds_gen_' + str(i) + '.csv'
    print(file_)
    with open(file_) as fp:
        if i != 0:
            seeds.append(fp.readlines())
        else:
            seeds.append(list(fp.readlines()))
		
scores = []
for i in range(223):
    file_ = 'seeds_score_gen_' + str(i) + '.csv'
    print(file_)
    with open(file_) as fp:
        scores.append(fp.readlines())
		
# flatten the lists
seeds_f = []
for i in range(len(seeds)):
    tmp = []
    for j in range(len(seeds[i])):
	    tmp.append(seeds[i][j].split('\n')[0])
    seeds_f.append(tmp)

scores_f = []
for s in scores:
    for c in s:
       scores_f.append(int(float(c.split('\n')[0])))

scores = scores_f
seeds = []
for s in seeds_f:
    for c in s:
        seeds.append(c)

ord1, ord2 = zip(*sorted(zip(scores, seeds)))
ord1 = ord1[::-1]
ord2 = ord2[::-1]

np_ord1 = np.array(ord1)
np_ord2 = np.array(ord2)

# histogram' values
np.histogram(np_ord1)

# nets with minimum/maximum score
len(np_ord1[np_ord1==min(np_ord1)])
len(np_ord1[np_ord1==max(np_ord1)])

# plot histogram
n, bins, patches = plt.hist(np_ord1, 10, facecolor='blue', alpha=0.5)
plt.xlabel('Punteggio')
plt.ylabel('Numero reti')
plt.title('Reti neurali e punteggio')
plt.show()

# save seeds and scores (9gb! for the seeds!)
np.savetxt("seeds_ordered_224gen5000nn.csv", np_ord2, delimiter=",")
np.save("seeds_ordered_224gen5000nn.csv", np_ord2, allow_pickle=True)