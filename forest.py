import hiive.mdptoolbox 
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import datetime

p, r = hiive.mdptoolbox.example.forest(S=1000, r1=8, r2=4, p=0.1, is_sparse=False)

vi = hiive.mdptoolbox.mdp.ValueIteration(p, r, 0.97, epsilon=.04)
x = vi.run()
time = 0
print(x[-1])
for i in x:
    time+= i["Time"] 
print('\nValue Itteration:')
#print('value itteration optimal policy: ' + str(vi.policy))
print('value itteration total time: ' + str(time))
print('value itteration time to converge: '+ str(vi.time))
print('value itteration ave time per itteration : ' + str(time/len(x)))
print('value itteration # itterations : ' + str(len(x)))
print('value itteration Max Value : ' + str(x[-1]['Max V']))


print('\n\nPolicy Itteration:')
pi = hiive.mdptoolbox.mdp.PolicyIteration(p ,r, 0.98)
x= pi.run()
time = 0
for i in x:
    time+= i["Time"] 
#print('policy itteration optimal policy: ' + str(pi.policy))
print('policy itteration total time: ' + str(time))
print('policy itteration time to converge: '+ str(pi.time))
print('policy itteration ave time per itteration : ' + str(time/len(x)))
print('policy itteration # itterations : ' + str(len(x)))
print('policy itteration Max Value : ' + str(x[-1]['Max V']))

print('\n\nQ-learning:')
start = datetime.datetime.now()
q = hiive.mdptoolbox.mdp.QLearning(p ,r, 0.98, epsilon=.09)
x = q.run()
stop = datetime.datetime.now()
time = 0
for i in x:
    time+= i["Time"] 
#print('q-learning optimal policy: ' + str(q.policy))
t= ((stop - start).total_seconds())
print('q-learning total time: ' + str(t))
print('q-learning time to converge: '+ str(q.time))
print('q-learning ave time per itteration : ' + str(time/len(x)))
print('q-learning # itterations : ' + str(len(x)))
print('q-learning itteration Max Value : ' + str(x[-1]['Max V']))
print('\n')
