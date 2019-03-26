
input='''










8.8	4.41	5.87	6	4.78	5.32	2.8	3.78	3.22





'''

lst = input.strip().split('\t')
newlst = []
for i in [3,4,5]:
    newlst.append(lst[i])
for i in [6,7,8]:
    newlst.append(lst[i])
for i in [0,1,2]:
    newlst.append(lst[i])

print ' & '.join(['%.1f'%float(val) for val in newlst])