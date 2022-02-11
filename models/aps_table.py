import numpy as np

metric_m = []
metric_q = []
for i in range(10):
    for d in ('m', 'q'):
        f = open('output/results_'+d+'_'+str(i)+'_.txt')
        lines = f.readlines()
        f.close()
        if d == 'm':
            print(i+1,d,'& median & ',lines[3].split(' ')[1][:6], ' & ',lines[14].split(' ')[1][:-1], '/',lines[9].split(' ')[1][:-1],\
                  ' & ', lines[15].split(' ')[1][:-1], '/', lines[10].split(' ')[1][:-1], '\\\\')
            metric_m.append([float(lines[3].split(' ')[1]),float(lines[14].split(' ')[1][:-1]),float(lines[9].split(' ')[1][:-1]),\
                             float(lines[15].split(' ')[1][:-1]), float( lines[10].split(' ')[1][:-1])])
            
        elif d == 'q':
            print(i+1,d,'& quantile & ',lines[3].split(' ')[1][:6], ' & ',lines[14].split(' ')[1][:-1], '/',lines[9].split(' ')[1][:-1],\
                  ' & ', lines[15].split(' ')[1][:-1], '/', lines[10].split(' ')[1][:-1], '\\\\')
            metric_q.append([float(lines[3].split(' ')[1]),float(lines[14].split(' ')[1][:-1]),float(lines[9].split(' ')[1][:-1]),\
                             float(lines[15].split(' ')[1][:-1]), float( lines[10].split(' ')[1][:-1])])
            
print(np.mean(metric_m, axis=0))
print(np.mean(metric_q, axis=0))
print((np.mean(metric_q, axis=0) - np.mean(metric_m, axis=0) ) / np.mean(metric_m, axis=0) * 100, '%')


ps = input('EB table')

metric_m = []
metric_q = []
for i in range(10):
    for d in ('m', 'q'):
        f = open('output/results_'+d+'_'+str(i)+'_.txt')
        lines = f.readlines()
        f.close()
        if d == 'm':
            print(i+1,d,'& median & ',lines[1].split(' ')[1][:6], ' & ',lines[26].split(' ')[1][:-1], '/',lines[21].split(' ')[1][:-1],\
                  ' & ', lines[27].split(' ')[1], '/', lines[22].split(' ')[1][:-1], '\\\\')
            metric_m.append([float(lines[1].split(' ')[1]),float(lines[26].split(' ')[1][:-1]),float(lines[21].split(' ')[1][:-1]),\
                             float(lines[27].split(' ')[1][:-1]), float( lines[22].split(' ')[1][:-1])])
            
        elif d == 'q':
            print(i+1,d,'& quantile & ',lines[1].split(' ')[1][:6], ' & ',lines[26].split(' ')[1][:-1], '/',lines[21].split(' ')[1][:-1],\
                  ' & ', lines[27].split(' ')[1], '/', lines[22].split(' ')[1][:-1], '\\\\')
            metric_q.append([float(lines[1].split(' ')[1]),float(lines[26].split(' ')[1][:-1]),float(lines[21].split(' ')[1][:-1]),\
                             float(lines[27].split(' ')[1][:-1]), float( lines[22].split(' ')[1][:-1])])


print(np.mean(metric_m, axis=0))
print(np.mean(metric_q, axis=0))
print((np.mean(metric_q, axis=0) - np.mean(metric_m, axis=0) ) / np.mean(metric_m, axis=0) * 100,'%')


''' 
for i in range(10):
    for d in ('m', 'q'):
        f = open('output/results_'+d+'_'+str(i)+'_.txt')
        lines = f.readlines()
        f.close()
        print(i+1,d,lines[1].split(' ')[1][:6], lines[26].split(' ')[1][:-1], lines[21].split(' ')[1][:-1],\
              lines[27].split(' ')[1], lines[22].split(' ')[1][:-1])



print('PC table')
#for metric in (3, 71, 56, 72, 57):
for metric in (3, 71, 76, 72, 77): ## aps, precision (t=0.5),  precision (t=0.25), recall (t=0.5),  recall (t=0.25)
    q_aps = []
    m_aps = []
    for i in range(10):
        #print(i, end=',')
        for d in ('m', 'q'):
            f = open('output/results_'+d+'_'+str(i)+'_.txt')
            lines = f.readlines()
            f.close()
            #print(d, float(lines[metric].split(' ')[1]), end=',')
            if d =='m':
                m_aps.append(float(lines[metric].split(' ')[1]))
            elif d =='q':
                q_aps.append(float(lines[metric].split(' ')[1]))
        #print(' ')

    print(np.mean(m_aps), np.mean(q_aps), (np.mean(q_aps)- np.mean(m_aps))/np.mean(m_aps)*100,'%')

print('EB table')
#for metric in (1, 34, 19, 35, 20):
for metric in (1, 34, 39, 35, 40):
    q_aps = []
    m_aps = []
    for i in range(10):
        #print(i, end=',')
        for d in ('m', 'q'):
            f = open('output/results_'+d+'_'+str(i)+'_.txt')
            lines = f.readlines()
            f.close()
            #print(d, float(lines[metric].split(' ')[1]), end=',')
            if d =='m':
                m_aps.append(float(lines[metric].split(' ')[1]))
            elif d =='q':
                q_aps.append(float(lines[metric].split(' ')[1]))
        #print(' ')

    print(np.mean(m_aps), np.mean(q_aps), (np.mean(q_aps)- np.mean(m_aps))/np.mean(m_aps)*100,'%')'''

'''
for i in range(10):
    print(i, end=',')
    for d in ('m', 'q'):
        f = open('output/results_'+d+'_'+str(i)+'_.txt')
        lines = f.readlines()
        f.close()
        print(d, float(lines[1].split(' ')[1]), end=',')
        if d =='m':
            m_aps.append(float(lines[1].split(' ')[1]))
        elif d =='q':
            q_aps.append(float(lines[1].split(' ')[1]))
    print(' ')

print(np.mean(m_aps), np.mean(q_aps),(np.mean(q_aps)- np.mean(m_aps))/np.mean(m_aps)*100,'%')

for i in range(10):
    print(i, end=',')
    for d in ('m', 'q'):
        f = open('output/results_'+d+'_'+str(i)+'_.txt')
        lines = f.readlines()
        f.close()
        print(d, float(lines[72].split(' ')[1]), end=',')
        if d =='m':
            m_aps.append(float(lines[72].split(' ')[1]))
        elif d =='q':
            q_aps.append(float(lines[73].split(' ')[1]))
    print(' ')

print(np.mean(m_aps), np.mean(q_aps),(np.mean(q_aps)- np.mean(m_aps))/np.mean(m_aps)*100,'%')'''
