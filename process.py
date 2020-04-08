import scipy.io as sio

cats = ["Peace","Affection","Esteem","Anticipation","Engagement","Confidence","Happiness","Pleasure","Excitement","Surprise","Sympathy","Doubt/Confusion","Disconnection","Fatigue","Embarrassment","Yearning","Disapproval","Aversion","Annoyance","Anger","Sensitivity","Sadness","Disquietment","Fear","Pain","Suffering"]

mapp = {}
for i in range(26):
    mapp[cats[i]] = i

a = sio.loadmat("Annotations.mat")

train = a['train']
val = a['val']
test = a['test']

def process(d, name):
    f = open(name, 'w')

    d = d[0]
    tot = d.shape[0]
    print(name, tot)
    for i in range(tot):
        di = d[i]
        name = di[0].item()
        folder = di[1].item()
        person = di[4][0]
        totp = person.shape[0]
        for j in range(totp):
            p = person[j].item()
            bbox = p[0][0]
            pc = p[1][0][0][0][0]
            label = [0] * 26
            totpc = len(pc)
            for k in range(totpc):
                label[mapp[pc[k].item()]] = 1
            label = [str(item) for item in label]
            label = ' '.join(label)
            f.write('%s/%s %f %f %f %f %s\n'%(folder, name, bbox[0], bbox[1], bbox[2], bbox[3], label))


process(train, 'train.txt')
#process(val, 'val.txt')
#process(test, 'test.txt')
