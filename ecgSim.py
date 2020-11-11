import math
import random
import matplotlib.pyplot as plt

class Simulator2:



    def init(self, nHeartRateMin,  nHeartRateMax,  nDeviationRR,  nWaveLength,  fAmplitudeInMV):
        self.nHeartRateMin = nHeartRateMin
        self.nHeartRateMax = nHeartRateMax
        self.nDeviationRR = nDeviationRR
        self.nWaveLength = nWaveLength
        self.dAmplitudeInMV = fAmplitudeInMV
        self.wCurve = [426, 426, 426,
                   426, 426, 426, 426, 426, 426, 426, 426, 426, 426, 425, 424, 423, 422, 420, 418, 415, 412, 408, 405,
                   400, 396, 393, 391, 389, 387,
                   386, 385, 385, 385, 385, 386, 387, 388, 389, 392, 393, 396, 399, 402, 405, 408, 412, 415, 419, 423,
                   428, 432, 436, 440, 444, 447,
                   450, 453, 455, 456, 457, 457, 457, 455, 451, 446, 435, 409, 386, 362, 338, 316, 293, 271, 249, 226,
                   205, 183, 162, 139, 119, 97, 75,
                   53, 31, 6, 36, 81, 136, 193, 244, 290, 339, 388, 442, 498, 552, 604, 593, 583, 573, 561, 550, 540,
                   528, 518, 509, 499, 490, 481, 472,
                   464, 457, 450, 442, 437, 433, 429, 427, 425, 423, 421, 420, 418, 417, 415, 413, 412, 410, 408, 406,
                   404, 403, 400, 398, 396, 394,
                   392, 390, 387, 385, 382, 379, 377, 374, 371, 367, 364, 361, 357, 354, 350, 346, 342, 338, 334, 330,
                   326, 322, 319, 315, 312, 309,
                   306, 303, 301, 299, 297, 295, 294, 293, 293, 293, 293, 294, 296, 298, 300, 302, 305, 309, 312, 315,
                   319, 324, 328, 333, 337, 342,
                   348, 354, 360, 366, 372, 378, 384, 389, 395, 400, 405, 409, 412, 416, 418, 421, 423, 424, 425, 426,
                   426, 427, 427, 427, 428, 428,
                   428, 428, 428, 428, 428, 428, 428, 428, 428, 428, 427, 427, 427, 427, 427, 427, 427, 427, 427, 427,
                   426, 426, 426, 426, 426, 426]

        self.curveCount=self.wCurve.__len__()
        dScaleTo1Mv = 0.683 * self.dAmplitudeInMV

        nFirstValue = self.wCurve[0]
        for i in range(0,self.wCurve.__len__()):
            self.wCurve[i]=(2047-(nFirstValue) * dScaleTo1Mv + (self.wCurve[i]) * dScaleTo1Mv)


        self.nRRMin=60000 / self.nHeartRateMax
        self.nRRMax=60000 / self.nHeartRateMin
        self.nTransitions=self.nWaveLength / 2
        self.nDeviation=self.nDeviationRR

        self.nPoint=240
        self.fRad=(3.14) * 1.5
        self.nRR=self.nRRMin


    def getEcgValue(self):

        wValue = 0
        if (self.nPoint < self.curveCount):

            wValue = self.wCurve[self.nPoint]
            self.nPoint += 1

        else:
            wValue = self.wCurve[self.wCurve.__len__()-1]
            self.nPoint += 1

        if (self.nPoint >= self.nRR):

            self.nPoint = 0
            self.fRad += 3.14 / (self.nTransitions)

            t0 = self.fRad

            t1 = (0.5 * (math.sin(t0) + 1.0))


            t2 = (self.nRRMax - self.nRRMin)
            self.nRR = self.nRRMin + (t1 * t2 * 0.5)

            if (self.nDeviation != 0):

                dScaleRR = 1.0 + random.random() % ((((self.nDeviation) + 1) - (self.nDeviation) / 2) / 100)
                if (dScaleRR < 0.0):
                    dScaleRR=0

                self.nRR=((self.nRR) * dScaleRR)




        return wValue

    def simulate(self):
        out=[]
        for i in range(0,11):
            out.append(self.getEcgValue())
        return out


a=Simulator2()
a.init(100,150,100,200,2.0)
b=[]
for i in range(0,1000):
    c=a.simulate()
    for j in range(0,c.__len__()):
        b.append(c[j])

fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
myplot = fig.add_subplot(1,1,1)
myplot.plot(b, linewidth=0.5, color='b')
plt.show()