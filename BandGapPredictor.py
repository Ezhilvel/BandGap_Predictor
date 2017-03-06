from pymatgen import *
from numpy import zeros, mean
from sklearn import *
import matplotlib.pyplot as plt

trainFile = open("bandgapDFT.csv", "r").readlines()


def naiveVectorize(composition):
    vector = zeros((MAX_Z))
    for element in composition:
        fraction = composition.get_atomic_fraction(element)
        vector[element.Z - 1] = fraction
    return(vector)

materials = []
bandgaps = []
naiveFeatures = []

MAX_Z = 100

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    materials.append(material)
    naiveFeatures.append(naiveVectorize(material))
    bandgaps.append(float(split[1]))

# plt.plot(bandgaps)
# plt.show()

baselineError = mean(abs(mean(bandgaps) - bandgaps))
print("The MAE of always guessing the average band gap is: " +
      str(round(baselineError, 3)) + " eV")


linear = linear_model.Ridge(alpha=0.5)

cv = cross_validation.ShuffleSplit(len(bandgaps),
                                   n_iter=10, test_size=0.1, random_state=0)

scores = cross_validation.cross_val_score(
    linear,
    naiveFeatures,
    bandgaps,
    cv=cv,
    scoring='mean_absolute_error')

print("The MAE of the linear ridge using the naive features: " +
      str(round(abs(mean(scores)), 3)) + " eV")

physicalFeatures = []

atmno = []
plotter = {}
it = 0

for material in materials:
    theseFeatures = []
    fraction = []
    atomicNo = []
    eneg = []
    group = []

    for element in material:
        fraction.append(material.get_atomic_fraction(element))
        atomicNo.append(float(element.Z))
        eneg.append(element.X)
        group.append(float(element.group))
    mustReverse = False

    if fraction[1] > fraction[0]:
        mustReverse = True

    for features in [fraction, atomicNo, eneg, group]:
        if mustReverse:
            features.reverse()
    theseFeatures.append(fraction[0] / fraction[1])
    theseFeatures.append(eneg[0] - eneg[1])
    theseFeatures.append(group[0])
    theseFeatures.append(group[1])
    physicalFeatures.append(theseFeatures)
    ZZ = 0
    for z in atomicNo:
        ZZ += z
    atmno.append(ZZ)
    plotter[bandgaps[it]] = ZZ
    it += 1

linear = linear_model.Ridge(alpha=0.5)


plt.plot(atmno, bandgaps)
plt.show()

cv = cross_validation.ShuffleSplit(len(bandgaps),
                                   n_iter=10, test_size=0.1, random_state=0)

scores = cross_validation.cross_val_score(
    linear,
    physicalFeatures,
    bandgaps,
    cv=cv,
    scoring='mean_absolute_error')

print("The MAE of the linear ridge using the physicalFeatures: " +
      str(round(abs(mean(scores)), 3)) + " eV")

rfr = ensemble.RandomForestRegressor(n_estimators=10)
scores = cross_validation.cross_val_score(
    rfr,
    physicalFeatures,
    bandgaps,
    cv=cv,
    scoring='mean_absolute_error')

print("The MAE of random forrest using physicalFeatures feature set is: " +
      str(round(abs(mean(scores)), 3)) + " eV")
