from pymatgen import *
from numpy import zeros, mean
from sklearn import *
import matplotlib.pyplot as plt
import bucket
from sklearn.metrics import accuracy_score

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
    if(float(split[1]) == 0):
        x = 1
    material = Composition(split[0])
    materials.append(material)
    naiveFeatures.append(naiveVectorize(material))
    bandgaps.append(float(split[1]))

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
plotter2 = {}
plotter3 = {}
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
    theseFeatures.append(atomicNo[0] + atomicNo[1])
    physicalFeatures.append(theseFeatures)
    ZZ = 0
    for z in atomicNo:
        ZZ += z
    atmno.append(ZZ)
    plotter[bandgaps[it]] = ZZ
    plotter2[bandgaps[it]] = eneg[0] - eneg[1]
    plotter3[bandgaps[it]] = fraction[0] / fraction[1]
    it += 1

linear = linear_model.Ridge(alpha=0.5)

f, (ax1, ax2, ax3) = plt.subplots(3)

c = sorted(plotter3.iteritems(), key=lambda (x, y): float(x))
key1 = []
val1 = []
for j in c:
    key1.append(j[0])
    val1.append(j[1])
ax1.plot(val1, key1, 'g.')
ax1.set_title('Atomic fraction, Electro negativity difference, ' +
              ' Molecular Weight')

# ax1.xlabel('Atomic Fraction')
# ax1.ylabel('Band Gap')


d = sorted(plotter2.iteritems(), key=lambda (x, y): float(x))
key2 = []
val2 = []
for k in d:
    key2.append(k[0])
    val2.append(k[1])
ax2.plot(val2, key2, 'b.')

# ax2.xlabel('Electro negativity difference')
# ax2.ylabel('Band Gap')


b = sorted(plotter.iteritems(), key=lambda (x, y): float(x))

key = []
val = []
for i in b:
    key.append(i[0])
    val.append(i[1])

# ax3.xlabel('Molecular weight')
# ax3.ylabel('Band Gap')
ax3.plot(val, key, "r.")

f.text(0.06, 0.5, 'Band Gap', ha='center',
       va='center', rotation='vertical')

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


# Using Decision Tree to classify
# training set size:3000 test set size:1096
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
train_X = physicalFeatures[0:3000]
convertedBandgap = bucket.create_bucket(bandgaps)
train_Y = convertedBandgap[0:3000]
test_X = physicalFeatures[3000:4096]
test_Y = convertedBandgap[3000:4096]
clf = DecisionTreeClassifier()
clf.fit(train_X, train_Y)
predict = clf.predict(test_X)
print ("Accuracy using Decision Tree Classifier is : " +
       str(accuracy_score(test_Y, predict) * 100) + "%")
print ("Number of correct predictions:" +
       str(accuracy_score(test_Y, predict, normalize=False)))
