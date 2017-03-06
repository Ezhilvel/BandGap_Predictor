from pymatgen import *
from numpy import zeros, mean
from sklearn import *


trainFile = open("bandgapDFT.csv", "r").readlines()


def naiveVectorize(composition):
    vector = zeros((MAX_Z))  # creates a vector of 100 dummy elements
    for element in composition:
        fraction = composition.get_atomic_fraction(element)
        vector[element.Z - 1] = fraction
    return(vector)

# Extract materials and band gaps into lists, and construct naive feature set
materials = []
bandgaps = []
naiveFeatures = []

MAX_Z = 100  # maximum length of vector to hold naive feature set

for line in trainFile:
    split = str.split(line, ',')  # "H2O,1.134" => ["H2O", "1.123"]
    material = Composition(split[0])  # H2, O1
    materials.append(material)  # store chemical formulas
    # create features from chemical formula
    naiveFeatures.append(naiveVectorize(material))
    bandgaps.append(float(split[1]))  # store numerical values of band gaps


# Establish baseline accuracy by "guessing the average" of the band gap set
# A good model should never do worse.
baselineError = mean(abs(mean(bandgaps) - bandgaps))
print("The MAE of always guessing the average band gap is: " +
      str(round(baselineError, 3)) + " eV")


# alpha is a tuning parameter affecting how regression deals with
# collinear inputs
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

for material in materials:
    theseFeatures = []
    fraction = []
    atomicNo = []
    eneg = []
    group = []
    oxidation_states = []
    for element in material:
        fraction.append(material.get_atomic_fraction(element))
        atomicNo.append(float(element.Z))
        eneg.append(element.X)
        group.append(float(element.group))
        oxidation_states.append(element.min_oxidation_state)
    # print eneg
    # We want to sort this feature set
    # according to which element in the binary compound is more abundant
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
    theseFeatures.append(oxidation_states[0])
    theseFeatures.append(oxidation_states[1])
    physicalFeatures.append(theseFeatures)

# alpha is a tuning parameter affecting how regression deals with
# collinear inputs
linear = linear_model.Ridge(alpha=0.5)

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


# using random forest trees
rfr = ensemble.RandomForestRegressor(n_estimators=50)
scores = cross_validation.cross_val_score(
    rfr,
    physicalFeatures,
    bandgaps,
    cv=cv,
    scoring='mean_absolute_error')

print("The MAE of random forrest using physicalFeatures feature set is: " +
      str(round(abs(mean(scores)), 3)) + " eV")
