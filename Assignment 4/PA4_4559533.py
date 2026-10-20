import pysubdisc
import pandas
import matplotlib.pyplot as plt


# Load the Adult data
data = pandas.read_csv('adult.txt')

# Examine input data
table = pysubdisc.loadDataFrame(data)
print(table.describeColumns())



print('\n\n******* Section 1 *******\n')

# SECTION 1
# Set up SD with default settings, based on a 'single nominal' setting
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')

# Print the default settings
print(sd.describeSearchParameters())

# Do the actual run
sd.run()

# Print the subgroups
print(sd.asDataFrame())


print('\n\n******* Section 2 *******\n')

# SECTION 2
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.qualityMeasureMinimum = 0.1

# BEGIN IMPLEMENTATION
sd.numericStrategy = 'NUMERIC_BEST'
# END IMPLEMENTATION

sd.run(verbose=False)

print(sd.asDataFrame())


print('\n\n******* Section 3 *******\n')

# SECTION 3
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'

# BEGIN IMPLEMENTATION
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasureMinimum = 0.25
# END IMPLEMENTATION

sd.run(verbose=False)

print(sd.asDataFrame())



print('\n\n******* Section 4 *******\n')

# SECTION 4
sd_no_filter = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd_no_filter.qualityMeasure = 'CORTANA_QUALITY'

# BEGIN IMPLEMENTATION
sd_no_filter.numericStrategy = 'NUMERIC_BEST'
sd_no_filter.searchDepth = 2
sd_no_filter.qualityMeasureMinimum = 0.25
sd_no_filter.filterSubgroups = False
# END IMPLEMENTATION

sd_no_filter.run(verbose=False)

print(sd_no_filter.asDataFrame())

print("Subgroup count with filtering turned ON: ", len(sd.asDataFrame()))	# reusing the result from Section 3 here
print("Subgroup count with filtering turned OFF: ", len(sd_no_filter.asDataFrame()))

# Compute pattern team of size 3 from the found subgroups

# BEGIN IMPLEMENTATION

patternTeam, grouping = sd_no_filter.getPatternTeam(3, returnGrouping=True)
print(patternTeam)

# END IMPLEMENTATION

print('\n\n******* Section 5 *******\n')

# SECTION 5
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')

# BEGIN IMPLEMENTATION
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasure = "RELATIVE_LIFT"
sd.qualityMeasureMinimum = 0
# END IMPLEMENTATION

sd.run(verbose=False)

print(sd.asDataFrame())



print('\n\n******* Section 6 *******\n')

# SECTION 6

# BEGIN IMPLEMENTATION
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasure = "RELATIVE_LIFT"
sd.qualityMeasureMinimum = 3
sd.minimumCoverage = 5
# END IMPLEMENTATION

sd.run(verbose=False)

print(sd.asDataFrame())



print('\n\n******* Section 7 *******\n')

# SECTION 7
# BEGIN IMPLEMENTATION
sd = pysubdisc.singleNumericTarget(data, 'age')
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasure = "Z_SCORE"
sd.qualityMeasureMinimum = 0
sd.minimumCoverage = 100
# END IMPLEMENTATION

sd.run(verbose=False)

print("Average age in the data: ", data['age'].mean())
print(sd.asDataFrame())



print('\n\n******* Section 8 *******\n')

# SECTION 8
# run 100 swap-randomised SD runs in order to determine the minimum required quality to reach a significance level alpha = 0.05

# BEGIN IMPLEMENTATION
sd = pysubdisc.singleNumericTarget(data, 'age')
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasure = "Z_SCORE"
sd.minimumCoverage = 100
threshold = sd.computeThreshold(significanceLevel=0.05, method='SWAP_RANDOMIZATION', amount=100, setAsMinimum=True)
# END IMPLEMENTATION

sd.run(verbose=False)

print("Minimum quality for significance: ", sd.qualityMeasureMinimum)
print(sd.asDataFrame())

