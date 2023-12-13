import pandas as pd
import os
from GA import select

current_dir = os.getcwd()
data_folder_path = os.path.join(current_dir, 'examples/data')
file_path = os.path.join(data_folder_path, 'communities.data')
data = pd.read_csv(file_path, delimiter=',')
crime_data = data.iloc[:, 5:] #remove some parameters
new_header = ["population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp",
              "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome",
              "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
              "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
              "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
              "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
              "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
              "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
              "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig",
              "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
              "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
              "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup",
              "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone",
              "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
              "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters",
              "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85",
              "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq",
              "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
              "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
              "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars",
              "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
              "ViolentCrimesPerPop"] #outcome is violent crimes per pop

# replace with the new header
crime_data.columns = new_header

# remove columns that contain '?'
rows_with_question_mark = crime_data[crime_data.eq('?').any(axis=1)].shape[0]
cols_contains_question_mark = (crime_data == '?').sum()
crime_data_clean = crime_data.loc[:, ~(crime_data == '?').any()]
nfields = crime_data_clean.shape[1] - 1

print(select(crime_data_clean, chromosome_length=nfields,outcome_index=nfields,population_size=40, generations=100, num_sets=10, mutation_rate=0.02, max_features=70, 
                     objective_function="AIC", log_outcome=False, regression_type="OLS", print_all_generation_data=True, plot_all_generation_data=True, with_progress_bar=True, plot_output_path='/Users/robertthompson/code/robert-thompson/GA-dev/examples',
                     exit_condition_scalar=.00005))









