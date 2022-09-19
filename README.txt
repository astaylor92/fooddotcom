~~~~~~Team 169 Recipe Exploration Tool ~~~~~~

###### TABLE OF CONTENTS ######

1. Package Description
2. File / Directory Overview
3. Installation Instructions
4. Demo Execution Instructions


###### 1 - PACKAGE DESCRIPTION ######
This repository contains a project whose goal was to create a python-based recipe exploration tool with a frontend built using plotly and dash. Within the DOC folder, the team\'92s final report and post can be found. Within the CODE folder, there are various subfolders. Although more detail is given in section 2, generally, folders 01 - 04 contain Jupyter notebooks which were used to clean and prepare the data, and do not need to be rerun for the demo in folder 05 to function. Folder 05 contains everything needed to launch a demo of the product (instructions for which can be found in sections 3 and 4 of this readme).


###### 2 - FILE / DIRECTORY OVERVIEW ######
SOURCE DATA
--------------------------------------
Source data, referenced in 01_DataCleansing\data_cleaning.ipynb, was downloaded from Kaggle at https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions.


FOLDER: 01_DataCleansing
--------------------------------------
DESCRIPTION: This folder contains the Jupyter notebook which reads in the raw data from Kaggle and outputs cleaned and filtered data which feeds many of the remaining Jupyter notebooks.
Note - these notebooks are NOT needed to run the demo

FILES:
 > data_cleaning.ipynb - File which cleans raw data from Kaggle and outputs a number of different files, to be used in later analysis and the final product


FOLDER: 02_UserSummaryPrep
--------------------------------------
DESCRIPTION: This folder contains the Jupyter notebooks which reads in the outputs of 01_DataCleansing and outputs summary files which are used for the User Summary page in the final product.
Note - these notebooks are NOT needed to run the demo
FILES:
 > user_summary_prep.ipynb - File which summarizes cleaned data and creates four input files for the demo. Feeds like product.


FOLDER: 03_AlgorithmDevelopment
--------------------------------------
DESCRIPTION: This folder contains the Jupyter notebooks which precompute various scoring for use in the final product.
Note - these notebooks are NOT needed to run the demo
FILES:
> comm_01_data_prep.ipynb - Notebook which creates a file to be used for network analysis. Needs to be run before all Comm_02 workflows.
> comm_02_louvain_01_numthresh-2.ipynb - Notebook which analyzes the network and communities where edges were only considered if 2 or more users are shared. Does not feed any future notebooks.
> comm_02_louvain_02_pctthresh-08_aws.ipynb - Notebook which analyzes the network and communities where edges were only considered if 8% or more users are shared. Needs to be run before comm_03_split_recalc_distances_pctthresh-08_split.ipynb. This was run on AWS SageMaker servers, reading data from S3 buckets.
> comm_02_louvain_03_nodesample-50.ipynb - Notebook which analyzes the network and communities where 50% of nodes were uniformly sampled. Does not feed any future notebooks.
> comm_02_louvain_03_nodesample-75.ipynb - Notebook which analyzes the network and communities where 75% of nodes were uniformly sampled. Does not feed any future notebooks.
> comm_02_louvain_03_nodesample-90_aws.ipynb - Notebook which analyzes the network and communities where 90% of nodes were uniformly sampled. Does not feed any future notebooks. This was run on AWS SageMaker servers, reading data from S3 buckets
> comm_03_split_recalc_distances_pctthresh-08_split.ipynb - Notebook which splits the largest clusters for the selected clustering. Needs to be run before Comm_04_gapfillassignments_pctthresh-08_split.ipynb.
> comm_04_gapfillassignments_pctthresh-08_split.ipynb - Notebook which gapfills community assignments. Feeds live product.
> comm_05_calcsimilarity_03_pctthresh-08_split.ipynb - Notebook which calculates similarity scores. Feeds live product.
> comm_06_createusercounts.ipynb - Notebook which calculates community-user assignments with counts. Feeds live product.
> ingredients_01_calcsimilarity.ipynb - Notebook which calculates ingredient to ingredient similarity. Feeds live product.


FOLDER: 04_Testing
--------------------------------------
DESCRIPTION: This folder contains the Jupyter notebooks used to conduct testing, as well as Excel files which document testing performed by team members.
Note - these files are NOT needed to run the demo
FILES:
> comm_similarity_testing.ipynb - Notebook which validates community similarity scores.
> cuisine_similarity_testing.ipynb - Notebook which validates cuisine similarity scores.
> technique_similarity_testing.ipynb - Notebook which validates technique similarity scores.
> ingredient_similarity_testing.ipynb - Notebook which validates ingredient similarity scores.
> functinality_testing.ipynb - Notebook which supports functionality testing
> functionality_testing.xlsx - Excel file which documents functionality testing
> performance_test.xlsx - Excel file which documents performance testing
> results_testing.xlsx - Excel file which documents results testing 
> user_sampling.ipynb - Notebook which samples user IDs for results and performance testing
> test_user_ids.csv - File containing user IDs used for results testing
> combinations.csv  - File containing combinations of similarity preferences used for results testing


FOLDER: 05_Testing
--------------------------------------
DESCRIPTION: This folder contains all code and data to run the live product on Dash. No other folders are needed to run the product demo. 
> assets/ - subfolder containing css and font files used by dash
> data/ - subfolder containing data files references by the application
	> community_assignments_pctthresh-08_split_filled_filter.parquet - file containing assignments of recipes to communities, output of comm_04_gapfillassignments_pctthresh-08_split.ipynb
	> community_distances_pctthresh-08_split_adj.parquet - file containing community to community assignments, output of comm_05_calcsimilarity_03_pctthresh-08_split.ipynb
	> community_graph_edges_pctthresh-08_split.parquet - file containing graph edges for community visualization, outputs of comm_04_gapfillassignments_pctthresh-08_split.ipynb
	> node_sizes.csv - file containing the count of recipes in each commmunity, output of comm_04_gapfillassignments_pctthresh-08_split.ipynb
	> nutrition.csv - file containing nutrition info for all recipes, output of data_cleaning.ipynb
	> recipes_in_count2_mean4.parquet - file containing cleaned recipes data, output of data_cleaning.ipynb
	> recipes_ingredient_vec.parquet - file containing ingredient-ingredient similarity from word2vec algorithm, output of ingredients_01_calcsimilarity.ipynb
	> user_assign_community_with_recipe_count/parquet - file containing user-community assignments with counts of user recipes, output of comm_06_createusercounts.ipynb
	> user_cuisines.parquet - file containing user cuisine counts, output of user_summary_prep.ipynb 
	> user_ing_count.parquet - file containing user ingredient counts, output of user_summary_prep.ipynb 
	> user_ingredients.parquet - file containing user ingredient combinations, output of user_summary_prep.ipynb 
	> user_techniques.parquet - file containing user technique counts, output of user_summary_prep.ipynb 
	> users_in_count2_mean4.csv - file containing cleaned user data, output of data_cleaning.ipynb
	> website.csv - file containing url data, output of data_cleaning.ipynb
	

###### 3 - DEMO SETUP INSTRUCTIONS ######
1. Navigate to the CODE > 05_RecipeExploration_Tool folder using Terminal / shell, with your new environment activated
2. Create a virtual environment running with Python 3.9.12 (we used conda and the command 'conda create -n cse6242-project-demo python=3.9.12'
3. Activate the new environment (we used conda and the command 'conda activate cse6242-project-demo')
4. Install requirements from requirements.txt file at CODE > 05_RecipeExploration_Tool > requirements.txt into that environment using 'pip install -r requirements.txt'
5. Type the command 'python index.py' into the terminal to start the Dash application. Take note of the local address where the Dash app is being hosted
6. Using a browser (preferably Google Chrome), go to the address from the previous step (it will be something like http://127.0.0.1:8050)
7. Use the application!

###### 4 - DEMO EXECUTION INSTRUCTIONS ######
Because this live product does not include a user login screen, to demo this product, a sample user ID should be used. Any contiguous integer between 0 and 20000 should be a valid user ID. The default value of 788 also serves as a good demo ID. Other suggested IDs include 1855, 887, 1782, and 7817.

In order to demo the product, first follow the steps in section 3 to setup and launch the site. Then, from that page, we recommend the following flow:
1. Enter a user ID in the 'User ID' field. This is intended to represent the login process which an official recipe site may have.
2. Review the visualizations on the 'User Summary' tab. This is intended to be where a user might learn about their own cooking habits (generally represented by blue), compared to the average food.com user (generally represented by gray).
3. Navigate to the 'Try New Recipes' page by clicking the tab at the top
4. Select any number of filters for the recipes to explore in step (1)
5. Select any number and degree of exploration desired in step (2). Theoretically, a user would notice something in the User Summary page that inspires them to explore (e.g. 'Wow, I don't cook with many different ingredients! I should explore different ingredients.')
6. Click 'Submit'
7. Review the list of recommendations and visualizations to see the recommended recipes to explore. The user's cooking habits are again generally represented in blue, and the recommendations are generally represented by orange.