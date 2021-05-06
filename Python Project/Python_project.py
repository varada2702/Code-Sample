######################################################################
# OVERVIEW :
# The code is organised as follows: 
# 1) Import all the required packages
# 2) All the functions we created for the various data analysis
# 3) Getting data from PDF file
# 4) Getting data from Chicago Health Atlas
# 5) Making Spatial Maps using Geopandas
# 6) Quantitative Analysis (OLS model and Machine Learning Model)


# PLEASE NOTE:
# Package tabula-py is used to scrape data from PDF. This requires Java Run
# Environment. Please make sure that is installed in the machine before installing
# tabula-py (permission to use this package was taken taken from Prof.Levy)
# Refer to https://tabula-py.readthedocs.io/en/latest/ for instructions on 
# creating an environment.


######################################################################


import os
import requests
import pandas as pd
import tabula
import geopandas
import descartes
import numpy as np
import statsmodels.api as sm
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from ipywidgets import interact, interact_manual
from zipfile import ZipFile
from functools import reduce
from geopandas import GeoDataFrame
from tabulate import tabulate
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, make_scorer
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter, MaxNLocator
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
output_notebook()
from bokeh.models import NumeralTickFormatter
from bokeh.models import DatetimeTickFormatter
from bokeh.models import Legend
from bokeh.palettes import Pastel1
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.models import LabelSet
from bokeh.models import HoverTool
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import row
from math import pi
pd.options.mode.chained_assignment = None

"""
General grading comments:
- Only import libraries you use
- Be consistent with your use of spaces, e.g. no spaces around the = sign function argument calls, but with spaces for assignment
- You can pick ' or " for strings, but be consistent everywhere possible
- Using "i" as an iterator should, by convention, only happens when it's counting up values, e.g. for i in range(10). For anything else, use a
  more descriptive variable name.
- All matplotlib operations should use axis or figure methods where possible, and you should especially avoid mixing axis and plt operations
- Your early code made very good use of functions, but your later code abandonded functions all together
- There were quite a few places where code was copy-pasted, instead of using some combination of loops, functions, and containers
"""

import xlrd
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True



## Insert your github directory
os.chdir(r"D:\GitHub\Code-Sample\Python Project")
path1 = os.getcwd()


######################################################################
# ALL FUNCTIONS
######################################################################

def get_pdf(url, filename, path):
    response = requests.get(url)
    with open(os.path.join(path, filename), 'wb') as ofile:
        ofile.write(response.content)


def get_table(path, filename, pages):
    with open(os.path.join(path, filename), 'rb') as ifile:
        pages = tabula.read_pdf(ifile, pages=pages,multiple_tables=True)
        return pages

def combine_clean(df1, df2, cols,flag):

    # The first column head also has data. So I take the column using list and put it
    # as first row in both the dfs
    # https://stackoverflow.com/questions/43408621/add-a-row-at-top-in-pandas-dataframe

    list0 = list(df1)
    list1 = list(df2)

    #JL: this code is copy-pasted, and therefore should have been a loop and/or function
    df1.loc[-1] = list0  # adding a row
    df1.index = df1.index + 1  # shifting index
    df1.sort_index(inplace=True)
    df1.columns = cols

    df2.loc[-1] = list1  # adding a row
    df2.index = df2.index + 1  # shifting index
    df2.sort_index(inplace=True)
    df2.columns = cols

    #Combine data0 and data1 to get 1st table
    frames = [df1, df2]
    table = pd.concat(frames)
    col_names = table.columns[table.columns != 'Geo_Group']
    if(flag == 0): #JL: omit the parentheses
        for col in col_names:
            table[col] = table[col].replace({'%': '', ',': '', 'Unnamed: 0': '0','Unnamed: 1': '0'}, regex=True)
    #https://datatofish.com/convert-string-to-float-dataframe/
    for col in col_names:
        table[col] = table[col].astype(float)
    table = table[table['Geo_Group'] != 1]

    return table


def get_webdata(url, filename, path):
    response = requests.get(url)
    output = response.content
    with open(filename, 'wb') as ofile:
        ofile.write(output)

def get_webtable(path, filename):
    df = pd.read_excel(os.path.join(path, filename))
    return df

def clean_atlas(df, columns,new_col_name1,new_col_name2,new_col_name3):
    df = df[df['Geography'].isin(['Community Area'])]
    df = df.drop(columns, axis = 1)
    df = df.rename(columns = {"Number":new_col_name1,'Percent': new_col_name2,'Crude_Rate':new_col_name3})

    return df

def download_file(url,filename,path):
    if filename not in os.listdir():
        print('downloading document from {}'.format(url))
        get_webdata(url, filename, path)
    else:
        print('document already in {}'.format(path))


def create_tables(url,filename,path,col1,col2,col3,colstodrop):
    download_file(url,filename,path)
    table_interim = get_webtable(path, filename)
    #Code from Sarah's Lab Discussion Session 1
    table = clean_atlas(table_interim, colstodrop,col1,col2,col3)
    return table


##https://www.geeksforgeeks.org/working-zip-files-python/
def unzip(file_name):
    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()

        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall()
        print('Done!')
    return

##########################################################################################
# Below Code not written by me (Until line specified). This was a group project
##########################################################################################
#This function calculates change in population between two years
def population_change(df,col1,col2,new_col1):
    df[new_col1] = (df[col2]-df[col1])
    return df


#This function converts the dataframe from long to wide and changes column names
def long_wide(df,col_name,var):
    df_long =  df.pivot_table(index=["Geo_Group", "Geo_ID"],
                    columns='Year',
                    values=col_name).reset_index()

    df_long = df_long.rename(columns = {"2006-2010":var+"_2006_10","2011-2015":var+"_2011_15","2012-2016":var+"_2012_16",2017:var+"_2017"})
    return df_long


#This dataframe merges a dataset with the boundary spatial file of Chicago
def merge_spatial_df(df,df_boundary):
    df['Area'] = df['Geo_Group'].apply(lambda x: x.split("-")[-1]).str.upper()
    df=df_boundary.merge(df, on='Area')
    df= GeoDataFrame(df)
    return df


#This function just calls the above function and calculates change between two years or total
#population in a particular year
def format_df(df,col_name,flag):
    df_long = long_wide(df,col_name,"pop")

    if(flag == 1):
        df_change = population_change(df_long,"pop_2006_10","pop_2012_16","change_10_16")
        df_change = merge_spatial_df(df_change,df_boundary)
    else:
        df_change = pd.DataFrame(columns = ['Race','2010','2016'])
        df_change = df_change.append({'Race' : col_name, '2010' : df_long["pop_2006_10"].sum(),'2016' : df_long["pop_2012_16"].sum()},ignore_index = True)
    return df_change


#This function plots the change in population in communities of Chicago
def plot_change(df1,race1,df2,race2):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 12))
    ax1, ax2 = axes

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.1)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.1)


    ax1 = df1.plot(ax=ax1, column='change_10_16', legend=True, cax=cax1,linewidth=0.3,edgecolor='black')
    df1.apply(lambda x: ax1.annotate(text=x.Geo_ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    ax1.axis('off')
    plot_title1 = "Change in Chicago’s " + race1 +" population"
    ax1.set_title(plot_title1);

    ax2 = df2.plot(ax=ax2, column='change_10_16', legend=True, cax=cax2,linewidth=0.3,edgecolor='black')
    df2.apply(lambda x: ax2.annotate(text=x.Geo_ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    ax2.axis('off')
    plot_title2 = "Change in Chicago’s " + race2 +" population"
    ax2.set_title(plot_title2);
    print(tabulate(change_pop, headers = 'keys', tablefmt = 'psql'))
    fig.tight_layout()
    plt.savefig('Population change.png')
    return #JL: you can use a stand-alone return statement when it helps with your workflow, e.g. in conditionals, but it does nothing here


#This function calls the above functions to create the final dataset for mapping
#This function is used when I just want to map a single column without any manipulation
def df_to_spatial(table,df_boundary,col_name,var_name):
    table_long = long_wide(table,col_name,var_name)
    table_long = merge_spatial_df(table_long,df_boundary)
    table_long.to_csv(os.path.join(path1,col_name+'.csv'))
    return table_long


#This function is used to plot multiple columns of a dataframe on the map of chicago
def plot_multiple_columns(df,col1,col2,col3):
    fig, axes = plt.subplots(ncols=3, figsize=(12, 12))
    ax1, ax2,ax3 = axes

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.1)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.1)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.1)

    ax1 = df.plot(ax=ax1, column=col1, legend=True, cax=cax1,linewidth=0.1,edgecolor='white')
    #df.apply(lambda x: ax1.annotate(text=x.Geo_ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    ax1.axis('off')
    ax1.set_title("2006-2010");

    ax2 = df.plot(ax=ax2, column=col2, legend=True, cax=cax2,linewidth=0.1,edgecolor='white')
    #df.apply(lambda x: ax2.annotate(text=x.Geo_ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    ax2.axis('off')
    ax2.set_title("2011-2015");

    ax3 = df.plot(ax=ax3, column=col3, legend=True, cax=cax3,linewidth=0.1,edgecolor='white')
    #df.apply(lambda x: ax3.annotate(text=x.Geo_ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    ax3.axis('off')
    ax3.set_title("2012-2016");
    #print(tabulate(change_pop, headers = 'keys', tablefmt = 'psql'))
    fig.tight_layout()
    plt.savefig('Per capita income distribution.png')
    return

#This function is used to plot a single column on the map of chicago
def plot_columns(df,col1,title):
    fig, ax = plt.subplots(figsize=(12, 12))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)

    ax = df.plot(ax=ax, column=col1, legend=True, cax=cax,linewidth=0.1,edgecolor='white')
    df.apply(lambda x: ax.annotate(text=x.Geo_ID, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
    ax.axis('off')
    ax.set_title(title);
    plt.savefig(col1+'.png')
    return

#This function is primarily used to display top 5 and bottom 5 communities by life expectancy
def print_five(table,ascending,print_word):
    table.sort_values(by=["life_expectancy_2017"], inplace=True, ascending=ascending)
    table= table.reset_index(drop = True)
    statement = "\033[1m" + "Community areas with " + print_word + " life expectancy" +"\033[0m"
    print(statement)
    print(table.Area[0].title(), "=",table.life_expectancy_2017[0])
    print(table.Area[1].title(), "=",table.life_expectancy_2017[1])
    print(table.Area[2].title(), "=",table.life_expectancy_2017[2])
    print(table.Area[3].title(), "=",table.life_expectancy_2017[3])
    print(table.Area[4].title(), "=",table.life_expectancy_2017[4])
    return


#This dataframe plots points and some socioeconomic variable
def plot_points(df_boundary,grocery,farmer,title, flag = 0, df_socio = pd.DataFrame(),col_name = ""):

    fig, ax = plt.subplots(figsize=(12,12))
    if(flag == 1):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)

        ax = df_socio.plot(ax=ax, column=col_name, legend=True, cax=cax,linewidth=0.1,edgecolor='white')

    if(flag == 0):
        df_boundary.plot(ax=ax, color='white', edgecolor='black')

    grocery.plot(ax=ax, color='red',legend = True)
    farmer.plot(ax=ax, color='blue',legend = True)

    food_options = ['Grocery stores', "Farmer's market"]
    col_dict = {'Grocery stores':'red',
               "Farmer's market" : 'blue'}


    patch_list =[]
    for food in food_options:
        label = food.capitalize()
        color = col_dict[food]
        patch_list.append(patches.Patch(facecolor=color,
                                        label=label,
                                        alpha=0.9,
                                        linewidth=2,
                                        edgecolor='black'))

    # Creates a legend with the list of patches above.
    ax.legend(handles=patch_list, fontsize=15, loc='lower left',
            bbox_to_anchor = (.1,0.04), title_fontsize=45)

    ax.axis('off')
    ax.set_title(title)
    plt.savefig(title+'.png')
    plt.show()

##########################################################################################
# Until here is not written by me. 
# below code is written by me
##########################################################################################

def filterdf_2016(df):
    df_2016 = df[df['Year'].isin(['2012-2016'])]
    return df_2016


def filterdf_2011(df):
    df_2011 = df[df['Year'].isin(['2011-2015'])]
    return df_2011


def ols_df_merge(f_year, df1_PDF, df2, df3, df4, df5, df6, df7, df8):

    df1_n = df1_PDF[df1_PDF['Geo_ID'] != 0]
    df2_n = f_year(df2)
    df3_n = f_year(df3)
    df4_n = f_year(df4)
    df5_n = f_year(df5)
    df6_n = f_year(df6)
    df7_n = f_year(df7)
    df8_n = f_year(df8)

    #Merging together
    dataframes = [df1_n, df2_n , df3_n, df4_n, df5_n, df6_n, df7_n, df8_n]
    df_merged = reduce(lambda left,right: pd.merge(left,right,on=['Geo_ID'],
                                            how='outer'), dataframes)
    # https://pythonpedia.com/en/knowledge-base/44327999/python--pandas-merge-multiple-dataframes


    #Dropping columns that are repeated:
    colskeep = ['Geo_ID',
                'Geo_Group_x',
                '% with Private Insurance (12-16)',
                '% with Public Insurance (12-16)',
                'Uninsured (12-16)',
                'Year_x',
                'Geography_x',
                'black_pop',
                'black_pop_percent',
                'hisp_pop',
                'hisp_pop_percent',
                'PCI',
                'PCI_percent',
                'Inft_Mort_Rate',
                'Active_transp',
                'Active_transp_percent',
                'SNAP_hh',
                'SNAP_hh_percent',
                'unemployed',
                'unemployed_percent']

    model_df = df_merged[colskeep]

    return model_df

def ols_model(np_array_y, np_array_X):

    # adding the constant term
    np_array_X = sm.add_constant(np_array_X)

    #fitting the model
    model = sm.OLS(np_array_y, np_array_X)
    results = model.fit()

    #printing the summary table
    print(results.summary())


def y_array(df, col_name):
    y_array = np.array(df[col_name], dtype = 'float')

    print('The median of the Infant Mortality data is:', np.median(y_array))

    #Accordingly we make infant mortality into a binary variable:

    for i in range(0,len(y_array)):
        if y_array[i] <= 7:
            y_array[i] = 1
        elif y_array[i] > 7:
            y_array[i] = 2

    return(y_array)


def ml_model(X_train, y_train, X_test, y_test, measure):
    # Code for cross-validation form Prof. Levy's Piazza reply.
    models = [('Dec Tree', DecisionTreeClassifier()),
              ('Lin Disc', LinearDiscriminantAnalysis()),
              ('Gauss', GaussianNB()),
              ('SVC', SVC(gamma='auto')),
              ('KNN', KNeighborsClassifier(n_neighbors=3))]

    results = []

    if measure == 'accuracy':
        for name, model in models:
            res = cross_val_score(model, X_train, y_train, scoring=measure)
            res_mean = round(res.mean(), 4)
            res_std  = round(res.std(), 4)
            results.append((name, res_mean, res_std))


        for line in results:
            print(line[0].ljust(10), str(line[1]).ljust(6), str(line[2]))

    else:
        scorer = make_scorer(measure, average='weighted')

        for name, model in models:
            res = cross_val_score(model, X_train, y_train, scoring=scorer)
            res_mean = round(res.mean(), 4)
            res_std  = round(res.std(), 4)
            results.append((name, res_mean, res_std))

        for line in results:
            print(line[0].ljust(10), str(line[1]).ljust(6), str(line[2]))


def conf_matrix(chosen_model, X_train, y_train, X_test, y_test):
    model = chosen_model
    model.fit(X_train, y_train)
    predict = model.predict(X_test)

    mat = confusion_matrix(y_test, predict)
    ax = sns.heatmap(mat, square=True, annot=True, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual');

    fig = ax.get_figure()
    fig.savefig(os.path.join(path1))

    print(classification_report(y_test, predict))
    print('The accuracy score is:', accuracy_score(y_test, predict))

    return predict


######################################################################
# GETTING DATA FROM PDF FILE
######################################################################
go_online = True #JL: a toggle like this should be at the top of the file
url1 = 'https://www.chicago.gov/content/dam/city/depts/fss/supp_info/ChildrenServices/DFSS2019CommunityAssssment.pdf'
filename1 = 'DFSS2019CommunityAssssment.pdf'

#JL: all this code should be in a function
if filename1 not in os.listdir():
    print('downloading document from {}'.format(url1))
    get_pdf(url1, filename1, path1)
else:
    print('document already in {}'.format(path1))

dfs = get_table(path1, filename1, "151-175")

data0 = pd.DataFrame(dfs[0]) #JL: too much copy-paste; this could be a loop and a container
data1 = pd.DataFrame(dfs[1])
data2 = pd.DataFrame(dfs[2])
data3 = pd.DataFrame(dfs[3])
data4 = pd.DataFrame(dfs[4])
data5 = pd.DataFrame(dfs[5])

# Since one table is over two pages, combine_clean() takes the two dfs and column names,
# cleans and adds column names and combines it to one table.


# Have to convert all these columns to float type. They are currently in object
# format.


cols1 = ['Geo_ID', 'Geo_Group',
         'Food Access Number (2015)', 'Food Access Rate (2015)']

cols2 = ['Geo_ID',
         'Geo_Group',
         '#children in SNAP, age 0-5 (2016)',
         '%children in SNAP, age 0-5 (2016)',
         '#children in SNAP, age 0-17 (2016)',
         '%children in SNAP, age 0-17 (2016)']

cols3 = ['Geo_ID',
         'Geo_Group',
         '% with Private Insurance (12-16)',
         '% with Public Insurance (12-16)',
         'Uninsured (12-16)']

#table1 = Food Access Rate by Chicago Community Area, 2015
table1 = combine_clean(data0, data1, cols1,1)

#table2 = Child and Youth Supplemental Nutrition Assistance	Program	(SNAP)Enrollment, 2016
table2 = combine_clean(data2, data3, cols2,0)

#table3 = Heatlh Insurance Coverage by Type, 2012-2016 (ACS 5-year estimate)
table3 = combine_clean(data4, data5, cols3,0)

dataframes = [table1, table2, table3]

PDFData = reduce(lambda left,right: pd.merge(left,right,on=['Geo_ID'], how='outer'), dataframes).drop(['Geo_Group_x', 'Geo_Group_y'], axis = 1)

# PDFData is the final merged dataframe.

# Since not everyone will have a JRE environment, I am download PDFData as
# a csv file and pushing it to the GitHub repo too.

PDFData.to_csv(os.path.join(path1,'PDFData.csv'))



######################################################################
# GETTING DATA FROM CHICAGO HEALTH ATLAS
######################################################################

colstodrop = ['Category',
 'SubCategory',
 'Indicator',
 'Demography',
 'Demo_Group',
 'Cum_Number',
 'Ave_Annual_Number',
 'Lower_95CI_Crude_Rate',
 'Upper_95CI_Crude_Rate',
 'Age_Adj_Rate',
 'Lower_95CI_Age_Adj_Rate',
 'Upper_95CI_Age_Adj_Rate',
 'Lower_95CI_Percent',
 'Upper_95CI_Percent',
 'Weight_Number',
 'Weight_Percent',
 'Lower_95CI_Weight_Percent',
 'Upper_95CI_Weight_Percent']


# Cleaning data #JL: this whole section looks like it would work very well in a loop and container
# Thus table 4 contains data for the number and % of Non-Hispanic African American or
# Black population in Chicago Community Areas for the years 2006 - 2010, 2011 - 2015,
# 2012 - 2016
#path is same as the first path, (Github Repo)
url2 = 'https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/441/Non-Hispanic_African-Amercian_or_Black_population.xlsx'
filename2 = 'Non-Hispanic_African-Amercian_or_Black_population.xlsx'
table4 = create_tables(url2,filename2,path1,"black_pop","black_pop_percent","black_pop_crude",colstodrop)


# Table 5 contains data for the number and % of Hispanic and Latino
# population in Chicago Community Areas for the years 2006 - 2010, 2011 - 2015,
# 2012 - 2016
url3 = 'https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/531/Hispanic_or_Latino_population.xlsx'
filename3 = 'Hispanic_or_Latino_population.xlsx'
table5 = create_tables(url3,filename3,path1,'hisp_pop','hisp_pop_percent',"hisp_pop_crude",colstodrop)

# Table 6 contains data for the percapita income Chicago Community Areas for the years 2006 - 2010, 2011 - 2015,
# 2012 - 2016
url4 = "https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/691/Per_capita_income.xlsx"
filename4 = 'Per_capita_income.xlsx'
table6 = create_tables(url4,filename4,path1,'PCI','PCI_percent',"PCI_crude",colstodrop)


# table 7 gives the infant mortality rate in Chicago Community Areas
url5 = "https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/715/Infant_mortality.xlsx"
filename5 = 'Infant_mortality.xlsx'
table7 = create_tables(url5,filename5,path1,'Inft_Mort_number','Inft_Mort_percent',"Inft_Mort_Rate",colstodrop)


# table 8 contains data on Active Transportation
url6 = "https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/675/Active_transportation.xlsx"
filename6 = 'Active_transportation.xlsx'
table8 =  create_tables(url6,filename6,path1,'Active_transp','Active_transp_percent',"Active_transp_crude",colstodrop)


# table 9 contains data on Number of households receiving food stamps/SNAP
url7 = "https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/687/Food_stamps_SNAP.xlsx"
filename7 = 'Food_stamps_SNAP.xlsx'
table9 = create_tables(url7,filename7,path1,"SNAP_hh","SNAP_hh_percent","SNAP_hh_crude",colstodrop)


#table 10 gives data on unemployment
url8 = 'https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/682/Unemployment.xlsx'
filename8 = 'Unemployment.xlsx'
table10 = create_tables(url8,filename8,path1,"unemployed","unemployed_percent","unemployed_crude",colstodrop)


# Table 11 contains data for the life expectancy in Chicago Community Areas for the years 2006 - 2010, 2011 - 2015,
# 2012 - 2016
url9 = 'https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/601/Life_Expectancy.xlsx'
filename9 = 'Life_expectancy.xlsx'
table11 = create_tables(url9,filename9,path1,"life_expectancy","life_expectancy_percent","life_expectancy_crude",colstodrop)

#Farmers Market
zipurl1 = 'https://data.cityofchicago.org/api/views/x5xx-pszi/rows.csv?accessType=DOWNLOAD'
zipname1 = 'Farmers_Markets_-_2015.csv'
download_file(zipurl1,zipname1,path1)

#Boundaries
zipurl2 = 'https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&format=Shapefile'
zipname2 = 'Boundaries - Community Areas (current).zip'
download_file(zipurl2,zipname2,path1)

#Grocery
url10 = 'https://data.cityofchicago.org/api/views/53t8-wyrc/rows.csv?accessType=DOWNLOAD'
filename10 = 'Grocery_Stores_-_2013.csv'
download_file(url10,filename10,path1)




######################################################################
#SPATIAL ANALYSIS USING GEOPANDAS --- THIS SECTION IS NOT WRITTEN BY ME
######################################################################
# specifying the zip file name #JL: this whole section should be in functions also
file_name = "Boundaries - Community Areas (current).zip"
unzip(file_name)

boundary = os.path.join(path1, 'geo_export_429dbc45-1796-4ca5-a31b-2a9ad5b2b1e6.shp')
df_boundary  = geopandas.read_file(boundary)
df_boundary=df_boundary.rename(columns={"community":"Area"})


table4_long = format_df(table4,"black_pop",1)
table4_long.to_csv(os.path.join(path1,'Black population.csv'))
table5_long = format_df(table5,"hisp_pop",1)
table5_long.to_csv(os.path.join(path1,'Hispanic population.csv'))
table4_change = format_df(table4,"black_pop",0)
table5_change = format_df(table5,"hisp_pop",0)
change_pop = pd.concat([table4_change,table5_change]).reset_index(drop = True)
change_pop["Race"] = ["Black population","Hispanic Population"]

#Map - Change in Chicago's Population
plot_change(table4_long,"black",table5_long,"Hispanic")


#Map - Change in Chciago's PCI
table6_long = df_to_spatial(table6,df_boundary,"PCI","income")
plot_multiple_columns(table6_long,"income_2006_10","income_2011_15","income_2012_16")


#Map - Life expectancy by community areas
table11_long = df_to_spatial(table11,df_boundary,"life_expectancy","life_expectancy")

print_five(table11_long,False,"highest")
print_five(table11_long,True,"lowest")
plot_columns(table11_long,"life_expectancy_2017","Life Expectancy by community areas, 2017")



#Map Grocery and Farmer's Markets
grocery = pd.read_csv("Grocery_Stores_-_2013.csv")
far_mkt = pd.read_csv("Farmers_Markets_-_2015.csv")

grocery_gdf = geopandas.GeoDataFrame(grocery, geometry=geopandas.points_from_xy(grocery['LONGITUDE'], grocery['LATITUDE']))
far_mkt_gdf = geopandas.GeoDataFrame(far_mkt, geometry=geopandas.points_from_xy(far_mkt['LONGITUDE'], far_mkt['LATITUDE']))

plot_points(df_boundary,grocery = grocery_gdf,farmer = far_mkt_gdf,title = "Location of Grocery stores and farmers markets across Chicago")


# Distribution of grocery staores and markets against transportation activity
table8_long = df_to_spatial(table8,df_boundary,"Active_transp_percent","Active_transp")
plot_points(df_boundary,grocery = grocery_gdf,farmer = far_mkt_gdf,flag = 1, df_socio = table8_long,col_name = "Active_transp_2012_16",title = "Distribution of grocery staores and markets against transportation activity")


# Map- Food Access Against Unemp 
table10_map =  df_to_spatial(table10,df_boundary,"unemployed_percent","unemployed")
plot_points(df_boundary,grocery = grocery_gdf,farmer = far_mkt_gdf,flag = 1, df_socio = table10_map,col_name = "unemployed_2012_16",title = "food access against unemployment")

# Map - Food access against infant mortality
table7_map =  df_to_spatial(table7,df_boundary,"Inft_Mort_Rate","mortality")
plot_points(df_boundary,grocery = grocery_gdf,farmer = far_mkt_gdf,flag = 1, df_socio = table7_map,col_name = "mortality_2012_16",title = "food access against infant mortality")


# Map - food access against % uninsured
table3_map =  merge_spatial_df(table3,df_boundary)
plot_points(df_boundary,grocery = grocery_gdf,farmer = far_mkt_gdf,flag = 1, df_socio = table3_map,col_name = "Uninsured (12-16)",title = "food access against percent uninsured in Chicago")

appendix = table1[["Geo_ID","Geo_Group"]]
appendix.to_csv("Community areas in Chicago.csv")



######################################################################
#QUANTITATIVE ANALYSIS I - OLS MODEL - THIS IS WRITTEN BY ME
######################################################################

df_2016 = ols_df_merge(filterdf_2016, table3, table4, table5, table6, table7, table8, table9, table10)

#Converting pandas df to numpy arrays
y1 = np.array(df_2016['Inft_Mort_Rate'], dtype = 'float')

features1 = ['Uninsured (12-16)',
             'black_pop_percent',
             'PCI',
             'Active_transp_percent',
             'SNAP_hh',
             'unemployed_percent']

X1 = np.array(df_2016[features1], dtype = 'float')


#Running a model using statsmodel
ols_model(y1, X1)

## This model shows that only the variable % Uninsured, % of Black Population
## PCI, %Active Transport are significant (p < 0.05).


######################################################################
# QUANTITATIVE ANALYSIS II - Machine Learning Model - THIS IS WRITTEN BY ME
######################################################################

# Use data for 2011-15 to predict the infant mortality in 2012-2016. This is
# the training data
df_2011 = ols_df_merge(filterdf_2011, table3, table4, table5, table6, table7, table8, table9, table10)

# Use data for 2011-15 to predict the infant mortality in 2012-2016. This is
# test data -- df_2016

features2 = ['PCI',
             'Active_transp_percent',
             'SNAP_hh_percent',
             'unemployed_percent',
             'black_pop_percent']


# Extracting numpy arrays from train dfs
y_train = y_array(df_2011, 'Inft_Mort_Rate')
X_train = np.array(df_2011[features2], dtype = 'float')

# Extracting to numpy arrays for test matrices
y_test = y_array(df_2016, 'Inft_Mort_Rate')
X_test = np.array(df_2016[features2], dtype = 'float')


#Checking the rank of X_train matrix to see if my columns are linearly independent
print("rank(X_train) =", la.matrix_rank(X_train))
# Since the rank is 5, the columns are linearly independent


#Running multiple models to determine which fits best
ml_model(X_train, y_train, X_test, y_test, precision_score)
ml_model(X_train, y_train, X_test, y_test, 'accuracy')

#Linear Discriminat Analysis has the best precision score. We choose precision
# score since the cost of a false positive is very high and we want to
# minimise it.
conf_matrix(LinearDiscriminantAnalysis(), X_train, y_train, X_test, y_test)

# The confusion matrix shows that the model does a good prediction of the community areas
# which have would have a high infant mortality rate

conf_matrix(GaussianNB(), X_train, y_train, X_test, y_test)
# Although GaussianNB also has a high precisions score, it has more false
# positives than LDA.

