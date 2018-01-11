
# coding: utf-8

# # App info

# In[1]:

#__author__ = "Simon Gonzalez (Writer and developer) and Nicole Harris (Testing and Improvements)"
#__year__ = "2018"
#__credits__ = ["Centre of Excellence for the Dynamics of Language, Sydney Speaks, Transcription Acceleration Project, Appen"]
#__license__ = "GPL"
#__version__ = "1.0.1"
#__maintainer__ = "Simon Gonzalez"
#__email__ = "simon.gonzalez@anu.edu.au"


# # Import libraries

# In[6]:


get_ipython().magic('matplotlib inline')
import functools
import pandas as pd
import numpy as np
from ipywidgets import interact
from ipywidgets import *
import ipywidgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from statsmodels.formula.api import ols, gls, wls, glm, rlm, mixedlm, Logit, Probit, MNLogit, Poisson
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic
from pivottablejs import pivot_ui
from beakerx import *

import warnings

warnings.filterwarnings('ignore')


# # Read input data

# The user can input the name of the file to import. The three components are:
# 
# * (Optional) Path of the file. If no PATH is speficied, then the program reads as it is in the same directory.
# * File name to be imported
# * File extenstion. Extensions allowed: .xlsx, .csv, .txt. For .txt files, these have to be tab-delimited files.
# 
# Example: 
# * With PATH: /path/to/file/myFile.xlsx
# * Without PATH: myFile.xlsx
# * Incorrect: myFile
# 
# The second input text widget (Sheet), specifies the index of the sheet to be read only if the file is an .xlsx file. If the file is not an .xlsx then this input is ignored.

# In[7]:

#read file
fileInput = widgets.Text(
    placeholder='Enter file name with extension',
    description='File:',
    disabled=False
)

fileInput.set_title  = 'fileInput'

#specify sheet index

sheet_number = widgets.IntText(
    value=0,
    description='Sheet',
    disabled=False
)

sheet_number.set_title  = 'sheet_number'

#display widgets
display(fileInput, sheet_number)


# # Import file into a python format

# Convert the input file into a Pandas dataframe.
# 
# Two dataframes are created:
# 
# * df: this is the working dataframe which is used for visualisation and analysis. This is also the file that is filtered (subset). After filtering, it can be reverted to the original input dataset.
# 
# * original_data: this is the imported dataframe and it is not changed throughout the program (as opposed to df). This is the file that df is reverted to whenever requested by the user.

# In[8]:

#get file name
inFile = fileInput.value
inFile = inFile.replace("\\", "/")
#get file extension
fileExtension = inFile.split(".")[-1]
#print(fileExtension)

if fileExtension == 'xlsx':
    df = pd.read_excel(inFile, sheet_number.value)
elif fileExtension == 'csv':
    df = pd.read_csv(inFile)
elif fileExtension == 'txt':
    df = pd.read_csv(inFile, sep='\t', lineterminator='\r')

df = pd.DataFrame(df)
original_data = df


# # Get column names

# Get the name of all columns in the dataframe.
# 
# There are three list created:
#     
#     1. df_cols: all columns in the dataframe
#     2. str_cols: only columns which have categorical values
#     3. numeric_cols: only columns which have numeric values
#     
# Important note: if a column has mixed values (numeric and categorical), the program does not work properly.

# In[9]:

#get data column names, all, numeric and categorical
df_cols = list(df.columns.values)
df_cols.insert(0, 'Select')
str_cols = list(df.select_dtypes(include=['object']).columns.values)
str_cols.insert(0, 'Select')
numeric_cols = list(df.select_dtypes(include=['number']).columns.values)
numeric_cols.insert(0, 'Select')


# # Create widgets for the VISUALISE tab

# These widgets manipulate the visualisation of the data.
# 
# * headers_x: Name of the column which has the values to be plotted in the X axis
# * headers_y: Name of the column which has the values to be plotted in the Y axis
# * colour_headers: Name of the column which has the values to be used for colour subset of the visualisation
# 
# Important note: The default option shown is the 'Select' option. This option creates an empty selection, i.e. no selection has been made.

# In[10]:


headers_x = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='X'
        )

headers_x.set_title  = 'headers_x'

headers_y = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Y'
        )

headers_y.set_title  = 'headers_y'

colour_headers = widgets.Dropdown(
            options=str_cols,
            value=str_cols[0],
            description='Colour'
        )

colour_headers.set_title  = 'colour_headers'


# # Interaction function for the VISUALISE tab

# This function is meant to be as intuitive as possible concernign the type of plot. 
# 
# The type of plot is based on the type of values requested for plotting. SO far, there are six options:
# 
# 1. X numeric only: if there is only one column selected and it is NUMERIC, then the plot is a HISTOGRAM.
# 
# 2. X categorical only: if there is only one column selected and it is CATEGORICAL, then the plot is a BAR PLOT.
# 
# 3. X numeric and Y numeric: if X and Y are NUMERIC, then the plot is a SCATTERPLOT.
# 
# 4. X numeric and Y categorical: if X is NUMERIC and Y is CATEGORICAL, then the plot is an inverted VIOLIN PLOT.
# 
# 5. X categorical and Y categorical: if X and Y are CATEGORICAL, then the plot is a MOSAIC PLOT.
# 
# 6. X categorical and Y numeric: if X is CATEGORICAL and Y is numeric, then the plot is a BOXPLOT.

# In[11]:


@interact
def view_image(headers_x:headers_x, headers_y:headers_y, 
               colour_headers:colour_headers):

        data = df
        
        #no selection made
        #========================================================================
        if headers_x == 'Select' and headers_y == 'Select':
            sns.set_context("notebook", font_scale=1.1)
        
        #X selected but noy Y
        #========================================================================
        elif headers_x != 'Select' and headers_y == 'Select':
            
            sns.set_context("notebook", font_scale=1.1)
            x=headers_x

            if df[headers_x].dtype == np.float or df[headers_x].dtype == np.int:
                x_type = 'is_numeric'
            elif df[headers_x].dtype == np.object:
                x_type = 'is_string'
            
            #plot when x is a string
            #--------------------------------------------------------------------
            if x_type == 'is_string':
                #if colours have not been selected
                #................................................................
                if colour_headers == 'Select':
                    
                        g=sns.countplot(x=x, data=data)
                        loc, labels = plt.xticks()
                        g.set_xticklabels(labels, rotation=90)
                        plt.show()
                        #sns_plot.savefig("output.png")
                #if colours have been selected
                #................................................................
                else:
                        g=sns.countplot(x=x, hue=colour_headers, data=data)
                        loc, labels = plt.xticks()
                        g.set_xticklabels(labels, rotation=90)
                        plt.show()
                        
            #plot when x is numeric
            #--------------------------------------------------------------------
            else:
                #if colours have not been selected
                #................................................................
                if colour_headers == 'Select':
                    xplot = data[x]
                    sns.distplot(xplot)
                    plt.show()
                #if colours have been selected   
                #................................................................
                else:
                    g = sns.FacetGrid(data, hue=colour_headers)
                    g = g.map(sns.distplot, x)
                    plt.show()
        #if only Y has been selected
        #========================================================================
        elif headers_x == 'Select' and headers_y != 'Select':
            sns.set_context("notebook", font_scale=1.1)
        #if both X and Y have been selected
        
        #========================================================================
        elif headers_x != 'Select' and headers_y != 'Select':
            x=headers_x
            y=headers_y

            if df[headers_x].dtype == np.float or df[headers_x].dtype == np.int:
                x_type = 'is_numeric'
            elif df[headers_x].dtype == np.object:
                x_type = 'is_string'

            if df[headers_y].dtype == np.float or df[headers_y].dtype == np.int:
                y_type = 'is_numeric'
            elif df[headers_y].dtype == np.object:
                y_type = 'is_string'
                
            sns.set_context("notebook", font_scale=1.1)
            sns.set_style("ticks")
            
            #Numeric vs Numeric
            #------------------------------------------------------------------------
            if x_type == 'is_numeric' and y_type == 'is_numeric':
                # Create scatterplot of dataframe
                #if colours have not been selected
                #................................................................
                if colour_headers == 'Select':
                    g = sns.lmplot(x=x, # Horizontal axis
                           y=y, # Vertical axis
                           data=data, # Data source
                           fit_reg=False, # Don't fix a regression line
                           scatter_kws={"marker": "D"},
                          legend = True)
                    plt.show()
                    
                #if colours have been selected
                #................................................................
                else:
                    g = sns.lmplot(x=x, # Horizontal axis
                           y=y, # Vertical axis
                           data=data, # Data source
                           fit_reg=False, # Don't fix a regression line
                           hue=colour_headers, # Set color
                           scatter_kws={"marker": "D"},# S marker size
                          legend = True)
                    plt.show()
                    
            #Numeric vs String
            #------------------------------------------------------------------------
            elif x_type == 'is_numeric' and y_type == 'is_string':
                sns.set_style("ticks")
                
                #if colours have not been selected
                #................................................................
                if colour_headers == 'Select':
                    g=sns.violinplot(x=x, y=y, data=data)
                    plt.show()
                #if colours have been selected   
                #................................................................
                else:
                    g=sns.violinplot(x=x, y=y, hue=colour_headers, data=data)
                    plt.show()
            #String vs Numeric
            #------------------------------------------------------------------------
            elif x_type == 'is_string' and y_type == 'is_numeric':
                
                #if colours have not been selected
                #................................................................
                if colour_headers == 'Select':
                    sns.set_style("ticks")
                    g=sns.boxplot(x=x, y=y, data=data)
                    plt.show()
                #if colours have been selected   
                #................................................................
                else:
                    sns.set_style("ticks")
                    g=sns.boxplot(x=x, y=y, hue=colour_headers, data=data)
                    plt.show()
                    

            #String vs String
            #------------------------------------------------------------------------
            elif x_type == 'is_string' and y_type == 'is_string':
                if headers_x != headers_y:
                    g=mosaic(data, [x, y])
                    plt.show()
                elif headers_x == headers_y:
                    g=sns.countplot(x=x, data=data)
                    loc, labels = plt.xticks()
                    g.set_xticklabels(labels, rotation=90)
                    plt.show()

        


# # Function for saving the created plots

# This function replicates the plot in the VISUALISATION tab and saves the plot to a PNG file in the same direcory of the notebook.

# In[12]:


def save_image(b):

        data = df
        
        #no selection made
        #========================================================================
        if headers_x.value == 'Select' and headers_y.value == 'Select':
            sns.set_context("notebook", font_scale=1.1)
        
        #X selected but noy Y
        #========================================================================
        elif headers_x.value != 'Select' and headers_y.value == 'Select':
            sns.set_context("notebook", font_scale=1.1)
            x=headers_x.value

            if df[headers_x.value].dtype == np.float or df[headers_x.value].dtype == np.int:
                x_type = 'is_numeric'
            elif df[headers_x.value].dtype == np.object:
                x_type = 'is_string'
            
            #plot when x is a string
            #--------------------------------------------------------------------
            if x_type == 'is_string':
                #if colours have not been selected
                #................................................................
                if colour_headers.value == 'Select':
                    
                        g=sns.countplot(x=x, data=data)
                        loc, labels = plt.xticks()
                        g.set_xticklabels(labels, rotation=90)
                        g.figure.savefig("xCategoricalNoColour.png")
                        plt.close()
                #if colours have been selected
                #................................................................
                else:
                        g=sns.countplot(x=x, hue=colour_headers.value, data=data)
                        loc, labels = plt.xticks()
                        g.set_xticklabels(labels, rotation=90)
                        g.figure.savefig("xCategoricalColour.png")
                        plt.close()
                        #plt.show()
                        
            #plot when x is numeric
            #--------------------------------------------------------------------
            else:
                #if colours have not been selected
                #................................................................
                if colour_headers.value == 'Select':
                    xplot = data[x]
                    g=sns.distplot(xplot)
                    g.figure.savefig("xNumericNoColour.png")
                    plt.close()
                    #plt.show()
                #if colours have been selected   
                #................................................................
                else:
                    g = sns.FacetGrid(data, hue=colour_headers.value)
                    g = g.map(sns.distplot, x)
                    g.savefig("xNumericColour.png")
                    plt.close()
                    #plt.show()
        #if only Y has been selected
        #========================================================================
        elif headers_x.value == 'Select' and headers_y.value != 'Select':
            sns.set_context("notebook", font_scale=1.1)
        #if both X and Y have been selected
        
        #========================================================================
        elif headers_x.value != 'Select' and headers_y.value != 'Select':
            x=headers_x.value
            y=headers_y.value

            if df[headers_x.value].dtype == np.float or df[headers_x.value].dtype == np.int:
                x_type = 'is_numeric'
            elif df[headers_x.value].dtype == np.object:
                x_type = 'is_string'

            if df[headers_y.value].dtype == np.float or df[headers_y.value].dtype == np.int:
                y_type = 'is_numeric'
            elif df[headers_y.value].dtype == np.object:
                y_type = 'is_string'
                
            sns.set_context("notebook", font_scale=1.1)
            sns.set_style("ticks")
            
            #Numeric vs Numeric
            #------------------------------------------------------------------------
            if x_type == 'is_numeric' and y_type == 'is_numeric':
                # Create scatterplot of dataframe
                #if colours have not been selected
                #................................................................
                if colour_headers.value == 'Select':
                    g = sns.lmplot(x=x, # Horizontal axis
                           y=y, # Vertical axis
                           data=data, # Data source
                           fit_reg=False, # Don't fix a regression line
                           scatter_kws={"marker": "D", # Set marker style
                                        "s": pointSize.value,
                                        "alpha":pointAlpha.value},# S marker size
                          legend = True)
                    g.savefig("NumericVsNumericNoColour.png")
                    plt.close()
                    #plt.show()
                    
                #if colours have been selected
                #................................................................
                else:
                    g = sns.lmplot(x=x, # Horizontal axis
                           y=y, # Vertical axis
                           data=data, # Data source
                           fit_reg=False, # Don't fix a regression line
                           hue=colour_headers.value, # Set color
                           scatter_kws={"marker": "D", # Set marker style
                                        "s": pointSize.value,
                                        "alpha":pointAlpha.value},# S marker size
                          legend = True)
                    g.savefig("NumericVsNumericColour.png")
                    plt.close()
                    #plt.show()
                    
            #Numeric vs String
            #------------------------------------------------------------------------
            elif x_type == 'is_numeric' and y_type == 'is_string':
                sns.set_style("ticks")
                g=sns.violinplot(x=x, y=y, data=data)
                g.figure.savefig("NumericVsCategorical.png")
                plt.close()
                #plt.show()
            #String vs Numeric
            #------------------------------------------------------------------------
            elif x_type == 'is_string' and y_type == 'is_numeric':
                sns.set_style("ticks")
                g=sns.boxplot(x=x, y=y, data=data)
                g.figure.savefig("CategoricalVsNumeric.png")
                plt.close()
                #plt.show()
            #String vs String
            #------------------------------------------------------------------------
            elif x_type == 'is_string' and y_type == 'is_string':
                plotting = mosaic(data, [x, y])
                plt.savefig('categoricalVsCategorical.png')
                plt.close()

        


# # Create widget button for saving plots in the VISUALISE tab

# This button widget controlls the saving of the plot. When it is clicked, it calles the save_image function.

# In[13]:



button = widgets.Button(description="Save Plot")
button.on_click(save_image)
display(button)


# # Create widgets for the DATA tab 

# This widget controls the display of the data as input.

# In[14]:


view_data = widgets.Dropdown(
            options=['df', 'original_data'],
            value='df',
            description='Data'
        )

view_data.set_title  = 'view_data'


# # Interaction function for the DATA tab

# This function displays the input data.

# In[15]:


@interact
def view_Data(view_data:view_data):
    data = globals()[view_data]
    display(data)


# # Create widgets for the PIVOT tab

# This widget controls the selection of the data to be manipulated in the Pivot tab.

# In[16]:


plot_data = widgets.Dropdown(
           options=['df', 'original_data'],
           value='df',
           description='Data'
       )

plot_data.set_title  = 'plot_data'


# # Interaction function for the PIVOT tab

# Function to manipulate data in the Pivot tab. This is manipulated through the library pivottablejs. Columns can be dragged into the X or Y axes and a pivot table is created. There are further options to plot the created pivot tables.

# In[17]:


@interact
def view_pivotTable(plot_data:plot_data):
    data = globals()[plot_data]
    display(pivot_ui(data))


# # Function to identidy the column type (string or numeric)

# Function to identify whether values in a specific column are nuremic or categorical.

# In[18]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


# # Create widgets for the FILTER tab 

# Widgets to manioulate the Filter tab. There are six widgets:
# 
# 1. headers_x: Name of the column which has the values to be filtered.
# 2. filter_options: options for the type of filtering:
#     * equals: dataframe keeps value(s) from the headers_x selection which are equal to the value(s) entered in the filter_input widget.
#     * notEqual: dataframe keeps value(s) from the headers_x selection which are NOT equal to the value(s) entered in the filter_input widget.
#     * largerThan: dataframe keeps value(s) from the headers_x selection which are larger than the value(s) entered in the filter_input widget.
#     * lowerThan: dataframe keeps value(s) from the headers_x selection which are lower than the value(s) entered in the filter_input widget.
# 3. filter_input: value to be filtered in the column selected at the headers_x.
# 4. manual_filter: this option is used if the user enters the filter by typing it instead of using the widgets. The advantage of a manual filter is that it can be as specific as desired by the user. For example, user can filter many columns at the same time.
# 5. overwrite_data: if the data has been filtered, the user has the option to overwrite the working dataframe and run the visualisation/analysis on this.
# 6. restore_data: if the data has been filtered and overwritten, the user has the option to restore the working data to the original input data.

# In[19]:


headers_x_filter = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Filter Column'
        )

headers_x_filter.set_title  = 'headers_x_filter'

filter_options = widgets.Dropdown(
            options=['equals', 'notEqual', 'largerThan', 'lowerThan'],
            value='equals',
            description='Filter options'
        )

filter_options.set_title  = 'filter_options'

filter_input = widgets.Text(
    value='Select',
    placeholder='Enter filter condition',
    description='Condition',
    disabled=False
)

filter_input.set_title  = 'filter_input'

manual_filter = widgets.Text(
    value='',
    placeholder='Type filter',
    description='Manual filter',
    disabled=False
)

manual_filter.set_title  = 'manual_filter'

overwrite_data = widgets.RadioButtons(
    options=['no', 'yes'],
    value='no',
    description='Overwrite',
    disabled=False
)

overwrite_data.set_title = 'overwrite_data'

restore_data = widgets.RadioButtons(
    options=['no', 'yes'],
    value='no',
    description='Restore',
    disabled=False,
    
)

restore_data.set_title = 'restore_data'


# # Interaction function for the FILTER tab

# Function to filter data based on user's requests

# In[20]:


@interact
def view_filterTable(headers_x_filter:headers_x_filter,
                     filter_options:filter_options,
                     filter_input:filter_input,
                     manual_filter:manual_filter,
                     overwrite_data:overwrite_data,
                     restore_data:restore_data):
        
        global df
        
        if restore_data == 'yes':
            df = original_data
        
        data = df
        
        filter_value = 'noInput'
        
        if manual_filter != '':
            filter_value = 'input'
                
        if filter_value == 'input':
            inside_df = data.query(manual_filter)
            display(inside_df)
            
            if overwrite_data == 'yes':
                df = inside_df
        else: 
            if is_number(filter_input):
                filter_input = float(filter_input)

            if headers_x_filter != 'Select' and filter_input != 'Select':
                if filter_options == 'equals':
                    inside_df = data[data[headers_x_filter] == filter_input]
                elif filter_options == 'notEqual':
                    inside_df = data[data[headers_x_filter] != filter_input]
                elif filter_options == 'largerThan':
                    inside_df = data[data[headers_x_filter] > filter_input]
                elif filter_options == 'lowerThan':
                    inside_df = data[data[headers_x_filter] < filter_input]

                display(inside_df)
                if overwrite_data == 'yes':
                    df = inside_df


# # Create widgets for the CONTINGENCY tables tab 

# Widgets to manipulate the Contingency tables tab. There are four widgets:
# 
# 1. summary_type: the type of summary for the contingency table
# 2. row_headers: name of columns containing the values to be plotted in the rows of the table
# 3. column_headers: name of columns containing the values to be plotted in the columns of the table
# 3. column_headers_extra: name of columns containing the values to be plotted in the columns of the table when there are three variables to be compared

# In[21]:


summary_type = widgets.Dropdown(
            options=['Original',
                     'Fitted Values',
                     'Residuals',
                     'p_value',
                     'chi2_contributions'],
            value='Original',
            description='Type'
        )

summary_type.set_title  = 'summary_type'

row_headers = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Row'
        )

row_headers.set_title  = 'row_headers'

column_headers = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Column'
        )

column_headers.set_title  = 'column_headers'

column_headers_extra = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Column 2'
        )

column_headers_extra.set_title  = 'column_headers_extra'


# # Interaction function for the CONTINGENCY tab

# Function to display the Contingency tables

# In[22]:


@interact
def view_Contingency(summary_type:summary_type,
                     row_headers:row_headers,
                     column_headers:column_headers,
                     column_headers_extra:column_headers_extra):

        data = df
        
        if row_headers != 'Select' and column_headers != 'Select' and column_headers_extra == 'Select':
            table = sm.stats.Table.from_data(data[[row_headers, column_headers]])
            
            if summary_type == 'Original':
                print(table.table_orig)
            elif summary_type == 'Fitted Values':
                print(table.fittedvalues)
            elif summary_type == 'Residuals':
                print(table.resid_pearson)
            elif summary_type == 'p_value':
                rslt = table.test_nominal_association()
                print(rslt.pvalue)
            elif summary_type == 'chi2_contributions':
                print(table.chi2_contribs)
    
        elif row_headers != 'Select' and column_headers != 'Select' and column_headers_extra != 'Select':
            table = pd.crosstab(index=data[row_headers], 
                             columns=[data[column_headers],
                                      data[column_headers_extra]],
                             margins=True)
            
            if summary_type == 'Original':
                print(table)
            else:
                print(table.T/table["All"])
                
            
            
            
                


# # Create widgets for the ANALYSIS tab 

# Widgets to control the Analysis tab. There are five widgets:
# 
# 1. model_type: the name of the model for the analysis. These can be grouped in three categories: linear models, generalized linear models and mixed effects models.
# 2. headers_dependent: name of columns containing the values to be entered as the dependent variable in the formula
# 3. headers_factor: name of columns containing the values to be entered as the independent factor in the formula
# 4. headers_groups: name of columns containing the values to be entered as the random factors in the formula. This only applies to mixed effects models
# 5. analysis_formula: the user is given the option to write the analysis formula. This overwrites any input in the widgets.

# In[23]:


model_type = widgets.Dropdown(
            options=['Ordinary Least Squares',
                     'Generalized Linear Models',
                     'Robust Linear Models',
                     'Linear Mixed Effects Models',
                     'Discrete - Regression with binary - Logit',
                     'Discrete - Regression with binary - Probit',
                     'Discrete - Regression with nominal - MNLogit',
                     'Discrete - Regression with count - Poisson'],
            value='Ordinary Least Squares',
            description='Model'
        )

model_type.set_title  = 'model_type'

headers_dependent = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Dependent'
        )

headers_dependent.set_title  = 'headers_dependent'

headers_factor = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Factor'
        )

headers_factor.set_title  = 'headers_factor'

headers_groups = widgets.Dropdown(
            options=df_cols,
            value=df_cols[0],
            description='Random'
        )

headers_groups.set_title  = 'headers_groups'

analysis_formula = widgets.Text(
    value='',
    placeholder='Enter formula',
    description='Formula',
    disabled=False
)

analysis_formula.set_title  = 'analysis_formula'


# # Interaction function for the ANALYSIS tab

# Function to do the analysis, based on the widgets in the Analysis tab.

# In[24]:


@interact
def view_Analysis(model_type:model_type,
                     headers_dependent:headers_dependent,
                     headers_factor:headers_factor,
                  headers_groups:headers_groups,
                  analysis_formula:analysis_formula):

        data = df
        
        mdl_string = 'noInput'
        
        if analysis_formula != '':
            mdl_string = analysis_formula
        else:
            if headers_dependent != 'Select' and headers_factor != 'Select':
                 mdl_string = headers_dependent + ' ~ ' + headers_factor
        
        if mdl_string != 'noInput':
            
            if analysis_formula != '':
                mdl_string = analysis_formula
            else:
                mdl_string = headers_dependent + ' ~ ' + headers_factor
            
            if model_type == 'Ordinary Least Squares':
                model = ols(mdl_string, data).fit()
            elif model_type == 'Generalized Linear Models':
                model = glm(mdl_string, data, family=sm.families.Gamma()).fit()
            elif model_type == 'Robust Linear Models':
                model = rlm(mdl_string, data, M=sm.robust.norms.HuberT()).fit()
            elif model_type == 'Linear Mixed Effects Models':
                if headers_groups != 'Select':
                    model = mixedlm(mdl_string, data, groups=data[headers_groups]).fit()
            elif model_type == 'Discrete - Regression with binary - Logit':
                model = Logit(data[headers_dependent], 
                              data[headers_factor].astype(float)).fit()
            elif model_type == 'Discrete - Regression with binary - Probit':
                model = Probit(data[headers_dependent], 
                              data[headers_factor].astype(float)).fit()
            elif model_type == 'Discrete - Regression with nominal - MNLogit':
                y = data[headers_factor]
                x = sm.add_constant(data[headers_dependent], prepend = False)
                model = sm.MNLogit(y, x).fit()
            elif model_type == 'Discrete - Regression with count - Poisson':
                model = Poisson(data[headers_dependent], 
                              data[headers_factor].astype(float)).fit()
                
            display(model.summary())


# # Assign tab contents

# This section assigns the different widgets and functions to a variable, which is passed to a box widget.

# In[25]:



#VISUALISATION
widgetInteract = interactive(view_image)
tab_visualise = widgets.HBox([widgetInteract,button])

#DATA
widgetData = interactive(view_Data)
tab_data = widgets.HBox([widgetData,])

#FILTER
widgetFilter = interactive(view_filterTable)
tab_filter = widgets.HBox([widgetFilter,])

#PIVOT
widgetPivot = interactive(view_pivotTable)
tab_pivot = widgets.HBox([widgetPivot,])

#CONTINGENCY
widgetContingency = interactive(view_Contingency)
tab_Contingency = widgets.HBox([widgetContingency,])

#ANALYSIS
widgetAnalyses = interactive(view_Analysis)
tab_Analyses = widgets.HBox([widgetAnalyses,])


# # Create the main interactive section

# This section puts together the previously created widgets and boxes into a single interactive section.

# In[26]:


tab_nest = widgets.Tab()
tab_nest.children = [tab_visualise,tab_data,widgetPivot,tab_filter,tab_Contingency,tab_Analyses]
tab_nest.set_title(0, 'Visualise')
tab_nest.set_title(1, 'Data')
tab_nest.set_title(2, 'Pivot')
tab_nest.set_title(3, 'Filter')
tab_nest.set_title(4, 'Contingency')
tab_nest.set_title(5, 'Analysis')
tab_nest

