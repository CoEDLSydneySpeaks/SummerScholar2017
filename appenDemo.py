
# coding: utf-8

# In[ ]:


#__author__ = "Simon Gonzalez and Nicole Harris"
#__year__ = "2018"
#__credits__ = ["Centre of Excellence for the Dynamics of Language, Sydney Speaks, Transcription Acceleration Project, Appen"]
#__license__ = "GPL"
#__version__ = "1.0.1"
#__maintainer__ = "Simon Gonzalez"
#__email__ = "simon.gonzalez@anu.edu.au"


# In[1]:


#import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import functools
import pandas as pd
import numpy as np
from ipywidgets import interact
from ipywidgets import *
import ipywidgets as widgets
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, push_notebook, show
from bokeh.palettes import brewer
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


# In[2]:


#read input data

fileInput = widgets.Text(
    #value='Me_Possessive.xlsx',
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


# In[3]:


#Import file
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


# In[4]:


#get data column names, all, numeric and categorical
df_cols = list(df.columns.values)
df_cols.insert(0, 'Select')
str_cols = list(df.select_dtypes(include=['object']).columns.values)
str_cols.insert(0, 'Select')
numeric_cols = list(df.select_dtypes(include=['number']).columns.values)
numeric_cols.insert(0, 'Select')


# In[5]:


#define widgets for the VISUALISE tab
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


# In[6]:


#interaction function for the VISUALISE tab
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
                mosaic(data, [x, y])
                plt.show()

        


# In[7]:


#function to save plots
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

        


# In[8]:


#define widgets for the VISUALISE tab - save button

button = widgets.Button(description="Save Plot")
button.on_click(save_image)
display(button)


# In[9]:


#function ton identidy whether a specifed column is a string or numeric
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


# In[10]:


#define widgets for the FILTER tab 
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


# In[11]:


#interaction function for the FILTER tab
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


# In[12]:


#define widgets for the ANALYSIS tab 
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
            description='Groups'
        )

headers_groups.set_title  = 'headers_groups'

analysis_formula = widgets.Text(
    value='',
    placeholder='Enter formula',
    description='Formula',
    disabled=False
)

analysis_formula.set_title  = 'analysis_formula'


# In[13]:


#interaction function for the ANALYSIS tab
@interact
def view_Analyses(model_type:model_type,
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


# In[14]:


#define widgets for the CONTINGENCY tables tab 
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


# In[15]:


#interaction function for the CONTINGENCY tab
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
                
            
            
            
                


# In[16]:


#define widgets for the DATA tab 
view_data = widgets.Dropdown(
            options=['df', 'original_data'],
            value='df',
            description='Data'
        )

view_data.set_title  = 'view_data'


# In[17]:


#interaction function for the DATA tab
@interact
def view_Data(view_data:view_data):
    data = globals()[view_data]
    display(data)


# In[18]:


#define widgets for the PIVOT tab 
plot_data = widgets.Dropdown(
            options=['df', 'original_data'],
            value='df',
            description='Data'
        )

plot_data.set_title  = 'plot_data'


# In[19]:


#interaction function for the PIVOT tab
@interact
def view_pivotTable(plot_data:plot_data):
    data = globals()[plot_data]
    display(pivot_ui(data))


# In[20]:


#assign widgets to tabs

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
widgetAnalyses = interactive(view_Analyses)
tab_Analyses = widgets.HBox([widgetAnalyses,])


# In[21]:


#create the main section of the app
tab_nest = widgets.Tab()
tab_nest.children = [tab_visualise,tab_data,tab_filter,widgetPivot,tab_Contingency,tab_Analyses]
tab_nest.set_title(0, 'Visualise')
tab_nest.set_title(1, 'Data')
tab_nest.set_title(2, 'Filter')
tab_nest.set_title(3, 'Pivot')
tab_nest.set_title(4, 'Contingency')
tab_nest.set_title(5, 'Analyses')
tab_nest

