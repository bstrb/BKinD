# tooltips.py

# Descriptions for tooltips
TOOLTIP_CRYSTAL_NAME = "Enter the name of the crystal for your 3DED diffraction data (only required for naming the output files)."

TOOLTIP_XDS_DIR = """Please specify the XDS directory that contains the files 
INTEGRATE.HKL, XDS_ASCII.HKL, and xds.inp."""

TOOLTIP_SHELX_DIR = {
    "default": "Please specify the SHELX directory that contains ONE .ins file",
    "xray": "Please specify the SHELX directory that contains at least one .ins file and one .hkl file"
}

TOOLTIP_OUTPUT_DIR = """Please specify the directory where you would like the output folder to be created.
This will be the location where all generated files and results will be saved."""

TOOLTIP_COMPLETENESS = """Description:
This parameter specifies the target final completeness level 
to achieve after completing the entire filtering process.

Purpose:
The parameter sets the stopping criterion for the iterative filtering process,
indicating when to halt once the desired completeness level is reached.

Example Value:
For instance, if set to 90 (the default value), the script will iteratively
filter the dataset until the data completeness is reduced to 90%."""

TOOLTIP_STEP_SIZE = """Description: 
This parameter specifies the incremental step size (in percentage points)
for intermediate stages during the filtering process. It does not affect the
final filtering outcome but provides intermediate steps that can be tracked,
for example, when plotting.

Purpose: 
This parameter controls the granularity of the output,
generating intermediate results that the user can inspect,
such as for plotting purposes.                          

Example Value: 
If set to 1 (the default value), each step of the filtering process
will reduce the current completeness by 1 percentage point. For instance,
if the starting completeness is 100% and the target is 90%, the procedure
will present 10 intermediate steps with a step size of 1 percentage point."""

TOOLTIP_FILTERING_PERCENTAGE = """Description: 
This parameter indicates the percentage of total data to be filtered out in each 
iterative step, based on their deviation from the mean value.

Purpose: 
Determines which proportion of the total dataset to be removed in each filtering iteration, 
focusing on removing the most extreme data points as defined by their deviation from the mean.

Example Value: 
If set to 0.1 (the default value), each iteration removes the 0.1% of the dataset that deviates
most from the mean, regardless of how much the completeness changes."""

TOOLTIP_WGHT_REFINEMENT = """Description: 
When enabled, the WGHT factor for filtered data is dynamically updated during an iterative SHELXL refinement process. 
This process continues until the WGHT factor stabilizes or a maximum of 10 iterations is reached."""
